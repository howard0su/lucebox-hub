#!/usr/bin/env python3
"""OpenAI-compatible HTTP server on top of test_laguna_daemon.

Mirrors the surface of scripts/server.py for the qwen35 stack but speaks the
simpler `generate <in> <n_gen> <out>` stdin protocol of test_laguna_daemon.
Greedy by default; sampler params (temperature/top_p/top_k/seed/freq_pen) are
appended as ` samp=` tail and parsed by the daemon.

Usage:
    python3 scripts/laguna_serve.py \\
        --target /root/models/laguna-xs2-Q4_K_M.gguf \\
        --laguna-tok /root/models/Laguna_XS_2 \\
        --laguna-bin ./build/test_laguna_daemon \\
        --port 8000

    curl http://localhost:8000/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model":"luce-laguna",
             "messages":[{"role":"user","content":"Hi"}],
             "max_tokens":32}'

Doesn't yet do PFlash compression (caller must trim long prompts) or
speculative decode (no Laguna draft model exists). Single in-flight request.
"""
from __future__ import annotations
import argparse, asyncio, json, os, struct, subprocess, sys, tempfile, time, uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL_NAME = "luce-laguna"


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    chat_template_kwargs: dict | None = None


def _content_to_str(content):
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _samp_suffix(req: ChatRequest) -> str:
    t = float(req.temperature or 0.0)
    if t <= 0.0:
        return ""
    tp = float(req.top_p or 1.0)
    tk = int(req.top_k or 0)
    rp = float(req.frequency_penalty or 0.0) + 1.0
    seed = int(req.seed or 0)
    return f" samp={t:.4f},{tp:.4f},{tk},{rp:.4f},{seed}"


class LagunaDaemon:
    """Single-process wrapper around test_laguna_daemon's stdin protocol."""
    def __init__(self, bin_path: Path, target: Path, max_ctx: int, kv: str, chunk: int):
        cmd = [str(bin_path), str(target),
               "--max-ctx", str(max_ctx),
               "--kv", kv,
               "--chunk", str(chunk)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        for _ in range(200):
            line = self.proc.stdout.readline().decode(errors="replace")
            if not line:
                raise RuntimeError("laguna daemon exited before ready")
            sys.stderr.write("  laguna init: " + line)
            if line.startswith("[laguna-daemon] ready"):
                return
        raise RuntimeError("test_laguna_daemon did not become ready")

    def generate(self, ids: list[int], n_gen: int, samp_suffix: str) -> tuple[list[int], dict]:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            in_path = Path(f.name)
        out_path = in_path.with_suffix(".out.bin")
        with in_path.open("wb") as f:
            f.write(struct.pack("<I", len(ids)))
            f.write(struct.pack(f"<{len(ids)}i", *ids))
        cmd = f"generate {in_path} {n_gen} {out_path}{samp_suffix}\n"
        self.proc.stdin.write(cmd.encode())
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().decode(errors="replace").strip()
        if not line.startswith("ok"):
            in_path.unlink(missing_ok=True)
            raise RuntimeError(f"laguna daemon: {line}")
        stats = {}
        for kv in line.split()[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                stats[k] = v
        data = out_path.read_bytes()
        n = struct.unpack_from("<I", data, 0)[0]
        out = list(struct.unpack_from(f"<{n}i", data, 4))
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        return out, stats

    def close(self):
        try:
            self.proc.stdin.write(b"quit\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def build_app(daemon: LagunaDaemon, tokenizer: AutoTokenizer, max_ctx: int) -> FastAPI:
    app = FastAPI(title="Luce Laguna OpenAI server")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                        allow_methods=["*"], allow_headers=["*"])
    daemon_lock = asyncio.Lock()

    @app.get("/health")
    def health():
        if daemon.proc.poll() is not None:
            return JSONResponse({"status": "error", "detail": "daemon exited"}, status_code=503)
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{
            "id": MODEL_NAME, "object": "model", "owned_by": "luce",
            "created": 1700000000,
            "context_length": max_ctx, "max_context_length": max_ctx,
        }]}

    def _render_prompt(req: ChatRequest) -> list[int]:
        msgs = [{"role": m.role, "content": _content_to_str(m.content)} for m in req.messages]
        kwargs = {"add_generation_prompt": True}
        if req.chat_template_kwargs:
            kwargs.update({k: v for k, v in req.chat_template_kwargs.items()
                            if k in ("enable_thinking", "tools", "add_generation_prompt")})
        text = tokenizer.apply_chat_template(msgs, tokenize=False, **kwargs)
        return tokenizer.encode(text, add_special_tokens=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest, raw: Request):
        prompt_ids = _render_prompt(req)
        n_gen = max(1, min(req.max_tokens, max_ctx - len(prompt_ids) - 4))
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        samp = _samp_suffix(req)

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    head = {"id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": MODEL_NAME,
                            "choices": [{"index": 0, "delta": {"role": "assistant"},
                                          "finish_reason": None}]}
                    yield f"data: {json.dumps(head)}\n\n"
                    try:
                        gen_ids, _stats = await asyncio.to_thread(
                            daemon.generate, prompt_ids, n_gen, samp)
                    except Exception as e:
                        err = {"error": str(e)}
                        yield f"data: {json.dumps(err)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    for tok_id in gen_ids:
                        delta = tokenizer.decode([tok_id])
                        chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                 "created": created, "model": MODEL_NAME,
                                 "choices": [{"index": 0, "delta": {"content": delta},
                                              "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                    tail = {"id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": MODEL_NAME,
                            "choices": [{"index": 0, "delta": {},
                                          "finish_reason": "stop"}]}
                    yield f"data: {json.dumps(tail)}\n\n"
                    yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        async with daemon_lock:
            gen_ids, stats = await asyncio.to_thread(daemon.generate, prompt_ids, n_gen, samp)
        text = tokenizer.decode(gen_ids)
        return {
            "id": completion_id, "object": "chat.completion",
            "created": created, "model": MODEL_NAME,
            "choices": [{"index": 0, "finish_reason": "stop",
                          "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": len(prompt_ids),
                      "completion_tokens": len(gen_ids),
                      "total_tokens": len(prompt_ids) + len(gen_ids)},
            "x_laguna_stats": stats,
        }

    @app.on_event("shutdown")
    def _close():
        daemon.close()

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--laguna-tok", required=True, type=Path,
                     help="HF tokenizer dir (NO dots in path, transformers parses dots as repo_id).")
    ap.add_argument("--laguna-bin", required=True, type=Path)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--kv", choices=["q4_0", "q5_0", "q8_0", "f16"], default="q8_0")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"[laguna-serve] loading tokenizer {args.laguna_tok}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(args.laguna_tok), trust_remote_code=True)

    print(f"[laguna-serve] starting daemon", file=sys.stderr)
    daemon = LagunaDaemon(args.laguna_bin, args.target, args.max_ctx, args.kv, args.chunk)

    app = build_app(daemon, tok, args.max_ctx)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
