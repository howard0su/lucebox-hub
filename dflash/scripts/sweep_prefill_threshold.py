#!/usr/bin/env python3
"""Sweep --prefill-threshold to find the fast/slow sweet spot.

This runner starts `scripts/server.py` once per threshold, replays a fixed
prompt matrix (short/medium/long), and chooses the winner by:
  1) quality gate (non-empty response rate),
  2) best long-context mean latency,
  3) long-context throughput tie-break.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = ROOT / "scripts" / "server.py"
DEFAULT_TARGET = Path(os.environ.get("DFLASH_TARGET", str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf")))
DEFAULT_DRAFT = Path(os.environ.get("DFLASH_DRAFT", str(ROOT / "models" / "draft")))
DEFAULT_BIN = Path(os.environ.get("DFLASH_BIN", str(ROOT / "build" / "test_dflash")))
DEFAULT_DRAFTER = Path(os.environ.get("DFLASH_PREFILL_DRAFTER", str(ROOT / "models" / "Qwen3-0.6B-BF16.gguf")))
DEFAULT_TOKENIZER = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.6-27B")


@dataclass(frozen=True)
class SweepConfig:
    max_ctx: int = 131072
    budget: int = 22
    fa_window: int = 0
    keep_ratio: float = 0.05
    alpha: float = 0.70
    bsa_enabled: int = 1
    n_requests_per_bucket: int = 2
    max_tokens: int = 64
    timeout_s: int = 600
    min_non_empty_rate: float = 0.95


def _http_json(url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _wait_server_up(port: int, proc: subprocess.Popen, timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            _http_json(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionResetError):
            time.sleep(1)
    return False


def _build_prompt(tokenizer, target_tokens: int) -> str:
    filler = (
        "This is a long-context benchmarking filler paragraph about GPU inference, "
        "attention windows, KV cache behavior, and latency tradeoffs. "
    )
    text = filler
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < target_tokens:
        text += filler
        ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=False)


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _start_server(
    *,
    port: int,
    threshold: int,
    threshold_exit: int | None,
    cfg: SweepConfig,
    target: Path,
    draft: Path,
    bin_path: Path,
    drafter: Path,
    tokenizer_id: str,
    log_path: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-u",
        str(SERVER_SCRIPT),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--target",
        str(target),
        "--draft",
        str(draft),
        "--bin",
        str(bin_path),
        "--budget",
        str(cfg.budget),
        "--max-ctx",
        str(cfg.max_ctx),
        "--fa-window",
        str(cfg.fa_window),
        "--tokenizer",
        tokenizer_id,
        "--prefill-compression",
        "auto",
        "--prefill-threshold",
        str(threshold),
        "--prefill-keep-ratio",
        str(cfg.keep_ratio),
        "--prefill-drafter",
        str(drafter),
    ]
    if threshold_exit is not None:
        cmd.extend(["--prefill-threshold-exit", str(threshold_exit)])

    env = os.environ.copy()
    env["DFLASH_FP_USE_BSA"] = str(cfg.bsa_enabled)
    env["DFLASH_FP_ALPHA"] = str(cfg.alpha)
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    proc._sweep_log_file = log_f  # type: ignore[attr-defined]
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    finally:
        log_f = getattr(proc, "_sweep_log_file", None)
        if log_f:
            log_f.close()


def _run_threshold(
    *,
    threshold: int,
    threshold_exit: int | None,
    port: int,
    prompts_by_bucket: dict[str, list[str]],
    cfg: SweepConfig,
    target: Path,
    draft: Path,
    bin_path: Path,
    drafter: Path,
    tokenizer_id: str,
) -> dict[str, Any]:
    log_path = Path(f"/tmp/sweep_prefill_threshold_{threshold}.log")
    proc = _start_server(
        port=port,
        threshold=threshold,
        threshold_exit=threshold_exit,
        cfg=cfg,
        target=target,
        draft=draft,
        bin_path=bin_path,
        drafter=drafter,
        tokenizer_id=tokenizer_id,
        log_path=log_path,
    )
    if not _wait_server_up(port, proc):
        _stop_server(proc)
        raise RuntimeError(f"server failed to start for threshold={threshold} (log={log_path})")

    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in prompts_by_bucket}
    try:
        for bucket, prompts in prompts_by_bucket.items():
            for prompt in prompts:
                payload = {
                    "model": "luce-dflash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": cfg.max_tokens,
                    "stream": False,
                }
                t0 = time.time()
                try:
                    resp = _http_json(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        payload=payload,
                        timeout=cfg.timeout_s,
                    )
                    dt = time.time() - t0
                    msg = resp["choices"][0]["message"].get("content", "")
                    usage = resp.get("usage", {}) or {}
                    completion_tokens = int(usage.get("completion_tokens", 0))
                    rows[bucket].append({
                        "latency_s": dt,
                        "completion_tokens": completion_tokens,
                        "non_empty": bool(str(msg).strip()),
                        "tokens_per_s": (completion_tokens / dt) if dt > 0 else 0.0,
                    })
                except Exception as exc:
                    rows[bucket].append({
                        "latency_s": float("inf"),
                        "completion_tokens": 0,
                        "non_empty": False,
                        "tokens_per_s": 0.0,
                        "error": str(exc),
                    })

        policy = _http_json(f"http://127.0.0.1:{port}/v1/debug/prefill-policy", timeout=10).get("prefill_policy", {})
    finally:
        _stop_server(proc)

    bucket_summary: dict[str, dict[str, float]] = {}
    all_rows = [r for bucket_rows in rows.values() for r in bucket_rows]
    for bucket, bucket_rows in rows.items():
        bucket_summary[bucket] = {
            "mean_latency_s": _mean([float(r["latency_s"]) for r in bucket_rows]),
            "mean_tokens_per_s": _mean([float(r["tokens_per_s"]) for r in bucket_rows]),
            "non_empty_rate": _mean([1.0 if r["non_empty"] else 0.0 for r in bucket_rows]),
        }

    return {
        "threshold": threshold,
        "threshold_exit": threshold_exit,
        "policy": policy,
        "bucket_summary": bucket_summary,
        "overall_non_empty_rate": _mean([1.0 if r["non_empty"] else 0.0 for r in all_rows]),
        "rows": rows,
    }


def _pick_winner(results: list[dict[str, Any]], cfg: SweepConfig) -> dict[str, Any] | None:
    valid = [
        r for r in results
        if float(r.get("overall_non_empty_rate", 0.0)) >= cfg.min_non_empty_rate
    ]
    if not valid:
        return None
    valid.sort(
        key=lambda r: (
            float(r["bucket_summary"]["long"]["mean_latency_s"]),
            -float(r["bucket_summary"]["long"]["mean_tokens_per_s"]),
        )
    )
    return valid[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep prefill threshold sweet spot")
    ap.add_argument("--thresholds", type=int, nargs="+",
                    default=[8000, 16000, 24000, 32000, 40000, 48000, 64000])
    ap.add_argument("--threshold-exit", type=int, default=None,
                    help="Optional fixed exit threshold for hysteresis during sweep.")
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--prefill-drafter", type=Path, default=DEFAULT_DRAFTER)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--port-base", type=int, default=18900)
    ap.add_argument("--out", type=Path, default=Path("/tmp/sweep_prefill_threshold.json"))
    ap.add_argument("--max-ctx", type=int, default=131072)
    ap.add_argument("--budget", type=int, default=22)
    ap.add_argument("--fa-window", type=int, default=0)
    ap.add_argument("--keep-ratio", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--requests-per-bucket", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--min-non-empty-rate", type=float, default=0.95)
    args = ap.parse_args()

    cfg = SweepConfig(
        max_ctx=args.max_ctx,
        budget=args.budget,
        fa_window=args.fa_window,
        keep_ratio=args.keep_ratio,
        alpha=args.alpha,
        n_requests_per_bucket=args.requests_per_bucket,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        min_non_empty_rate=args.min_non_empty_rate,
    )

    if not args.target.is_file():
        raise SystemExit(f"target not found: {args.target}")
    if not args.bin.is_file():
        raise SystemExit(f"binary not found: {args.bin}")
    if not args.prefill_drafter.is_file():
        raise SystemExit(f"prefill drafter GGUF not found: {args.prefill_drafter}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    prompts_by_bucket: dict[str, list[str]] = {
        "short": [],
        "medium": [],
        "long": [],
    }
    for _ in range(cfg.n_requests_per_bucket):
        prompts_by_bucket["short"].append(_build_prompt(tok, 2000))
        prompts_by_bucket["medium"].append(_build_prompt(tok, 16000))
        prompts_by_bucket["long"].append(_build_prompt(tok, 64000))

    results: list[dict[str, Any]] = []
    for idx, threshold in enumerate(args.thresholds):
        port = args.port_base + idx
        print(f"[sweep] threshold={threshold} port={port}", flush=True)
        res = _run_threshold(
            threshold=threshold,
            threshold_exit=args.threshold_exit,
            port=port,
            prompts_by_bucket=prompts_by_bucket,
            cfg=cfg,
            target=args.target,
            draft=args.draft,
            bin_path=args.bin,
            drafter=args.prefill_drafter,
            tokenizer_id=args.tokenizer,
        )
        results.append(res)
        print(
            "[sweep] long mean latency="
            f"{res['bucket_summary']['long']['mean_latency_s']:.2f}s "
            f"non_empty={res['overall_non_empty_rate']:.2%} "
            f"compress={res.get('policy', {}).get('compress', 0)}/"
            f"{res.get('policy', {}).get('total', 0)}",
            flush=True,
        )

    winner = _pick_winner(results, cfg)
    output = {
        "config": asdict(cfg),
        "thresholds": args.thresholds,
        "threshold_exit": args.threshold_exit,
        "winner_threshold": winner["threshold"] if winner else None,
        "winner": winner,
        "results": results,
    }
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {args.out}")
    if winner:
        print(
            f"[sweep] winner={winner['threshold']} "
            f"long_mean_latency={winner['bucket_summary']['long']['mean_latency_s']:.2f}s"
        )
    else:
        print("[sweep] no winner passed quality gate", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

