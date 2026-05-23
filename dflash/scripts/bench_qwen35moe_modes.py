#!/usr/bin/env python3
"""Benchmark qwen35moe baseline vs hybrid-placement daemon modes.

This compares:
  - baseline qwen35moe daemon (all-GPU + draft/spec path)
  - hybrid qwen35moe daemon (placement file enabled; current AR-only policy)

The script writes a tiny counted prompt, runs one generate command against each
mode, parses the daemon status line, and prints a compact comparison table.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


OK_RE = re.compile(
    r"ok N=(?P<n>\d+) gen=(?P<gen>\d+) "
    r"prefill_s=(?P<prefill>[0-9.]+) "
    r"decode_s=(?P<decode>[0-9.]+) "
    r"decode_tok_s=(?P<toks>[0-9.]+)"
)


def write_counted_i32(path: Path, ids: list[int]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(ids)))
        if ids:
            f.write(struct.pack("<" + "i" * len(ids), *ids))


def read_until(proc: subprocess.Popen[str], predicate, timeout_s: float = 180.0) -> list[str]:
    import time

    lines: list[str] = []
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line == "":
            raise RuntimeError("daemon exited before expected output")
        lines.append(line.rstrip("\n"))
        if predicate(line):
            return lines
    raise TimeoutError("timed out waiting for daemon output")


def run_mode(bin_path: Path,
             target: Path,
             draft: Path,
             prompt_counted: Path,
             n_gen: int,
             placement: Path | None) -> dict[str, float | int | str]:
    cmd = [str(bin_path), str(target), str(draft), "--daemon"]
    env = None
    if placement is not None:
        env = dict(**dict(**subprocess.os.environ))
        env["DFLASH_QWEN35MOE_PLACEMENT"] = str(placement)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        ready_lines = read_until(proc, lambda line: "[daemon] ready" in line)
        mode = "hybrid" if placement else "baseline"
        out_path = prompt_counted.parent / f"out_{mode}.bin"

        assert proc.stdin is not None
        proc.stdin.write(f"generate {prompt_counted} {n_gen} {out_path}\n")
        proc.stdin.write("quit\n")
        proc.stdin.flush()

        lines = read_until(proc, lambda line: line.startswith("ok N="), 300.0)
        match = None
        for line in reversed(lines):
            m = OK_RE.search(line)
            if m:
                match = m
                break
        if not match:
            raise RuntimeError("failed to parse daemon status line")
        proc.wait(timeout=30)
        return {
            "mode": mode,
            "prefill_s": float(match.group("prefill")),
            "decode_s": float(match.group("decode")),
            "decode_tok_s": float(match.group("toks")),
            "n_prompt": int(match.group("n")),
            "n_gen": int(match.group("gen")),
            "ready_lines": len(ready_lines),
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=repo / "dflash/build/test_dflash")
    ap.add_argument("--target", type=Path, default=repo / "dflash/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    ap.add_argument("--draft", type=Path, default=repo / "dflash/models/draft/draft-Qwen3.6-35B-A3B.gguf")
    ap.add_argument("--placement", type=Path, default=Path("/tmp/q35moe-hybrid-5nVjld/placement.json"))
    ap.add_argument("--n-gen", type=int, default=4)
    ap.add_argument("--prompt", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    ids = [int(tok) for tok in args.prompt.split(",") if tok.strip()]
    with tempfile.TemporaryDirectory(prefix="q35moe-bench-") as td:
        tmp = Path(td)
        prompt_counted = tmp / "prompt_counted.bin"
        write_counted_i32(prompt_counted, ids)

        baseline = run_mode(args.bin, args.target, args.draft, prompt_counted, args.n_gen, None)
        hybrid = run_mode(args.bin, args.target, args.draft, prompt_counted, args.n_gen, args.placement)

        result = {
            "baseline": baseline,
            "hybrid": hybrid,
        }
        if args.json_out:
            args.json_out.write_text(json.dumps(result, indent=2))

        print("| mode | prefill_s | decode_s | decode_tok_s |")
        print("|---|---:|---:|---:|")
        for row in (baseline, hybrid):
            print(f"| {row['mode']} | {row['prefill_s']:.3f} | {row['decode_s']:.3f} | {row['decode_tok_s']:.2f} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
