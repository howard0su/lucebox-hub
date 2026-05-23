#!/usr/bin/env python3
"""Smoke-check qwen35moe daemon parity flows.

Exercises the dedicated qwen35moe daemon path through:
  1. generate
  2. SNAPSHOT + RESTORE
  3. park / generate-while-parked / unpark / generate
  4. compress

Exits non-zero on the first missing/incorrect protocol signal.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def write_counted_i32(path: Path, ids: list[int]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(ids)))
        if ids:
            f.write(struct.pack("<" + "i" * len(ids), *ids))


def write_uncounted_i32(path: Path, ids: list[int]) -> None:
    with path.open("wb") as f:
        if ids:
            f.write(struct.pack("<" + "i" * len(ids), *ids))


def read_counted_i32(path: Path) -> list[int]:
    with path.open("rb") as f:
        n_raw = f.read(4)
        if len(n_raw) != 4:
            return []
        n = struct.unpack("<I", n_raw)[0]
        if n == 0:
            return []
        data = f.read(4 * n)
        return list(struct.unpack("<" + "i" * n, data))


def read_until(proc: subprocess.Popen[str], predicate, timeout_s: float = 120.0) -> list[str]:
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


def send_and_wait(proc: subprocess.Popen[str], cmd: str, predicate, timeout_s: float = 120.0) -> list[str]:
    assert proc.stdin is not None
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()
    return read_until(proc, predicate, timeout_s)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=repo / "dflash/build/test_dflash")
    ap.add_argument("--target", type=Path, default=repo / "dflash/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    ap.add_argument("--draft", type=Path, default=repo / "dflash/models/draft/draft-Qwen3.6-35B-A3B.gguf")
    ap.add_argument("--compress-drafter", type=Path, default=repo / "dflash/models/Qwen3-0.6B-BF16.gguf")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="q35moe-parity-") as td:
        tmp = Path(td)
        prompt = [1, 2, 3, 4]
        prompt_counted = tmp / "prompt_counted.bin"
        prompt_uncounted = tmp / "prompt_uncounted.bin"
        out_counted = tmp / "out_counted.bin"
        out_after_unpark = tmp / "out_after_unpark.bin"
        restore_uncounted = tmp / "restore_uncounted.bin"

        write_counted_i32(prompt_counted, prompt)
        write_uncounted_i32(prompt_uncounted, prompt)

        proc = subprocess.Popen(
            [str(args.bin), str(args.target), str(args.draft), "--daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            read_until(proc, lambda line: "[daemon] ready" in line, 180.0)

            send_and_wait(
                proc,
                f"generate {prompt_counted} 1 {out_counted}",
                lambda line: line.startswith("ok N="),
                120.0,
            )
            gen = read_counted_i32(out_counted)
            if len(gen) != 1:
                raise RuntimeError(f"expected 1 generated token, got {gen}")

            send_and_wait(
                proc,
                "SNAPSHOT 0",
                lambda line: "[snap] inline slot=0" in line,
                30.0,
            )

            write_uncounted_i32(restore_uncounted, prompt + gen)
            send_and_wait(
                proc,
                f"RESTORE 0 {restore_uncounted} 1",
                lambda line: line.startswith("ok N=") and "RESTORE slot=0" in line,
                120.0,
            )

            send_and_wait(
                proc,
                "park target",
                lambda line: "[park] target released" in line,
                30.0,
            )
            send_and_wait(
                proc,
                f"generate {prompt_counted} 1 {tmp/'out_after_park.bin'}",
                lambda line: line.startswith("err target_parked"),
                30.0,
            )
            send_and_wait(
                proc,
                "unpark target",
                lambda line: "[unpark] target restored" in line,
                180.0,
            )
            send_and_wait(
                proc,
                f"generate {prompt_counted} 1 {out_after_unpark}",
                lambda line: line.startswith("ok N="),
                120.0,
            )
            if len(read_counted_i32(out_after_unpark)) != 1:
                raise RuntimeError("generate after unpark did not produce one token")

            send_and_wait(
                proc,
                f"compress {prompt_uncounted} 500 {args.compress_drafter}",
                lambda line: "[compress] " in line and " -> " in line and "tokens" in line,
                180.0,
            )

            assert proc.stdin is not None
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=30)
            if proc.returncode != 0:
                raise RuntimeError(f"daemon exited with {proc.returncode}")

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    print("qwen35moe parity smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
