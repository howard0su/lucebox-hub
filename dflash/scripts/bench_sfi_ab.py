#!/usr/bin/env python3
"""A/B benchmark: baseline (window-only) vs SFI (window + periodic refresh).

Runs `bench_niah_cpp.py` twice — once with `DFLASH27B_FA_REFRESH_INTERVAL=0`
(baseline, pure sliding window) and once with a positive interval (SFI slow
refresh), then prints a side-by-side comparison.

Usage (RTX 2080 Ti example):
    python dflash/scripts/bench_sfi_ab.py \
        --ctx 32768 --n 1 \
        --refresh-interval 4096 \
        --fa-window 4096 \
        --bin dflash/build/test_dflash \
        --target dflash/models/Qwen3.6-27B-Q4_K_M.gguf \
        --draft-spec dflash/models/draft/draft-Qwen3.6-27B.gguf \
        --drafter-gguf dflash/models/Qwen3-0.6B-BF16.gguf
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NIAH_GEN = ROOT / "pflash" / "tests" / "niah_gen.py"
BENCH_NIAH = ROOT / "pflash" / "tests" / "bench_niah_cpp.py"


def generate_cases(ctx: int, n: int, tokenizer: str) -> str:
    """Generate NIAH test cases and return path to JSONL file."""
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="sfi_bench_")
    os.close(fd)
    cmd = [
        sys.executable, str(NIAH_GEN),
        "--tokenizer", tokenizer,
        "--ctx", str(ctx),
        "--n", str(n),
        "--out", path,
    ]
    print(f"[gen] {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)
    return path


def run_bench(label: str, cases_path: str, args, refresh_interval: int,
              sfi_budget: int = 0) -> dict:
    """Run bench_niah_cpp.py with the given refresh interval and parse output."""
    env = os.environ.copy()
    env["DFLASH27B_FA_REFRESH_INTERVAL"] = str(refresh_interval)
    env["DFLASH27B_SFI_BUDGET"] = str(sfi_budget)
    env.pop("DFLASH_FP_USE_BSA", None)  # BSA off for 2080 Ti

    cmd = [
        sys.executable, str(BENCH_NIAH),
        "--cases", cases_path,
        "--n", str(args.n),
        "--bin", str(args.bin),
        "--target", str(args.target),
        "--draft-spec", str(args.draft_spec),
        "--drafter-gguf", str(args.drafter_gguf),
        "--fa-window", str(args.fa_window),
        "--kv-tq3", str(args.kv_tq3),
        "--bsa", "0",
        "--no-thinking",
    ]
    if args.auto_max_ctx:
        cmd.append("--auto-max-ctx")

    print(f"\n{'='*60}", flush=True)
    print(f"[{label}] refresh_interval={refresh_interval}", flush=True)
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    elapsed = time.time() - t0

    output = result.stdout + result.stderr
    print(output, flush=True)

    # Parse key metrics from output
    metrics = {"label": label, "wall_s": elapsed, "refresh_interval": refresh_interval}

    # Extract per-case metrics
    for line in output.split("\n"):
        m = re.search(r"score_s=([\d.]+)", line)
        if m:
            metrics["score_s"] = float(m.group(1))
        m = re.search(r"gen_s=([\d.]+)", line)
        if m:
            metrics["gen_s"] = float(m.group(1))
        m = re.search(r"ttft=([\d.]+)", line)
        if m:
            metrics["ttft"] = float(m.group(1))
        m = re.search(r"compressed=(\d+)", line)
        if m:
            metrics["compressed"] = int(m.group(1))
        m = re.search(r"ratio=([\d.]+)x", line)
        if m:
            metrics["ratio"] = float(m.group(1))
        m = re.search(r"accuracy:\s*(\d+)/(\d+)", line)
        if m:
            metrics["correct"] = int(m.group(1))
            metrics["total"] = int(m.group(2))

    if result.returncode != 0:
        metrics["error"] = True
        print(f"[{label}] FAILED (exit {result.returncode})", flush=True)

    return metrics


def print_comparison(baseline: dict, sfi: dict, ctx: int):
    """Print side-by-side comparison table."""
    print(f"\n{'='*70}")
    print(f"  SFI A/B Comparison @ {ctx//1024}K context")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Baseline':>15} {'SFI':>15} {'Delta':>15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")

    def row(name, key, unit="", lower_better=True):
        bv = baseline.get(key)
        sv = sfi.get(key)
        if bv is None or sv is None:
            print(f"{name:<25} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
            return
        delta = sv - bv
        pct = (delta / bv * 100) if bv != 0 else 0
        sign = "↓" if (delta < 0) == lower_better else "↑"
        qual = "✓" if (delta < 0) == lower_better else "✗"
        if not lower_better:
            qual = "✓" if delta > 0 else "✗"
            sign = "↑" if delta > 0 else "↓"
        print(f"{name:<25} {bv:>13.2f}{unit:>2} {sv:>13.2f}{unit:>2} "
              f"{sign}{abs(pct):>5.1f}% {qual}")

    row("Score time", "score_s", "s")
    row("Gen time", "gen_s", "s")
    row("TTFT", "ttft", "s")
    row("Wall time", "wall_s", "s")

    # Accuracy
    bc = baseline.get("correct", "?")
    bt = baseline.get("total", "?")
    sc = sfi.get("correct", "?")
    st = sfi.get("total", "?")
    print(f"{'Accuracy':<25} {str(bc)+'/'+str(bt):>15} {str(sc)+'/'+str(st):>15}")

    print(f"{'='*70}\n")


def main():
    ap = argparse.ArgumentParser(description="A/B benchmark: baseline vs SFI refresh")
    ap.add_argument("--ctx", type=int, default=32768, help="Context length in tokens")
    ap.add_argument("--n", type=int, default=1, help="Number of NIAH cases")
    ap.add_argument("--refresh-interval", type=int, default=4096,
                    help="SFI refresh interval (baseline always uses 0)")
    ap.add_argument("--sfi-budget", type=int, default=2048,
                    help="SFI sparse budget (0=disabled for baseline/refresh-only runs)")
    ap.add_argument("--fa-window", type=int, default=4096)
    ap.add_argument("--kv-tq3", type=int, choices=[0, 1], default=1)
    ap.add_argument("--auto-max-ctx", action="store_true", default=True)
    ap.add_argument("--bin", default="dflash/build/test_dflash")
    ap.add_argument("--target", default="dflash/models/Qwen3.6-27B-Q4_K_M.gguf")
    ap.add_argument("--draft-spec",
                    default="dflash/models/draft/draft-Qwen3.6-27B.gguf")
    ap.add_argument("--drafter-gguf",
                    default="dflash/models/Qwen3-0.6B-BF16.gguf")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out-json", default=None, help="Save results as JSON")
    args = ap.parse_args()

    print(f"[config] ctx={args.ctx}, n={args.n}, "
          f"refresh_interval={args.refresh_interval}, fa_window={args.fa_window}",
          flush=True)

    # Generate test cases
    cases_path = generate_cases(args.ctx, args.n, args.tokenizer)
    print(f"[cases] {cases_path}", flush=True)

    try:
        # Run baseline (no refresh, no SFI)
        baseline = run_bench("baseline", cases_path, args,
                             refresh_interval=0, sfi_budget=0)

        # Run refresh-only (periodic full attention, no sparse gather)
        refresh_only = run_bench("refresh-only", cases_path, args,
                                 refresh_interval=args.refresh_interval,
                                 sfi_budget=0)

        # Run full SFI (refresh + sparse gather)
        sfi_result = run_bench("sfi-full", cases_path, args,
                               refresh_interval=args.refresh_interval,
                               sfi_budget=args.sfi_budget)

        # Print comparisons
        print_comparison(baseline, refresh_only, args.ctx)
        print_comparison(baseline, sfi_result, args.ctx)

        # Save JSON if requested
        if args.out_json:
            results = {
                "ctx": args.ctx,
                "n": args.n,
                "refresh_interval": args.refresh_interval,
                "sfi_budget": args.sfi_budget,
                "fa_window": args.fa_window,
                "baseline": baseline,
                "refresh_only": refresh_only,
                "sfi_full": sfi_result,
            }
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[saved] {args.out_json}", flush=True)

    finally:
        os.unlink(cases_path)


if __name__ == "__main__":
    main()
