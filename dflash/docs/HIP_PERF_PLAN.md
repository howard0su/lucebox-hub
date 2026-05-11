# HIP perf — diagnosis + kernel-side optimization plan

_Drafted 2026-05-11 against `Luce-Org/lucebox-hub` post-#122 with HIP/ROCm
support landing in the upcoming PR. Numbers below are from the canonical
DFlash bench (Qwen3.6-27B-Q4_K_M + z-lab DFlash drafter,
`--fast-rollback --ddtree --ddtree-budget=22`, HE-style 128-tok prompt
md5 `4280413edc0b45c2b09e1a45f4f5ee60`, n_gen=256, warmup + 2 measurement
runs)._

## tl;dr

Lucebox HIP decode on gfx1100 (7900 XTX) runs **3.22× over AR** today —
within 6% of the README's CUDA RTX 3090 3.43× anchor. That's the headline
DFlash speedup. But the **absolute throughput is 50 tok/s on gfx1100 vs
~200 tok/s on hipfire's RDNA-native engine** on the same physical card
(same prompt, same target, same context). The 4× gap is **not** in
attention — it's in `mul_mat_q` for `q4_K` / `q4_0` / `q5_0`. The fix is
upstream in `ggml-cuda/mmq.cuh` + `mmvq.cuh`.

Tier 1 of this plan is **already empirically verified**: setting
`--ddtree-budget=8` instead of the default 22 on gfx1100 lifts decode
from 49.81 tok/s to 76.02 tok/s — a **53% speedup from a single config
flag**, no kernel work. Same flag is a -9% regression on gfx1201, so the
ship is arch-aware. Details in the Tier 1 section below.

This doc traces the rocprofv3 evidence, identifies the dispatch decisions
that route lucebox onto the slow path, and proposes a four-tier
optimization plan for the lucebox `llama.cpp-dflash-ggml` fork.

## rocprofv3 top-10 hot kernels on gfx1100

Captured via `scripts/lucebox_kernel_atlas.py` (kernel-trace + summary +
ISA manifest) on the canonical DFlash bench above. Total profiled wall
~112s; per-kernel totals below cover all 256 generated tokens.

| Time   | Calls | Kernel | Notes |
|-------:|------:|---|---|
| 2076 ms | 1820 | `mul_mat_q<q4_K, 32, false>` | **target q4_K matmul, DDTree batch-tile 32** |
| 1247 ms | 8064 | `mul_mat_q<q4_0, 32, false>` | KV cache q4_0 matmul |
|  741 ms | 3456 | `Cijk_Alik_Bljk_SB_MT64x64x8_SN_1LDSB0_...` | rocBLAS strided batched GEMM (no WMMA) |
|  211 ms | 2304 | `mul_mat_q<q4_0, 16, false>` | smaller-tile MMQ |
|  205 ms | 1344 | `mul_mat_q<q5_0, 32, false>` | MMQ q5_0 |
|  130 ms |  420 | `Cijk_Alik_Bljk_HB_MT64x64x32_MI16x16x16x1_...` | rocBLAS GEMM **with** WMMA |
|  125 ms |  540 | `mul_mat_q<q4_K, 16, false>` | smaller-tile MMQ for q4_K |
|   72 ms | 1344 | `gated_delta_net_cuda<128, false, true, __half>` | DeltaNet hybrid path |
|   47 ms |  149 | `Cijk_..._MI16x16x16x1_...` | rocBLAS WMMA, second shape |
|   27 ms |  448 | `flash_attn_tile<256, 256, 32, 1, false>` | **FA tile — 0.5% of total** |

**~76% of GPU time is `mul_mat_q` variants. FlashAttention is 0.5%.**
The "huge tax" frame is correct, but it doesn't live in the missing
`flashprefill_kernels.hip.cu` (that path is short-prompt-cold here) — it
lives in `mmq.cuh`'s `q4_K` MMA path on RDNA3+.

## The dispatch trace — why DDTree always lands on MMQ

`ggml-cuda.cu:2294` decides MMVQ vs MMQ:

```cpp
bool use_mul_mat_vec_q = ggml_is_quantized(src0->type)
                        && ... && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
```

`MMVQ_MAX_BATCH_SIZE = 8` (`mmvq.cuh:3`).

DDTree budget=22 → speculation batch = 22 → `src1->ne[1] = 22 > 8` →
**always falls to MMQ.** MMQ uses WMMA on RDNA3+ (via
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` in `mma.cuh`) but the tile
shape is `32 × mmq_y × K` — designed for big-batch prefill. On a 22-wide
spec-verify batch it does 32 columns of work and discards 10 → **31%
wasted GPU compute on every speculation step**.

For the non-spec AR baseline (batch=1 decode), MMVQ kicks in correctly,
which is why the AR baseline at 28 tok/s is closer to hipfire's per-call
throughput than DFlash's 50 tok/s.

## Where MMVQ stops scaling — the second hard wall

`mmvq.cu:calc_nwarps()` and `calc_rows_per_block()` are explicitly tuned
for `ncols_dst ∈ {1..8}`. The fall-through for `ncols_dst ≥ 9` returns
`nwarps=1, rows_per_block=1` — i.e. no parallelism, one batch per warp,
launch-overhead-bound. Even if you bumped `MMVQ_MAX_BATCH_SIZE` to 32,
the kernel would behave badly because the per-arch
`MMVQ_PARAMETERS_RDNA3_0` / `RDNA4` cases gate on `ncols_dst == 1`
specifically (`mmvq.cu:326-345`).

So MMVQ as-shipped is not a drop-in fix for the 22-batch spec-verify
shape. It needs new instantiations.

## Four-tier optimization plan

### Tier 1 — Config-only (15 min, zero risk, **arch-specific**)

Try `--ddtree-budget=8` on the HIP backend. Routes spec-verify through
MMVQ instead of MMQ.

**Empirically validated 2026-05-11, n_gen=256 on the canonical HE bench
above, warmup + 2 measurement runs each:**

| Arch | Card | budget=22 (MMQ) | budget=8 (MMVQ) | Delta |
|---|---|---:|---:|---:|
| gfx1100 | 7900 XTX | 49.81 tok/s | **76.02 tok/s** | **+53%** |
| gfx1151 | Strix Halo iGPU | **34.78 tok/s** | 30.71 tok/s | -13% |
| gfx1201 | R9700 | **84.70 tok/s** | 77.23 tok/s | -9% |

The win is **gfx110x-only** (vanilla RDNA3 desktop dGPUs: 7900 XTX/XT,
7800 XT, 7700 XT/S, 7600). RDNA3.5 (Strix Halo gfx1151) and RDNA4
(gfx1201) both prefer MMQ at budget=22 — likely a combination of:

- RDNA4 has well-tuned MMQ tile shapes that make wasted columns of a
  batch-32 tile cheap proportionally.
- RDNA3.5 Strix Halo's LPDDR5X UMA (~270 GB/s vs 7900 XTX's 960 GB/s
  GDDR6) makes one-big-MMQ-launch's launch-amortization more valuable
  than tile-utilization. MMVQ's per-batch separate launches hurt UMA.

The dispatch analysis is correct on all three archs; the threshold is
just on the wrong side specifically for the desktop RDNA3 SKUs.

**Suggested ship**: arch-aware default in the daemon's CLI parsing or
`server.py` — set `--ddtree-budget=8` when running on gfx1100, gfx1101,
gfx1102 (desktop RDNA3 only). Keep 22 on gfx115x (RDNA3.5), gfx120x
(RDNA4), and CUDA. Single-PR change, zero kernel work, recovers most of
the gfx110x-specific gap.

```bash
./test_dflash $T $D prompt.bin 256 out.bin --fast-rollback --ddtree --ddtree-budget=22  # current default
./test_dflash $T $D prompt.bin 256 out.bin --fast-rollback --ddtree --ddtree-budget=8   # MMVQ-routed
```

### Tier 2 — Extend MMVQ template instantiations to `ncols_dst ∈ {9..32}` (3-5 days)

In `mmvq.cu`:

- Add `MMVQ_PARAMETERS_RDNA3_0` + `MMVQ_PARAMETERS_RDNA4` cases for
  `ncols_dst ∈ {16, 24, 32}` with `nwarps ∈ {2, 4}` and
  `rows_per_block ∈ {2, 4}`.
- Bump `MMVQ_MAX_BATCH_SIZE` to 32 (or add a per-type ceiling like
  `get_mmvq_max_batch(type, cc)` parallel to `get_mmvq_mmid_max_batch_*`).
- Template instantiations for `q4_K`, `q5_K`, `q6_K`, `q4_0`, `q5_0`,
  `q8_0` × `ncols_dst ∈ {16, 24, 32}` × `RDNA3+` only.

This lets DDTree budget ≤ 32 stay on the GEMV path. Expected impact:
**~1.5-2× on DFlash spec-verify** (cutting the dominant
`mul_mat_q<q4_K, 32>` kernel time by half).

### Tier 3 — Multi-row decode GEMV à la hipfire (1-2 weeks)

Hipfire's `kernels/src/gemv_hfq4g256_multirow.gfx1100.hip` processes
R=2/4/8 output rows per warp, sharing the X (activation) register state
across rows. For a wide decode batch (DDTree budget=22-32) this is
exactly the right shape:

- One warp processes 4 output cols × 22 batches → 88 dot products with
  one X-load
- Register pressure: ~38 VGPRs for R=4 multirow (hipfire-measured on
  gfx1100), still 16 waves/CU occupancy
- vs MMQ's tile-based approach which does 32 cols × mmq_y rows but
  burns more shared memory and launches more thread blocks

To port to ggml's q4_K shape, the kernel needs:
- The q4_K block layout reader (super-blocks of 256 with 6-bit scales +
  6-bit mins per sub-block of 32)
- A wave32 fast path using `v_dot4_i32_i8` for non-WMMA inner loops on
  gfx1010/1030, and `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` for
  gfx1100+
- Register-packed batch dimension to avoid LDS-staging for activations

Reference patterns in hipfire's tree (not directly portable, but the
tile/loop structure transfers):
- `crates/rdna-compute/src/dispatch.rs:601-613` — the per-arch ROCm-min +
  WMMA-builtin routing table
- `crates/rdna-compute/src/kernels.rs:multirow_*` — the R-selection logic
- `kernels/src/gemv_hfq4g256_multirow.gfx1100.hip` — the actual kernel

Expected impact: **2-3× on DFlash spec-verify**, bringing q4_K decode
within ~20% of hipfire on the same hardware.

### Tier 4 — gfx1010 / gfx1030 scalar-fallback score kernel (3-5 days)

Orthogonal to the q4_K decode work. Required to unblock PFlash on
RDNA1/RDNA2 cards where today the score-blocks kernel hangs (gfx1010 —
missing `v_dot4`) or runs ~7× slower than Strix Halo (gfx1030 — uses
SDWA fallback but no WMMA available).

Pattern: hipfire's `kernels/src/gemv_hfq4g256_multirow.gfx1010.hip`.
Wave32 RDNA1, scalar fp16 accumulation, no WMMA dependency.

## Ranked priority

1. **Tier 1 today** — needs only a flag change, gives us a data point on
   whether the analysis above is directionally correct.
2. **Tier 2 first** — biggest impact / effort ratio. Mostly mechanical:
   add per-arch template instantiations + dispatch bump. Lands in
   `mmvq.cu` + `mmvq.cuh`. Upstream-able to `ggml-org/llama.cpp` after
   landing on the dflash fork.
3. **Tier 3 second** — the real engineering work. Reaches hipfire-class
   per-token throughput on RDNA3+/4.
4. **Tier 4 in parallel** — unlocks RDNA1/RDNA2 PFlash entirely.

Path B (rocWMMA port of `flashprefill_kernels.cu`) addresses the PFlash
*prefill* tax (compress + target_prefill on long ctx). It is **orthogonal
to this plan** — it helps long-context TTFT, not decode tok/s. Both
should ship.

## Sequence of PRs against `Luce-Org/llama.cpp-dflash-ggml`

1. **PR-A**: extend MMVQ to `ncols_dst ≤ 32` for q4_K on RDNA3+ (Tier 2).
2. **PR-B**: same for q4_0 / q5_0 / q5_1 / q6_K / q8_0 (covers all
   common quants in lucebox configs).
3. **PR-C**: multi-row decode GEMV for q4_K (Tier 3, biggest payoff).
4. **PR-D**: gfx1010/1030 scalar-fallback score kernel (Tier 4).
5. **PR-E (separate, against `lucebox-hub`)**: rocWMMA port of
   `flashprefill_kernels.cu` → `flashprefill_kernels.hip.cu` (Path B,
   prefill tax).

Each PR is independent, separately bench-able, separately revertible.

## Validation per PR

- Bit-identical token output vs CUDA baseline (gfx1100 vs RTX 30/40-series
  on the same prompt). Lucebox already has `test_vs_oracle` for this.
- DFlash 3-tier coherence smoke (Path-A attractor / 3gram density /
  EOS-immediate) — port from hipfire `crates/hipfire-detect/` as a small
  Python script.
- NIAH retrieval at 8K / 32K / 64K / 128K — already verified end-to-end
  on the HIP support PR.
- Prompt-md5 disciplined bench: warmup + ≥2 measurement runs, fresh
  binary, prompt md5 logged. See lucebox's existing `bench_he.py` setup.
