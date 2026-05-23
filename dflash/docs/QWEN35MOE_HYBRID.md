# Qwen35MoE hybrid workflow

This document captures the current `qwen35moe` hybrid CPU/GPU workflow in `dflash`, including:

- offline routing-stat collection
- per-layer expert placement generation
- hybrid storage enablement
- parity / smoke checks
- benchmark workflow

It reflects the code landed in:

- `6e91961` — all-GPU `qwen35moe` backend
- `e089931` — parity smoke + routing-stats scaffold
- `7b27545` — router capture + calibration + per-layer placement tools
- `a1a5965` — hybrid storage loader
- `3bbd128` — hybrid FFN helper + swap planning
- `3134499` — runtime swap hooks
- `87facc7` — benchmark script

## Current implementation state

The current implementation is split into three tiers:

| Tier | Status | Notes |
|---|---|---|
| All-GPU `qwen35moe` backend | **implemented** | Baseline functional path |
| Hybrid storage / planning | **implemented** | Per-layer hot/cold placement, CPU cold tensors, GPU hot tensors |
| Full live hybrid runtime execution | **partial** | Helper/evaluator exists, but the end-to-end daemon path is still conservative |

Important current behavior:

1. **Placement is per-layer**, not uniform. Hot expert allocation is derived from each layer’s own routing distribution.
2. **Hybrid mode currently falls back to AR-only** by policy.
3. **Cold experts are stored on CPU**, so higher system RAM usage is expected even before CPU execution is fully wired into the live generation path.

## 1. Collect routing statistics

Use the offline calibration tool:

```bash
cd dflash
./build/collect_qwen35moe_routing_stats \
  ./models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  /tmp/q35moe_stats.json \
  /path/to/prompt1.bin /path/to/prompt2.bin
```

Prompt files are raw `int32` token streams (uncounted).

Output:

- JSON routing stats with per-layer, per-expert activation counts

Relevant code:

- `test/collect_qwen35moe_routing_stats.cpp`
- `src/qwen35moe/qwen35moe_routing_stats.{h,cpp}`

## 2. Build a placement file

Placement is intentionally **layer-aware**. The code already supports per-layer hot expert lists via:

- `src/qwen35moe/qwen35moe_expert_placement.{h,cpp}`

The intended flow is:

1. rank experts independently for each layer
2. allocate hot slots by marginal gain across layers
3. emit a placement JSON with:
   - `hot_counts`
   - `hot_expert_ids`

The existing test coverage is:

- `test/test_qwen35moe_expert_placement.cpp`

## 3. Run hybrid storage mode

Set a placement file before starting the daemon:

```bash
export DFLASH_QWEN35MOE_PLACEMENT=/path/to/placement.json
./build/test_dflash \
  ./models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  ./models/draft/draft-Qwen3.6-35B-A3B.gguf \
  --daemon
```

On startup you should see a line like:

```text
[qwen35moe] hybrid storage ready: total_hot=... total_cold=... placement=... (AR-only mode)
```

What this means:

- hot experts were compacted into GPU tensors
- cold experts were materialized into CPU tensors
- the runtime recognized the placement file successfully

Relevant code:

- `src/qwen35moe/qwen35moe_hybrid_storage.{h,cpp}`
- `src/qwen35moe/qwen35moe_backend.{h,cpp}`

## 4. Runtime stats + request-boundary swap hooks

Optional environment variables:

```bash
export DFLASH_QWEN35MOE_RUNTIME_STATS_OUT=/tmp/runtime_stats.json
export DFLASH_QWEN35MOE_NEXT_PLACEMENT_OUT=/tmp/next_placement.json
export DFLASH_QWEN35MOE_SWAP_MAX=1
export DFLASH_QWEN35MOE_SWAP_MIN_GAIN=1
```

Current behavior:

- runtime router selections are accumulated when enabled
- request-boundary swap planning can produce a next placement
- live placement rebuild hooks exist

This is still a conservative path; request-boundary swap integration is intentionally safer than mid-request remapping.

## 5. Smoke / parity checks

### Full parity smoke

```bash
python3 dflash/scripts/qwen35moe_parity_smoke.py
```

Covers:

- daemon init
- generate
- snapshot / restore
- park / unpark
- compress

### Router capture smoke

```bash
./build/smoke_qwen35moe_router_capture \
  ./models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

### Hybrid storage smoke

```bash
./build/smoke_qwen35moe_hybrid_storage \
  ./models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

### Hybrid FFN smoke

```bash
./build/smoke_qwen35moe_hybrid_ffn \
  ./models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

### Unit tests

```bash
./build/test_qwen35moe_routing_stats
./build/test_qwen35moe_expert_placement
./build/test_qwen35moe_swap_manager
```

## 6. Benchmark baseline vs hybrid

Use the benchmark script:

```bash
python3 dflash/scripts/bench_qwen35moe_modes.py --n-gen 2
```

It compares:

- baseline daemon mode
- hybrid daemon mode with `DFLASH_QWEN35MOE_PLACEMENT`

Current caveat:

- this is **not yet a final apples-to-apples performance claim**
- hybrid mode is still **AR-only fallback**
- the current runtime has storage split and planning in place, but the fully integrated live hot/cold expert execution path is still incomplete

So treat the benchmark as:

- a regression tracker
- a way to compare current runtime modes
- not yet the final performance result for the intended PowerInfer-style system

## 7. Expected resource profile today

Current hybrid mode may show:

- **high system RAM**
- **low CPU utilization**
- **little or no throughput improvement**

That is expected for the current stage, because:

1. cold experts are now materialized in host memory
2. full live CPU cold-expert execution is not yet on the end-to-end daemon path
3. hybrid mode intentionally uses AR-only fallback for safety

Once the remaining runtime path is integrated, CPU usage and end-to-end hybrid behavior should become meaningful to measure.
