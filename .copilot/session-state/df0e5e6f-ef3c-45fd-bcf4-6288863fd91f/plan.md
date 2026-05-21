# Plan: BSA Support for Qwen3.5 Prefill (Option 1 — Graph-based)

## Problem

Qwen3.5's 16 full-attention layers (every 4th of 64 total) use `ggml_flash_attn_ext` inside a monolithic ggml graph, chunked into 512-token ubatches. At long contexts, these layers attend to the entire KV history with dense FA — BSA (Block-Sparse Attention) would be significantly faster.

## Approach

**Swap `ggml_flash_attn_ext` → `ggml_flash_attn_sparse` in the graph, processing the full prompt in one pass (no chunking).**

Key constraint: The BSA kernel requires `Q_len == KV_len` (single `seq_len` for both). This rules out chunked prefill (where chunk 2+ has Q=512 but KV=1024+). Solution: process the entire prompt as one graph computation when BSA is active.

### Why this works for Qwen3.5

- DeltaNet layers (48/64) handle arbitrary `n_tokens` — they're unrolled recurrences, no length limitation
- Full-attention layers (16/64) get BSA via `ggml_flash_attn_sparse`
- Memory: ggml allocator reuses tensors across layers; peak ≈ 2-3× one layer buffer, not 64×
- BSA handles causality internally — no external mask needed

### Scope

- **On by default** — no env var gate. The `ggml_flash_attn_sparse` op has a built-in fallback: when no BSA kernel is registered (e.g., 2080Ti / sm_75), `fattn-sparse.cu` transparently redirects to `ggml_flash_attn_ext` (dense FA). Zero behavior change on unsupported hardware.
- **Fresh prefill only** (kv_offset == 0): The restore-and-generate delta path has kv_offset > 0 → Q_len ≠ KV_len → must stay dense. Deltas are typically 2-10 tokens anyway (too short for BSA benefit).
- **Threshold**: Only activates for prompt_len > 1024 (below that, BSA block-scoring overhead exceeds benefit; keep chunked dense path).
- **Alpha**: Uses existing `DFLASH_FP_ALPHA` env (default 0.12). In the ggml op, alpha < 1.0 triggers BSA; alpha ≥ 1.0 falls through to dense.
- **Hardware fallback**: `pflash_register_ggml_kernel()` is guarded by `#ifdef DFLASH27B_HAVE_SM80_FLASHPREFILL`. On 2080Ti (no BF16 WMMA), the kernel stays unregistered → sparse op falls back to dense FA automatically.

## Changes

### 1. Register pflash kernel at init (conditional on hardware)

**File: `qwen35_backend.cpp`**

Call `pflash_register_ggml_kernel()` in the backend constructor, guarded by `#ifdef DFLASH27B_HAVE_SM80_FLASHPREFILL`. On 2080Ti this ifdef is false → kernel stays unregistered → sparse op falls back to dense FA. On sm_80+ (3090, 4090, etc.) → BSA is active.

### 2. Add `use_sparse_attn` flag to `QwenGraphInputs`

**File: `internal.h`**

Add `bool use_sparse_attn = false;` and `float sparse_alpha = 0.12f;` to `QwenGraphInputs`. Controls whether `build_full_attn_block` uses sparse or dense FA.

### 3. Modify `build_full_attn_block` to support sparse FA

**File: `qwen35_target_graph.cpp` (~line 619)**

When `use_sparse_attn` is true:
- Use `ggml_flash_attn_sparse(ctx, Qfa, Kfa, Vfa, kq_scale, alpha)` instead of `ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask, kq_scale, 0, 0)`
- The `attn_mask` parameter becomes unused when sparse is active (BSA handles causality)

### 4. Thread the flag through `build_target_step` and `build_qwen35_graph`

**Files: `graph_builders.cpp`, `qwen35_target_graph.cpp`**

- `build_target_step`: accept `use_sparse_attn` param, pass to `QwenGraphInputs`
- `build_qwen35_graph`: pass `in.use_sparse_attn` down to `build_full_attn_block`
- When sparse is active, skip mask creation (`with_mask=false`)

### 5. BSA prefill path in `do_prefill`

**File: `qwen35_backend.cpp`**

When BSA is enabled and conditions are met (kv_offset == 0, prompt_len > threshold):
- Process the full prompt in ONE graph compute (set `prefill_ubatch = prompt_len`)
- Build graph with `use_sparse_attn=true`, `with_mask=false`
- Handle inline snapshots after the full pass instead of mid-stream

When conditions are NOT met, fall through to the existing chunked path (no behavior change).

## Notes

- `pflash_register_ggml_kernel()` is a one-time init — it registers the pFlash kernel as the CUDA backend handler for `GGML_OP_FLASH_ATTN_SPARSE`. Without this, the op falls back to dense FA (same as `ggml_flash_attn_ext`).
- The `ggml_flash_attn_sparse` op stores `{scale, alpha}` in op_params. The CUDA backend reads these, converts Q/K/V from F32/F16 → BF16 with S↔H transpose, calls pFlash, converts output BF16 → F32.
- Inline snapshots: In the non-chunked BSA path, snap_pos handling becomes simpler — just save after the full forward. The KV cache contains all positions; `cache_.cur_pos = committed` is set after graph compute. Call `snapshot_save` if snap_pos falls within range.
- No changes to the spec-decode or verify paths — they use small n_tokens (1 or ≤16) where BSA has no benefit.
