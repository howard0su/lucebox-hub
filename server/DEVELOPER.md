# DFlash Developer Guide

## Prerequisites

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | NVIDIA Turing (sm_75, e.g. RTX 2080) | Ampere+ (sm_86, e.g. RTX 3090) |
| VRAM | 22 GB | 24 GB |
| OS | Ubuntu 22.04 (jammy) | Ubuntu 24.04 (noble) |

> **Note:** FlashPrefill and BSA (Block-Sparse Attention) require **sm_80+** (Ampere or newer).
> On Turing (sm_75) the drafter falls back to ggml's `flash_attn_ext`.

### System packages

```
build-essential  cmake  git  git-lfs  nvcc (CUDA Toolkit)
```

A setup script is provided that installs everything (run as root):

```bash
sudo bash server/scripts/setup_system.sh
```

This installs build tools, `hf` (via pipx), and the CUDA Toolkit.

### Python

- **Python 3.11+** (tested with 3.11.2)
- Virtual environment recommended

```bash
python3 -m venv venv
source venv/bin/activate
```

### Python packages

Install the required packages:

```bash
pip install fastapi uvicorn transformers pydantic starlette
```

For running tests:

```bash
pip install pytest
```

---

## Building the C++ daemon

DFlash uses **CMake** with CUDA. The build produces `test_dflash`, the speculative-decoding
daemon that the Python server drives via stdin/stdout.

```bash
cd dflash

# Initialize the remaining submodule (Block-Sparse-Attention)
git submodule update --init --recursive

# Configure
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build the daemon binary
cmake --build build --target test_dflash -j
```

The binary lands at `server/build/test_dflash`.

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_CUDA_ARCHITECTURES` | `75;86` (auto-extended) | Target GPU architectures |
| `DFLASH27B_FA_ALL_QUANTS` | `ON` | Build all FA KV-quant pairs (3× longer compile) |
| `DFLASH27B_ENABLE_BSA` | `ON` | Block-Sparse Attention for spec-prefill (needs sm_80+) |
| `DFLASH27B_TESTS` | `ON` | Build C++ numerics tests |

---

## Model files

Download models before running the server:

```bash
# Target model (Q4_K_M quantized Qwen3.6-27B)
hf download <repo-id> --local-dir server/models/

# Draft model (0.98 GB default Qwen3.6 GGUF draft)
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf --local-dir server/models/draft/
```

Expected layout:

```
server/models/
├── Qwen3.6-27B-Q4_K_M.gguf          # --target (GGUF)
└── draft/
    └── dflash-draft-3.6-q4_k_m.gguf   # --draft  (GGUF)
```

The target path can also be set via the `DFLASH_TARGET` environment variable.

---

## Running the server

```bash
cd dflash
./build/dflash_server models/Qwen3.6-27B-Q4_K_M.gguf --port 8080
```

### Server CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Port |
| `--target` | `models/Qwen3.6-27B-Q4_K_M.gguf` | Target GGUF model |
| `--draft` | `models/draft` | Draft model directory |
| `--bin` | `build/test_dflash` | Path to the daemon binary |
| `--budget` | `22` | DDTree speculation budget |
| `--max-ctx` | `16384` | Maximum context length |
| `--kv-f16` | off | Force F16 KV cache |
| `--cache-type-k` / `--ctk` | auto | KV cache type for keys (f16/q4_0/q8_0/tq3_0/...) |
| `--cache-type-v` / `--ctv` | auto | KV cache type for values |
| `--fa-window` | auto | Sliding window size for flash attention (0 = full) |
| `--tokenizer` | auto (from GGUF) | HuggingFace tokenizer ID |
| `--prefix-cache-slots` | `4` | Number of prefix-cache slots |
| `--prefill-cache-slots` | `4` | Number of prefill-cache slots |
| `--daemon` | off | Run as background daemon |

### API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List models (OpenAI + Codex format) |
| `POST /v1/chat/completions` | OpenAI Chat Completions API |
| `POST /v1/responses` | OpenAI Responses API (Codex) |
| `POST /v1/messages` | Anthropic Messages API |

---

## Tests

### C++ tests (require GPU + model files)

After building:

```bash
cd server/build

# Numerics tests
./test_vs_oracle --target ../models/Qwen3.6-27B-Q4_K_M.gguf \
                 --draft ../models/draft/dflash-draft-3.6-q4_k_m.gguf

# Smoke tests
./smoke_load_target --target ../models/Qwen3.6-27B-Q4_K_M.gguf
./smoke_load_draft --draft ../models/draft/dflash-draft-3.6-q4_k_m.gguf
./smoke_draft_graph --draft ../models/draft/dflash-draft-3.6-q4_k_m.gguf
```

### Integration tests (require running server)

These scripts start their own server subprocess and need the server binary + models:

```bash
cd server/scripts
python test_server_prefix_cache.py
python test_multi_turn_prefix_cache.py
python test_full_compress_cache.py
```

Or run against an already-running server:

```bash
python test_server_prefix_cache.py --url http://localhost:8000
python test_multi_turn_prefix_cache.py --url http://localhost:8000
```

---

## CODA fused kernels (`DFLASH_CODA`)

CODA ([arXiv:2605.19269](https://arxiv.org/abs/2605.19269) §3) rewrites transformer
blocks as GEMM-epilogue programs, fusing the memory-bound residual / norm / activation
work into the surrounding GEMMs so the epilogue is hidden behind the matmul. This repo
implements the ggml-native subset for the qwen-family, Laguna, Gemma4, and DeepSeek4
graphs. `DFLASH_CODA` enables the full CODA set; `DFLASH_CODA_GLU`,
`DFLASH_CODA_RESIDUAL`, and `DFLASH_CODA_RMS` enable the activation, residual-epilogue,
and RMS partial-stats paths independently. All unset = default, unchanged behavior.

What engages:

| Pattern | Gate | Path | How |
| --- | --- | --- | --- |
| **SwiGLU** (§3.2.2): `silu(gate) * up` | `DFLASH_CODA_GLU` or umbrella | prefill + decode | qwen35 and qwen3 emit `ggml_glu_split(..., SWIGLU)`, triggering ggml's fused `{MUL_MAT, MUL_MAT, GLU}` kernel. qwen35 only does this when gate/up weight scales are 1.0 (non-NVFP4). Laguna and qwen35moe already used `ggml_swiglu_split`. |
| **Packed SwiGLU**: `silu(gate) * up` from one `[gate\|up]` projection | `DFLASH_CODA_GLU` or umbrella | decode / MoE hot paths | Laguna shared-expert and qwen35moe combined gate-up paths use `ggml_swiglu(...)` on the packed projection output, eliminating split views / cont copies while preserving exact `silu(gate) * up` semantics. |
| **GEGLU**: `gelu(gate) * up` | `DFLASH_CODA_GLU` or umbrella | prefill + decode | gemma4 emits `ggml_glu_split(..., GEGLU)` for dense/shared FFN and per-layer injection. Dense/shared FFN uses `{MUL_MAT, MUL_MAT, GLU}`; per-layer injection and routed combined-gate-up paths get activation-level GLU fusion. |
| **Clamped SwiGLU**: `silu(clamp(gate)) * clamp(up)` | `DFLASH_CODA_GLU` or umbrella | prefill + decode | deepseek4 keeps the asymmetric clamps explicit and uses `ggml_glu_split(..., SWIGLU)` after them. This is exact; it is **not** `SWIGLU_OAI`. Clamp nodes prevent GEMM-epilogue fusion, but collapse the activation from `{SILU, MUL}` into one GLU op. |
| **GEMM-Residual** (§3.2.1): `mul_mat(W,x) + residual` | existing upstream path | decode (M=1) | Already fused upstream via the mmvq (`{MUL_MAT, ADD}`) mat-vec epilogue. |
| **GEMM-Residual** (§3.2.1) | `DFLASH_CODA_RESIDUAL`, `DFLASH_CODA_RMS`, or umbrella | prefill / verify (M>1) | Forked `mmq` kernel adds a dst-shaped residual in its write-back epilogue. Detected automatically from any `{MUL_MAT, ADD}` with a contiguous, dst-shaped residual (non-MoE); **no graph rewrite needed** for pre-norm qwen/laguna/deepseek dense projections. Gemma4's sandwich/post-norm breaks this adjacency. |
| **GEMM-Residual + RMS partial stats** (§3.2.1 prototype) | `DFLASH_CODA_RMS` or umbrella | prefill / verify (quantized mmq, M>8) | A named graph side-output tensor `coda_partial_ms:<tag>` lets the mmq residual epilogue also write per-token partial mean-square blocks over `h = mul_mat(W,x)+residual`. This validates ggml multi-output graph lifetime and the CODA RMSNorm stats path. |
| **RMS partial-stats consumer** (§3.2.1) | `DFLASH_CODA_RMS` or umbrella | prefill / verify (qwen/qwen35 eligible residual→norm sites) | A CUDA RMSNorm helper consumes tagged `coda_partial_ms:<tag>` side outputs by reducing block means instead of recomputing `sum(h^2)` over all features. qwen35 and qwen3 graph builders emit tagged side-output/consumer pairs only when the residual input is a direct `{MUL_MAT, ADD}` with M>8 and 256-feature block alignment. |

Set `DFLASH_CODA_DEBUG=1` to trace when the forked mmq residual epilogue engages.
Set `DFLASH_CODA_RMS_POST_STATS=1` with `DFLASH_CODA_RMS=1` to test the experimental
post-reduction stats path: mmq writes only `h`, then a separate CUDA reduction fills
`coda_partial_ms:<tag>` immediately before the RMS consumer. This is correctness-covered
but not an e2e win in current measurements.
Set `GGML_CUDA_DISABLE_FUSION=1` to disable all ggml-cuda fusion (baseline).

Core changes: `deps/llama.cpp/ggml/src/ggml-cuda/{mmq.cuh,mmq.cu,ggml-cuda.cu}`
(residual and optional RMS partial-stats side-output threaded through `mmq_args` →
`mul_mat_q` → `mmq_write_back_*`) plus `ggml-cuda/norm.cu` for the RMS stats
consumer and GLU rewrites in
`src/qwen35/qwen35_target_graph.cpp`, `src/qwen3/`, `src/gemma4/`, and
`src/deepseek4/deepseek4_graph.cpp`.

### Local model-free tests (no weights, any CUDA arch incl. sm_75)

These validate the fused kernels against the unfused ggml op sequence with small
synthetic tensors, so they run without the 27B model:

```bash
cd server/build

# GLU-family fusion: SWIGLU, GEGLU, clamped SWIGLU, packed SWIGLU correctness + microbenchmarks
DFLASH_CODA_GLU=1 ./test_coda_swiglu

# GEMM-Residual mmq epilogue: GPU fused-vs-unfused parity (rel ~1e-7) + GPU-vs-CPU
DFLASH_CODA_RESIDUAL=1 ./test_coda_residual              # exercise the fused mmq path (M>1)
DFLASH_CODA_DEBUG=1 DFLASH_CODA_RESIDUAL=1 ./test_coda_residual   # + trace engagement
GGML_CUDA_DISABLE_FUSION=1 ./test_coda_residual # unfused baseline (bench compare)

# CODA RMS side-output/consumer: validates two observable graph outputs, quantized
# mmq residual+partial-mean-square side-output, tagged graph association, and
# RMSNorm consumption of those partial stats.
DFLASH_CODA_RMS=1 ./test_coda_rms_side_output
```

End-to-end logit/token parity and paper-aligned block / prefill-decode perf require the
27B model on a GPU with enough memory and are run on the remote validation host.

---

## Project structure

```
server/
├── CMakeLists.txt              # C++ build (cmake)
├── include/                    # C++ headers
├── src/                        # C++ sources (target/draft graph, KV cache, FlashPrefill)
├── test/                       # C++ test sources (test_dflash.cpp, smoke_*, test_*)
├── deps/
│   ├── llama.cpp/              # Vendored ggml snapshot + extracted helpers
│   └── Block-Sparse-Attention/ # BSA kernels (submodule)
├── models/                     # Model files (not in git)
│   ├── Qwen3.6-27B-Q4_K_M.gguf
│   └── draft/dflash-draft-3.6-q4_k_m.gguf
├── scripts/
│   ├── run.py                  # CLI text generation
│   ├── test_server_prefix_cache.py    # Integration test (--url or auto-spawn)
│   ├── test_multi_turn_prefix_cache.py # Integration test (--url or auto-spawn)
│   ├── test_full_compress_cache.py    # Integration test
│   └── setup_system.sh         # System dependency installer
├── README.md
└── DEVELOPER.md                # This file
```

---

## Using with OpenAI Codex CLI

The server natively supports the **Responses API** (`/v1/responses`) used by
[OpenAI Codex](https://github.com/openai/codex).

### Configuration

Create `~/.codex/config.toml`:

```toml
model = "luce-dflash"
model_provider = "dflash"

[model_providers.dflash]
name = "DFlash"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
supports_websockets = false
```

No `env_key` is needed for local use.

### Running

```bash
# Start the server
./build/dflash_server models/Qwen3.6-27B-Q4_K_M.gguf --port 8080

# In another terminal
codex --provider dflash "Explain this codebase"
```
