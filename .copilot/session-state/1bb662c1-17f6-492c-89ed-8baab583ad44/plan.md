# Normalize Multi-GPU Support Across Model Backends

## Problem Statement

Multi-GPU support is currently qwen35-only and hardcoded in `test/test_dflash.cpp`:
- `run_target_layer_split_daemon()` (~500 lines) assumes qwen35 arch
- `inspect_target_layer_count()` reads `qwen35.block_count` 
- `compute_layer_ranges()` is generic but trapped in test file
- GPU placement is scattered across per-arch DaemonArgs with no shared structure
- Draft model placement, peer access, layer-split are all ad-hoc per-arch

**Goal**: Extract reusable GPU placement infrastructure into `src/`, create a
`Qwen35LayerSplitBackend` as a concrete ModelBackend, and add a shared
`DevicePlacement` struct so all backends describe GPU allocation uniformly.

## Design Principles (from rubber-duck critique)

1. **Narrow device abstraction** — `DevicePlacement` for target GPU only; draft/IPC config stays arch-specific
2. **Layer split is backend-internal** — not a generic wrapper; each arch gets its own split backend
3. **Don't overgeneralize** — laguna/qwen3/gemma4 don't need draft/IPC support yet
4. **Extract qwen35 layer-split first** — then generalize only when a second arch needs it
5. **GGUF inspection returns arch + layer count** — not just block_count

## Phase 1: Extract generic utilities to `src/common/`

### `src/common/device_placement.h`
```cpp
struct DevicePlacement {
    int gpu = 0;                          // primary GPU
    std::vector<int> layer_split_gpus;    // empty = single GPU
    std::vector<double> layer_split_weights;
    bool peer_access = false;
    int max_ctx = 8192;
    
    bool is_layer_split() const;
    int primary_gpu() const;
};
```

### `src/common/gguf_inspect.{h,cpp}`
```cpp
struct GgufModelInfo {
    std::string arch;
    int n_layer = -1;
};
GgufModelInfo inspect_gguf_model_info(const char * path);
```

### `src/common/layer_split_utils.{h,cpp}`
```cpp
// Pure utility — works with any arch
std::vector<std::pair<int,int>> compute_layer_ranges(
    int n_layer, int n_gpus, const std::vector<double> & weights);
// Validation
bool validate_device_placement(const DevicePlacement & dp, int cuda_device_count);
```

## Phase 2: Add DevicePlacement to existing backend configs

Update each backend's args struct to include `DevicePlacement devices` instead of
bare `int gpu`. Backends initially only consume `devices.primary_gpu()`.

```cpp
struct Qwen35DaemonArgs {
    const char * target_path;
    DevicePlacement devices;       // replaces target_gpu
    // Draft config stays qwen35-specific
    const char * draft_path;
    int draft_gpu = -1;
    // ... other qwen35-specific fields ...
};

struct LagunaDaemonArgs {
    const char * target_path;
    DevicePlacement devices;       // replaces implicit gpu=0
    // ... laguna-specific fields ...
};
```

## Phase 3: Extract `Qwen35LayerSplitBackend`

Move `run_target_layer_split_daemon()` + supporting functions from test_dflash.cpp
to `src/qwen35/qwen35_layer_split.{h,cpp}`:

- `Qwen35LayerSplitBackend : ModelBackend`
- Owns shard management, partial GGUF loading, inter-GPU activation copy
- Draft model placement on specified GPU
- Park/unpark per shard
- Snapshot: currently unsupported in split mode (preserve existing behavior)

**test_dflash.cpp** dispatch becomes:
```cpp
if (target_gpus.size() > 1 && daemon_mode) {
    return run_qwen35_layer_split_daemon(args);  // → src/qwen35/
}
```

## Phase 4: Wire up and test

- Update test_dflash.cpp dispatch to use new DevicePlacement + split backend
- Build-verify all targets
- The layer-split daemon behavior must be identical to current

## Files Summary

### Create
| File | Phase | Description |
|------|-------|-------------|
| `src/common/device_placement.h` | 1 | DevicePlacement struct |
| `src/common/gguf_inspect.{h,cpp}` | 1 | Arch + layer count inspection |
| `src/common/layer_split_utils.{h,cpp}` | 1 | compute_layer_ranges + validation |
| `src/qwen35/qwen35_layer_split.{h,cpp}` | 3 | Qwen35LayerSplitBackend |

### Modify
| File | Phase | Description |
|------|-------|-------------|
| `src/qwen35/qwen35_daemon.h` | 2 | Add DevicePlacement to args |
| `src/laguna/laguna_daemon.h` | 2 | Add DevicePlacement to args |
| `src/qwen3/qwen3_daemon.h` | 2 | Add DevicePlacement to args |
| `src/gemma4/gemma4_daemon.h` | 2 | Add DevicePlacement to args |
| `test/test_dflash.cpp` | 3-4 | Remove layer-split code, wire new backend |
| `CMakeLists.txt` | 1 | Add new .cpp files |

