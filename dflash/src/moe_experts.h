// MoE expert weight management for qwen35moe models.
// All expert weights are pinned in VRAM — no runtime swapping.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dflash27b {

// Source of expert weight data in the mmap'd GGUF file.
struct MoeExpertSource {
    const uint8_t * mmap_base = nullptr;

    struct LayerOffsets {
        size_t gate_offset = 0;  // byte offset from mmap_base to blk.L.ffn_gate_exps data
        size_t up_offset   = 0;
        size_t down_offset = 0;
    };
    std::vector<LayerOffsets> layers;  // [n_layer]

    size_t gate_expert_bytes = 0;  // bytes for one expert's gate weights
    size_t up_expert_bytes   = 0;
    size_t down_expert_bytes = 0;

    ggml_type gate_type = GGML_TYPE_COUNT;
    ggml_type up_type   = GGML_TYPE_COUNT;
    ggml_type down_type = GGML_TYPE_COUNT;

    // Per-layer down type info (may vary across layers).
    std::vector<ggml_type> layer_down_types;  // [n_layer]
    std::vector<size_t>    layer_down_bytes;  // [n_layer]

    int hidden_dim     = 0;
    int expert_ffn_dim = 0;
    int n_experts      = 0;  // 256
    int n_layers       = 0;  // 40
};

// All 256 experts pinned in VRAM per layer.
struct PinnedExperts {
    struct LayerTensors {
        ggml_tensor * gate = nullptr;  // [hidden, expert_ffn, 256]
        ggml_tensor * up   = nullptr;  // [hidden, expert_ffn, 256]
        ggml_tensor * down = nullptr;  // [expert_ffn, hidden, 256]
    };

    // Allocates VRAM and bulk-loads all experts for specified layers.
    bool init(ggml_backend_t backend, const MoeExpertSource & source,
              const std::vector<int> & pinned_layer_ids);

    bool is_pinned(int layer) const {
        return layer < (int)pinned_.size() && pinned_[layer];
    }
    const LayerTensors & get(int layer) const { return layers_[layer]; }

    size_t total_bytes() const { return total_bytes_; }
    void destroy();
    ~PinnedExperts() { destroy(); }

private:
    std::vector<bool> pinned_;
    std::vector<LayerTensors> layers_;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    size_t total_bytes_ = 0;
};

} // namespace dflash27b
