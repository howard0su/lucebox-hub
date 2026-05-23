// Phase 3 hybrid expert storage for qwen35moe.

#pragma once

#include "qwen35moe_expert_placement.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen35MoeHybridLayerStorage {
    ggml_context * hot_ctx = nullptr;
    ggml_backend_buffer_t hot_buf = nullptr;
    ggml_tensor * gate_hot = nullptr;
    ggml_tensor * up_hot = nullptr;
    ggml_tensor * down_hot = nullptr;
    ggml_tensor * gate_up_hot = nullptr;

    ggml_context * cold_ctx = nullptr;
    ggml_backend_buffer_t cold_buf = nullptr;
    ggml_tensor * gate_cold = nullptr;
    ggml_tensor * up_cold = nullptr;
    ggml_tensor * down_cold = nullptr;
    ggml_tensor * gate_up_cold = nullptr;

    std::vector<int32_t> hot_expert_ids;
    std::vector<int32_t> cold_expert_ids;
    std::vector<int32_t> hot_local_by_global;
    std::vector<int32_t> cold_local_by_global;

    bool fused_gate_up = false;
    size_t gate_expert_bytes = 0;
    size_t up_expert_bytes = 0;
    size_t down_expert_bytes = 0;
    size_t gate_up_expert_bytes = 0;

    std::vector<uint8_t> gate_cold_bytes;
    std::vector<uint8_t> up_cold_bytes;
    std::vector<uint8_t> down_cold_bytes;
    std::vector<uint8_t> gate_up_cold_bytes;
};

struct Qwen35MoeHybridStorage {
    Qwen35MoeHybridStorage() = default;
    Qwen35MoeHybridStorage(const Qwen35MoeHybridStorage &) = delete;
    Qwen35MoeHybridStorage & operator=(const Qwen35MoeHybridStorage &) = delete;
    Qwen35MoeHybridStorage(Qwen35MoeHybridStorage && other) noexcept;
    Qwen35MoeHybridStorage & operator=(Qwen35MoeHybridStorage && other) noexcept;
    ~Qwen35MoeHybridStorage();

    ggml_backend_t cpu_backend = nullptr;
    Qwen35MoeExpertPlacement placement;
    std::vector<Qwen35MoeHybridLayerStorage> layers;

    bool matches(const TargetWeights & w) const;
    bool empty() const;
};

bool build_qwen35moe_hybrid_storage(const TargetWeights & w,
                                    ggml_backend_t backend,
                                    const Qwen35MoeExpertPlacement & placement,
                                    Qwen35MoeHybridStorage & out,
                                    std::string * err = nullptr);

}  // namespace dflash::common
