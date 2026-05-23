// Single-token hybrid qwen35moe FFN evaluation helpers.

#pragma once

#include "internal.h"
#include "qwen35moe_hybrid_storage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

bool eval_qwen35moe_reference_ffn_single(
    ggml_backend_t         gpu_backend,
    const TargetWeights &  w,
    const TargetLayer &    L,
    const float *          cur_host,
    const int32_t *        selected_ids,
    const float *          selected_weights,
    int                    n_selected,
    std::vector<float> &   out,
    std::string *          err = nullptr);

bool eval_qwen35moe_hybrid_ffn_single(
    ggml_backend_t                      gpu_backend,
    const TargetWeights &               w,
    const TargetLayer &                 L,
    const Qwen35MoeHybridLayerStorage & storage,
    ggml_backend_t                      cpu_backend,
    const float *                       cur_host,
    const int32_t *                     selected_ids,
    const float *                       selected_weights,
    int                                 n_selected,
    std::vector<float> &                out,
    std::string *                       err = nullptr);

}  // namespace dflash::common
