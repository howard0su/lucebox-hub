// Shared qwen35-family graph helpers used by both dense qwen35 and qwen35moe.

#pragma once

#include "common/ggml_graph_precision.h"
#include "ggml.h"

#include <cstdlib>
#include <cstring>

namespace dflash::common {

inline bool coda_feature_enabled(const char * feature_env) {
    return std::getenv("DFLASH_CODA") != nullptr || std::getenv(feature_env) != nullptr;
}

inline ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * weight, float eps) {
    x = rms_norm_input_f32(ctx, x);
    weight = graph_tensor_f32(ctx, weight);
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

inline ggml_tensor * coda_rms_norm_mul_after_residual(
    ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x,
    ggml_tensor * weight, float eps, const char * tag) {
    x = rms_norm_input_f32(ctx, x);
    weight = graph_tensor_f32(ctx, weight);

    static const bool coda = coda_feature_enabled("DFLASH_CODA_RMS");
    constexpr int partial_block = 256;
    const bool has_mul_mat_residual =
        x->op == GGML_OP_ADD && x->type == GGML_TYPE_F32 &&
        x->ne[2] == 1 && x->ne[3] == 1 &&
        x->ne[1] > 8 && x->ne[0] % partial_block == 0 &&
        x->src[0] && x->src[1] &&
        (x->src[0]->op == GGML_OP_MUL_MAT || x->src[1]->op == GGML_OP_MUL_MAT);

    if (coda && gf && tag && has_mul_mat_residual) {
        ggml_set_name(x, tag);
        ggml_tensor * partial_ms = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0] / partial_block, x->ne[1]);
        ggml_format_name(partial_ms, "coda_partial_ms:%s", tag);
        ggml_set_output(partial_ms);
        ggml_build_forward_expand(gf, partial_ms);

        ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
        ggml_format_name(n, "coda_rms_from_partial:%s", tag);
        return ggml_mul(ctx, n, weight);
    }

    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

inline ggml_tensor * coda_deferred_rms_weight_after_residual(
    ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x,
    ggml_tensor * weight, float eps, const char * tag, bool * deferred) {
    x = rms_norm_input_f32(ctx, x);
    weight = graph_tensor_f32(ctx, weight);
    if (deferred) {
        *deferred = false;
    }

    static const bool coda = coda_feature_enabled("DFLASH_CODA_RMS");
    constexpr int partial_block = 256;
    const bool has_mul_mat_residual =
        x->op == GGML_OP_ADD && x->type == GGML_TYPE_F32 &&
        x->ne[2] == 1 && x->ne[3] == 1 &&
        x->ne[1] > 8 && x->ne[0] % partial_block == 0 &&
        x->src[0] && x->src[1] &&
        (x->src[0]->op == GGML_OP_MUL_MAT || x->src[1]->op == GGML_OP_MUL_MAT);

    if (coda && gf && tag && has_mul_mat_residual) {
        ggml_set_name(x, tag);
        ggml_tensor * partial_ms = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0] / partial_block, x->ne[1]);
        ggml_format_name(partial_ms, "coda_partial_ms:%s", tag);
        ggml_set_output(partial_ms);
        ggml_build_forward_expand(gf, partial_ms);
        if (deferred) {
            *deferred = true;
        }
        return ggml_mul(ctx, x, weight);
    }

    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

inline ggml_tensor * coda_apply_deferred_rstd(
    ggml_context * ctx, ggml_tensor * x, float eps, const char * tag, bool deferred) {
    if (!deferred || !tag) {
        return x;
    }
    ggml_tensor * y = ggml_scale(ctx, x, 1.0f);
    ggml_format_name(y, "coda_apply_rstd:%s", tag);
    std::memcpy(y->op_params + 1, &eps, sizeof(float));
    return y;
}

// NVFP4 scale2: if weight has a per-tensor scale, multiply the matmul result
// by that scale. No-op when scale==1.0f (non-NVFP4 models).
inline ggml_tensor * apply_scale2(ggml_context * ctx, ggml_tensor * mm_result,
                                  float scale) {
    if (scale == 1.0f) return mm_result;
    return ggml_scale(ctx, mm_result, scale);
}

}  // namespace dflash::common
