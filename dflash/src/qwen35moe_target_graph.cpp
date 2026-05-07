// MoE forward pass for Qwen3.5-MoE (qwen35moe) — fused mega-graph.
//
// All 256 experts per layer are pinned in VRAM. A single CUDA graph
// covers all 40 layers + lm_head, replayed each decode step.

#include "internal.h"
#include "moe_experts.h"
#include "qwen35_blocks.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

// Debug flags (can be set from test harness)
bool g_moe_debug_sync = false;
int  g_moe_debug_call = 0;

// Timing accumulators (reset per forward call)
static double g_time_alloc_ms       = 0;
static double g_time_graph_fused_ms = 0;

// Cached mega-graph for CUDA graph replay (one per n_tokens value)
struct CachedMegaGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    int            n_tokens_cached = 0;
    int            n_pinned_cached = 0;
    bool           valid = false;

    void reset() {
        if (gallocr) { ggml_gallocr_free(gallocr); gallocr = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        gf = nullptr;
        n_tokens_cached = 0;
        n_pinned_cached = 0;
        valid = false;
    }
    ~CachedMegaGraph() {
        gallocr = nullptr;
        ctx = nullptr;
    }
};

static constexpr int MEGA_CACHE_MAX = 8;
static CachedMegaGraph g_mega_cache[MEGA_CACHE_MAX];

static CachedMegaGraph * find_mega_cache(int n_tokens, int n_pinned) {
    for (int i = 0; i < MEGA_CACHE_MAX; i++) {
        if (g_mega_cache[i].valid &&
            g_mega_cache[i].n_tokens_cached == n_tokens &&
            g_mega_cache[i].n_pinned_cached == n_pinned) {
            return &g_mega_cache[i];
        }
    }
    return nullptr;
}

static CachedMegaGraph * alloc_mega_cache_slot() {
    for (int i = 0; i < MEGA_CACHE_MAX; i++) {
        if (!g_mega_cache[i].valid) return &g_mega_cache[i];
    }
    g_mega_cache[0].reset();
    for (int i = 0; i < MEGA_CACHE_MAX - 1; i++) {
        g_mega_cache[i] = g_mega_cache[i+1];
    }
    g_mega_cache[MEGA_CACHE_MAX-1] = CachedMegaGraph{};
    return &g_mega_cache[MEGA_CACHE_MAX-1];
}

namespace dflash27b {

static constexpr float MOE_EPS = 1e-6f;

// Builds a single fused graph for one MoE layer: attention + router + expert FFN.
// All 256 experts are in VRAM so top-k IDs feed directly into ggml_mul_mat_id.
static ggml_tensor * build_moe_graph_fused(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const TargetWeights & w,
    TargetCache &         cache,
    int                   layer_idx,
    ggml_tensor *         inp,         // [hidden, n_tokens]
    ggml_tensor *         positions,
    ggml_tensor *         attn_mask,
    int                   kv_start,
    int                   n_tokens,
    ggml_tensor *         gate_t,      // pinned [hidden, expert_ffn, 256]
    ggml_tensor *         up_t,        // pinned [hidden, expert_ffn, 256]
    ggml_tensor *         down_t,      // pinned [expert_ffn, hidden, 256]
    int                   fa_window = 0,
    ggml_tensor *         parent_ids = nullptr,
    DeltaNetCapture *     cap = nullptr)
{
    const int hidden  = w.n_embd;
    const int n_used  = w.n_experts_active;
    const int n_total = w.n_experts;
    const TargetLayer & L = w.layers[layer_idx];
    const bool is_attn = (((layer_idx + 1) % w.full_attention_interval) == 0);

    // ── Attention ──
    ggml_tensor * inpSA = inp;
    ggml_tensor * cur = rms_norm_mul(ctx, inp, L.attn_norm, MOE_EPS);

    if (is_attn) {
        int fa_idx = 0;
        for (int il = 0; il < layer_idx; il++)
            if (((il + 1) % w.full_attention_interval) == 0) fa_idx++;
        cur = build_full_attn_block(ctx, gf, w, L, cur, positions,
                                     cache.attn_k[fa_idx], cache.attn_v[fa_idx],
                                     attn_mask, kv_start, n_tokens,
                                     cache.kv_k_type, cache.kv_v_type, fa_window);
    } else {
        int dn_idx = 0;
        for (int il = 0; il < layer_idx; il++)
            if (((il + 1) % w.full_attention_interval) != 0) dn_idx++;
        cur = build_delta_net_block(ctx, gf, w, L, cur,
                                     cache.conv_state[dn_idx], cache.ssm_state[dn_idx],
                                     n_tokens, cap, parent_ids);
    }

    // Residual after attention
    cur = ggml_add(ctx, cur, inpSA);
    ggml_tensor * ffn_residual = cur;

    // Post-attention norm → input to MoE FFN
    ggml_tensor * post = rms_norm_mul(ctx, cur, L.attn_post_norm, MOE_EPS);

    // ── Router + top-k ──
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, post);
    ggml_tensor * probs  = ggml_soft_max(ctx, logits);
    ggml_tensor * selected = ggml_cont(ctx, ggml_argsort_top_k(ctx, probs, n_used));

    // ── Weight extraction + normalization ──
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_total, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected);

    ggml_tensor * weights_2d  = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
    ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights_2d);
    weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5f, INFINITY);
    weights_2d  = ggml_div(ctx, weights_2d, weights_sum);
    weights     = ggml_reshape_3d(ctx, weights_2d, 1, n_used, n_tokens);

    // ── Expert FFN via mul_mat_id ──
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, post, hidden, 1, n_tokens);

    ggml_tensor * gate_out = ggml_mul_mat_id(ctx, gate_t, cur_3d, selected);
    ggml_tensor * up_out   = ggml_mul_mat_id(ctx, up_t,   cur_3d, selected);
    ggml_tensor * swiglu   = ggml_swiglu_split(ctx, gate_out, up_out);
    ggml_tensor * experts  = ggml_mul_mat_id(ctx, down_t, swiglu, selected);

    experts = ggml_mul(ctx, experts, weights);

    // ── Sum over experts (single op: repeat_back sums dim 1) ──
    ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, n_tokens);
    ggml_tensor * moe_sum   = ggml_repeat_back(ctx, experts, sum_shape);
    ggml_tensor * moe_out   = ggml_reshape_2d(ctx, moe_sum, hidden, n_tokens);

    // ── Shared expert FFN ──
    TargetLayer shared_L{};
    shared_L.w_gate = L.shared_w_gate;
    shared_L.w_up   = L.shared_w_up;
    shared_L.w_down = L.shared_w_down;
    ggml_tensor * shared_ffn = build_swiglu_ffn(ctx, post, shared_L);

    if (L.ffn_gate_inp_shexp) {
        ggml_tensor * shared_gate_logit = ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, post);
        ggml_tensor * shared_gate = ggml_sigmoid(ctx, shared_gate_logit);
        shared_ffn = ggml_mul(ctx, shared_ffn, shared_gate);
    }

    // Combine routed experts + shared expert + residual
    ggml_tensor * ffn_out = ggml_add(ctx, moe_out, shared_ffn);
    return ggml_add(ctx, ffn_out, ffn_residual);
}

// ─── Full forward pass (all layers + lm_head) ────────────────────────

bool run_qwen35moe_forward(
    ggml_backend_t         backend,
    const TargetWeights &  w,
    TargetCache &          cache,
    PinnedExperts &        pinned,
    ggml_tensor *          act_a,
    ggml_tensor *          act_b,
    int                    n_tokens,
    ggml_tensor *          positions,
    ggml_tensor *          attn_mask,
    int                    kv_start,
    bool                   capture,
    int                    fa_window,
    ggml_tensor *          logits_out,
    ggml_tensor *          argmax_out,
    ggml_tensor *          parent_ids,
    std::vector<DeltaNetCapture> * delta_captures)
{
    const int hidden  = w.n_embd;
    const int n_layer = w.n_layer;

    g_moe_debug_call++;
    if (g_moe_debug_sync) {
        std::fprintf(stderr, "[MoE-DBG] forward call=%d n_tokens=%d kv_start=%d\n",
            g_moe_debug_call, n_tokens, kv_start);
    }

    g_time_graph_fused_ms = 0;
    g_time_alloc_ms       = 0;

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const int n_pinned = n_layer;

    ggml_tensor * final_act = act_a;
    bool lm_head_fused = false;

    {
        auto t_build_start = std::chrono::steady_clock::now();

        CachedMegaGraph * cached = find_mega_cache(n_tokens, n_pinned);
        bool rebuild = (cached == nullptr);

        if (rebuild) {
            cached = alloc_mega_cache_slot();

            ggml_init_params ip{};
            ip.mem_size   = 512 * 1024 * 1024;
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;

            ggml_context * ctx = ggml_init(ip);
            if (!ctx) return false;

            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

            ggml_tensor * inpL = ggml_view_2d(ctx, act_a,
                hidden, n_tokens, act_a->nb[1], 0);
            ggml_set_input(inpL);

            for (int il = 0; il < n_pinned; il++) {
                const auto & lt = pinned.get(il);

                DeltaNetCapture * cap_ptr = nullptr;
                if (delta_captures && !((il + 1) % w.full_attention_interval == 0)) {
                    int dn_idx = 0;
                    for (int j = 0; j < il; j++)
                        if (((j + 1) % w.full_attention_interval) != 0) dn_idx++;
                    if (dn_idx < (int)delta_captures->size())
                        cap_ptr = &(*delta_captures)[dn_idx];
                }

                inpL = build_moe_graph_fused(
                    ctx, gf, w, cache, il,
                    inpL, positions, attn_mask, kv_start, n_tokens,
                    lt.gate, lt.up, lt.down,
                    fa_window, parent_ids, cap_ptr);

                // DFlash feature capture
                if (capture && cache.target_feat) {
                    int capture_idx = -1;
                    for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
                        if (w.capture_layer_ids[k] == il) { capture_idx = k; break; }
                    }
                    if (capture_idx >= 0) {
                        const size_t elt        = ggml_element_size(cache.target_feat);
                        const size_t col_stride = cache.target_feat->nb[1];
                        const int    cap_sz     = cache.target_feat_cap;
                        const int    slot_start = kv_start % cap_sz;
                        const int    pre_n      = std::min(n_tokens, cap_sz - slot_start);
                        const int    post_n     = n_tokens - pre_n;

                        ggml_tensor * cur_2d = ggml_reshape_2d(ctx, inpL, hidden, n_tokens);
                        {
                            const size_t offset =
                                (size_t)slot_start * col_stride +
                                (size_t)capture_idx * hidden * elt;
                            ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                                hidden, pre_n, col_stride, offset);
                            ggml_tensor * src = ggml_view_2d(ctx, cur_2d,
                                hidden, pre_n, cur_2d->nb[1], 0);
                            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                        }
                        if (post_n > 0) {
                            const size_t offset =
                                (size_t)capture_idx * hidden * elt;
                            ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                                hidden, post_n, col_stride, offset);
                            ggml_tensor * src = ggml_view_2d(ctx, cur_2d,
                                hidden, post_n, cur_2d->nb[1],
                                (size_t)pre_n * cur_2d->nb[1]);
                            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                        }
                    }
                }
            }

            // Fuse lm_head into mega-graph
            if (n_pinned == n_layer && logits_out) {
                ggml_tensor * cur = ggml_rms_norm(ctx, inpL, 1e-6f);
                cur = ggml_mul(ctx, cur, w.out_norm);

                ggml_tensor * logits = ggml_mul_mat(ctx, w.output, cur);
                ggml_set_name(logits, "logits");

                ggml_tensor * logits_dst = ggml_view_2d(ctx, logits_out,
                    logits_out->ne[0], n_tokens, logits_out->nb[1], 0);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, logits, logits_dst));

                if (argmax_out) {
                    ggml_tensor * argmax = ggml_argmax(ctx, logits);
                    ggml_tensor * argmax_dst = ggml_view_1d(ctx, argmax_out, n_tokens, 0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, argmax, argmax_dst));
                }

                ggml_tensor * out_view = ggml_view_2d(ctx, act_b,
                    hidden, n_tokens, act_b->nb[1], 0);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, inpL, out_view));
            } else {
                ggml_tensor * out_view = ggml_view_2d(ctx, act_b,
                    hidden, n_tokens, act_b->nb[1], 0);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, inpL, out_view));
            }

            ggml_gallocr_t gallocr = ggml_gallocr_new(buft);
            if (!ggml_gallocr_alloc_graph(gallocr, gf)) {
                std::fprintf(stderr, "[MoE] mega-graph alloc failed\n");
                ggml_gallocr_free(gallocr);
                ggml_free(ctx);
                return false;
            }

            cached->ctx = ctx;
            cached->gf  = gf;
            cached->gallocr = gallocr;
            cached->n_tokens_cached = n_tokens;
            cached->n_pinned_cached = n_pinned;
            cached->valid = true;

            if (g_moe_debug_call <= 10) {
                std::fprintf(stderr, "[MoE] call=%d BUILT graph (n_tok=%d nodes=%d)\n",
                    g_moe_debug_call, n_tokens, ggml_graph_n_nodes(gf));
            }
        } else {
            if (g_moe_debug_call <= 10) {
                std::fprintf(stderr, "[MoE] call=%d REUSING graph (n_tok=%d nodes=%d)\n",
                    g_moe_debug_call, n_tokens, ggml_graph_n_nodes(cached->gf));
            }
        }

        final_act = act_b;
        lm_head_fused = (n_pinned == n_layer && logits_out != nullptr);

        auto t_build_end = std::chrono::steady_clock::now();

        auto status = ggml_backend_graph_compute(backend, cached->gf);
        auto t_compute_end = std::chrono::steady_clock::now();

        cudaDeviceSynchronize();
        auto t_sync_end = std::chrono::steady_clock::now();

        g_time_alloc_ms += std::chrono::duration<double,std::milli>(t_build_end - t_build_start).count();
        g_time_graph_fused_ms += std::chrono::duration<double,std::milli>(t_sync_end - t_build_end).count();

        if (g_moe_debug_call <= 10) {
            double build_ms = std::chrono::duration<double,std::milli>(t_build_end - t_build_start).count();
            double compute_ms = std::chrono::duration<double,std::milli>(t_compute_end - t_build_end).count();
            double gpu_wait_ms = std::chrono::duration<double,std::milli>(t_sync_end - t_compute_end).count();
            int n_nodes = ggml_graph_n_nodes(cached->gf);
            std::fprintf(stderr, "[MoE] call=%d nodes=%d: build=%.1f dispatch=%.1f gpu=%.1f ms (%.1f us/node)\n",
                g_moe_debug_call, n_nodes, build_ms, compute_ms, gpu_wait_ms,
                compute_ms * 1000.0 / n_nodes);
        }

        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[MoE] mega-graph compute failed status=%d\n", (int)status);
            return false;
        }
    }

    // Timing summary
    if (g_moe_debug_sync || g_moe_debug_call <= 20) {
        std::fprintf(stderr, "[MoE-PERF] call=%d n_tok=%d: fused=%.1f alloc=%.1f ms\n",
            g_moe_debug_call, n_tokens, g_time_graph_fused_ms, g_time_alloc_ms);
    }

    // LM HEAD (fallback when not fused into mega-graph)
    if (!lm_head_fused) {
        ggml_init_params ip{};
        ip.mem_size   = 64 * 1024 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;

        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);

        ggml_tensor * inp_view = ggml_view_2d(ctx, final_act,
            hidden, n_tokens, final_act->nb[1], 0);
        ggml_set_input(inp_view);

        ggml_tensor * cur = ggml_rms_norm(ctx, inp_view, 1e-6f);
        cur = ggml_mul(ctx, cur, w.out_norm);

        ggml_tensor * logits = ggml_mul_mat(ctx, w.output, cur);
        ggml_set_name(logits, "logits");

        ggml_tensor * logits_dst = ggml_view_2d(ctx, logits_out,
            logits_out->ne[0], n_tokens, logits_out->nb[1], 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, logits, logits_dst));

        if (argmax_out) {
            ggml_tensor * argmax = ggml_argmax(ctx, logits);
            ggml_tensor * argmax_dst = ggml_view_1d(ctx, argmax_out, n_tokens, 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, argmax, argmax_dst));
        }

        ggml_gallocr_t gallocr = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(gallocr, gf)) {
            std::fprintf(stderr, "[MoE] lm_head graph alloc failed\n");
            ggml_gallocr_free(gallocr);
            ggml_free(ctx);
            return false;
        }

        auto status = ggml_backend_graph_compute(backend, gf);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);

        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[MoE] lm_head graph compute failed status=%d\n", (int)status);
            return false;
        }
    }

    return true;
}

} // namespace dflash27b
