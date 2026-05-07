// MoE generation for Qwen3.5/3.6-35B-A3B.
// Linked into the test_dflash binary; provides run_moe_autoregressive and run_moe_dflash.

#include "dflash27b.h"
#include "internal.h"
#include "dflash_graph.h"
#include "qwen3_drafter.h"
#include "moe_experts.h"
#include "test_helpers.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

extern "C" void dflash27b_launch_f16_to_f32(const void * src,
                                            void * dst,
                                            size_t n_elems,
                                            cudaStream_t stream);
extern "C" void dflash27b_launch_bf16_to_f32(const void * src,
                                             void * dst,
                                             size_t n_elems,
                                             cudaStream_t stream);

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace dflash27b;

#define IS_EOS_TOK(tok, w) \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id) \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

// Debug sync flag (defined in qwen35moe_target_graph.cpp)
extern bool g_moe_debug_sync;
extern int  g_moe_debug_call;

static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), sz);
    return out;
}

static bool write_int32_file(const std::string & path, const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *)v.data(), v.size() * sizeof(int32_t));
    return (bool)f;
}

static int argmax_f32(const float * x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

static std::vector<std::string> load_gguf_vocab(const char * path) {
    std::vector<std::string> vocab;
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path, gip);
    if (!gctx) return vocab;
    int64_t kid = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (kid >= 0) {
        size_t n = gguf_get_arr_n(gctx, kid);
        vocab.resize(n);
        for (size_t i = 0; i < n; i++) {
            vocab[i] = gguf_get_arr_str(gctx, kid, i);
        }
    }
    gguf_free(gctx);
    if (meta_ctx) ggml_free(meta_ctx);
    return vocab;
}

static std::string detokenize(const std::vector<std::string> & vocab,
                              const std::vector<int32_t> & ids) {
    std::string out;
    for (int32_t id : ids) {
        if (id >= 0 && id < (int32_t)vocab.size()) out += vocab[id];
    }
    // Decode GPT-2 style byte-level BPE: Ġ→space, Ċ→newline, etc.
    std::string decoded;
    for (size_t i = 0; i < out.size(); ) {
        unsigned char c = (unsigned char)out[i];
        // UTF-8 two-byte sequences used by GPT-2 BPE for byte values
        if (c == 0xC4 && i + 1 < out.size()) {
            unsigned char c2 = (unsigned char)out[i + 1];
            if (c2 == 0xa0) { decoded += ' '; i += 2; continue; }   // Ġ → space
            if (c2 == 0x8a) { decoded += '\n'; i += 2; continue; }   // Ċ → newline
        }
        if (c == 0xC5 && i + 1 < out.size()) {
            unsigned char c2 = (unsigned char)out[i + 1];
            if (c2 == 0x82) { decoded += '\t'; i += 2; continue; }   // ł → tab
        }
        decoded += out[i];
        i++;
    }
    return decoded;
}

// ─── Simple autoregressive MoE generation (all experts in VRAM) ───────────
int run_moe_autoregressive(
    const char * target_path,
    const char * prompt_path,
    int n_gen,
    const char * out_path,
    ggml_backend_t backend,
    TargetWeights & w)
{
    using namespace dflash27b;
    const int hidden = w.n_embd;
    const int vocab  = (int)w.embedder.n_vocab;
    const int max_ctx = g_max_ctx_override > 0 ? g_max_ctx_override : 4096;

    std::printf("[moe-ar] hidden=%d vocab=%d n_layer=%d experts=%d active=%d\n",
                hidden, vocab, w.n_layer, w.n_experts, w.n_experts_active);

    // ── Pin ALL layers into VRAM ──
    PinnedExperts pinned;
    {
        std::vector<int> pin_ids;
        for (int l = 0; l < w.n_layer; l++) pin_ids.push_back(l);
        if (!pinned.init(backend, w.expert_source, pin_ids)) {
            std::fprintf(stderr, "[moe-ar] PinnedExperts init failed\n");
            return 1;
        }
    }


    // ── KV Cache ──
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, /*max_verify_tokens=*/1, backend, cache, /*prefill_only=*/true)) {
        std::fprintf(stderr, "[moe-ar] cache: %s\n", dflash27b_last_error());
        return 1;
    }

    // ── Activation buffers (n_tokens=1) ──
    ggml_init_params act_ip{};
    act_ip.mem_size = 8 * ggml_tensor_overhead() + 4096;
    act_ip.no_alloc = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    ggml_tensor * act_a = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * act_b = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, 1);
    ggml_set_name(act_a, "act_a"); ggml_set_name(act_b, "act_b");
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, backend);

    // ── Output tensors ──
    ggml_init_params out_ip{};
    out_ip.mem_size = 8 * ggml_tensor_overhead() + 4096;
    out_ip.no_alloc = true;
    ggml_context * out_ctx = ggml_init(out_ip);
    ggml_tensor * logits_out = ggml_new_tensor_2d(out_ctx, GGML_TYPE_F32, vocab, 1);
    ggml_set_name(logits_out, "logits_out");
    ggml_backend_buffer_t out_buf = ggml_backend_alloc_ctx_tensors(out_ctx, backend);

    // ── Positions + mask ──
    ggml_init_params pos_ip{};
    pos_ip.mem_size = 8 * ggml_tensor_overhead() + 4096;
    pos_ip.no_alloc = true;
    ggml_context * pos_ctx = ggml_init(pos_ip);
    ggml_tensor * positions = ggml_new_tensor_1d(pos_ctx, GGML_TYPE_I32, 4);
    ggml_set_name(positions, "positions");
    ggml_tensor * attn_mask = ggml_new_tensor_2d(pos_ctx, GGML_TYPE_F16,
        align_up(max_ctx + 1, KQ_MASK_PAD), align_up(1, KQ_MASK_PAD));
    ggml_set_name(attn_mask, "attn_mask");
    ggml_backend_buffer_t pos_buf = ggml_backend_alloc_ctx_tensors(pos_ctx, backend);

    // ── Load prompt ──
    std::vector<int32_t> prompt_tokens = read_int32_file(prompt_path);
    if (prompt_tokens.empty()) {
        std::fprintf(stderr, "[moe-ar] failed to read prompt: %s\n", prompt_path);
        return 1;
    }
    std::printf("[moe-ar] prompt: %d tokens, gen: %d tokens\n", (int)prompt_tokens.size(), n_gen);


    // ── Prefill (token-by-token) ──
    auto t_pf0 = std::chrono::steady_clock::now();
    int committed = 0;
    int last_tok = -1;
    std::vector<float> embed(hidden);
    std::vector<float> logit_buf(vocab);

    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        w.embedder.embed(&prompt_tokens[i], 1, embed.data());
        ggml_backend_tensor_set(act_a, embed.data(), 0, sizeof(float) * hidden);

        int32_t pos4[4] = {i, i, i, 0};
        ggml_backend_tensor_set(positions, pos4, 0, sizeof(int32_t) * 4);

        const int kv_len = i + 1;
        std::vector<uint16_t> mask_data;
        ggml_tensor * mask_ptr = nullptr;
        if (kv_len > 1) {
            build_causal_mask(mask_data, kv_len, 1, i);
            const int kv_pad = align_up(kv_len, KQ_MASK_PAD);
            attn_mask->ne[0] = kv_pad;
            attn_mask->ne[1] = align_up(1, KQ_MASK_PAD);
            attn_mask->nb[1] = (size_t)kv_pad * ggml_element_size(attn_mask);
            ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                sizeof(uint16_t) * mask_data.size());
            mask_ptr = attn_mask;
        }

        if (!run_qwen35moe_forward(backend, w, cache, pinned,
                act_a, act_b, 1, positions, mask_ptr, i, false, 0, logits_out, nullptr,
                nullptr, nullptr)) {
            std::fprintf(stderr, "[moe-ar] prefill failed @%d\n", i);
            return 1;
        }

        ggml_backend_tensor_get(logits_out, logit_buf.data(), 0, sizeof(float) * vocab);
        last_tok = argmax_f32(logit_buf.data(), vocab);
        committed = i + 1;
    }
    auto t_pf1 = std::chrono::steady_clock::now();
    double pf_ms = std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count();
    std::printf("[moe-ar] prefill %d tokens in %.1f ms (%.1f tok/s)\n",
                committed, pf_ms, committed * 1000.0 / pf_ms);

    // ── Generation ──
    auto t_gen0 = std::chrono::steady_clock::now();
    int n_generated = 0;
    std::vector<int32_t> out_all;

    while (n_generated < n_gen) {
        w.embedder.embed(&last_tok, 1, embed.data());
        ggml_backend_tensor_set(act_a, embed.data(), 0, sizeof(float) * hidden);

        const int pos = committed;
        int32_t pos4[4] = {pos, pos, pos, 0};
        ggml_backend_tensor_set(positions, pos4, 0, sizeof(int32_t) * 4);

        const int kv_len = pos + 1;
        std::vector<uint16_t> mask_data;
        ggml_tensor * mask_ptr = nullptr;
        if (kv_len > 1) {
            build_causal_mask(mask_data, kv_len, 1, pos);
            const int kv_pad = align_up(kv_len, KQ_MASK_PAD);
            attn_mask->ne[0] = kv_pad;
            attn_mask->ne[1] = align_up(1, KQ_MASK_PAD);
            attn_mask->nb[1] = (size_t)kv_pad * ggml_element_size(attn_mask);
            ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                sizeof(uint16_t) * mask_data.size());
            mask_ptr = attn_mask;
        }

        if (!run_qwen35moe_forward(backend, w, cache, pinned,
                act_a, act_b, 1, positions, mask_ptr, pos, false, 0, logits_out, nullptr,
                nullptr, nullptr)) {
            std::fprintf(stderr, "[moe-ar] forward failed @pos=%d\n", pos);
            return 1;
        }

        ggml_backend_tensor_get(logits_out, logit_buf.data(), 0, sizeof(float) * vocab);
        last_tok = argmax_f32(logit_buf.data(), vocab);
        out_all.push_back(last_tok);
        committed++;
        n_generated++;

        if (IS_EOS_TOK(last_tok, w)) {
            std::printf("[moe-ar] EOS after %d tokens\n", n_generated);
            break;
        }
    }

    auto t_gen1 = std::chrono::steady_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_gen1 - t_gen0).count();
    std::printf("\n=== MoE Autoregressive Results ===\n");
    std::printf("Generated %d tokens in %.1f ms (%.1f tok/s)\n",
                n_generated, gen_ms, n_generated * 1000.0 / gen_ms);

    // Detokenize and print
    auto vocab_strs = load_gguf_vocab(target_path);
    if (!vocab_strs.empty()) {
        std::string text = detokenize(vocab_strs, out_all);
        std::printf("\n--- Generated text ---\n%s\n---\n", text.c_str());
    } else {
        std::printf("\n--- Generated token IDs (%d) ---\n", (int)out_all.size());
        for (int i = 0; i < std::min((int)out_all.size(), 128); i++) std::printf("%d ", out_all[i]);
        std::printf("\n---\n");
    }

    if (out_path) write_int32_file(out_path, out_all);

    // Cleanup
    ggml_backend_buffer_free(pos_buf); ggml_free(pos_ctx);
    ggml_backend_buffer_free(out_buf); ggml_free(out_ctx);
    ggml_backend_buffer_free(act_buf); ggml_free(act_ctx);
    free_target_cache(cache);
    return 0;
}

int run_moe_dflash(
    const char * target_path,
    const char * draft_path,
    const char * prompt_path,
    int n_gen,
    const char * out_path,
    int budget,
    ggml_backend_t backend,
    ggml_backend_t draft_backend,
    TargetWeights & w,
    DraftWeights & dw,
    bool ddtree_mode = false,
    int  ddtree_budget = 22,
    float ddtree_temp = 1.0f,
    bool ddtree_chain_seed = true)
{
    using namespace dflash27b;
    const int hidden = w.n_embd;
    const int vocab  = (int)w.embedder.n_vocab;
    const int draft_block = DFLASH27B_DRAFT_BLOCK_SIZE;  // draft always produces 16
    const int q_len  = std::min(budget, draft_block);    // verify at most budget tokens (chain mode)
    const int max_ctx = g_max_ctx_override > 0 ? g_max_ctx_override : 4096;
    // DDTree needs room for 1 + ddtree_budget tree nodes in intermediate buffers.
    const int max_verify = ddtree_mode
        ? std::max(q_len + 1, ddtree_budget + 1)
        : q_len + 1;

    std::printf("[moe] hidden=%d vocab=%d n_layer=%d experts=%d active=%d budget=%d ddtree=%d(%d)\n",
                hidden, vocab, w.n_layer, w.n_experts, w.n_experts_active, q_len,
                (int)ddtree_mode, ddtree_budget);

    // ── Pin ALL layers into VRAM (no swap support) ──
    PinnedExperts pinned;
    {
        std::vector<int> pin_ids;
        for (int l = 0; l < w.n_layer; l++) pin_ids.push_back(l);
        if (!pinned.init(backend, w.expert_source, pin_ids)) {
            std::fprintf(stderr, "[moe] PinnedExperts init failed\n");
            return 1;
        }
    }



    // ── TargetCache ──
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, max_verify, backend, cache, /*prefill_only=*/true)) {
        std::fprintf(stderr, "[moe] cache: %s\n", dflash27b_last_error());
        return 1;
    }

    // ── Activation buffers ──
    ggml_init_params act_ip{};
    act_ip.mem_size = 8 * ggml_tensor_overhead() + 4096;
    act_ip.no_alloc = true;
    ggml_context * act_ctx = ggml_init(act_ip);
    ggml_tensor * act_a = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, max_verify);
    ggml_tensor * act_b = ggml_new_tensor_2d(act_ctx, GGML_TYPE_F32, hidden, max_verify);
    ggml_set_name(act_a, "act_a"); ggml_set_name(act_b, "act_b");
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(act_ctx, backend);

    // ── Output tensors ──
    ggml_init_params out_ip{};
    out_ip.mem_size = 8 * ggml_tensor_overhead() + 4096;
    out_ip.no_alloc = true;
    ggml_context * out_ctx = ggml_init(out_ip);
    ggml_tensor * logits_out = ggml_new_tensor_2d(out_ctx, GGML_TYPE_F32, vocab, max_verify);
    ggml_tensor * argmax_out = ggml_new_tensor_1d(out_ctx, GGML_TYPE_I32, max_verify);
    ggml_set_name(logits_out, "logits_out"); ggml_set_name(argmax_out, "argmax_out");
    ggml_backend_buffer_t out_buf = ggml_backend_alloc_ctx_tensors(out_ctx, backend);

    // ── Positions + mask + parent_ids ──
    ggml_init_params pos_ip{};
    pos_ip.mem_size = 12 * ggml_tensor_overhead() + 4096;
    pos_ip.no_alloc = true;
    ggml_context * pos_ctx = ggml_init(pos_ip);
    ggml_tensor * positions = ggml_new_tensor_1d(pos_ctx, GGML_TYPE_I32, 4 * max_verify);
    ggml_set_name(positions, "positions");
    ggml_tensor * attn_mask = ggml_new_tensor_2d(pos_ctx, GGML_TYPE_F16,
        align_up(max_ctx + max_verify, KQ_MASK_PAD), align_up(max_verify, KQ_MASK_PAD));
    ggml_set_name(attn_mask, "attn_mask");
    ggml_tensor * parent_ids_t = nullptr;
    if (ddtree_mode) {
        parent_ids_t = ggml_new_tensor_1d(pos_ctx, GGML_TYPE_I32, max_verify);
        ggml_set_name(parent_ids_t, "parent_ids");
    }
    ggml_backend_buffer_t pos_buf = ggml_backend_alloc_ctx_tensors(pos_ctx, backend);

    // ── Load prompt ──
    std::vector<int32_t> prompt_tokens = read_int32_file(prompt_path);
    if (prompt_tokens.empty()) {
        std::fprintf(stderr, "[moe] failed to read prompt: %s\n", prompt_path);
        return 1;
    }
    std::printf("[moe] prompt: %d tokens\n", (int)prompt_tokens.size());

    // ── Prefill ──
    auto t_pf0 = std::chrono::steady_clock::now();
    int committed = 0;
    int last_tok = -1;
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        std::vector<float> embed(hidden);
        w.embedder.embed(&prompt_tokens[i], 1, embed.data());
        ggml_backend_tensor_set(act_a, embed.data(), 0, sizeof(float) * hidden);

        std::vector<int32_t> pos4 = {i, i, i, 0};
        positions->ne[0] = 4;
        ggml_backend_tensor_set(positions, pos4.data(), 0, sizeof(int32_t) * 4);

        const int kv_len = i + 1;
        std::vector<uint16_t> mask_data;
        ggml_tensor * mask_ptr = nullptr;
        if (kv_len > 1) {
            build_causal_mask(mask_data, kv_len, 1, i);
            const int kv_pad = align_up(kv_len, KQ_MASK_PAD);
            attn_mask->ne[0] = kv_pad;
            attn_mask->ne[1] = align_up(1, KQ_MASK_PAD);
            attn_mask->nb[1] = (size_t)kv_pad * ggml_element_size(attn_mask);
            ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                sizeof(uint16_t) * mask_data.size());
            mask_ptr = attn_mask;
        }

        if (!run_qwen35moe_forward(backend, w, cache, pinned,
                act_a, act_b, 1, positions, mask_ptr, i, true, 0, logits_out, nullptr,
                nullptr, nullptr)) {
            std::fprintf(stderr, "[moe] prefill failed @%d\n", i);
            return 1;
        }

        std::vector<float> logit_buf(vocab);
        ggml_backend_tensor_get(logits_out, logit_buf.data(), 0, sizeof(float) * vocab);
        last_tok = argmax_f32(logit_buf.data(), vocab);
        committed = i + 1;
    }
    auto t_pf1 = std::chrono::steady_clock::now();
    double pf_ms = std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count();
    std::printf("[moe] prefill %d tokens in %.1f ms (%.1f tok/s), last_tok=%d\n",
                committed, pf_ms, committed * 1000.0 / pf_ms, last_tok);

    // ════════════════════════════════════════════════════════════════════
    // Simple autoregressive generation (budget==1): no draft model needed
    // ════════════════════════════════════════════════════════════════════
    if (budget == 1) {
        g_moe_debug_sync = false;
        auto t_gen0 = std::chrono::steady_clock::now();
        int n_generated = 0;
        std::vector<int32_t> out_all;
        std::vector<float> embed(hidden);
        std::vector<float> logit_buf(vocab);

        while (n_generated < n_gen) {
            // Embed the last token
            w.embedder.embed(&last_tok, 1, embed.data());
            ggml_backend_tensor_set(act_a, embed.data(), 0, sizeof(float) * hidden);

            // Position
            const int pos = committed;
            std::vector<int32_t> pos4 = {pos, pos, pos, 0};
            positions->ne[0] = 4;
            ggml_backend_tensor_set(positions, pos4.data(), 0, sizeof(int32_t) * 4);

            // Mask (kv_len > 1 needs causal mask for flash-attn)
            const int kv_len = pos + 1;
            std::vector<uint16_t> mask_data;
            ggml_tensor * mask_ptr = nullptr;
            if (kv_len > 1) {
                build_causal_mask(mask_data, kv_len, 1, pos);
                const int kv_pad = align_up(kv_len, KQ_MASK_PAD);
                attn_mask->ne[0] = kv_pad;
                attn_mask->ne[1] = align_up(1, KQ_MASK_PAD);
                attn_mask->nb[1] = (size_t)kv_pad * ggml_element_size(attn_mask);
                ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                    sizeof(uint16_t) * mask_data.size());
                mask_ptr = attn_mask;
            }

            // Forward (n_tok=1, no capture)
            if (!run_qwen35moe_forward(backend, w, cache, pinned,
                    act_a, act_b, 1, positions, mask_ptr, pos, false, 0, logits_out, nullptr,
                    nullptr, nullptr)) {
                std::fprintf(stderr, "[moe] forward failed @pos=%d\n", pos);
                return 1;
            }

            // Read logits → argmax
            ggml_backend_tensor_get(logits_out, logit_buf.data(), 0, sizeof(float) * vocab);
            last_tok = argmax_f32(logit_buf.data(), vocab);
            out_all.push_back(last_tok);
            committed++;
            n_generated++;

            if (IS_EOS_TOK(last_tok, w)) {
                std::printf("[moe] EOS after %d tokens\n", n_generated);
                break;
            }
        }

        auto t_gen1 = std::chrono::steady_clock::now();
        double gen_ms = std::chrono::duration<double, std::milli>(t_gen1 - t_gen0).count();
        std::printf("\n=== MoE Autoregressive Results ===\n");
        std::printf("Generated %d tokens in %.1f ms (%.1f tok/s)\n",
                    n_generated, gen_ms, n_generated * 1000.0 / gen_ms);

        // Detokenize and print
        auto vocab_strs = load_gguf_vocab(target_path);
        if (!vocab_strs.empty()) {
            std::string text = detokenize(vocab_strs, out_all);
            std::printf("\n--- Generated text ---\n%s\n---\n", text.c_str());
        } else {
            std::printf("\n--- Generated token IDs (%d) ---\n", (int)out_all.size());
            for (int i = 0; i < std::min((int)out_all.size(), 128); i++) std::printf("%d ", out_all[i]);
            std::printf("\n---\n");
        }

        if (out_path) write_int32_file(out_path, out_all);

        // Cleanup
        ggml_backend_buffer_free(pos_buf); ggml_free(pos_ctx);
        ggml_backend_buffer_free(out_buf); ggml_free(out_ctx);
        ggml_backend_buffer_free(act_buf); ggml_free(act_ctx);
        free_target_cache(cache);
        return 0;
    }

    // ── Migrate cache for rollback (needed for spec-decode path) ──
    if (!migrate_prefill_cache(w, max_ctx, max_verify, backend, cache)) {
        std::fprintf(stderr, "[moe] migrate: %s\n", dflash27b_last_error());
        return 1;
    }

    // ── Draft context ──
    struct MoeDraftCtx {
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ggml_tensor * inp_embed = nullptr;
        ggml_tensor * target_hidden_cat = nullptr;
        ggml_tensor * positions_q = nullptr;
        ggml_tensor * positions_k = nullptr;
        ggml_tensor * logits = nullptr;
        ggml_tensor * argmax_tokens = nullptr;
    };
    auto draft_free = [](MoeDraftCtx & d) {
        if (d.alloc) { ggml_gallocr_free(d.alloc); d.alloc = nullptr; }
        if (d.ctx) { ggml_free(d.ctx); d.ctx = nullptr; }
        d.gf = nullptr;
    };
    auto draft_build = [&](MoeDraftCtx & d, int ctx_len) -> bool {
        draft_free(d);
        ggml_init_params ip{};
        ip.mem_size = 256 * 1024 * 1024;
        ip.no_alloc = true;
        d.ctx = ggml_init(ip);
        if (!d.ctx) return false;
        const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * w.n_embd;
        d.inp_embed = ggml_new_tensor_3d(d.ctx, GGML_TYPE_F32, hidden, draft_block, 1);
        ggml_set_name(d.inp_embed, "inp_embed"); ggml_set_input(d.inp_embed);
        d.target_hidden_cat = ggml_new_tensor_3d(d.ctx, GGML_TYPE_F32, fc_in, ctx_len, 1);
        ggml_set_name(d.target_hidden_cat, "target_hidden_cat"); ggml_set_input(d.target_hidden_cat);
        d.positions_q = ggml_new_tensor_1d(d.ctx, GGML_TYPE_I32, draft_block);
        ggml_set_name(d.positions_q, "positions_q"); ggml_set_input(d.positions_q);
        d.positions_k = ggml_new_tensor_1d(d.ctx, GGML_TYPE_I32, ctx_len + draft_block);
        ggml_set_name(d.positions_k, "positions_k"); ggml_set_input(d.positions_k);
        d.gf = ggml_new_graph_custom(d.ctx, 4096, false);
        DraftGraphInputs gi{};
        gi.ctx_len = ctx_len;
        gi.noise_embed = d.inp_embed;
        gi.target_hidden_cat = d.target_hidden_cat;
        gi.positions_q = d.positions_q;
        gi.positions_k = d.positions_k;
        gi.lm_head = w.output;
        DraftGraphOutputs go = build_draft_graph(d.ctx, dw, gi);
        d.logits = go.logits;
        if (!d.logits) return false;
        d.argmax_tokens = ggml_argmax(d.ctx, d.logits);
        ggml_set_name(d.argmax_tokens, "draft_argmax");
        ggml_set_output(d.argmax_tokens);
        ggml_build_forward_expand(d.gf, d.argmax_tokens);
        if (!d.alloc) d.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(draft_backend));
        return ggml_gallocr_alloc_graph(d.alloc, d.gf);
    };

    // ── Decode loop ──
    g_moe_debug_sync = false;
    auto t_gen0 = std::chrono::steady_clock::now();
    int n_generated = 0, n_draft_steps = 0, n_accept_sum = 0;
    std::vector<int32_t> out_all;
    std::vector<float> noise_embed_buf((size_t)hidden * draft_block);
    std::vector<int32_t> noise_ids(draft_block);
    std::vector<int32_t> draft_tok(q_len), target_tok(q_len);
    MoeDraftCtx dctx;
    int draft_ctx_len = -1;
    constexpr int DRAFT_CTX_MAX = 2048;
    const int mask_tok = DFLASH27B_DRAFT_MASK_TOKEN_ID;

    while (n_generated < n_gen) {
        // ─ 1. Draft ─
        noise_ids[0] = last_tok;
        for (int i = 1; i < draft_block; i++) noise_ids[i] = mask_tok;
        w.embedder.embed(noise_ids.data(), draft_block, noise_embed_buf.data());

        const int cur_draft_ctx = std::min(committed, DRAFT_CTX_MAX);
        const int draft_start = committed - cur_draft_ctx;
        if (cur_draft_ctx != draft_ctx_len) {
            if (!draft_build(dctx, cur_draft_ctx)) {
                std::fprintf(stderr, "[moe] draft build failed\n"); return 1;
            }
            draft_ctx_len = cur_draft_ctx;
        }
        ggml_backend_tensor_set(dctx.inp_embed, noise_embed_buf.data(), 0,
                                sizeof(float) * noise_embed_buf.size());

        // Copy target_feat → draft (bf16 → f32)
        if (cache.target_feat && cur_draft_ctx > 0) {
            const int cap = cache.target_feat_cap;
            const size_t fc_in = (size_t)DFLASH27B_DRAFT_N_TARGET_LAYERS * w.n_embd;
            const size_t elt_feat = ggml_element_size(cache.target_feat);
            const int slot0 = draft_start % cap;
            const int pre_n = std::min(cur_draft_ctx, cap - slot0);
            const int post_n = cur_draft_ctx - pre_n;
            dflash27b_launch_bf16_to_f32(
                (const char *)cache.target_feat->data + (size_t)slot0 * elt_feat * fc_in,
                dctx.target_hidden_cat->data, (size_t)pre_n * fc_in, nullptr);
            if (post_n > 0) {
                dflash27b_launch_bf16_to_f32(
                    (const char *)cache.target_feat->data,
                    (char *)dctx.target_hidden_cat->data + (size_t)pre_n * fc_in * sizeof(float),
                    (size_t)post_n * fc_in, nullptr);
            }
            cudaDeviceSynchronize();
        }

        // Draft positions + compute (always 16 tokens)
        std::vector<int32_t> pos_q(draft_block), pos_k(cur_draft_ctx + draft_block);
        for (int i = 0; i < draft_block; i++) pos_q[i] = cur_draft_ctx + i;
        for (int i = 0; i < cur_draft_ctx + draft_block; i++) pos_k[i] = i;
        ggml_backend_tensor_set(dctx.positions_q, pos_q.data(), 0, sizeof(int32_t) * draft_block);
        ggml_backend_tensor_set(dctx.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * (cur_draft_ctx + draft_block));
        ggml_backend_graph_compute(draft_backend, dctx.gf);
        // Read first q_len tokens from the 16 draft outputs
        ggml_backend_tensor_get(dctx.argmax_tokens, draft_tok.data(), 0, sizeof(int32_t) * q_len);
        draft_tok[0] = last_tok;

        // ── DDTree path ──
        if (ddtree_mode) {
            const int L = q_len - 1;  // max tree depth
            const int ddK = (ddtree_budget > L) ? 8 : 1;

            // Extract top-K from draft logits (positions 1..q_len-1)
            std::vector<float>   top_log_probs((size_t)L * ddK, 0.0f);
            std::vector<int32_t> top_token_ids((size_t)L * ddK, 0);
            if (ddK == 1) {
                for (int i = 0; i < L; i++) {
                    top_log_probs[i] = 0.0f;
                    top_token_ids[i] = draft_tok[i + 1];
                }
            } else {
                std::vector<float> draft_logits_buf((size_t)vocab * q_len);
                ggml_backend_tensor_get(dctx.logits, draft_logits_buf.data(), 0,
                                        sizeof(float) * vocab * q_len);
                extract_draft_topk(draft_logits_buf.data() + (size_t)vocab,
                                   L, vocab, ddK,
                                   top_log_probs.data(), top_token_ids.data(),
                                   ddtree_temp);
            }

            // Build DDTree
            DDTree tree = build_ddtree(top_log_probs.data(), top_token_ids.data(),
                                       L, ddK, ddtree_budget, ddtree_chain_seed);
            const int N_actual = 1 + tree.n_nodes;
            const int N = ddtree_budget + 1;  // fixed alloc size for gallocr reuse

            // Snapshot SSM state
            snapshot_ssm_state(cache);

            // Embeddings: [last_tok, tree.token_ids[0..n_nodes-1], padding...]
            std::vector<int32_t> flat_tokens(N, 0);
            flat_tokens[0] = last_tok;
            for (int i = 0; i < tree.n_nodes; i++) flat_tokens[1 + i] = tree.token_ids[i];

            std::vector<float> tree_embed((size_t)hidden * N, 0.0f);
            w.embedder.embed(flat_tokens.data(), N_actual, tree_embed.data());
            ggml_backend_tensor_set(act_a, tree_embed.data(), 0, sizeof(float) * hidden * N);

            // M-RoPE positions (axis-major)
            std::vector<int32_t> pos4(4 * N, 0);
            for (int i = 0; i < N_actual; i++) {
                int p = committed + (i == 0 ? 0 : tree.depths[i - 1]);
                pos4[0 * N + i] = p;
                pos4[1 * N + i] = p;
                pos4[2 * N + i] = p;
                pos4[3 * N + i] = 0;
            }
            positions->ne[0] = 4 * N;
            ggml_backend_tensor_set(positions, pos4.data(), 0, sizeof(int32_t) * 4 * N);

            // Ancestor-only tree mask
            const int tree_kv_len = committed + N;
            const int kv_pad_m = align_up(tree_kv_len, KQ_MASK_PAD);
            const int q_pad_m  = align_up(N, KQ_MASK_PAD);
            std::vector<uint16_t> tree_mask((size_t)kv_pad_m * q_pad_m, F16_NEG_INF);
            for (int q = 0; q < N_actual; q++) {
                // Past KV positions: visible to all tree nodes
                for (int k = 0; k < committed; k++) {
                    tree_mask[(size_t)q * kv_pad_m + k] = F16_ZERO;
                }
                // Tree self-visibility
                for (int j = 0; j < N_actual; j++) {
                    if (tree.visibility[(size_t)q * N_actual + j]) {
                        tree_mask[(size_t)q * kv_pad_m + (committed + j)] = F16_ZERO;
                    }
                }
            }
            attn_mask->ne[0] = kv_pad_m;
            attn_mask->ne[1] = q_pad_m;
            attn_mask->nb[1] = (size_t)kv_pad_m * ggml_element_size(attn_mask);
            ggml_backend_tensor_set(attn_mask, tree_mask.data(), 0,
                sizeof(uint16_t) * tree_mask.size());

            // parent_ids tensor
            std::vector<int32_t> parent_ids_data(N, 0);
            parent_ids_data[0] = -1;
            for (int i = 1; i < N_actual; i++) parent_ids_data[i] = (int32_t)tree.parents[i];
            ggml_backend_tensor_set(parent_ids_t, parent_ids_data.data(), 0,
                sizeof(int32_t) * N);

            // DeltaNetCapture: point at cache's intermediate buffers
            const int n_delta = (int)cache.ssm_state.size();
            std::vector<DeltaNetCapture> delta_caps(n_delta);
            for (int il = 0; il < n_delta; il++) {
                delta_caps[il].ssm_intermediate_states = cache.ssm_intermediate[il];
                delta_caps[il].conv_input              = cache.conv_input_cache[il];
            }

            // Run tree-structured verify through MoE forward
            auto t_verify0 = std::chrono::steady_clock::now();
            if (!run_qwen35moe_forward(backend, w, cache, pinned,
                    act_a, act_b, N, positions, attn_mask,
                    committed, true, 0, logits_out, argmax_out,
                    parent_ids_t, &delta_caps)) {
                std::fprintf(stderr, "[moe] DDTree verify failed step %d\n", n_draft_steps);
                return 1;
            }
            auto t_verify1 = std::chrono::steady_clock::now();
            double verify_ms = std::chrono::duration<double, std::milli>(t_verify1 - t_verify0).count();

            // Read argmax for the actual tree slots
            std::vector<int32_t> posterior(N_actual);
            ggml_backend_tensor_get(argmax_out, posterior.data(), 0, sizeof(int32_t) * N_actual);

            // Walk tree
            int next_token = -1;
            int bonus_node_idx = 0;
            std::vector<int> accepted = follow_verified_tree(tree, posterior.data(), next_token, &bonus_node_idx);
            const int accept_depth = (int)accepted.size();

            int commit_n = accept_depth;
            const int need_budget = n_gen - n_generated;
            if (commit_n > need_budget) commit_n = need_budget;

            // Commit accepted tokens
            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                const int dfs_idx = accepted[i];
                const int32_t tok = (dfs_idx == 0) ? last_tok : tree.token_ids[dfs_idx - 1];
                out_all.push_back(tok);
                if (tok == w.eos_id || tok == w.eos_chat_id) hit_eos = true;
            }
            last_tok = next_token;

            // Rollback SSM + conv state
            const int rollback_dfs = (commit_n > 0) ? accepted[commit_n - 1] : 0;
            bool walked_sibling = false;
            for (int i = 0; i < commit_n; i++) {
                if (accepted[i] != i) { walked_sibling = true; break; }
            }
            {
                cudaStream_t stream = nullptr;
                for (int il = 0; il < n_delta; il++) {
                    const DeltaNetCapture & cap = delta_caps[il];
                    if (!cap.ssm_intermediate_states || !cap.conv_input) continue;

                    // SSM state: f16 intermediate → f32 state
                    const size_t ssm_elems =
                        (size_t)cache.ssm_state[il]->ne[0] *
                        (size_t)cache.ssm_state[il]->ne[1] *
                        (size_t)cache.ssm_state[il]->ne[2];
                    const size_t ssm_src_offset =
                        (size_t)rollback_dfs * cap.ssm_intermediate_states->nb[3];
                    const void * ssm_src =
                        (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
                    dflash27b_launch_f16_to_f32(ssm_src, cache.ssm_state[il]->data,
                                                ssm_elems, stream);

                    // Conv state rollback
                    const int K_conv = 4;
                    const int row_cnt = (int)cap.conv_input->ne[1];
                    const size_t elt = ggml_element_size(cap.conv_input);
                    const size_t dpitch = (K_conv - 1) * elt;
                    const size_t spitch = cap.conv_input->nb[1];
                    if (!walked_sibling) {
                        const int conv_off = rollback_dfs + 1;
                        const void * conv_src =
                            (const char *)cap.conv_input->data + (size_t)conv_off * elt;
                        cudaMemcpy2DAsync(cache.conv_state[il]->data, dpitch,
                                           conv_src, spitch,
                                           (K_conv - 1) * elt, row_cnt,
                                           cudaMemcpyDeviceToDevice, stream);
                    } else {
                        int virt[K_conv - 1];
                        virt[K_conv - 2] = rollback_dfs;
                        for (int k = K_conv - 3; k >= 0; k--) {
                            const int prev = virt[k + 1];
                            virt[k] = (prev >= 0) ? (int)tree.parents[prev] : (prev - 1);
                        }
                        for (int k = 0; k < K_conv - 1; k++) {
                            const int sx_slot = (K_conv - 1) + virt[k];
                            const void * src_col =
                                (const char *)cap.conv_input->data + (size_t)sx_slot * elt;
                            char * dst_col =
                                (char *)cache.conv_state[il]->data + (size_t)k * elt;
                            cudaMemcpy2DAsync(dst_col, dpitch,
                                               src_col, spitch,
                                               elt, row_cnt,
                                               cudaMemcpyDeviceToDevice, stream);
                        }
                    }
                }

                // KV compaction for full-attention layers
                const int n_full_attn = (int)cache.attn_k.size();
                for (int d = 0; d < commit_n; d++) {
                    const int src_dfs = accepted[d];
                    if (src_dfs == d) continue;
                    for (int l = 0; l < n_full_attn; l++) {
                        ggml_tensor * ck = cache.attn_k[l];
                        ggml_tensor * cv = cache.attn_v[l];
                        const size_t slot_bytes = ck->nb[1];
                        const size_t src_off = (size_t)(committed + src_dfs) * slot_bytes;
                        const size_t dst_off = (size_t)(committed + d) * slot_bytes;
                        const int n_kv = (int)ck->ne[2];
                        for (int h = 0; h < n_kv; h++) {
                            const size_t head_src = src_off + (size_t)h * ck->nb[2];
                            const size_t head_dst = dst_off + (size_t)h * ck->nb[2];
                            cudaMemcpyAsync((char *)ck->data + head_dst,
                                            (const char *)ck->data + head_src,
                                            slot_bytes, cudaMemcpyDeviceToDevice, stream);
                            cudaMemcpyAsync((char *)cv->data + head_dst,
                                            (const char *)cv->data + head_src,
                                            slot_bytes, cudaMemcpyDeviceToDevice, stream);
                        }
                    }
                }

                // target_feat compaction
                if (cache.target_feat) {
                    const size_t elt = ggml_element_size(cache.target_feat);
                    const size_t col_stride = cache.target_feat->nb[1];
                    const int    tcap = cache.target_feat_cap;
                    const int    fc_in = (int)cache.target_feat->ne[0];
                    for (int d = 1; d < commit_n; d++) {
                        const int src_dfs = accepted[d];
                        if (src_dfs == d) continue;
                        const int src_slot = (committed + src_dfs) % tcap;
                        const int dst_slot = (committed + d) % tcap;
                        const size_t src_off = (size_t)src_slot * col_stride;
                        const size_t dst_off = (size_t)dst_slot * col_stride;
                        cudaMemcpyAsync((char *)cache.target_feat->data + dst_off,
                                        (const char *)cache.target_feat->data + src_off,
                                        (size_t)fc_in * elt,
                                        cudaMemcpyDeviceToDevice, stream);
                    }
                }
            }

            committed    += commit_n;
            n_generated  += commit_n;
            n_accept_sum += accept_depth;
            n_draft_steps++;

            std::printf("[step %d] DDTree N=%d accept=%d commit=%d | verify: %.1fms | next=%d\n",
                        n_draft_steps - 1, N_actual, accept_depth, commit_n,
                        verify_ms, next_token);

            if (hit_eos) { std::printf("[moe] EOS after %d tokens\n", n_generated); break; }
            continue;  // skip chain path
        }

        // ─ 2. Snapshot ─ (chain path)
        snapshot_ssm_state(cache);

        // ─ 3. Verify ─
        std::vector<float> verify_embed((size_t)hidden * q_len);
        w.embedder.embed(draft_tok.data(), q_len, verify_embed.data());
        ggml_backend_tensor_set(act_a, verify_embed.data(), 0, sizeof(float) * verify_embed.size());

        std::vector<int32_t> pos4(4 * q_len);
        for (int i = 0; i < q_len; i++) {
            int p = committed + i;
            pos4[0 * q_len + i] = p; pos4[1 * q_len + i] = p;
            pos4[2 * q_len + i] = p; pos4[3 * q_len + i] = 0;
        }
        positions->ne[0] = 4 * q_len;
        ggml_backend_tensor_set(positions, pos4.data(), 0, sizeof(int32_t) * 4 * q_len);

        const int verify_kv_len = committed + q_len;
        std::vector<uint16_t> verify_mask;
        build_causal_mask(verify_mask, verify_kv_len, q_len, committed);
        const int vkv_pad = align_up(verify_kv_len, KQ_MASK_PAD);
        const int vq_pad = align_up(q_len, KQ_MASK_PAD);
        attn_mask->ne[0] = vkv_pad;
        attn_mask->ne[1] = vq_pad;
        attn_mask->nb[1] = (size_t)vkv_pad * ggml_element_size(attn_mask);
        ggml_backend_tensor_set(attn_mask, verify_mask.data(), 0,
            sizeof(uint16_t) * vkv_pad * vq_pad);

        auto t_verify0 = std::chrono::steady_clock::now();
        if (!run_qwen35moe_forward(backend, w, cache, pinned,
                act_a, act_b, q_len, positions, attn_mask,
                committed, true, 0, logits_out, argmax_out,
                nullptr, nullptr)) {
            std::fprintf(stderr, "[moe] verify failed step %d\n", n_draft_steps);
            return 1;
        }
        auto t_verify1 = std::chrono::steady_clock::now();
        double verify_ms = std::chrono::duration<double, std::milli>(t_verify1 - t_verify0).count();
        ggml_backend_tensor_get(argmax_out, target_tok.data(), 0, sizeof(int32_t) * q_len);

        // ─ 4. Accept ─
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        const int budget = n_gen - n_generated;
        if (commit_n > budget) { commit_n = budget; if (commit_n <= accept_n) bonus_tok = -1; }

        // ─ 5. Restore + Replay ─
        restore_ssm_state(cache);
        std::vector<int32_t> replay_tok(commit_n);
        for (int i = 0; i < commit_n; i++)
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;

        std::vector<float> replay_embed((size_t)hidden * commit_n);
        w.embedder.embed(replay_tok.data(), commit_n, replay_embed.data());
        ggml_backend_tensor_set(act_a, replay_embed.data(), 0, sizeof(float) * replay_embed.size());

        std::vector<int32_t> rpos4(4 * commit_n);
        for (int i = 0; i < commit_n; i++) {
            int p = committed + i;
            rpos4[0 * commit_n + i] = p; rpos4[1 * commit_n + i] = p;
            rpos4[2 * commit_n + i] = p; rpos4[3 * commit_n + i] = 0;
        }
        positions->ne[0] = 4 * commit_n;
        ggml_backend_tensor_set(positions, rpos4.data(), 0, sizeof(int32_t) * 4 * commit_n);

        const int replay_kv_len = committed + commit_n;
        std::vector<uint16_t> replay_mask;
        build_causal_mask(replay_mask, replay_kv_len, commit_n, committed);
        const int rkv_pad = align_up(replay_kv_len, KQ_MASK_PAD);
        const int rq_pad = align_up(commit_n, KQ_MASK_PAD);
        attn_mask->ne[0] = rkv_pad;
        attn_mask->ne[1] = rq_pad;
        attn_mask->nb[1] = (size_t)rkv_pad * ggml_element_size(attn_mask);
        ggml_backend_tensor_set(attn_mask, replay_mask.data(), 0,
            sizeof(uint16_t) * rkv_pad * rq_pad);

        auto t_replay0 = std::chrono::steady_clock::now();
        if (!run_qwen35moe_forward(backend, w, cache, pinned,
                act_a, act_b, commit_n, positions, attn_mask,
                committed, true, 0, logits_out, nullptr,
                nullptr, nullptr)) {
            std::fprintf(stderr, "[moe] replay failed step %d\n", n_draft_steps);
            return 1;
        }
        auto t_replay1 = std::chrono::steady_clock::now();
        double replay_ms = std::chrono::duration<double, std::milli>(t_replay1 - t_replay0).count();

        std::vector<float> replay_logits(vocab);
        ggml_backend_tensor_get(logits_out, replay_logits.data(),
                                sizeof(float) * (size_t)vocab * (commit_n - 1),
                                sizeof(float) * vocab);
        last_tok = argmax_f32(replay_logits.data(), vocab);

        // ─ 6. Commit ─
        bool hit_eos = false;
        for (int i = 0; i < commit_n; i++) {
            out_all.push_back(replay_tok[i]);
            if (replay_tok[i] == w.eos_id || replay_tok[i] == w.eos_chat_id) hit_eos = true;
        }
        committed += commit_n;
        n_generated += commit_n;
        n_accept_sum += accept_n;
        n_draft_steps++;

        std::printf("[step %d] accept=%d commit=%d | verify: %.1fms | replay: %.1fms\n",
                    n_draft_steps - 1, accept_n, commit_n,
                    verify_ms,
                    replay_ms);

        if (hit_eos) { std::printf("[moe] EOS after %d tokens\n", n_generated); break; }
    }

    auto t_gen1 = std::chrono::steady_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_gen1 - t_gen0).count();
    std::printf("\n=== MoE Results ===\n");
    std::printf("Generated %d tokens in %.1f ms (%.1f tok/s)\n",
                n_generated, gen_ms, n_generated * 1000.0 / gen_ms);
    std::printf("Draft steps: %d, avg accept: %.2f\n",
                n_draft_steps, n_draft_steps > 0 ? (double)n_accept_sum / n_draft_steps : 0.0);

    // Detokenize and print
    auto vocab_strs = load_gguf_vocab(target_path);
    if (!vocab_strs.empty()) {
        std::string text = detokenize(vocab_strs, out_all);
        std::printf("\n--- Generated text ---\n%s\n---\n", text.c_str());
    } else {
        std::printf("\n--- Generated token IDs (%d) ---\n", (int)out_all.size());
        for (int i = 0; i < std::min((int)out_all.size(), 128); i++) std::printf("%d ", out_all[i]);
        std::printf("\n---\n");
    }

    // Write output file
    if (out_path) write_int32_file(out_path, out_all);

    // Cleanup
    draft_free(dctx);
    ggml_backend_buffer_free(pos_buf); ggml_free(pos_ctx);
    ggml_backend_buffer_free(out_buf); ggml_free(out_ctx);
    ggml_backend_buffer_free(act_buf); ggml_free(act_ctx);
    free_target_cache(cache);
    return 0;
}

