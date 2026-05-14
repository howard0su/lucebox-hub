// Generic DFlash speculative decode loop.
//
// See spec_decode_loop.h for interface documentation.

#include "common/spec_decode_loop.h"
#include "common/dflash_target.h"
#include "internal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace dflash27b {

bool generic_spec_decode(
        DFlashTarget & target,
        DraftWeights & dw,
        ggml_backend_t draft_backend,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int last_tok,
        std::function<void(int32_t)> emit,
        SpecDecodeStats * stats) {

    const int hidden = target.hidden_size();
    const int q_len = dw.block_size;
    const int mask_id = target.mask_token_id();

    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    int committed = (int)prompt.size();
    int n_generated = 0;
    int n_draft_steps = 0;
    int n_accept_sum = 0;

    auto t_dec0 = std::chrono::steady_clock::now();

    while (n_generated < n_gen) {
        // 1. Build noise input: [last_tok, MASK, MASK, ...]
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = mask_id;
        if (!target.embed_tokens(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "generic-spec-decode: noise embed failed\n");
            return false;
        }

        // 2. Run draft model forward
        // TODO: Build draft graph with target features and run it.
        // This requires either local draft (build_draft_step) or IPC.
        // For now, this is a stub that shows the structure.
        //
        // The full implementation would:
        //   a) Sync target features to draft feature ring
        //   b) Build and compute draft graph → hidden states
        //   c) Project hidden states through target lm_head → draft_tok
        //   d) Verify draft tokens with target
        //   e) Accept matching prefix

        // Stub: fall back to autoregressive single-token decode
        int next_tok = 0;
        std::vector<int32_t> single_tok = {last_tok};
        if (!target.verify_batch(single_tok, committed, next_tok, nullptr)) {
            std::fprintf(stderr, "generic-spec-decode: verify failed\n");
            return false;
        }

        committed++;
        n_generated++;
        last_tok = next_tok;
        emit(next_tok);

        if (target.is_eos(next_tok)) {
            emit(-1);
            break;
        }
    }

    auto t_dec1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->n_generated = n_generated;
        stats->n_draft_steps = n_draft_steps;
        stats->n_accept_sum = n_accept_sum;
        stats->decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    }

    return true;
}

}  // namespace dflash27b
