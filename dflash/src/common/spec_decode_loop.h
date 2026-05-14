// Generic DFlash speculative decode loop.
//
// Drives the DFlash draft model with ANY target that implements DFlashTarget.
// The loop follows the standard DFlash spec-decode protocol:
//   1. Embed noise input (last_tok + mask tokens) using target's embedder
//   2. Run draft model to get hidden-state proposals
//   3. Project draft hidden states through target's lm_head to get candidate tokens
//   4. Run target's verify_batch on candidates
//   5. Accept matching prefix, rollback on mismatch
//   6. Repeat until EOS or n_gen reached
//
// This is the model-agnostic version of qwen35/spec_decode.cpp. It uses the
// DFlashTarget interface instead of direct qwen35 infrastructure.

#pragma once

#include "common/dflash_target.h"
#include "internal.h"         // DraftWeights

#include <cstdint>
#include <functional>
#include <vector>

namespace dflash27b {

struct SpecDecodeStats {
    int n_generated = 0;
    int n_draft_steps = 0;
    int n_accept_sum = 0;
    double decode_s = 0.0;
};

// Run a generic DFlash speculative decode loop.
//
// Parameters:
//   target       — implements DFlashTarget (verify, embed, project, snapshot/restore)
//   dw           — draft model weights (loaded)
//   draft_backend— GPU backend for the draft model
//   prompt       — tokenized prompt (already prefilled into target KV)
//   n_gen        — max tokens to generate
//   last_tok     — last token from prefill (first decode input)
//   emit         — callback to emit each accepted token (pass -1 for EOS sentinel)
//   stats        — output stats (optional)
//
// Returns true on success (EOS or n_gen reached), false on error.
bool generic_spec_decode(
    DFlashTarget & target,
    DraftWeights & dw,
    ggml_backend_t draft_backend,
    const std::vector<int32_t> & prompt,
    int n_gen,
    int last_tok,
    std::function<void(int32_t)> emit,
    SpecDecodeStats * stats = nullptr);

}  // namespace dflash27b
