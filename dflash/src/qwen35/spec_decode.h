// spec_decode.h — Speculative decoding loop for qwen35 layer-split inference.
//
// Runs the draft model → verify → replay loop that accepts multiple tokens per
// step, achieving higher throughput than plain autoregressive decoding.

#pragma once

#include "layer_split_types.h"
#include "layer_split_forward.h"
#include "draft_ipc.h"
#include "draft_feature_mirror.h"
#include "step_graph.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace dflash27b {

// Run the speculative decode loop: draft → verify → replay.
// Returns true on success, false on any error.
//
// `kq_stride_pad`, `fa_window`, `draft_ctx_max` replace the globals that the
// original test_dflash.cpp code read directly.
bool run_target_layer_split_dflash_decode(
        std::vector<TargetLayerSplitShard> & shards,
        DraftWeights & draft_weights,
        ggml_backend_t draft_backend,
        int draft_gpu,
        DraftFeatureMirror & feature_ring,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int last_tok,
        const char * out_path,
        int kq_stride_pad,
        int fa_window,
        int draft_ctx_max,
        int stream_fd = -1,
        DFlashDraftIpcClient * remote_draft = nullptr);

} // namespace dflash27b
