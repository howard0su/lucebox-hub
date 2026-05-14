// Feature-copy helpers for the qwen35 speculative decode pipeline.
//
// During speculative decoding, the target model's intermediate hidden states
// (from designated "capture" layers) are accumulated into a ring buffer on the
// draft GPU (DraftFeatureMirror). These helpers move data between the target
// activations, the ring buffer, and draft graph input tensors.

#pragma once

#include "draft_feature_mirror.h"
#include "internal.h"  // TargetWeights, DFLASH27B_* constants

#include "ggml.h"

namespace dflash27b {

// Return the capture-layer index (0..N_TARGET_LAYERS-1) for the given
// absolute layer index, or -1 if this layer is not captured.
int target_capture_index(const TargetWeights & w, int layer_idx);

// Copy one capture slice from a target layer's activation output into the
// DraftFeatureMirror ring buffer. src_device is the GPU device of act_out.
bool copy_capture_slice_to_draft_ring(
    DraftFeatureMirror & feature_ring,
    int capture_idx,
    const ggml_tensor * act_out,
    int src_device,
    int chunk_start,
    int start_pos,
    int n_tokens);

// Copy n_tokens rows from the DraftFeatureMirror ring buffer into a
// destination tensor (typically the draft graph's target_hidden_cat input).
bool copy_feature_ring_range_to_tensor(
    const DraftFeatureMirror & feature_ring,
    ggml_tensor * dst,
    int start_pos,
    int n_tokens);

}  // namespace dflash27b
