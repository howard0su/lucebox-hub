#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace dflash27b::sfi {

// Helper: parse an integer environment variable with a default.
inline int parse_env_int(const char * name, int default_val) {
    const char * raw = std::getenv(name);
    if (!raw || !raw[0]) return default_val;
    return std::atoi(raw);
}

struct AttnWindowSlice {
    int  win_start;
    int  win_len;
    int  win_len_padded;
    int  effective_window;
    bool used_slow_refresh;
};

inline AttnWindowSlice resolve_attn_window_slice(
    int  kv_start,
    int  n_tokens,
    bool allow_slow_refresh,
    int  fa_window,
    int  refresh_interval,
    bool uses_256_stride
) {
    const bool do_slow_refresh =
        allow_slow_refresh &&
        refresh_interval > 0 &&
        kv_start > 0 &&
        (kv_start % refresh_interval) == 0;
    const int effective_window = do_slow_refresh ? 0 : fa_window;
    const int win_start = (effective_window > 0 && kv_start > effective_window)
                              ? (kv_start - effective_window) : 0;
    const int kv_len = kv_start + n_tokens;
    const int win_len = kv_len - win_start;
    const int stride = uses_256_stride ? 256 : 1;
    const int win_len_padded = ((win_len + stride - 1) / stride) * stride;
    return AttnWindowSlice{
        win_start,
        win_len,
        win_len_padded,
        effective_window,
        do_slow_refresh,
    };
}

inline std::vector<int> merge_sparse_index_sets(
    int kv_len,
    int sink_tokens,
    int recent_tokens,
    const std::vector<int> & selected_indices
) {
    if (kv_len <= 0) return {};

    const int sink_end = std::clamp(sink_tokens, 0, kv_len);
    const int recent_start = std::max(0, kv_len - std::max(recent_tokens, 0));

    std::vector<int> merged;
    merged.reserve((size_t)sink_end + selected_indices.size() + (size_t)(kv_len - recent_start));

    for (int i = 0; i < sink_end; ++i) {
        merged.push_back(i);
    }

    for (int idx : selected_indices) {
        if (idx < 0 || idx >= kv_len) continue;
        if (idx < sink_end) continue;
        if (idx >= recent_start) continue;
        merged.push_back(idx);
    }

    for (int i = recent_start; i < kv_len; ++i) {
        merged.push_back(i);
    }

    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    return merged;
}

// ── SFI Selector Helpers ─────────────────────────────────────────────

// Update selector scores with EMA of new attention weights.
// `attn_weights` has `kv_len` entries (sum of attention across heads for the
// current decode step). `alpha` controls EMA decay (paper uses ~0.9).
inline void update_selector_scores(
    std::vector<float> & scores,
    const float * attn_weights,
    int kv_len,
    float alpha = 0.9f
) {
    for (int i = 0; i < kv_len && i < (int)scores.size(); ++i) {
        scores[i] = alpha * scores[i] + (1.0f - alpha) * attn_weights[i];
    }
}

// Extract Top-K indices from selector scores (excluding sink/recent which are
// always included). Returns sorted indices in ascending order.
inline std::vector<int> topk_from_scores(
    const std::vector<float> & scores,
    int kv_len,
    int k,
    int sink_tokens,
    int recent_tokens
) {
    if (k <= 0 || kv_len <= 0) return {};

    const int sink_end = std::clamp(sink_tokens, 0, kv_len);
    const int recent_start = std::max(0, kv_len - std::max(recent_tokens, 0));

    // Collect candidate indices (middle region only)
    std::vector<int> candidates;
    candidates.reserve(std::max(0, recent_start - sink_end));
    for (int i = sink_end; i < recent_start; ++i) {
        candidates.push_back(i);
    }

    if ((int)candidates.size() <= k) {
        return candidates;  // all middle tokens fit in budget
    }

    // Partial sort to get top-k by score (descending)
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                      [&scores](int a, int b) { return scores[a] > scores[b]; });
    candidates.resize(k);
    std::sort(candidates.begin(), candidates.end());  // restore position order
    return candidates;
}

// Full SFI selection: compute merged sparse indices from selector state.
inline std::vector<int> compute_sfi_indices(
    const std::vector<float> & scores,
    int kv_len,
    int budget,
    int sink_tokens = 4,
    int recent_tokens = 256
) {
    // Budget covers total tokens; subtract sink and recent to get selected budget
    const int sink_end = std::clamp(sink_tokens, 0, kv_len);
    const int recent_count = std::min(std::max(recent_tokens, 0), kv_len);
    const int selected_budget = std::max(0, budget - sink_end - recent_count);

    std::vector<int> selected = topk_from_scores(
        scores, kv_len, selected_budget, sink_tokens, recent_tokens);

    return merge_sparse_index_sets(kv_len, sink_tokens, recent_tokens, selected);
}

// Initialize/refresh selector scores with a heuristic (position-based decay).
// Used as a bootstrap before real attention weights are available.
// Assigns higher scores to positions that are "landmarks" (evenly spaced)
// with a slight recency bias, simulating the attention pattern observed in
// long-context transformers.
inline void refresh_selector_heuristic(
    std::vector<float> & scores,
    int kv_len
) {
    scores.resize(kv_len, 0.0f);
    if (kv_len == 0) return;
    // Strategy: uniform spacing (every N-th token gets a boost) + recency decay.
    // This ensures the selected set covers the full context evenly.
    const float base_score = 0.1f;
    const float spacing_boost = 0.5f;
    const int stride = std::max(1, kv_len / 512);  // ~512 landmark positions
    for (int i = 0; i < kv_len; i++) {
        float s = base_score;
        // Spacing landmarks
        if (i % stride == 0) s += spacing_boost;
        // Gentle recency bias (linear decay from newest to oldest)
        s += 0.3f * (float)i / (float)kv_len;
        scores[i] = s;
    }
}

} // namespace dflash27b::sfi
