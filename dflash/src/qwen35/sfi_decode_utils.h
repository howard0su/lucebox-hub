#pragma once

#include <algorithm>
#include <vector>

namespace dflash27b::sfi {

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

} // namespace dflash27b::sfi
