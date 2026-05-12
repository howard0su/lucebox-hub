#include "sfi_decode_utils.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using dflash27b::sfi::AttnWindowSlice;

namespace {

int n_pass = 0;
int n_fail = 0;

int check(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++n_fail;
        return 1;
    }
    ++n_pass;
    return 0;
}

// ── Windowed-slice tests ────────────────────────────────────────────

int test_windowed_slice_q8() {
    const AttnWindowSlice s = dflash27b::sfi::resolve_attn_window_slice(
        /*kv_start=*/8192, /*n_tokens=*/16,
        /*allow_slow_refresh=*/true, /*fa_window=*/4096,
        /*refresh_interval=*/0, /*uses_256_stride=*/false);
    int rc = 0;
    rc |= check(!s.used_slow_refresh, "windowed q8 should not use slow refresh");
    rc |= check(s.effective_window == 4096, "windowed q8 effective window");
    rc |= check(s.win_start == 4096, "windowed q8 win_start");
    rc |= check(s.win_len == 4112, "windowed q8 win_len");
    rc |= check(s.win_len_padded == 4112, "windowed q8 win_len_padded");
    return rc;
}

int test_refresh_only_when_allowed() {
    const AttnWindowSlice no_refresh = dflash27b::sfi::resolve_attn_window_slice(
        /*kv_start=*/4096, /*n_tokens=*/16,
        /*allow_slow_refresh=*/false, /*fa_window=*/2048,
        /*refresh_interval=*/4096, /*uses_256_stride=*/false);
    const AttnWindowSlice with_refresh = dflash27b::sfi::resolve_attn_window_slice(
        /*kv_start=*/4096, /*n_tokens=*/16,
        /*allow_slow_refresh=*/true, /*fa_window=*/2048,
        /*refresh_interval=*/4096, /*uses_256_stride=*/false);
    int rc = 0;
    rc |= check(!no_refresh.used_slow_refresh, "masked path must not slow refresh");
    rc |= check(no_refresh.win_start == 2048, "masked path keeps windowed start");
    rc |= check(with_refresh.used_slow_refresh, "decode path should slow refresh");
    rc |= check(with_refresh.effective_window == 0, "slow refresh uses full attention");
    rc |= check(with_refresh.win_start == 0, "slow refresh starts from zero");
    rc |= check(with_refresh.win_len == 4112, "slow refresh win_len matches full prefix");
    return rc;
}

int test_tq3_padding() {
    const AttnWindowSlice s = dflash27b::sfi::resolve_attn_window_slice(
        /*kv_start=*/1, /*n_tokens=*/16,
        /*allow_slow_refresh=*/false, /*fa_window=*/0,
        /*refresh_interval=*/0, /*uses_256_stride=*/true);
    int rc = 0;
    rc |= check(s.win_start == 0, "tq3 win_start");
    rc |= check(s.win_len == 17, "tq3 win_len");
    rc |= check(s.win_len_padded == 256, "tq3 path pads to 256");
    return rc;
}

// ── Merge-sparse-index tests ────────────────────────────────────────

int test_merge_sparse_index_sets() {
    const std::vector<int> merged = dflash27b::sfi::merge_sparse_index_sets(
        /*kv_len=*/100, /*sink_tokens=*/4, /*recent_tokens=*/8,
        /*selected_indices=*/{-1, 0, 2, 4, 10, 10, 50, 93, 99, 100});
    const std::vector<int> expected = {
        0, 1, 2, 3, 4, 10, 50,
        92, 93, 94, 95, 96, 97, 98, 99
    };
    int rc = 0;
    rc |= check(merged == expected, "merge should retain sink + filtered selected + recent");
    return rc;
}

int test_merge_when_recent_covers_all() {
    const std::vector<int> merged = dflash27b::sfi::merge_sparse_index_sets(
        /*kv_len=*/6, /*sink_tokens=*/2, /*recent_tokens=*/6,
        /*selected_indices=*/{2, 3, 4});
    const std::vector<int> expected = {0, 1, 2, 3, 4, 5};
    return check(merged == expected, "recent covering full kv returns full range");
}

// ── Edge-case tests ─────────────────────────────────────────────────

int test_merge_empty_kv() {
    auto m = dflash27b::sfi::merge_sparse_index_sets(0, 4, 8, {1, 2});
    return check(m.empty(), "kv_len=0 returns empty");
}

int test_merge_empty_selected() {
    auto m = dflash27b::sfi::merge_sparse_index_sets(20, 4, 4, {});
    int rc = 0;
    rc |= check((int)m.size() == 8, "sink(4) + recent(4) with no overlap = 8");
    rc |= check(m.front() == 0 && m.back() == 19, "first=0, last=19");
    return rc;
}

int test_merge_sink_exceeds_kv() {
    auto m = dflash27b::sfi::merge_sparse_index_sets(3, 10, 0, {});
    const std::vector<int> expected = {0, 1, 2};
    return check(m == expected, "sink > kv_len clamps to kv_len");
}

int test_merge_zero_sink_zero_recent() {
    auto m = dflash27b::sfi::merge_sparse_index_sets(10, 0, 0, {3, 7});
    const std::vector<int> expected = {3, 7};
    return check(m == expected, "zero sink/recent keeps only selected");
}

int test_merge_all_selected_oob() {
    auto m = dflash27b::sfi::merge_sparse_index_sets(10, 2, 2, {-5, 10, 100});
    int rc = 0;
    rc |= check((int)m.size() == 4, "only sink(2)+recent(2)");
    rc |= check(m == (std::vector<int>{0, 1, 8, 9}), "no oob indices included");
    return rc;
}

int test_merge_sink_recent_overlap() {
    // kv_len=6, sink=4, recent=4 → sink covers [0,4), recent covers [2,6)
    auto m = dflash27b::sfi::merge_sparse_index_sets(6, 4, 4, {});
    const std::vector<int> expected = {0, 1, 2, 3, 4, 5};
    return check(m == expected, "overlapping sink/recent deduplicates to full range");
}

int test_window_kv_start_zero() {
    const AttnWindowSlice s = dflash27b::sfi::resolve_attn_window_slice(
        /*kv_start=*/0, /*n_tokens=*/8,
        /*allow_slow_refresh=*/true, /*fa_window=*/4096,
        /*refresh_interval=*/4096, /*uses_256_stride=*/false);
    int rc = 0;
    rc |= check(!s.used_slow_refresh, "kv_start=0 must not trigger refresh");
    rc |= check(s.win_start == 0, "kv_start=0 win_start=0");
    rc |= check(s.win_len == 8, "kv_start=0 win_len=n_tokens");
    return rc;
}

int test_window_non_aligned_no_refresh() {
    const AttnWindowSlice s = dflash27b::sfi::resolve_attn_window_slice(
        /*kv_start=*/4097, /*n_tokens=*/1,
        /*allow_slow_refresh=*/true, /*fa_window=*/2048,
        /*refresh_interval=*/4096, /*uses_256_stride=*/false);
    int rc = 0;
    rc |= check(!s.used_slow_refresh, "non-aligned kv_start should not refresh");
    rc |= check(s.win_start == 2049, "windowed start at kv_start - fa_window");
    return rc;
}

int test_window_multiple_refresh_points() {
    int rc = 0;
    for (int mult = 1; mult <= 4; ++mult) {
        const int kv = 4096 * mult;
        const AttnWindowSlice s = dflash27b::sfi::resolve_attn_window_slice(
            kv, 1, true, 2048, 4096, false);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "refresh at %d×4096", mult);
        rc |= check(s.used_slow_refresh, buf);
        rc |= check(s.win_start == 0, buf);
    }
    return rc;
}

// ── Selector helpers tests ───────────────────────────────────────────

int test_update_selector_scores() {
    std::vector<float> scores(10, 0.0f);
    float weights[10] = {0, 0, 0, 0, 0, 1.0f, 0, 0, 0, 0};  // only pos 5
    dflash27b::sfi::update_selector_scores(scores, weights, 10, 0.9f);
    int rc = 0;
    rc |= check(scores[5] > 0.09f && scores[5] < 0.11f, "first update: pos 5 ~ 0.1");
    rc |= check(scores[0] == 0.0f, "first update: pos 0 stays 0");

    // Second update accumulates
    dflash27b::sfi::update_selector_scores(scores, weights, 10, 0.9f);
    rc |= check(scores[5] > 0.18f && scores[5] < 0.20f, "second update: pos 5 ~ 0.19");
    return rc;
}

int test_topk_from_scores() {
    std::vector<float> scores = {0, 0, 0, 0, 0.5f, 0.8f, 0.3f, 0.9f, 0, 0};
    // kv_len=10, sink=2, recent=2 → middle is [2,8)
    auto top = dflash27b::sfi::topk_from_scores(scores, 10, 2, 2, 2);
    int rc = 0;
    rc |= check((int)top.size() == 2, "topk returns 2 indices");
    // Highest scores in middle: pos 7 (0.9), pos 5 (0.8)
    rc |= check(top[0] == 5, "topk first is pos 5 (sorted)");
    rc |= check(top[1] == 7, "topk second is pos 7 (sorted)");
    return rc;
}

int test_topk_budget_exceeds_candidates() {
    std::vector<float> scores = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    // kv_len=5, sink=2, recent=1 → middle is [2,4) = {2,3}
    auto top = dflash27b::sfi::topk_from_scores(scores, 5, 100, 2, 1);
    int rc = 0;
    rc |= check((int)top.size() == 2, "budget > candidates returns all middle");
    rc |= check(top[0] == 2 && top[1] == 3, "returns {2,3}");
    return rc;
}

int test_compute_sfi_indices() {
    std::vector<float> scores(1000, 0.0f);
    // Make positions 500-509 very high scoring
    for (int i = 500; i < 510; ++i) scores[i] = 1.0f;

    auto indices = dflash27b::sfi::compute_sfi_indices(
        scores, /*kv_len=*/1000, /*budget=*/280,
        /*sink_tokens=*/4, /*recent_tokens=*/256);
    int rc = 0;
    // Should contain: sink[0-3] + selected from [4,744) + recent[744-999]
    rc |= check(indices[0] == 0, "sfi indices start with sink");
    rc |= check(indices[3] == 3, "sfi includes sink end");
    // All 10 high-score positions should be in the result
    for (int i = 500; i < 510; ++i) {
        bool found = std::binary_search(indices.begin(), indices.end(), i);
        if (!found) { rc |= check(false, "high-score pos missing"); break; }
    }
    // Budget is 280: 4 sink + 256 recent + 20 selected = 280
    rc |= check((int)indices.size() <= 280, "total within budget");
    return rc;
}

int test_compute_sfi_indices_small_ctx() {
    // When kv_len < budget, should return full range
    std::vector<float> scores(50, 0.5f);
    auto indices = dflash27b::sfi::compute_sfi_indices(
        scores, 50, 2048, 4, 256);
    int rc = 0;
    rc |= check((int)indices.size() == 50, "small ctx returns all positions");
    return rc;
}

// ── Performance micro-benchmark ─────────────────────────────────────

void bench_merge_sparse() {
    const int kv_len = 131072;  // 128K context
    const int sink = 4;
    const int recent = 256;

    // Build a random selected set (~5% of total)
    std::mt19937 rng(42);
    const int n_selected = kv_len / 20;
    std::vector<int> selected(n_selected);
    std::uniform_int_distribution<int> dist(0, kv_len - 1);
    for (auto & v : selected) v = dist(rng);

    // Warm up
    for (int i = 0; i < 5; ++i) {
        auto m = dflash27b::sfi::merge_sparse_index_sets(kv_len, sink, recent, selected);
        (void)m;
    }

    const int N = 1000;

    // Sparse merge
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t sparse_sz = 0;
    for (int i = 0; i < N; ++i) {
        auto m = dflash27b::sfi::merge_sparse_index_sets(kv_len, sink, recent, selected);
        sparse_sz = m.size();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sparse_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    // Baseline: full range (dense attention equivalent)
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        std::vector<int> full(kv_len);
        std::iota(full.begin(), full.end(), 0);
        (void)full;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double dense_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

    // Window-slice: benchmark N decode steps to measure amortized cost
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int kv_pos = (i * 37) % kv_len;  // vary position
        auto ws = dflash27b::sfi::resolve_attn_window_slice(
            kv_pos, 1, true, 4096, 4096, false);
        (void)ws;
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double slice_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N;

    std::printf("\n=== SFI micro-benchmark (kv_len=%d, %d selected) ===\n", kv_len, n_selected);
    std::printf("  merge_sparse : %7.1f µs → %zu indices (%.1f%% of full)\n",
                sparse_us, sparse_sz, 100.0 * sparse_sz / kv_len);
    std::printf("  dense_alloc  : %7.1f µs → %d indices (100%%)\n",
                dense_us, kv_len);
    std::printf("  speedup      : %.2fx less data vs dense\n",
                (double)kv_len / sparse_sz);
    std::printf("  window_slice : %7.3f µs/call (amortized over %d steps)\n",
                slice_us, N);

    // Full SFI selection pipeline (scores → topk → merge)
    std::vector<float> scores(kv_len, 0.0f);
    std::uniform_real_distribution<float> score_dist(0.0f, 1.0f);
    for (auto & s : scores) s = score_dist(rng);

    auto t6 = std::chrono::high_resolution_clock::now();
    size_t sfi_sz = 0;
    for (int i = 0; i < N; ++i) {
        auto idx = dflash27b::sfi::compute_sfi_indices(scores, kv_len, 2048, 4, 256);
        sfi_sz = idx.size();
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    double sfi_us = std::chrono::duration<double, std::micro>(t7 - t6).count() / N;

    std::printf("  sfi_select   : %7.1f µs → %zu indices (budget=2048, %.1f%% of full)\n",
                sfi_us, sfi_sz, 100.0 * sfi_sz / kv_len);
}

} // namespace

int main(int argc, char ** argv) {
    bool run_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) run_bench = true;
    }

    int rc = 0;

    // Core tests
    rc |= test_windowed_slice_q8();
    rc |= test_refresh_only_when_allowed();
    rc |= test_tq3_padding();
    rc |= test_merge_sparse_index_sets();
    rc |= test_merge_when_recent_covers_all();

    // Edge-case tests
    rc |= test_merge_empty_kv();
    rc |= test_merge_empty_selected();
    rc |= test_merge_sink_exceeds_kv();
    rc |= test_merge_zero_sink_zero_recent();
    rc |= test_merge_all_selected_oob();
    rc |= test_merge_sink_recent_overlap();
    rc |= test_window_kv_start_zero();
    rc |= test_window_non_aligned_no_refresh();
    rc |= test_window_multiple_refresh_points();

    // Selector helper tests
    rc |= test_update_selector_scores();
    rc |= test_topk_from_scores();
    rc |= test_topk_budget_exceeds_candidates();
    rc |= test_compute_sfi_indices();
    rc |= test_compute_sfi_indices_small_ctx();

    std::printf("%d passed, %d failed\n", n_pass, n_fail);

    if (rc != 0) {
        return 1;
    }

    if (run_bench) {
        bench_merge_sparse();
    }

    return 0;
}
