#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>


// ggml_flash_attn_ext expects kv_len aligned to KQ_MASK_PAD (32) on the
// f16/Q* paths, and to FATTN_KQ_STRIDE (256) on the TurboQuant FA paths.
// The global `g_kq_stride_pad` below is set at init time and applied by
// both build_causal_mask and build_tree_mask so the mask dim matches the
// K/V view length used in build_attn_block.
static constexpr int KQ_MASK_PAD = 32;
static int g_kq_stride_pad = KQ_MASK_PAD;   // overridden to 256 when TBQ KV is active
static int g_max_ctx_override = 0;           // overridden by --max-ctx=N (default 4096)
static int g_fa_window       = 2048;         // overridden by DFLASH27B_FA_WINDOW=N
static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// F16 encoding for the two values we use: 0 and -inf.
// 0 in F16 is 0x0000. -inf is 0xFC00.
static constexpr uint16_t F16_ZERO = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static void build_causal_mask(std::vector<uint16_t> & out,
                              int kv_len, int n_tokens, int kv_start,
                              int win_start = 0) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    const int abs_end = win_start + kv_len;
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        const int min_k = std::max(0, win_start);
        const int max_k = abs_q;
        for (int k = min_k; k <= max_k && k < abs_end; k++) {
            out[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
    }
}

// ─── DDTree support (ported from liranringel/ddtree/ddtree.py) ────────

// Per-position top-K softmax extraction. Computes log-probabilities (needed
// so that cross-depth prefix comparisons in the best-first heap are valid)
// via a single pass over the vocab that also maintains top-K in a heap and
// computes logsumexp online. Runs on CPU since draft logits are already on
// host after ggml_backend_tensor_get.
//
// Input:  logits [n_positions × vocab] f32
// Output: out_log_probs [n_positions × K] f32, out_token_ids [n_positions × K] i32
//         both sorted by log-probability DESCENDING (rank 0 = argmax).
static void extract_draft_topk(const float * logits,
                               int n_positions, int vocab, int K,
                               float * out_log_probs,
                               int32_t * out_token_ids,
                               float temperature = 1.0f) {
    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };

    // Temperature scaling: dividing logits by T<1 sharpens the softmax,
    // widening the gap between top-1 and lower ranks. This compensates for
    // Q4_K_M quantization that flattens the draft's softmax — without it,
    // pure best-first picks shallow bushy trees instead of going deep.
    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    // Parallelize across positions — each i is independent.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(K);

        // Online log-sum-exp with running max. Single pass over the vocab,
        // simultaneously maintaining top-K.
        float running_max     = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;

            // Online logsumexp
            if (l > running_max) {
                // rescale previous sum to the new max
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }

            // Top-K maintenance
            if ((int)heap.size() < K) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(running_sum_exp);

        // Sort the K entries descending (largest logit first) and emit.
        // sort_heap with cmp_greater on a min-heap already produces descending
        // order (cppreference: "sort_heap leaves the range sorted in the same
        // order as sort would with the same comparator" — greater→descending).
        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}

// A flat DFS-ordered tree built from the draft's top-K softmax distributions.
// Slot 0 is the tree root (the bonus token from the previous spec round);
// slots 1..n_nodes are the DFS-ordered tree nodes. `parents[i]` gives each
// node's parent index in the same flat array (parents[0] = -1). `depth[i]`
// is the absolute depth within the block-diffusion prediction window, with
// the root at depth 0 and its children at depth 1. `child_maps[i]` maps a
// token_id to the child's flat index, used for the tree walk post-verify.
// `visibility[i][j]` (ancestor-only mask) is true iff j is an ancestor of i
// in the tree (including j == i); used to build the attention mask.
struct DDTree {
    int                         n_nodes = 0;          // excludes root
    std::vector<int32_t>        token_ids;            // size n_nodes
    std::vector<int>            depths;               // size n_nodes (1..L)
    std::vector<int>            parents;              // size n_nodes + 1
    std::vector<std::unordered_map<int32_t, int>> child_maps;  // size n_nodes + 1
    std::vector<uint8_t>        visibility;           // (1 + n_nodes)^2 row-major
};

// Port of build_ddtree_tree() from ddtree.py. Runs a best-first heap over
// prefixes of the per-position top-K distributions, pops until `budget`
// nodes are accumulated. Populates the flat DFS-ordered tree structure.
//
// top_log_probs: [L × K]  the drafter's per-position top-K log-probabilities
// top_token_ids: [L × K]  matching token ids, rank 0 = argmax per position
// L:             max tree depth (e.g. q_len - 1 for a block diffusion block)
// K:             top-K per position (same as used in extract_draft_topk)
// budget:        maximum number of non-root tree nodes
static DDTree build_ddtree(const float * top_log_probs,
                           const int32_t * top_token_ids,
                           int L, int K, int budget,
                           bool chain_seed = true) {
    DDTree tree;
    if (budget <= 0 || L <= 0) {
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();
        tree.visibility.assign(1, 1);
        return tree;
    }

    // Heap entry:
    //   neg_logw, ranks (encoded as a small vector), parent_index, depth, rank, logw
    // We sort by neg_logw ASCENDING, which is equivalent to logw DESCENDING.
    struct HeapEntry {
        float                neg_logw;
        std::vector<int>     ranks;        // rank tuple used only to prevent duplicate state; not strictly needed
        int                  parent_index; // index in the flat tree of this candidate's parent
        int                  depth;        // 1..L
        int                  rank;         // rank within top-K at depth-1 (0-indexed)
        float                logw;         // actual log-prob sum so far
    };
    struct HeapCmp {
        bool operator()(const HeapEntry & a, const HeapEntry & b) const {
            // std::priority_queue is a max-heap; we want SMALLEST neg_logw at the top
            // so that we pop the highest-probability prefix first.
            return a.neg_logw > b.neg_logw;
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    tree.token_ids.reserve(budget);
    tree.depths.reserve(budget);
    tree.parents.reserve(budget + 1);
    tree.parents.push_back(-1);                 // root
    tree.child_maps.emplace_back();             // root's children

    // Two seeding strategies:
    //   - chain_seed=true: pre-seed full top-1 chain (defensive, guarantees
    //     AL >= chain mode even with flat-softmax draft like Q4_K_M). Compensates
    //     for quantization that shrinks top-1/top-2 logp gap.
    //   - chain_seed=false: paper's pure best-first — heap starts with just
    //     the depth-1 top-1 child of the root. Tree shape emerges from log-prob
    //     ordering. Works only when the draft top-1 is dominant enough.
    if (chain_seed) {
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * K + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * K + 0];

            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;

            if (K > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * K + 0]
                    + top_log_probs[(size_t)(d - 1) * K + 1];
                heap.push({
                    /*neg_logw*/ -sibling_logw,
                    /*ranks   */ {1},
                    /*parent  */ prev_idx,
                    /*depth   */ d,
                    /*rank    */ 1,
                    /*logw    */ sibling_logw,
                });
            }
            prev_idx = cur_idx;
        }
    } else {
        // Paper-style pure best-first: seed heap with depth-1 top-1 only.
        const float root_logw = top_log_probs[0 * K + 0];
        heap.push({
            /*neg_logw*/ -root_logw,
            /*ranks   */ {0},
            /*parent  */ 0,  // root flat index
            /*depth   */ 1,
            /*rank    */ 0,
            /*logw    */ root_logw,
        });
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top();
        heap.pop();

        const int    depth_minus_1 = top.depth - 1;
        const int    rank          = top.rank;
        const int32_t token_id     = top_token_ids[(size_t)depth_minus_1 * K + rank];

        const int current_index = tree.n_nodes + 1;  // slot in flat tree
        tree.token_ids.push_back(token_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][token_id] = current_index;
        tree.n_nodes++;

        // Push next sibling (same depth, next-best rank at this depth).
        if (rank + 1 < K) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t)depth_minus_1 * K + rank]
                + top_log_probs[(size_t)depth_minus_1 * K + rank + 1];
            std::vector<int> sibling_ranks = top.ranks;
            sibling_ranks.back() = rank + 1;
            heap.push({
                /*neg_logw*/ -sibling_logw,
                /*ranks   */ std::move(sibling_ranks),
                /*parent  */ top.parent_index,
                /*depth   */ top.depth,
                /*rank    */ rank + 1,
                /*logw    */ sibling_logw,
            });
        }

        // Push first child (next depth, top-1 rank under this node).
        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth /*new depth_minus_1*/ * K + 0];
            std::vector<int> child_ranks = top.ranks;
            child_ranks.push_back(0);
            heap.push({
                /*neg_logw*/ -child_logw,
                /*ranks   */ std::move(child_ranks),
                /*parent  */ current_index,
                /*depth   */ top.depth + 1,
                /*rank    */ 0,
                /*logw    */ child_logw,
            });
        }
    }

    // Build ancestor-only visibility mask (flat row-major, (1+n)^2).
    const int N = 1 + tree.n_nodes;
    tree.visibility.assign((size_t)N * N, 0);
    tree.visibility[0 * N + 0] = 1;  // root sees itself
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[i];  // immediate parent
        // Inherit the parent's visibility row up to column i-1,
        // then mark self at column i.
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t)i * N + j] = tree.visibility[(size_t)p * N + j];
        }
        tree.visibility[(size_t)i * N + i] = 1;
    }

    return tree;
}

// Walk the verified tree following the target's argmax (posterior) at each
// node. Returns the list of flat-tree indices that make up the accepted path
// (starting at root), plus the next "bonus" token (target's argmax at the
// deepest accepted node, which didn't match any of that node's children).
static std::vector<int> follow_verified_tree(const DDTree & tree,
                                             const int32_t * posterior,
                                             int & out_next_token,
                                             int * out_node_idx = nullptr) {
    std::vector<int> accepted;
    accepted.reserve(tree.n_nodes + 1);
    accepted.push_back(0);

    int current_index = 0;
    int next_token    = posterior[current_index];
    while (true) {
        const auto & children = tree.child_maps[current_index];
        auto it = children.find(next_token);
        if (it == children.end()) break;
        current_index = it->second;
        accepted.push_back(current_index);
        next_token = posterior[current_index];
    }
    out_next_token = next_token;
    if (out_node_idx) *out_node_idx = current_index;
    return accepted;
}

// Build an f16 ancestor-only attention mask for tree verify:
//   mask[q=i][k<past_length]          = 0    (past KV cache, attend freely)
//   mask[q=i][k=past_length+j]        = 0 iff j is an ancestor of i in the tree
//                                              (including j == i)
//                                     = -inf otherwise
// Shape matches the ggml flash_attn_ext expectation: [kv_pad, q_pad] f16.
static void build_tree_mask(const DDTree & tree, int past_length,
                            std::vector<uint16_t> & out_mask,
                            int win_start = 0) {
    const int N      = 1 + tree.n_nodes;
    const int win_len = past_length + N - win_start;
    const int kv_pad = align_up(win_len, g_kq_stride_pad);
    const int q_pad  = align_up(N,      KQ_MASK_PAD);
    out_mask.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < N; q++) {
        for (int k = std::max(0, win_start); k < past_length; k++) {
            out_mask[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
        for (int j = 0; j < N; j++) {
            if (tree.visibility[(size_t)q * N + j]) {
                out_mask[(size_t)q * kv_pad + (past_length + j - win_start)] = F16_ZERO;
            }
        }
    }
}
