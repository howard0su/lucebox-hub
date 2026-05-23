// Smoke test for qwen35moe router-selection capture.
//
// Loads qwen35moe, builds a one-token forward step with capture_moe_router=1,
// runs it on CUDA, and verifies that routed expert ids can be observed into
// Qwen35MoeRoutingStats for every layer.

#include "internal.h"
#include "graph_builders.h"
#include "qwen35moe_routing_stats.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35moe.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "cuda init failed\n");
        return 1;
    }

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_target_gguf: %s\n", dflash27b_last_error());
        return 1;
    }
    if (!w.is_moe) {
        std::fprintf(stderr, "model is not qwen35moe\n");
        return 1;
    }

    TargetCache cache;
    if (!create_target_cache(w, /*max_ctx=*/64, /*max_verify_tokens=*/0, backend, cache)) {
        std::fprintf(stderr, "create_target_cache: %s\n", dflash27b_last_error());
        return 1;
    }

    StepGraph sg;
    if (!build_target_step(sg, w, cache, backend,
                           /*kv_start=*/0, /*n_tokens=*/1,
                           /*with_mask=*/false, /*capture=*/false,
                           /*capture_delta_intermediate=*/false,
                           /*fa_window=*/0,
                           /*last_token_logits_only=*/false,
                           KQ_MASK_PAD,
                           /*capture_moe_router=*/true)) {
        std::fprintf(stderr, "build_target_step failed\n");
        return 1;
    }

    std::vector<float> embed((size_t)w.n_embd);
    const int32_t tok = 1;
    if (!w.embedder.embed(&tok, 1, embed.data())) {
        std::fprintf(stderr, "embed failed\n");
        return 1;
    }
    ggml_backend_tensor_set(sg.inp_embed, embed.data(), 0, sizeof(float) * embed.size());

    int32_t pos4[4] = {0, 0, 0, 0};
    ggml_backend_tensor_set(sg.positions, pos4, 0, sizeof(pos4));

    auto st = ggml_backend_graph_compute(backend, sg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed: %d\n", (int)st);
        return 1;
    }

    Qwen35MoeRoutingStats stats;
    if (!stats.init_from_weights(w)) {
        std::fprintf(stderr, "stats init failed\n");
        return 1;
    }
    std::string err;
    int captured_layers = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        ggml_tensor * selected = (il < (int)sg.moe_selected.size()) ? sg.moe_selected[(size_t)il] : nullptr;
        if (!selected) continue;
        if (!stats.observe_selected_tensor(backend, il, selected, &err)) {
            std::fprintf(stderr, "observe_selected_tensor layer %d: %s\n", il, err.c_str());
            return 1;
        }
        if (stats.layer_totals[(size_t)il] != (uint64_t)w.n_expert_used) {
            std::fprintf(stderr, "unexpected layer total at %d: got %llu expected %d\n",
                         il,
                         (unsigned long long)stats.layer_totals[(size_t)il],
                         w.n_expert_used);
            return 1;
        }
        captured_layers++;
    }

    if (captured_layers != w.n_layer) {
        std::fprintf(stderr, "captured %d layers, expected %d\n", captured_layers, w.n_layer);
        return 1;
    }

    step_graph_destroy(sg);
    free_target_cache(cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("OK\n");
    return 0;
}
