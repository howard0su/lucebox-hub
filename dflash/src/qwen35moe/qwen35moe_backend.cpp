#include "qwen35moe_backend.h"

#include <cstdio>
#include <cstdlib>

namespace dflash::common {

Qwen35MoeBackend::Qwen35MoeBackend(const Qwen35Config & cfg)
    : Qwen35Backend(cfg) {}

bool Qwen35MoeBackend::load_target_model(ggml_backend_t backend, TargetWeights & out) {
    if (!load_target_gguf(cfg_.target_path, backend, out)) {
        return false;
    }

    if (const char * stats_path = std::getenv("DFLASH_QWEN35MOE_RUNTIME_STATS_OUT")) {
        routing_stats_ = std::make_shared<Qwen35MoeRoutingStats>();
        if (!routing_stats_->init_from_weights(out)) {
            set_last_error("qwen35moe runtime stats init failed");
            return false;
        }
        routing_stats_out_path_ = stats_path;
    }

    const char * placement_path = std::getenv("DFLASH_QWEN35MOE_PLACEMENT");
    if (!placement_path || !placement_path[0]) {
        return true;
    }

    Qwen35MoeExpertPlacement placement;
    std::string err;
    if (!Qwen35MoeExpertPlacement::load_json(placement_path, placement, &err)) {
        set_last_error(std::string("qwen35moe placement load failed: ") + err);
        return false;
    }

    auto hybrid = std::make_shared<Qwen35MoeHybridStorage>();
    if (!build_qwen35moe_hybrid_storage(out, backend, placement, *hybrid, &err)) {
        set_last_error(std::string("qwen35moe hybrid storage build failed: ") + err);
        return false;
    }
    out.moe_hybrid = std::move(hybrid);
    hybrid_mode_ = true;
    cfg_.draft_path = nullptr;  // policy: hybrid mode falls back to AR-only until hybrid FFN lands
    int total_cold = 0;
    for (const auto & layer : out.moe_hybrid->layers) {
        total_cold += (int)layer.cold_expert_ids.size();
    }
    std::printf("[qwen35moe] hybrid storage ready: total_hot=%d total_cold=%d placement=%s (AR-only mode)\n",
                out.moe_hybrid->placement.total_hot, total_cold, placement_path);
    if (const char * out_path = std::getenv("DFLASH_QWEN35MOE_NEXT_PLACEMENT_OUT")) {
        placement_out_path_ = out_path;
    }
    if (const char * swap_max = std::getenv("DFLASH_QWEN35MOE_SWAP_MAX")) {
        swap_policy_.max_swaps_total = std::max(0, std::atoi(swap_max));
    }
    if (const char * swap_gain = std::getenv("DFLASH_QWEN35MOE_SWAP_MIN_GAIN")) {
        swap_policy_.min_promote_gain = (uint64_t)std::max(1, std::atoi(swap_gain));
    }
    return true;
}

void Qwen35MoeBackend::after_target_compute(StepGraph & sg, int, int) {
    if (!routing_stats_) return;
    std::string err;
    for (int il = 0; il < target_weights().n_layer; ++il) {
        ggml_tensor * selected = (il < (int)sg.moe_selected.size()) ? sg.moe_selected[(size_t)il] : nullptr;
        if (!selected) continue;
        if (!routing_stats_->observe_selected_tensor(target_backend(), il, selected, &err)) {
            std::fprintf(stderr, "[qwen35moe] routing-stats observe failed at layer %d: %s\n",
                         il, err.c_str());
            break;
        }
    }
}

void Qwen35MoeBackend::maybe_post_request_swap() {
    if (!routing_stats_) return;

    if (!routing_stats_out_path_.empty()) {
        std::string err;
        if (!routing_stats_->save_json(routing_stats_out_path_, &err)) {
            std::fprintf(stderr, "[qwen35moe] failed to save runtime stats: %s\n", err.c_str());
        }
    }

    if (!hybrid_mode_ || !target_weights().moe_hybrid || swap_policy_.max_swaps_total <= 0) return;

    Qwen35MoeSwapPlan plan;
    std::string err;
    if (!build_qwen35moe_swap_plan(target_weights().moe_hybrid->placement, *routing_stats_,
                                   swap_policy_, plan, &err)) {
        std::fprintf(stderr, "[qwen35moe] swap plan failed: %s\n", err.c_str());
        return;
    }
    if (plan.actions.empty()) return;

    auto rebuilt = std::make_shared<Qwen35MoeHybridStorage>();
    if (!build_qwen35moe_hybrid_storage(target_weights(), target_backend(),
                                        plan.next_placement, *rebuilt, &err)) {
        std::fprintf(stderr, "[qwen35moe] swap rebuild failed: %s\n", err.c_str());
        return;
    }
    target_weights().moe_hybrid = std::move(rebuilt);
    if (!placement_out_path_.empty()) {
        if (!plan.next_placement.save_json(placement_out_path_, &err)) {
            std::fprintf(stderr, "[qwen35moe] failed to save next placement: %s\n", err.c_str());
        }
    }
    std::printf("[qwen35moe] applied %zu swap actions at request boundary\n", plan.actions.size());
}

GenerateResult Qwen35MoeBackend::generate(const GenerateRequest & req,
                                          const DaemonIO & io) {
    auto result = Qwen35Backend::generate(req, io);
    if (result.ok) {
        maybe_post_request_swap();
    }
    return result;
}

GenerateResult Qwen35MoeBackend::restore_and_generate(int slot,
                                                      const GenerateRequest & req,
                                                      const DaemonIO & io) {
    auto result = Qwen35Backend::restore_and_generate(slot, req, io);
    if (result.ok) {
        maybe_post_request_swap();
    }
    return result;
}

}  // namespace dflash::common
