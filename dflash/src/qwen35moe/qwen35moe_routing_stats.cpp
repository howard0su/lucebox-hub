#include "qwen35moe_routing_stats.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <numeric>

namespace dflash::common {

size_t Qwen35MoeRoutingStats::index_of(int layer_idx, int expert_idx) const {
    return (size_t)layer_idx * (size_t)n_expert + (size_t)expert_idx;
}

bool Qwen35MoeRoutingStats::init_from_weights(const TargetWeights & w) {
    if (!w.is_moe || w.n_layer <= 0 || w.n_expert <= 0 || w.n_expert_used <= 0) {
        return false;
    }
    n_layer = w.n_layer;
    n_expert = w.n_expert;
    n_expert_used = w.n_expert_used;
    counts.assign((size_t)n_layer * (size_t)n_expert, 0);
    layer_totals.assign((size_t)n_layer, 0);
    return true;
}

bool Qwen35MoeRoutingStats::matches(const TargetWeights & w) const {
    return w.is_moe &&
           n_layer == w.n_layer &&
           n_expert == w.n_expert &&
           n_expert_used == w.n_expert_used &&
           counts.size() == (size_t)n_layer * (size_t)n_expert &&
           layer_totals.size() == (size_t)n_layer;
}

bool Qwen35MoeRoutingStats::empty() const {
    return counts.empty();
}

uint64_t Qwen35MoeRoutingStats::count(int layer_idx, int expert_idx) const {
    if (layer_idx < 0 || layer_idx >= n_layer || expert_idx < 0 || expert_idx >= n_expert) {
        return 0;
    }
    return counts[index_of(layer_idx, expert_idx)];
}

bool Qwen35MoeRoutingStats::observe(int layer_idx, const int32_t * expert_ids, int n_ids) {
    if (!expert_ids || layer_idx < 0 || layer_idx >= n_layer || n_ids < 0) {
        return false;
    }
    for (int i = 0; i < n_ids; ++i) {
        const int expert_idx = expert_ids[i];
        if (expert_idx < 0 || expert_idx >= n_expert) {
            return false;
        }
    }
    for (int i = 0; i < n_ids; ++i) {
        const int expert_idx = expert_ids[i];
        counts[index_of(layer_idx, expert_idx)]++;
        layer_totals[(size_t) layer_idx]++;
    }
    return true;
}

std::vector<int> Qwen35MoeRoutingStats::ranked_experts(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= n_layer) return {};
    std::vector<int> ranked((size_t)n_expert);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::stable_sort(ranked.begin(), ranked.end(),
        [&](int a, int b) {
            const uint64_t ca = counts[index_of(layer_idx, a)];
            const uint64_t cb = counts[index_of(layer_idx, b)];
            if (ca != cb) return ca > cb;
            return a < b;
        });
    return ranked;
}

std::vector<int> Qwen35MoeRoutingStats::hot_experts(int layer_idx, int hot_count) const {
    std::vector<int> ranked = ranked_experts(layer_idx);
    if (hot_count < 0) hot_count = 0;
    if ((size_t) hot_count < ranked.size()) {
        ranked.resize((size_t) hot_count);
    }
    return ranked;
}

bool Qwen35MoeRoutingStats::save_json(const std::string & path, std::string * err) const {
    if (n_layer <= 0 || n_expert <= 0 || counts.size() != (size_t)n_layer * (size_t)n_expert) {
        if (err) *err = "routing stats not initialized";
        return false;
    }

    nlohmann::json j;
    j["arch"] = "qwen35moe";
    j["version"] = 1;
    j["n_layer"] = n_layer;
    j["n_expert"] = n_expert;
    j["n_expert_used"] = n_expert_used;
    j["layer_totals"] = layer_totals;
    j["counts"] = nlohmann::json::array();
    for (int il = 0; il < n_layer; ++il) {
        nlohmann::json row = nlohmann::json::array();
        for (int ie = 0; ie < n_expert; ++ie) {
            row.push_back(counts[index_of(il, ie)]);
        }
        j["counts"].push_back(std::move(row));
    }

    std::ofstream f(path);
    if (!f) {
        if (err) *err = "failed to open output file";
        return false;
    }
    f << j.dump(2);
    if (!f) {
        if (err) *err = "failed to write json";
        return false;
    }
    return true;
}

bool Qwen35MoeRoutingStats::load_json(const std::string & path,
                                      Qwen35MoeRoutingStats & out,
                                      std::string * err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "failed to open input file";
        return false;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception & ex) {
        if (err) *err = ex.what();
        return false;
    }

    if (j.value("arch", std::string()) != "qwen35moe") {
        if (err) *err = "unexpected arch";
        return false;
    }

    const int n_layer = j.value("n_layer", 0);
    const int n_expert = j.value("n_expert", 0);
    const int n_expert_used = j.value("n_expert_used", 0);
    if (n_layer <= 0 || n_expert <= 0 || n_expert_used <= 0) {
        if (err) *err = "invalid dimensions";
        return false;
    }

    const auto & counts_json = j["counts"];
    const auto & totals_json = j["layer_totals"];
    if (!counts_json.is_array() || (int)counts_json.size() != n_layer ||
        !totals_json.is_array() || (int)totals_json.size() != n_layer) {
        if (err) *err = "invalid counts shape";
        return false;
    }

    Qwen35MoeRoutingStats tmp;
    tmp.n_layer = n_layer;
    tmp.n_expert = n_expert;
    tmp.n_expert_used = n_expert_used;
    tmp.counts.assign((size_t)n_layer * (size_t)n_expert, 0);
    tmp.layer_totals.assign((size_t)n_layer, 0);

    for (int il = 0; il < n_layer; ++il) {
        const auto & row = counts_json[(size_t)il];
        if (!row.is_array() || (int)row.size() != n_expert) {
            if (err) *err = "invalid row width";
            return false;
        }
        for (int ie = 0; ie < n_expert; ++ie) {
            tmp.counts[tmp.index_of(il, ie)] = row[(size_t)ie].get<uint64_t>();
        }
        tmp.layer_totals[(size_t)il] = totals_json[(size_t)il].get<uint64_t>();
    }

    out = std::move(tmp);
    return true;
}

}  // namespace dflash::common
