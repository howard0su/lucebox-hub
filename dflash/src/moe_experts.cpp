// PinnedExperts: bulk-load all 256 experts per layer into VRAM.

#include "moe_experts.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

namespace dflash27b {

bool PinnedExperts::init(ggml_backend_t backend, const MoeExpertSource & source,
                         const std::vector<int> & pinned_layer_ids) {
    destroy();
    const int n_layers = source.n_layers;
    const int n_exp = source.n_experts;

    pinned_.assign(n_layers, false);
    layers_.resize(n_layers);
    for (int l : pinned_layer_ids) {
        if (l >= 0 && l < n_layers) pinned_[l] = true;
    }

    int n_pinned = 0;
    for (int l = 0; l < n_layers; l++) if (pinned_[l]) n_pinned++;
    if (n_pinned == 0) return true;

    ggml_init_params ip{};
    ip.mem_size = (size_t)(3 * n_pinned + 4) * ggml_tensor_overhead() + 16 * 1024;
    ip.no_alloc = true;
    ctx_ = ggml_init(ip);
    if (!ctx_) return false;

    for (int l = 0; l < n_layers; l++) {
        if (!pinned_[l]) continue;
        ggml_type down_type = source.down_type;
        if (!source.layer_down_types.empty())
            down_type = source.layer_down_types[l];

        layers_[l].gate = ggml_new_tensor_3d(ctx_, source.gate_type,
            source.hidden_dim, source.expert_ffn_dim, n_exp);
        layers_[l].up = ggml_new_tensor_3d(ctx_, source.up_type,
            source.hidden_dim, source.expert_ffn_dim, n_exp);
        layers_[l].down = ggml_new_tensor_3d(ctx_, down_type,
            source.expert_ffn_dim, source.hidden_dim, n_exp);

        char name[64];
        std::snprintf(name, sizeof(name), "pin_gate_L%d", l); ggml_set_name(layers_[l].gate, name);
        std::snprintf(name, sizeof(name), "pin_up_L%d", l);   ggml_set_name(layers_[l].up, name);
        std::snprintf(name, sizeof(name), "pin_down_L%d", l); ggml_set_name(layers_[l].down, name);
    }

    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buf_) {
        std::fprintf(stderr, "[PinnedExperts] GPU alloc failed for %d layers\n", n_pinned);
        destroy();
        return false;
    }

    // Bulk-load all experts from mmap into pinned tensors.
    cudaStream_t stream = cudaStreamPerThread;
    for (int l = 0; l < n_layers; l++) {
        if (!pinned_[l]) continue;
        const auto & li = source.layers[l];
        size_t down_bytes = source.layer_down_bytes.empty()
            ? source.down_expert_bytes : source.layer_down_bytes[l];

        for (int e = 0; e < n_exp; e++) {
            const uint8_t * gate_data = source.mmap_base + li.gate_offset
                                      + (size_t)e * source.gate_expert_bytes;
            const uint8_t * up_data = source.mmap_base + li.up_offset
                                    + (size_t)e * source.up_expert_bytes;
            const uint8_t * down_data = source.mmap_base + li.down_offset
                                      + (size_t)e * down_bytes;

            char * gate_dst = (char *)layers_[l].gate->data + (size_t)e * layers_[l].gate->nb[2];
            char * up_dst   = (char *)layers_[l].up->data   + (size_t)e * layers_[l].up->nb[2];
            char * down_dst = (char *)layers_[l].down->data + (size_t)e * layers_[l].down->nb[2];

            cudaMemcpyAsync(gate_dst, gate_data, source.gate_expert_bytes,
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(up_dst, up_data, source.up_expert_bytes,
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(down_dst, down_data, down_bytes,
                            cudaMemcpyHostToDevice, stream);
        }
    }
    cudaStreamSynchronize(stream);

    total_bytes_ = ggml_backend_buffer_get_size(buf_);
    std::printf("[PinnedExperts] %d layers pinned (%.1f MB)\n",
        n_pinned, total_bytes_ / (1024.0 * 1024.0));
    return true;
}

void PinnedExperts::destroy() {
    if (buf_) { ggml_backend_buffer_free(buf_); buf_ = nullptr; }
    if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }
    pinned_.clear();
    layers_.clear();
    total_bytes_ = 0;
}

} // namespace dflash27b
