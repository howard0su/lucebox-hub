#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Maximum k the kernel supports.  Shared memory is sized at compile time
// from this constant (see gpu_sampler.cu for details).
static constexpr int GPU_TOPK_K = 40;

// Persistent workspace to avoid per-call cudaMalloc overhead.
struct GpuTopKWorkspace {
    float   * d_vals = nullptr;
    int32_t * d_ids  = nullptr;
    int       capacity = 0;   // current allocation in elements (k * n_pos)

    void ensure(int needed);
    ~GpuTopKWorkspace();

    GpuTopKWorkspace() = default;
    GpuTopKWorkspace(const GpuTopKWorkspace &) = delete;
    GpuTopKWorkspace & operator=(const GpuTopKWorkspace &) = delete;
};

// Extract top-k logit values and their token indices from GPU logits tensor.
//
// logits_gpu:  device pointer to logits, layout [vocab, n_pos] (column-major,
//              each column is one position's logit vector of length `vocab`)
// vocab:       vocabulary size (e.g., 248320)
// n_pos:       number of positions to process (≥ 1)
// row_stride:  byte stride between consecutive positions (logits_tensor->nb[1])
// k:           number of top candidates to extract (≤ GPU_TOPK_K)
// out_vals:    host output [k * n_pos] top-k logit values (descending per pos)
// out_ids:     host output [k * n_pos] top-k token IDs
// ws:          persistent workspace (device buffers are grown lazily)
// stream:      CUDA stream (nullptr for default)
void gpu_top_k_extract(const float * logits_gpu, int vocab, int n_pos,
                       size_t row_stride, int k,
                       float * out_vals, int32_t * out_ids,
                       GpuTopKWorkspace & ws,
                       cudaStream_t stream = nullptr);
