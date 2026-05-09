// GPU-side top-k extraction for speculative decoding acceptance sampling.

#include "gpu_sampler.h"
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Kernel templated on K (number of top candidates).
//
// Each thread scans its stripe of the vocab maintaining an unsorted buffer
// of the K best candidates, tracked via an explicit minimum value so the
// hot path (reject) is a single comparison.
//
// After the scan, each thread insertion-sorts its buffer (O(K²), tiny for
// K≤40) and writes it to shared memory.  A tree reduction merges all
// threads' sorted lists into the final top-K.
//
// Template parameter K determines the per-thread stack frame size
// (K × 8 bytes).  Using K=10 with 256 threads keeps the stack at 80
// bytes/thread and shared memory at 20 KB — well within limits.
// ---------------------------------------------------------------------------

struct Pair { float val; int32_t id; };

template <int K>
__global__ void top_k_kernel(const float * __restrict__ logits,
                             int vocab, size_t row_stride_bytes,
                             float * __restrict__ d_out_vals,
                             int32_t * __restrict__ d_out_ids) {
    const int pos = blockIdx.x;
    const float * row = (const float *)((const char *)logits + pos * row_stride_bytes);

    // Thread-local top-K via unsorted min-replace buffer
    Pair local[K];
    int local_n = 0;
    float min_val = -INFINITY;
    int min_idx = 0;

    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = row[i];
        if (local_n < K) {
            local[local_n] = {v, i};
            local_n++;
            if (local_n == K) {
                min_val = local[0].val; min_idx = 0;
                for (int j = 1; j < K; j++) {
                    if (local[j].val < min_val) { min_val = local[j].val; min_idx = j; }
                }
            }
        } else if (v > min_val) {
            local[min_idx] = {v, i};
            min_val = local[0].val; min_idx = 0;
            for (int j = 1; j < K; j++) {
                if (local[j].val < min_val) { min_val = local[j].val; min_idx = j; }
            }
        }
    }
    for (int j = local_n; j < K; j++) local[j] = {-INFINITY, -1};

    // Insertion sort (K ≤ 40, so O(K²) is fine)
    for (int i = 1; i < K; i++) {
        Pair key = local[i];
        int j = i - 1;
        while (j >= 0 && local[j].val < key.val) {
            local[j + 1] = local[j]; j--;
        }
        local[j + 1] = key;
    }

    // Write to shared memory for tree reduction
    extern __shared__ Pair smem[];
    Pair * my = smem + threadIdx.x * K;
    for (int i = 0; i < K; i++) my[i] = local[i];
    __syncthreads();

    // Tree reduction: at each level, the lower-index thread merges with
    // its partner's sorted list.
    for (int stride = 1; stride < (int)blockDim.x; stride *= 2) {
        if ((threadIdx.x % (2 * stride)) == 0) {
            int partner = threadIdx.x + stride;
            if (partner < (int)blockDim.x) {
                Pair * a = smem + threadIdx.x * K;
                const Pair * b = smem + partner * K;
                Pair tmp[K];
                int ia = 0, ib = 0;
                for (int i = 0; i < K; i++) {
                    if (ia < K && (ib >= K || a[ia].val >= b[ib].val))
                        tmp[i] = a[ia++];
                    else
                        tmp[i] = b[ib++];
                }
                for (int i = 0; i < K; i++) a[i] = tmp[i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float   * ov = d_out_vals + pos * K;
        int32_t * oi = d_out_ids  + pos * K;
        for (int i = 0; i < K; i++) {
            ov[i] = smem[i].val;
            oi[i] = smem[i].id;
        }
    }
}

// ---- GpuTopKWorkspace ----

void GpuTopKWorkspace::ensure(int needed) {
    if (needed <= capacity) return;
    if (d_vals) { cudaFree(d_vals); d_vals = nullptr; }
    if (d_ids)  { cudaFree(d_ids);  d_ids  = nullptr; }
    int alloc = needed * 2;
    cudaError_t e1 = cudaMalloc(&d_vals, (size_t)alloc * sizeof(float));
    cudaError_t e2 = cudaMalloc(&d_ids,  (size_t)alloc * sizeof(int32_t));
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        std::fprintf(stderr, "GpuTopKWorkspace::ensure: cudaMalloc failed (%s, %s)\n",
                     cudaGetErrorString(e1), cudaGetErrorString(e2));
        if (d_vals) { cudaFree(d_vals); d_vals = nullptr; }
        if (d_ids)  { cudaFree(d_ids);  d_ids  = nullptr; }
        capacity = 0;
        return;
    }
    capacity = alloc;
}

GpuTopKWorkspace::~GpuTopKWorkspace() {
    if (d_vals) { cudaFree(d_vals); d_vals = nullptr; }
    if (d_ids)  { cudaFree(d_ids);  d_ids  = nullptr; }
}

// ---- Host entry point ----

// Dispatch to the right kernel instantiation.  The template parameter
// controls per-thread stack frame size and shared memory, so we keep a
// handful of instantiations for the most common K values.
void gpu_top_k_extract(const float * logits_gpu, int vocab, int n_pos,
                       size_t row_stride, int k,
                       float * out_vals, int32_t * out_ids,
                       GpuTopKWorkspace & ws,
                       cudaStream_t stream) {
    if (k <= 0 || k > GPU_TOPK_K) {
        std::fprintf(stderr, "gpu_top_k_extract: k=%d out of range [1,%d]\n",
                     k, GPU_TOPK_K);
        return;
    }
    if (n_pos <= 0 || vocab <= 0) return;
    if (k > vocab) k = vocab;

    const int total = k * n_pos;
    ws.ensure(total);
    if (!ws.d_vals || !ws.d_ids) return;

    // Use a dedicated non-blocking stream to avoid serialization with
    // ggml's default/legacy CUDA stream.
    static cudaStream_t s_stream = nullptr;
    if (!s_stream) cudaStreamCreateWithFlags(&s_stream, cudaStreamNonBlocking);
    cudaStream_t use_stream = stream ? stream : s_stream;

    // Select thread count and kernel instantiation based on k.
    // Shared memory budget: threads * k * sizeof(Pair) ≤ 48 KB.
    //   k=10: 256 threads, smem = 20 KB, stack = 80 B/thread
    //   k=20: 256 threads, smem = 40 KB, stack = 160 B/thread
    //   k=40: 64 threads,  smem = 20 KB, stack = 320 B/thread
    #define LAUNCH_TOPK(K_VAL, T_COUNT) do { \
        size_t smem_bytes = (size_t)(T_COUNT) * (K_VAL) * sizeof(Pair); \
        top_k_kernel<K_VAL><<<n_pos, T_COUNT, smem_bytes, use_stream>>>( \
            logits_gpu, vocab, row_stride, ws.d_vals, ws.d_ids); \
    } while (0)

    if (k <= 10) {
        LAUNCH_TOPK(10, 256);
    } else if (k <= 20) {
        LAUNCH_TOPK(20, 256);
    } else {
        LAUNCH_TOPK(40, 64);
    }
    #undef LAUNCH_TOPK

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "gpu_top_k_extract: kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        return;
    }

    cudaMemcpyAsync(out_vals, ws.d_vals,
                    (size_t)total * sizeof(float),
                    cudaMemcpyDeviceToHost, use_stream);
    cudaMemcpyAsync(out_ids, ws.d_ids,
                    (size_t)total * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, use_stream);
    err = cudaStreamSynchronize(use_stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "gpu_top_k_extract: sync failed: %s\n",
                     cudaGetErrorString(err));
    }
}
