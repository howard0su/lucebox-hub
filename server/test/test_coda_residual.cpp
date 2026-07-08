// Unit test + microbenchmark for the CODA GEMM-Residual epilogue (arXiv:2605.19269 §3.2.1),
// prefill path (mmq / M>1 quantized GEMM).
//
// Pattern under test:   h1 = mul_mat(W, x) + z          (residual add fused into GEMM epilogue)
//
// The fork already fuses {MUL_MAT, ADD} into the mat-vec epilogue for M==1
// (decode). For M>1 the add is a separate pass today; the DFLASH_CODA mmq fork
// adds the residual epilogue there too. This harness:
//   * correctness: compares the CUDA {mul_mat, add} result against the ggml CPU
//     backend (ground truth) for M covering both mmvq (M<=8) and mmq (M>8);
//   * microbenchmark: times the CUDA graph so the unfused baseline can be
//     compared against the fused mmq epilogue (toggle GGML_CUDA_DISABLE_FUSION).
//
// Runs on any CUDA arch (incl. sm_75); no model weights needed.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Max batch for which ggml-cuda uses the MMVQ (mat-vec) kernels; above this it
// uses MMQ (batched GEMM). Mirrors MMVQ_MAX_BATCH_SIZE in mmvq.cuh.
static constexpr int kMmvqMaxBatch = 8;

// Build h1 = mul_mat(W, x) + z on `backend`, returning the output in `out`.
// Weight bytes (already quantized/converted to wtype), x and z (both F32) are
// provided so the CPU and CUDA runs see identical inputs.
static void run_residual(ggml_backend_t backend, ggml_type wtype, int K, int N, int M,
                         const std::vector<uint8_t> & wbytes,
                         const std::vector<float> & xdata,
                         const std::vector<float> & zdata,
                         std::vector<float> & out,
                         double * ms_per_iter = nullptr, int iters = 0) {
    const size_t ctx_size = 64 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, N);            // [K, N]
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);    // [K, M]
    ggml_tensor * z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);    // [N, M] residual
    ggml_set_input(W);
    ggml_set_input(x);
    ggml_set_input(z);

    ggml_tensor * h0 = ggml_mul_mat(ctx, W, x);   // [N, M]
    ggml_tensor * h1 = ggml_add(ctx, h0, z);      // residual add -> fusible {MUL_MAT, ADD}
    ggml_set_output(h1);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, h1);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(W, wbytes.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(x, xdata.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(z, zdata.data(), 0, ggml_nbytes(z));

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    out.resize(ggml_nelements(h1));
    ggml_backend_tensor_get(h1, out.data(), 0, out.size() * sizeof(float));

    if (ms_per_iter && iters > 0) {
        // Warmup.
        for (int it = 0; it < 5; it++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; it++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::high_resolution_clock::now();
        *ms_per_iter = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

// Fill weight bytes for wtype from freshly generated F32 data.
static void make_weight(ggml_type wtype, int K, int N, std::vector<uint8_t> & wbytes) {
    std::vector<float> f32(K * N);
    for (auto & v : f32) v = (float)(rand() % 2000 - 1000) / 1000.0f;
    const size_t row = ggml_row_size(wtype, K);
    wbytes.resize(row * N);
    if (wtype == GGML_TYPE_F32) {
        memcpy(wbytes.data(), f32.data(), wbytes.size());
    } else if (wtype == GGML_TYPE_F16) {
        auto * h = (ggml_fp16_t *) wbytes.data();
        for (int i = 0; i < K * N; i++) h[i] = ggml_fp32_to_fp16(f32[i]);
    } else {
        ggml_quantize_chunk(wtype, f32.data(), wbytes.data(), 0, N, K, nullptr);
    }
}

static bool test_correctness(ggml_backend_t cuda, ggml_backend_t cpu,
                             ggml_type wtype, int K, int N, int M) {
    srand(7);
    std::vector<uint8_t> wbytes;
    make_weight(wtype, K, N, wbytes);
    std::vector<float> xdata(K * M), zdata(N * M);
    for (auto & v : xdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : zdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;

    std::vector<float> gpu, ref;
    run_residual(cuda, wtype, K, N, M, wbytes, xdata, zdata, gpu);
    run_residual(cpu,  wtype, K, N, M, wbytes, xdata, zdata, ref);

    float max_abs = 0.0f, max_diff = 0.0f;
    bool nonfinite = false;
    for (size_t i = 0; i < ref.size(); i++) {
        if (!std::isfinite(gpu[i]) || !std::isfinite(ref[i])) { nonfinite = true; break; }
        max_abs  = std::fmax(max_abs, std::fabs(ref[i]));
        max_diff = std::fmax(max_diff, std::fabs(gpu[i] - ref[i]));
    }
    const float rel = max_abs > 0.0f ? max_diff / max_abs : max_diff;
    // Cross-backend: CUDA quantizes activations to q8_1, so Q4_K differs from the
    // CPU F32 dequant path by quantization noise (~1e-2 rel). F16/F32 are tighter.
    const float tol = (wtype == GGML_TYPE_Q4_K) ? 3e-2f : 5e-3f;
    const char * path = (M <= kMmvqMaxBatch) ? "mmvq" : "mmq ";
    const bool pass = !nonfinite && rel < tol && max_abs > 1e-4f;
    printf("[correctness] %s wtype=%-6s K=%4d N=%4d M=%4d  rel=%.2e (tol=%.0e) %s\n",
           path, ggml_type_name(wtype), K, N, M, rel, tol,
           nonfinite ? "NONFINITE" : (pass ? "PASS" : "FAIL"));
    return pass;
}

// Build h0 = mul_mat(W, x) ALONE on `backend` (no add, so no residual fusion is
// possible), returning h0 in `out`. Used to construct the "unfused" GPU reference.
static void run_matmul_only(ggml_backend_t backend, ggml_type wtype, int K, int N, int M,
                            const std::vector<uint8_t> & wbytes,
                            const std::vector<float> & xdata,
                            std::vector<float> & out) {
    const size_t ctx_size = 64 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_input(W);
    ggml_set_input(x);

    ggml_tensor * h0 = ggml_mul_mat(ctx, W, x);
    ggml_set_output(h0);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, h0);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(W, wbytes.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(x, xdata.data(), 0, ggml_nbytes(x));

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    out.resize(ggml_nelements(h0));
    ggml_backend_tensor_get(h0, out.data(), 0, out.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

// Direct GPU fused-vs-unfused check: the fused single-graph {mul_mat, add} result
// (which uses the CODA epilogue when DFLASH_CODA is set) must equal a standalone
// GPU GEMM followed by the residual add done separately. This isolates the mmq
// residual-epilogue fork from cross-backend quantization noise: both sides run the
// identical GPU GEMM, so the fp32 residual add should agree essentially exactly.
static bool test_fused_vs_split(ggml_backend_t cuda, ggml_type wtype, int K, int N, int M) {
    srand(7);
    std::vector<uint8_t> wbytes;
    make_weight(wtype, K, N, wbytes);
    std::vector<float> xdata(K * M), zdata(N * M);
    for (auto & v : xdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : zdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;

    std::vector<float> fused, h0;
    run_residual(cuda, wtype, K, N, M, wbytes, xdata, zdata, fused);
    run_matmul_only(cuda, wtype, K, N, M, wbytes, xdata, h0);

    float max_abs = 0.0f, max_diff = 0.0f;
    bool nonfinite = false;
    for (size_t i = 0; i < fused.size(); i++) {
        const float ref = h0[i] + zdata[i];   // unfused residual add
        if (!std::isfinite(fused[i]) || !std::isfinite(ref)) { nonfinite = true; break; }
        max_abs  = std::fmax(max_abs, std::fabs(ref));
        max_diff = std::fmax(max_diff, std::fabs(fused[i] - ref));
    }
    const float rel = max_abs > 0.0f ? max_diff / max_abs : max_diff;
    // Same GPU GEMM on both sides -> fp32 add must match to rounding (~1e-6 rel).
    const float tol = 1e-5f;
    const char * path = (M <= kMmvqMaxBatch) ? "mmvq" : "mmq ";
    const bool pass = !nonfinite && rel < tol;
    printf("[fused==split] %s wtype=%-6s K=%4d N=%4d M=%4d  rel=%.2e (tol=%.0e) %s\n",
           path, ggml_type_name(wtype), K, N, M, rel, tol,
           nonfinite ? "NONFINITE" : (pass ? "PASS" : "FAIL"));
    return pass;
}

static void bench(ggml_backend_t cuda, ggml_type wtype, int K, int N, int M) {
    srand(7);
    std::vector<uint8_t> wbytes;
    make_weight(wtype, K, N, wbytes);
    std::vector<float> xdata(K * M), zdata(N * M);
    for (auto & v : xdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : zdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;

    std::vector<float> out;
    double ms = 0.0;
    run_residual(cuda, wtype, K, N, M, wbytes, xdata, zdata, out, &ms, 200);
    const char * path = (M <= kMmvqMaxBatch) ? "mmvq" : "mmq ";
    printf("[bench]       %s wtype=%-6s K=%4d N=%4d M=%4d  %.4f ms/iter\n",
           path, ggml_type_name(wtype), K, N, M, ms);
}

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) { fprintf(stderr, "CUDA backend not available\n"); return 1; }
    ggml_backend_t cpu = ggml_backend_cpu_init();

    static const bool fusion_off = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr);
    printf("=== CODA residual epilogue (fusion %s) ===\n", fusion_off ? "OFF" : "ON");

    bool ok = true;
    // Q4_K (production). M=1 -> mmvq (already fuses residual); M>8 -> mmq.
    for (int M : {1, 8, 32, 128, 512}) {
        ok &= test_correctness(cuda, cpu, GGML_TYPE_Q4_K, 512, 1024, M);
    }

    // Direct GPU fused-vs-unfused parity (validates the mmq residual-epilogue fork
    // itself; set DFLASH_CODA=1 to exercise the fused mmq path for M>8).
    printf("\n");
    for (int M : {1, 8, 32, 128, 512}) {
        ok &= test_fused_vs_split(cuda, GGML_TYPE_Q4_K, 512, 1024, M);
    }

    printf("\n");
    for (int M : {1, 32, 128, 512, 2048}) {
        bench(cuda, GGML_TYPE_Q4_K, 512, 1024, M);
    }

    ggml_backend_free(cpu);
    ggml_backend_free(cuda);
    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
