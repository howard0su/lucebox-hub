// Prototype/unit test for the CODA §3.2.1 "two-output" RMSNorm opportunity.
//
// The paper's GEMM-Residual-RMSNorm fusion wants a GEMM epilogue that writes:
//   1. h = mul_mat(W, x) + residual                         [N, M]
//   2. partial mean-square stats over h for the next RMSNorm [N / block, M]
//
// ggml tensors are single-output ops today, but a graph can retain multiple
// output tensors. This test models the desired contract without adding a new op:
// `h` is marked as one graph output and a side tensor made from partial
// sum_rows(sqr(view(h))) blocks is marked as a second output. That validates the
// graph allocator/lifetime shape we need before forking mmq to write both outputs
// directly from the epilogue.

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void make_weight(ggml_type wtype, int K, int N, std::vector<uint8_t> & wbytes) {
    std::vector<float> f32(K * N);
    for (auto & v : f32) v = (float)(rand() % 2000 - 1000) / 1000.0f;

    wbytes.resize(ggml_row_size(wtype, K) * N);
    if (wtype == GGML_TYPE_F32) {
        memcpy(wbytes.data(), f32.data(), wbytes.size());
    } else if (wtype == GGML_TYPE_F16) {
        auto * h = (ggml_fp16_t *) wbytes.data();
        for (int i = 0; i < K * N; ++i) h[i] = ggml_fp32_to_fp16(f32[i]);
    } else {
        ggml_quantize_chunk(wtype, f32.data(), wbytes.data(), 0, N, K, nullptr);
    }
}

static ggml_tensor * build_partial_ms(ggml_context * ctx, ggml_tensor * h, int block) {
    const int N = (int) h->ne[0];
    const int M = (int) h->ne[1];
    GGML_ASSERT(N % block == 0);

    ggml_tensor * stats = nullptr;
    for (int b = 0; b < N / block; ++b) {
        ggml_tensor * h_blk = ggml_view_2d(
            ctx, h, block, M, h->nb[1], (size_t) b * (size_t) block * ggml_element_size(h));
        // CUDA unary kernels require contiguous inputs; the block view is row-strided
        // for M > 1. A fused mmq side-output epilogue would write these partials
        // directly and avoid this materialization.
        h_blk = ggml_cont(ctx, h_blk);
        ggml_tensor * ms_blk = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, h_blk)), 1.0f / (float) block);
        stats = stats ? ggml_concat(ctx, stats, ms_blk, /*dim=*/0) : ms_blk;
    }
    return stats; // [N / block, M]
}

static void run_two_output_graph(ggml_backend_t backend, ggml_type wtype, int K, int N, int M, int block,
                                 const std::vector<uint8_t> & wbytes,
                                 const std::vector<float> & xdata,
                                 const std::vector<float> & zdata,
                                 std::vector<float> & h_out,
                                 std::vector<float> & stats_out,
                                 bool fused_side_output = false,
                                 double * ms_per_iter = nullptr,
                                 int iters = 0) {
    const size_t ctx_size = 128 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    ggml_set_input(W);
    ggml_set_input(x);
    ggml_set_input(z);

    ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, W, x), z);
    ggml_tensor * stats = nullptr;
    if (fused_side_output) {
        stats = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N / block, M);
        ggml_set_name(stats, "coda_partial_ms");
    } else {
        stats = build_partial_ms(ctx, h, block);
    }

    ggml_set_output(h);
    ggml_set_output(stats);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    // `h` and `stats` are two observable outputs. In fused_side_output mode the
    // stats tensor is a named leaf output written as a side effect by the mmq
    // residual epilogue prototype.
    ggml_build_forward_expand(gf, h);
    ggml_build_forward_expand(gf, stats);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(W, wbytes.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(x, xdata.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(z, zdata.data(), 0, ggml_nbytes(z));

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    h_out.resize(ggml_nelements(h));
    stats_out.resize(ggml_nelements(stats));
    ggml_backend_tensor_get(h, h_out.data(), 0, h_out.size() * sizeof(float));
    ggml_backend_tensor_get(stats, stats_out.data(), 0, stats_out.size() * sizeof(float));

    if (ms_per_iter && iters > 0) {
        for (int it = 0; it < 5; ++it) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const auto t1 = std::chrono::high_resolution_clock::now();
        *ms_per_iter = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static double bench_residual_only(ggml_backend_t backend, ggml_type wtype, int K, int N, int M,
                                  const std::vector<uint8_t> & wbytes,
                                  const std::vector<float> & xdata,
                                  const std::vector<float> & zdata,
                                  int iters) {
    const size_t ctx_size = 64 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    ggml_set_input(W);
    ggml_set_input(x);
    ggml_set_input(z);

    ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, W, x), z);
    ggml_set_output(h);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, h);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(W, wbytes.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(x, xdata.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(z, zdata.data(), 0, ggml_nbytes(z));

    for (int it = 0; it < 5; ++it) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    const auto t1 = std::chrono::high_resolution_clock::now();

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static bool check_partial_ms(const std::vector<float> & h, const std::vector<float> & stats,
                             int N, int M, int block, const char * label) {
    float max_abs = 0.0f;
    float max_diff = 0.0f;
    bool nonfinite = false;

    for (int m = 0; m < M; ++m) {
        for (int b = 0; b < N / block; ++b) {
            float ref = 0.0f;
            for (int i = 0; i < block; ++i) {
                const float v = h[(size_t) m * (size_t) N + (size_t) b * (size_t) block + (size_t) i];
                ref += v * v;
            }
            ref /= (float) block;
            const float got = stats[(size_t) m * (size_t) (N / block) + (size_t) b];
            if (!std::isfinite(ref) || !std::isfinite(got)) {
                nonfinite = true;
                break;
            }
            max_abs = std::fmax(max_abs, std::fabs(ref));
            max_diff = std::fmax(max_diff, std::fabs(got - ref));
        }
    }

    const float rel = max_abs > 0.0f ? max_diff / max_abs : max_diff;
    const bool pass = !nonfinite && rel < 2e-5f && max_abs > 1e-4f;
    printf("[coda_rms_side] %-8s N=%4d M=%4d block=%3d stats=%4d  rel=%.2e %s\n",
           label, N, M, block, N / block, rel, nonfinite ? "NONFINITE" : (pass ? "PASS" : "FAIL"));
    return pass;
}

static bool test_two_outputs(ggml_backend_t backend, ggml_type wtype, int K, int N, int M, int block) {
    srand(2026);
    std::vector<uint8_t> wbytes;
    make_weight(wtype, K, N, wbytes);

    std::vector<float> xdata(K * M), zdata(N * M);
    for (auto & v : xdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : zdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;

    std::vector<float> h, stats;
    std::vector<float> h_ref, stats_ref;
    run_two_output_graph(backend, wtype, K, N, M, block, wbytes, xdata, zdata, h_ref, stats_ref,
                         /*fused_side_output=*/false);

    bool ok = check_partial_ms(h_ref, stats_ref, N, M, block, ggml_type_name(wtype));

    const bool expect_fused = ggml_is_quantized(wtype) && M > 8;
    if (!expect_fused) {
        printf("[coda_rms_fused] %-8s N=%4d M=%4d block=%3d SKIP (mmq quantized M>8 only)\n",
               ggml_type_name(wtype), N, M, block);
        return ok;
    }

    run_two_output_graph(backend, wtype, K, N, M, block, wbytes, xdata, zdata, h, stats,
                         /*fused_side_output=*/true);

    ok &= check_partial_ms(h, stats, N, M, block, ggml_type_name(wtype));

    float max_h = 0.0f, max_h_diff = 0.0f, max_s = 0.0f, max_s_diff = 0.0f;
    for (size_t i = 0; i < h.size(); ++i) {
        max_h = std::fmax(max_h, std::fabs(h_ref[i]));
        max_h_diff = std::fmax(max_h_diff, std::fabs(h[i] - h_ref[i]));
    }
    for (size_t i = 0; i < stats.size(); ++i) {
        max_s = std::fmax(max_s, std::fabs(stats_ref[i]));
        max_s_diff = std::fmax(max_s_diff, std::fabs(stats[i] - stats_ref[i]));
    }
    const float h_rel = max_h > 0.0f ? max_h_diff / max_h : max_h_diff;
    const float s_rel = max_s > 0.0f ? max_s_diff / max_s : max_s_diff;
    const bool parity = h_rel < 1e-5f && s_rel < 1e-5f;
    printf("[coda_rms_fused] %-8s N=%4d M=%4d block=%3d h_rel=%.2e stats_rel=%.2e %s\n",
           ggml_type_name(wtype), N, M, block, h_rel, s_rel, parity ? "PASS" : "FAIL");
    return ok && parity;
}

static void bench_two_outputs(ggml_backend_t backend, ggml_type wtype, int K, int N, int M, int block) {
    srand(2026);
    std::vector<uint8_t> wbytes;
    make_weight(wtype, K, N, wbytes);

    std::vector<float> xdata(K * M), zdata(N * M);
    for (auto & v : xdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;
    for (auto & v : zdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;

    std::vector<float> h, stats;
    double ms = 0.0;
    run_two_output_graph(backend, wtype, K, N, M, block, wbytes, xdata, zdata, h, stats,
                         /*fused_side_output=*/false, &ms, 200);
    double fused_ms = 0.0;
    run_two_output_graph(backend, wtype, K, N, M, block, wbytes, xdata, zdata, h, stats,
                         /*fused_side_output=*/true, &fused_ms, 200);
    const double residual_ms = bench_residual_only(backend, wtype, K, N, M, wbytes, xdata, zdata, 200);
    printf("[bench_rms_side] wtype=%-6s K=%4d N=%4d M=%4d block=%3d  residual=%.4f ms composed=%.4f ms fused=%.4f ms fused/residual=%.2fx fused/composed=%.2fx\n",
           ggml_type_name(wtype), K, N, M, block, residual_ms, ms, fused_ms,
           residual_ms > 0.0 ? fused_ms / residual_ms : 0.0,
           ms > 0.0 ? fused_ms / ms : 0.0);
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "CUDA backend not available\n");
        return 1;
    }

    bool ok = true;
    // Q4_K exercises the production quantized mmq path (M > 8) and mmvq path (M <= 8).
    for (int M : {1, 32, 128}) {
        ok &= test_two_outputs(backend, GGML_TYPE_Q4_K, 512, 1024, M, 256);
    }
    // F16 keeps the same graph contract on a dense GEMM path.
    ok &= test_two_outputs(backend, GGML_TYPE_F16, 512, 1024, 32, 256);

    printf("\n");
    for (int M : {32, 128, 512}) {
        bench_two_outputs(backend, GGML_TYPE_Q4_K, 512, 1024, M, 256);
    }

    ggml_backend_free(backend);
    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
