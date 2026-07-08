// Local correctness + microbenchmark for CODA GLU-style GEMM epilogue rewrites
// (arXiv:2605.19269 §3.2.2).
//
// Model graph coverage:
//   qwen3    : SWIGLU       silu(gate(x)) * up(x)        -> ggml_glu_split(..., SWIGLU)
//   gemma4   : GEGLU        gelu(gate(x)) * up(x)        -> ggml_glu_split(..., GEGLU)
//   deepseek4: clamped SWIGLU silu(clamp(gate)) * clamp(up) -> clamp + ggml_glu_split(..., SWIGLU)
//
// qwen3/gemma4 produce the backend-fusible {MUL_MAT, MUL_MAT, GLU} pattern; the
// deepseek4 clamps intentionally remain explicit, so only the activation itself
// collapses from {SILU, MUL} into a GLU op. No model weights are needed.

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

enum class GluCase {
    SWIGLU,
    GEGLU,
    CLAMPED_SWIGLU,
};

static const char * case_name(GluCase c) {
    switch (c) {
        case GluCase::SWIGLU:         return "swiglu";
        case GluCase::GEGLU:          return "geglu";
        case GluCase::CLAMPED_SWIGLU: return "clamped_swiglu";
    }
    return "unknown";
}

static ggml_glu_op case_op(GluCase c) {
    return c == GluCase::GEGLU ? GGML_GLU_OP_GEGLU : GGML_GLU_OP_SWIGLU;
}

static void make_weight(ggml_type wtype, int K, int N, std::vector<uint8_t> & wbytes) {
    std::vector<float> f32(K * N);
    for (auto & v : f32) v = (float)(rand() % 2000 - 1000) / 1000.0f;

    wbytes.resize(ggml_row_size(wtype, K) * N);
    if (wtype == GGML_TYPE_F32) {
        memcpy(wbytes.data(), f32.data(), wbytes.size());
    } else if (wtype == GGML_TYPE_F16) {
        auto * h = (ggml_fp16_t *) wbytes.data();
        for (int i = 0; i < K * N; i++) h[i] = ggml_fp32_to_fp16(f32[i]);
    } else {
        ggml_quantize_chunk(wtype, f32.data(), wbytes.data(), 0, N, K, nullptr);
    }
}

static void run_glu_graph(ggml_backend_t backend, GluCase which, bool fused,
                          ggml_type wtype, int K, int N, int M,
                          const std::vector<uint8_t> & Wg_data,
                          const std::vector<uint8_t> & Wu_data,
                          const std::vector<float> & xdata,
                          std::vector<float> & out,
                          double * ms_per_iter = nullptr, int iters = 0) {
    const size_t ctx_size = 128 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * Wg = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * Wu = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_input(Wg);
    ggml_set_input(Wu);
    ggml_set_input(x);

    ggml_tensor * gate = ggml_mul_mat(ctx, Wg, x);
    ggml_tensor * up   = ggml_mul_mat(ctx, Wu, x);

    if (which == GluCase::CLAMPED_SWIGLU) {
        constexpr float limit = 10.0f;
        gate = ggml_clamp(ctx, gate, -INFINITY, limit);
        up   = ggml_clamp(ctx, up,   -limit, limit);
    }

    ggml_tensor * result = nullptr;
    if (fused) {
        result = ggml_glu_split(ctx, gate, up, case_op(which));
    } else {
        gate = (which == GluCase::GEGLU) ? ggml_gelu(ctx, gate) : ggml_silu(ctx, gate);
        result = ggml_mul(ctx, gate, up);
    }
    ggml_set_output(result);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(Wg, Wg_data.data(), 0, ggml_nbytes(Wg));
    ggml_backend_tensor_set(Wu, Wu_data.data(), 0, ggml_nbytes(Wu));
    ggml_backend_tensor_set(x,  xdata.data(),  0, ggml_nbytes(x));

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    out.resize(ggml_nelements(result));
    ggml_backend_tensor_get(result, out.data(), 0, out.size() * sizeof(float));

    if (ms_per_iter && iters > 0) {
        for (int it = 0; it < 5; ++it) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::high_resolution_clock::now();
        *ms_per_iter = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

static bool test_glu_fusion(ggml_backend_t backend, GluCase which,
                            ggml_type wtype, int K, int N, int M) {
    srand(1234);
    std::vector<uint8_t> Wg_data, Wu_data;
    make_weight(wtype, K, N, Wg_data);
    make_weight(wtype, K, N, Wu_data);
    std::vector<float> xdata(K * M);
    for (auto & v : xdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;

    std::vector<float> base, coda;
    run_glu_graph(backend, which, /*fused=*/false, wtype, K, N, M, Wg_data, Wu_data, xdata, base);
    run_glu_graph(backend, which, /*fused=*/true,  wtype, K, N, M, Wg_data, Wu_data, xdata, coda);

    float max_abs = 0.0f, max_diff = 0.0f;
    bool nonfinite = false;
    for (size_t i = 0; i < base.size(); i++) {
        if (!std::isfinite(base[i]) || !std::isfinite(coda[i])) {
            nonfinite = true;
            break;
        }
        max_abs  = std::fmax(max_abs, std::fabs(base[i]));
        max_diff = std::fmax(max_diff, std::fabs(base[i] - coda[i]));
    }

    const float rel = max_abs > 0.0f ? max_diff / max_abs : max_diff;
    const float tol = (wtype == GGML_TYPE_F16 && M == 1) ? 2e-3f : 1e-5f;
    const bool pass = !nonfinite && rel < tol && max_abs > 1e-4f;
    printf("[coda_glu] %-15s wtype=%-8s K=%4d N=%4d M=%2d  max_abs=%.5f max_diff=%.6f rel=%.2e %s\n",
           case_name(which), ggml_type_name(wtype), K, N, M, max_abs, max_diff, rel,
           nonfinite ? "NONFINITE" : (pass ? "PASS" : "FAIL"));
    return pass;
}

static void bench_glu_fusion(ggml_backend_t backend, GluCase which,
                             ggml_type wtype, int K, int N, int M) {
    srand(1234);
    std::vector<uint8_t> Wg_data, Wu_data;
    make_weight(wtype, K, N, Wg_data);
    make_weight(wtype, K, N, Wu_data);
    std::vector<float> xdata(K * M);
    for (auto & v : xdata) v = (float)(rand() % 2000 - 1000) / 1000.0f;

    std::vector<float> out;
    double base_ms = 0.0, coda_ms = 0.0;
    run_glu_graph(backend, which, /*fused=*/false, wtype, K, N, M, Wg_data, Wu_data, xdata, out, &base_ms, 200);
    run_glu_graph(backend, which, /*fused=*/true,  wtype, K, N, M, Wg_data, Wu_data, xdata, out, &coda_ms, 200);

    printf("[bench_glu] %-15s wtype=%-8s K=%4d N=%4d M=%2d  unfused=%.4f ms fused=%.4f ms speedup=%.2fx\n",
           case_name(which), ggml_type_name(wtype), K, N, M, base_ms, coda_ms,
           coda_ms > 0.0 ? base_ms / coda_ms : 0.0);
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "CUDA backend not available\n");
        return 1;
    }

    bool ok = true;
    for (GluCase which : { GluCase::SWIGLU, GluCase::GEGLU, GluCase::CLAMPED_SWIGLU }) {
        for (int M : {1, 8, 32}) {
            ok &= test_glu_fusion(backend, which, GGML_TYPE_Q4_K, 512, 256, M);
        }
        for (int M : {1, 8, 32}) {
            ok &= test_glu_fusion(backend, which, GGML_TYPE_F16, 512, 256, M);
        }
    }

    printf("\n");
    for (GluCase which : { GluCase::SWIGLU, GluCase::GEGLU, GluCase::CLAMPED_SWIGLU }) {
        for (int M : {1, 32}) {
            bench_glu_fusion(backend, which, GGML_TYPE_Q4_K, 512, 256, M);
        }
    }

    ggml_backend_free(backend);
    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
