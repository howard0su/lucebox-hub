// Local correctness test for the CODA SwiGLU-GEMM fusion (arXiv:2605.19269 §3.2.2).
//
// The qwen35 FFN rewrite (build_swiglu_ffn behind DFLASH_CODA) expresses the
// gate/up projections + SwiGLU activation as {MUL_MAT, MUL_MAT, GLU} so the
// ggml-cuda backend fuses the pairwise silu*mul activation into the gate/up
// GEMM epilogue (ggml_cuda_should_fuse_mul_mat).
//
// This test validates, on the local CUDA device (no full model needed), that
// the fused formulation produces numerically identical results to the original
// unfused path:
//
//   baseline: gu = silu(mul_mat(Wg, x)) * mul_mat(Wu, x)
//   coda:     gu = ggml_glu_split(mul_mat(Wg, x), mul_mat(Wu, x), SWIGLU)
//
// It exercises quantized (Q4_K, matching the production Q4_K_M weights) as well
// as F16 weights, and both M=1 (decode / mmvq-mmvf) and M>1 (prefill / mmq-mmf)
// token counts, which take different fused kernels internally.

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Build gate/up quantized (or F16) weights [K, N] and input x [K, M], then
// compare the unfused silu/mul path against the fused glu_split path.
static bool test_swiglu_fusion(ggml_backend_t backend, ggml_type wtype,
                               int K, int N, int M) {
    const size_t ctx_size = 128 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    // Weights: ne[0]=K (contraction), ne[1]=N (output features).
    ggml_tensor * Wg = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * Wu = ggml_new_tensor_2d(ctx, wtype, K, N);
    // Input activations: ne[0]=K, ne[1]=M (tokens). Shared by gate and up so the
    // fusion precondition ffn_up->src[1] == ffn_gate->src[1] holds.
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);

    ggml_set_input(Wg);
    ggml_set_input(Wu);
    ggml_set_input(x);

    // Baseline (unfused): silu(gate) * up
    ggml_tensor * g0   = ggml_mul_mat(ctx, Wg, x);
    ggml_tensor * gate = ggml_silu(ctx, g0);
    ggml_tensor * up0  = ggml_mul_mat(ctx, Wu, x);
    ggml_tensor * base = ggml_mul(ctx, gate, up0);

    // CODA (fusible): {MUL_MAT, MUL_MAT, GLU}
    ggml_tensor * g1   = ggml_mul_mat(ctx, Wg, x);
    ggml_tensor * u1   = ggml_mul_mat(ctx, Wu, x);
    ggml_tensor * coda = ggml_glu_split(ctx, g1, u1, GGML_GLU_OP_SWIGLU);

    ggml_set_output(base);
    ggml_set_output(coda);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, base);
    ggml_build_forward_expand(gf, coda);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    srand(1234);
    auto rnd = []() { return (float)(rand() % 2000 - 1000) / 1000.0f; };

    // Fill weights: generate F32 then quantize (or convert to F16) to the wtype.
    auto upload_weight = [&](ggml_tensor * W) {
        std::vector<float> f32(K * N);
        for (auto & v : f32) v = rnd();
        if (wtype == GGML_TYPE_F32) {
            ggml_backend_tensor_set(W, f32.data(), 0, ggml_nbytes(W));
        } else if (wtype == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> f16(K * N);
            for (int i = 0; i < K * N; i++) f16[i] = ggml_fp32_to_fp16(f32[i]);
            ggml_backend_tensor_set(W, f16.data(), 0, ggml_nbytes(W));
        } else {
            std::vector<uint8_t> q(ggml_nbytes(W));
            ggml_quantize_chunk(wtype, f32.data(), q.data(), 0, N, K, nullptr);
            ggml_backend_tensor_set(W, q.data(), 0, ggml_nbytes(W));
        }
    };
    upload_weight(Wg);
    upload_weight(Wu);

    std::vector<float> xbuf(K * M);
    for (auto & v : xbuf) v = rnd();
    ggml_backend_tensor_set(x, xbuf.data(), 0, ggml_nbytes(x));

    ggml_backend_graph_compute(backend, gf);

    const size_t n = ggml_nelements(base);
    std::vector<float> a(n), b(n);
    ggml_backend_tensor_get(base, a.data(), 0, n * sizeof(float));
    ggml_backend_tensor_get(coda, b.data(), 0, n * sizeof(float));

    float max_abs = 0.0f, max_diff = 0.0f;
    bool nonfinite = false;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { nonfinite = true; break; }
        max_abs  = std::fmax(max_abs, std::fabs(a[i]));
        max_diff = std::fmax(max_diff, std::fabs(a[i] - b[i]));
    }

    // The fused and unfused paths differ only in whether the identical silu*mul
    // epilogue runs inside or outside the GEMM kernel; the mul_mat itself is the
    // same op. For quantized weights the two paths are bit-identical; for F16 the
    // M=1 mat-vec may pick a differently-accumulating fused kernel, so we use a
    // relative tolerance sized to F16 precision. Require non-trivial output to
    // avoid a vacuous pass.
    const float rel = max_abs > 0.0f ? max_diff / max_abs : max_diff;
    const bool pass = !nonfinite && rel < 2e-3f && max_abs > 1e-4f;
    printf("[coda_swiglu] wtype=%-8s K=%4d N=%4d M=%2d  max_abs=%.5f max_diff=%.6f rel=%.2e %s\n",
           ggml_type_name(wtype), K, N, M, max_abs, max_diff, rel,
           nonfinite ? "NONFINITE" : (pass ? "PASS" : "FAIL"));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return pass;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "CUDA backend not available\n");
        return 1;
    }

    bool ok = true;
    // Q4_K matches the production Q4_K_M weights (K must be a multiple of 256).
    for (int M : {1, 8, 32}) {
        ok &= test_swiglu_fusion(backend, GGML_TYPE_Q4_K, 512, 256, M);
    }
    // F16 exercises the mmvf/mmf fused path.
    for (int M : {1, 8, 32}) {
        ok &= test_swiglu_fusion(backend, GGML_TYPE_F16, 512, 256, M);
    }

    ggml_backend_free(backend);
    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
