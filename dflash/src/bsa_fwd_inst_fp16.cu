// Instantiate BSA's hdim=128 FP16 forward block kernel for SM75 (Turing).
// We provide a custom specialization that SKIPS the Is_dropout=true template
// path entirely. The reason: even with `if constexpr`, nvcc still instantiates
// the full kernel signature including write_softmax_to_gmem's static_assert
// when taking &flash_fwd_block_kernel<..., Return_softmax=true>. SM75's
// DefaultCopy atom layout is incompatible with that code.
// Since inference never uses dropout (p_dropout == 1.0), this is safe.
//
// Slow to compile (CUTLASS templates) — separate TU for incremental builds.

#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_block_<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    // No dropout, no is_sm8x heuristic (SM75 always uses 128x64 tile).
    run_flash_fwd_block<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, cutlass::half_t>,
                        /*Is_dropout=*/false, /*Is_causal=*/false>(params, stream);
}

} // namespace FLASH_NAMESPACE
