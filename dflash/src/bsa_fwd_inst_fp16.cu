// Instantiate BSA's hdim=128 fp16 forward block kernel.
// Slow to compile (cutlass templates) — separate translation unit so incremental rebuilds skip this.
#include "flash_fwd_block_hdim128_fp16_sm80.cu"
