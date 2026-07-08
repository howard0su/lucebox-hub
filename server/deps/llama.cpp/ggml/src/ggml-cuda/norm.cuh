#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_from_partial_ms(ggml_backend_cuda_context & ctx,
                                           ggml_tensor * dst,
                                           const ggml_tensor * partial_ms,
                                           int partial_block);

void ggml_cuda_op_rms_norm_from_partial_ms_fused(ggml_backend_cuda_context & ctx,
                                                 ggml_tensor * dst,
                                                 ggml_tensor * mul_tensor,
                                                 const ggml_tensor * partial_ms,
                                                 int partial_block);

void ggml_cuda_op_coda_partial_ms(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor * src,
                                  ggml_tensor * partial_ms,
                                  int partial_block);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
