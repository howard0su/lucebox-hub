#pragma once

// Minimal CUDA/HIP runtime compatibility for dflash harness code that already
// uses cuda* names. ggml exposes the HIP backend through ggml-cuda.h-compatible
// APIs, so keeping the call sites unchanged avoids touching the hot-path logic.

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)

#include <hip/hip_runtime.h>

#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemset hipMemset
#define cudaSetDevice hipSetDevice
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

#else

#include <cuda_runtime.h>

#endif
