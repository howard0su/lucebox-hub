// HIP compatibility shim for <cuda_bf16.h>
#pragma once

// cuda_runtime.h (our compat) must be included first to ensure __HIP_PLATFORM_AMD__ is set
// before hip_bfloat16.h is parsed. If included in isolation, set it now.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#  define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_bfloat16.h>
#include <cstring>  // memcpy for raw bit reinterpretation on host

// Type alias: CUDA __nv_bfloat16 → AMD hip_bfloat16
using __nv_bfloat16 = hip_bfloat16;

// hip_bfloat162 does not exist in all ROCm versions; skip the alias.
// Tests and source code that reference __nv_bfloat162 will need guarding.

// Conversion intrinsics.
//
// When compiled by hipcc, hip_bfloat16's constructor and operator float() are
// __host__ __device__. When compiled by g++ (plain CXX sources), __HOST_DEVICE__
// collapses to __device__, making them unavailable on the host.
//
// Provide host-side helpers via raw bit manipulation so that test code and
// pure-CXX source files can use these conversions without the device compiler.

#ifdef __HIPCC__
// hipcc path: use the type's own constructors / conversions
__device__ __host__ inline float __bfloat162float(hip_bfloat16 x) {
    return static_cast<float>(x);
}
__device__ __host__ inline hip_bfloat16 __float2bfloat16(float x) {
    return hip_bfloat16(x);
}
__device__ __host__ inline hip_bfloat16 __float2bfloat16_rn(float x) {
    return hip_bfloat16(x);
}
#else
// g++ / plain CXX path: bit-cast approach, no device attributes
namespace __hip_bf16_compat_detail {
    // Truncating float→bf16: drop lower 16 mantissa bits.
    inline uint16_t float_to_bf16_bits(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return static_cast<uint16_t>(u >> 16);
    }
    inline float bf16_bits_to_float(uint16_t b) {
        uint32_t u = static_cast<uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
}
inline float __bfloat162float(hip_bfloat16 x) {
    return __hip_bf16_compat_detail::bf16_bits_to_float(x.data);
}
inline hip_bfloat16 __float2bfloat16(float x) {
    hip_bfloat16 r;
    r.data = __hip_bf16_compat_detail::float_to_bf16_bits(x);
    return r;
}
inline hip_bfloat16 __float2bfloat16_rn(float x) {
    return __float2bfloat16(x);
}
#endif
