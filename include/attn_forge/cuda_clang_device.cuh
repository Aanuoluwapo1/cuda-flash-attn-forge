#pragma once

// =============================================================================
// Minimal Clang/NVCC Standalone CUDA Runtime & Device Header Shim
// Enables compiling src/kernels.cu directly to SM 7.5 PTX via Clang's built-in
// NVPTX backend (clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_75)
// and launching live GPU kernels via Windows/Linux CUDA Driver API (nvcuda.dll)
// even when the standalone NVIDIA CUDA Toolkit (nvcc / cudart) is not installed.
// =============================================================================

#ifdef __CUDA_ARCH__

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

struct __cuda_builtin_threadIdx_t {
    __device__ unsigned int __fetch_x() const { return __nvvm_read_ptx_sreg_tid_x(); }
    __device__ unsigned int __fetch_y() const { return __nvvm_read_ptx_sreg_tid_y(); }
    __device__ unsigned int __fetch_z() const { return __nvvm_read_ptx_sreg_tid_z(); }
    __declspec(property(get = __fetch_x)) unsigned int x;
    __declspec(property(get = __fetch_y)) unsigned int y;
    __declspec(property(get = __fetch_z)) unsigned int z;
};

struct __cuda_builtin_blockIdx_t {
    __device__ unsigned int __fetch_x() const { return __nvvm_read_ptx_sreg_ctaid_x(); }
    __device__ unsigned int __fetch_y() const { return __nvvm_read_ptx_sreg_ctaid_y(); }
    __device__ unsigned int __fetch_z() const { return __nvvm_read_ptx_sreg_ctaid_z(); }
    __declspec(property(get = __fetch_x)) unsigned int x;
    __declspec(property(get = __fetch_y)) unsigned int y;
    __declspec(property(get = __fetch_z)) unsigned int z;
};

struct __cuda_builtin_blockDim_t {
    __device__ unsigned int __fetch_x() const { return __nvvm_read_ptx_sreg_ntid_x(); }
    __device__ unsigned int __fetch_y() const { return __nvvm_read_ptx_sreg_ntid_y(); }
    __device__ unsigned int __fetch_z() const { return __nvvm_read_ptx_sreg_ntid_z(); }
    __declspec(property(get = __fetch_x)) unsigned int x;
    __declspec(property(get = __fetch_y)) unsigned int y;
    __declspec(property(get = __fetch_z)) unsigned int z;
};

__device__ constexpr __cuda_builtin_threadIdx_t threadIdx{};
__device__ constexpr __cuda_builtin_blockIdx_t blockIdx{};
__device__ constexpr __cuda_builtin_blockDim_t blockDim{};

__device__ inline float __shfl_xor_sync(unsigned int mask, float var, int laneMask) {
    float ret;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, %3;"
                 : "=f"(ret)
                 : "f"(var), "r"(laneMask), "r"(mask));
    return ret;
}

__device__ inline float __expf(float x) {
    float ret;
    // Fast hardware base-2 exponential: exp(x) = 2^(x * log2(e))
    asm volatile("mul.f32 %0, %1, 0f3FB8AA3B;\n\t"
                 "ex2.approx.ftz.f32 %0, %0;"
                 : "=f"(ret)
                 : "f"(x));
    return ret;
}

__device__ inline float fmaxf(float a, float b) {
    float ret;
    asm volatile("max.f32 %0, %1, %2;" : "=f"(ret) : "f"(a), "f"(b));
    return ret;
}

__device__ inline int min(int a, int b) {
    return (a < b) ? a : b;
}

#endif // __CUDA_ARCH__
