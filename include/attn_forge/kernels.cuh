#pragma once

#include "attn_forge/types.hpp"
#include <cstdio>
#include <stdexcept>

#if ATTN_FORGE_ENABLE_CUDA
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            char msg__[512];                                                                 \
            std::snprintf(                                                                   \
                msg__, sizeof(msg__), "CUDA error at %s:%d in %s: %s (%d)", __FILE__,        \
                __LINE__, #call, cudaGetErrorString(err__), static_cast<int>(err__));        \
            throw std::runtime_error(msg__);                                                 \
        }                                                                                    \
    } while (0)
#else
#define CUDA_CHECK(call) ((void)(call))
#endif

namespace attn_forge {

// Returns true if compiled with nvcc and at least one CUDA-capable GPU is available at runtime
bool cuda_backend_available();

// Returns the active device description or CPU fallback explanation
std::string get_device_description();

// Kernel 1: Shared-memory tiled GEMM (TILE_M x TILE_N x TILE_K) with +1 bank-conflict padding
// Executes on GPU if available; otherwise falls back to exact tiled emulation on host
double launch_gemm_tiled(
    const float* h_A,
    const float* h_B,
    float* h_C,
    const GemmConfig& cfg,
    bool* out_used_gpu = nullptr
);

// Kernel 2: Warp-primitive row-wise max & sum softmax normalization (__shfl_xor_sync)
double launch_warp_rowmax_sum_softmax(
    const float* h_input,
    float* h_output,
    int rows,
    int cols,
    int iters,
    bool* out_used_gpu = nullptr
);

// Kernel 3: Tile-wise online softmax causal scaled dot-product attention
double launch_causal_attention_tiled(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_Out,
    const AttentionConfig& cfg,
    bool* out_used_gpu = nullptr
);

// Calculate occupancy & shared-memory footprint for a given kernel configuration
OccupancyEstimate estimate_gemm_occupancy();
OccupancyEstimate estimate_attention_occupancy(int head_dim);

// High-level benchmark & parity runners (implemented in host_runner.cpp)
BenchmarkResult run_gemm_benchmark(const GemmConfig& cfg);
BenchmarkResult run_attention_benchmark(const AttentionConfig& cfg);
BenchmarkResult run_warp_softmax_benchmark(int rows, int cols, int iters);

} // namespace attn_forge
