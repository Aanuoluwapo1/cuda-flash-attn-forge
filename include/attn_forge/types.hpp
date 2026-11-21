#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace attn_forge {

// Tile dimensions for GEMM and Flash-style tiled causal attention
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;
constexpr int WARP_SIZE = 32;
constexpr int MAX_HEAD_DIM = 128;

// Non-owning 2D/3D row-major tensor view
struct TensorView2D {
    float* data = nullptr;
    int rows = 0;
    int cols = 0;
    int ld = 0; // Leading dimension (stride between rows)

    TensorView2D() = default;
    TensorView2D(float* ptr, int r, int c, int leading_dim = 0)
        : data(ptr), rows(r), cols(c), ld(leading_dim > 0 ? leading_dim : c) {}

    inline float& at(int r, int c) { return data[static_cast<std::size_t>(r) * ld + c]; }
    inline const float& at(int r, int c) const { return data[static_cast<std::size_t>(r) * ld + c]; }
};

struct GemmConfig {
    int m = 256;
    int n = 256;
    int k = 256;
    float alpha = 1.0f;
    float beta = 0.0f;
    int iters = 10;
};

struct AttentionConfig {
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 128;
    int head_dim = 64;
    bool causal = true;
    int iters = 10;
};

struct ParityMetrics {
    double max_abs_err = 0.0;
    double mean_abs_err = 0.0;
    double cosine_similarity = 1.0;
    bool passed = false;
};

struct OccupancyEstimate {
    int threads_per_block = 0;
    int warps_per_block = 0;
    std::size_t shared_mem_bytes = 0;
    int active_blocks_per_sm = 0;
    double theoretical_occupancy_pct = 0.0;
};

struct BenchmarkResult {
    std::string kernel_name;
    std::string backend_name;      // e.g. "CUDA GPU (Live)" vs "CPU Host Fallback (No CUDA Device)"
    bool is_live_gpu = false;
    std::string shape_desc;
    int iterations = 1;
    double avg_latency_ms = 0.0;
    double achieved_gflops = 0.0;
    double effective_bandwidth_gbps = 0.0;
    double total_flops = 0.0;
    double total_bytes = 0.0;
    double arithmetic_intensity = 0.0; // FLOPs / Byte
    OccupancyEstimate occupancy;
    ParityMetrics parity;
};

} // namespace attn_forge
