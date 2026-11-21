#include "attn_forge/cpu_reference.hpp"
#include "attn_forge/kernels.cuh"
#include "attn_forge/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>
#include <vector>

namespace attn_forge {

#if ATTN_FORGE_ENABLE_CUDA
// Forward declarations from src/kernels.cu when compiled with nvcc
bool query_cuda_device_available();
std::string query_cuda_device_name();
double launch_gemm_tiled_cuda(const float* h_A, const float* h_B, float* h_C, const GemmConfig& cfg);
double launch_warp_rowmax_sum_cuda(const float* h_input, float* h_output, int rows, int cols, int iters);
double launch_causal_attention_tiled_cuda(
    const float* h_Q, const float* h_K, const float* h_V, float* h_Out, const AttentionConfig& cfg);
#endif

bool cuda_backend_available() {
#if ATTN_FORGE_ENABLE_CUDA
    return query_cuda_device_available();
#else
    return false;
#endif
}

std::string get_device_description() {
#if ATTN_FORGE_ENABLE_CUDA
    return query_cuda_device_name();
#else
    return "CPU Host Fallback (BUILD_CUDA_KERNELS=OFF / nvcc unavailable)";
#endif
}

double launch_gemm_tiled(
    const float* h_A,
    const float* h_B,
    float* h_C,
    const GemmConfig& cfg,
    bool* out_used_gpu
) {
#if ATTN_FORGE_ENABLE_CUDA
    if (query_cuda_device_available()) {
        if (out_used_gpu) *out_used_gpu = true;
        return launch_gemm_tiled_cuda(h_A, h_B, h_C, cfg);
    }
#endif
    if (out_used_gpu) *out_used_gpu = false;
    const int iters = std::max(1, cfg.iters);
    std::vector<float> c_initial(
        h_C, h_C + static_cast<std::size_t>(cfg.m) * static_cast<std::size_t>(cfg.n));

    // Warmup pass
    cpu_gemm_tiled_emulation(h_A, h_B, h_C, cfg.m, cfg.n, cfg.k, cfg.alpha, cfg.beta);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (cfg.beta != 0.0f) {
            std::copy(c_initial.begin(), c_initial.end(), h_C);
        }
        cpu_gemm_tiled_emulation(h_A, h_B, h_C, cfg.m, cfg.n, cfg.k, cfg.alpha, cfg.beta);
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return elapsed_ms / static_cast<double>(iters);
}

double launch_warp_rowmax_sum_softmax(
    const float* h_input,
    float* h_output,
    int rows,
    int cols,
    int iters,
    bool* out_used_gpu
) {
#if ATTN_FORGE_ENABLE_CUDA
    if (query_cuda_device_available()) {
        if (out_used_gpu) *out_used_gpu = true;
        return launch_warp_rowmax_sum_cuda(h_input, h_output, rows, cols, iters);
    }
#endif
    if (out_used_gpu) *out_used_gpu = false;
    const int run_iters = std::max(1, iters);
    cpu_row_softmax_reference(h_input, h_output, rows, cols);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < run_iters; ++i) {
        cpu_row_softmax_reference(h_input, h_output, rows, cols);
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return elapsed_ms / static_cast<double>(run_iters);
}

double launch_causal_attention_tiled(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_Out,
    const AttentionConfig& cfg,
    bool* out_used_gpu
) {
    if (cfg.head_dim <= 0 || cfg.head_dim > MAX_HEAD_DIM) {
        throw std::invalid_argument("head_dim must be in [1, MAX_HEAD_DIM (128)]");
    }
#if ATTN_FORGE_ENABLE_CUDA
    if (query_cuda_device_available()) {
        if (out_used_gpu) *out_used_gpu = true;
        return launch_causal_attention_tiled_cuda(h_Q, h_K, h_V, h_Out, cfg);
    }
#endif
    if (out_used_gpu) *out_used_gpu = false;
    const int iters = std::max(1, cfg.iters);
    cpu_causal_attention_tiled_online(
        h_Q, h_K, h_V, h_Out, cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.causal);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        cpu_causal_attention_tiled_online(
            h_Q, h_K, h_V, h_Out, cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.causal);
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return elapsed_ms / static_cast<double>(iters);
}

OccupancyEstimate estimate_gemm_occupancy() {
    OccupancyEstimate occ;
    occ.threads_per_block = TILE_M * TILE_N; // 256 threads
    occ.warps_per_block = occ.threads_per_block / WARP_SIZE; // 8 warps
    // Shared memory includes +1 padding per row on As and Bs
    occ.shared_mem_bytes =
        (TILE_M * (TILE_K + 1) + TILE_K * (TILE_N + 1)) * sizeof(float);

    // Reference SM architecture calculation (Ampere/Ada: 48 warps/SM or 64 warps/SM, 100KB+ SMEM)
    constexpr int max_warps_per_sm = 48;
    constexpr int max_blocks_per_sm = 16;
    constexpr std::size_t max_smem_per_sm = 65536;

    int by_warps = max_warps_per_sm / occ.warps_per_block;
    int by_smem = static_cast<int>(max_smem_per_sm / std::max<std::size_t>(1, occ.shared_mem_bytes));
    occ.active_blocks_per_sm = std::min({max_blocks_per_sm, by_warps, by_smem});
    occ.theoretical_occupancy_pct =
        100.0 * static_cast<double>(occ.active_blocks_per_sm * occ.warps_per_block) / max_warps_per_sm;
    return occ;
}

OccupancyEstimate estimate_attention_occupancy(int head_dim) {
    (void)head_dim;
    OccupancyEstimate occ;
    occ.threads_per_block = TILE_M; // blockDim.x = TILE_M (16 threads = 1 allocated hardware warp)
    occ.warps_per_block = (occ.threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
    // Static shared memory allocated by causal_attention_tiled_kernel:
    // s_K[TILE_N][MAX_HEAD_DIM + 1] + s_V[TILE_N][MAX_HEAD_DIM + 1]
    occ.shared_mem_bytes = 2 * TILE_N * (MAX_HEAD_DIM + 1) * sizeof(float);

    constexpr int max_warps_per_sm = 48;
    constexpr int max_blocks_per_sm = 16;
    constexpr std::size_t max_smem_per_sm = 65536;

    int by_warps = max_warps_per_sm / occ.warps_per_block;
    int by_smem = static_cast<int>(max_smem_per_sm / std::max<std::size_t>(1, occ.shared_mem_bytes));
    occ.active_blocks_per_sm = std::min({max_blocks_per_sm, by_warps, by_smem});
    occ.theoretical_occupancy_pct =
        100.0 * static_cast<double>(occ.active_blocks_per_sm * occ.warps_per_block) / max_warps_per_sm;
    return occ;
}

BenchmarkResult run_gemm_benchmark(const GemmConfig& cfg) {
    const std::size_t size_A = static_cast<std::size_t>(cfg.m) * cfg.k;
    const std::size_t size_B = static_cast<std::size_t>(cfg.k) * cfg.n;
    const std::size_t size_C = static_cast<std::size_t>(cfg.m) * cfg.n;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(size_A), B(size_B), C_ref(size_C, 0.0f), C_out(size_C, 0.0f);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    cpu_gemm_reference(A.data(), B.data(), C_ref.data(), cfg.m, cfg.n, cfg.k, cfg.alpha, cfg.beta);

    bool used_gpu = false;
    double avg_ms = launch_gemm_tiled(A.data(), B.data(), C_out.data(), cfg, &used_gpu);

    BenchmarkResult res;
    res.kernel_name = "gemm_tiled_kernel";
    res.backend_name = get_device_description();
    res.is_live_gpu = used_gpu;

    std::ostringstream oss;
    oss << "M=" << cfg.m << ", N=" << cfg.n << ", K=" << cfg.k;
    res.shape_desc = oss.str();
    res.iterations = cfg.iters;
    res.avg_latency_ms = avg_ms;

    // 2 * M * N * K FLOPs for matrix multiplication
    res.total_flops = 2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k;
    res.total_bytes = static_cast<double>(size_A + size_B + size_C) * sizeof(float);
    res.arithmetic_intensity = res.total_flops / std::max(1.0, res.total_bytes);

    const double avg_sec = std::max(1e-9, avg_ms * 1e-3);
    res.achieved_gflops = (res.total_flops * 1e-9) / avg_sec;
    res.effective_bandwidth_gbps = (res.total_bytes * 1e-9) / avg_sec;
    res.occupancy = estimate_gemm_occupancy();
    res.parity = compute_parity_metrics(C_ref.data(), C_out.data(), size_C, 1e-3, 0.99999);

    return res;
}

BenchmarkResult run_warp_softmax_benchmark(int rows, int cols, int iters) {
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<float> input(count), ref_out(count, 0.0f), act_out(count, 0.0f);
    for (auto& v : input) v = dist(rng);

    cpu_row_softmax_reference(input.data(), ref_out.data(), rows, cols);

    bool used_gpu = false;
    double avg_ms = launch_warp_rowmax_sum_softmax(
        input.data(), act_out.data(), rows, cols, iters, &used_gpu);

    BenchmarkResult res;
    res.kernel_name = "warp_rowmax_sum_kernel";
    res.backend_name = get_device_description();
    res.is_live_gpu = used_gpu;

    std::ostringstream oss;
    oss << "Rows=" << rows << ", Cols=" << cols;
    res.shape_desc = oss.str();
    res.iterations = iters;
    res.avg_latency_ms = avg_ms;

    // ~5 FLOPs per element (max, sub, exp, sum, div)
    res.total_flops = 5.0 * static_cast<double>(count);
    res.total_bytes = 2.0 * static_cast<double>(count) * sizeof(float);
    res.arithmetic_intensity = res.total_flops / std::max(1.0, res.total_bytes);

    const double avg_sec = std::max(1e-9, avg_ms * 1e-3);
    res.achieved_gflops = (res.total_flops * 1e-9) / avg_sec;
    res.effective_bandwidth_gbps = (res.total_bytes * 1e-9) / avg_sec;

    res.occupancy.threads_per_block = 4 * WARP_SIZE;
    res.occupancy.warps_per_block = 4;
    res.occupancy.shared_mem_bytes = 0;
    res.occupancy.active_blocks_per_sm = 12;
    res.occupancy.theoretical_occupancy_pct = 100.0;

    res.parity = compute_parity_metrics(ref_out.data(), act_out.data(), count, 1e-5, 0.999999);
    return res;
}

BenchmarkResult run_attention_benchmark(const AttentionConfig& cfg) {
    const int total_heads = cfg.batch_size * cfg.num_heads;
    const std::size_t count = static_cast<std::size_t>(total_heads) * cfg.seq_len * cfg.head_dim;

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> Q(count), K(count), V(count), Out_ref(count, 0.0f), Out_act(count, 0.0f);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    cpu_causal_attention_reference(
        Q.data(), K.data(), V.data(), Out_ref.data(),
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.causal);

    bool used_gpu = false;
    double avg_ms = launch_causal_attention_tiled(
        Q.data(), K.data(), V.data(), Out_act.data(), cfg, &used_gpu);

    BenchmarkResult res;
    res.kernel_name = "causal_attention_tiled_kernel";
    res.backend_name = get_device_description();
    res.is_live_gpu = used_gpu;

    std::ostringstream oss;
    oss << "B=" << cfg.batch_size << ", H=" << cfg.num_heads
        << ", N=" << cfg.seq_len << ", D=" << cfg.head_dim
        << (cfg.causal ? " (Causal)" : " (Full)");
    res.shape_desc = oss.str();
    res.iterations = cfg.iters;
    res.avg_latency_ms = avg_ms;

    // Effective causal attention FLOPs: 4 * B * H * (N * (N + 1) / 2) * D
    const double active_pairs = cfg.causal
        ? (0.5 * static_cast<double>(cfg.seq_len) * (cfg.seq_len + 1))
        : (static_cast<double>(cfg.seq_len) * cfg.seq_len);
    res.total_flops = 4.0 * static_cast<double>(total_heads) * active_pairs * cfg.head_dim;
    // Flash-style HBM traffic: read Q, K, V and write Out (4 * B * H * N * D * sizeof(float))
    res.total_bytes = 4.0 * static_cast<double>(count) * sizeof(float);
    res.arithmetic_intensity = res.total_flops / std::max(1.0, res.total_bytes);

    const double avg_sec = std::max(1e-9, avg_ms * 1e-3);
    res.achieved_gflops = (res.total_flops * 1e-9) / avg_sec;
    res.effective_bandwidth_gbps = (res.total_bytes * 1e-9) / avg_sec;
    res.occupancy = estimate_attention_occupancy(cfg.head_dim);
    res.parity = compute_parity_metrics(Out_ref.data(), Out_act.data(), count, 1e-4, 0.99999);

    return res;
}

} // namespace attn_forge
