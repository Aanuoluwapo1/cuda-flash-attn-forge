#if defined(ATTN_FORGE_CLANG_PTX_ONLY)
#include "attn_forge/cuda_clang_device.cuh"
namespace attn_forge {
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;
constexpr int WARP_SIZE = 32;
constexpr int MAX_HEAD_DIM = 128;
} // namespace attn_forge
#else
#include "attn_forge/kernels.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#endif

namespace attn_forge {

extern "C" {

// -----------------------------------------------------------------------------
// 1. gemm_tiled_kernel
// Shared-memory tiled matrix multiplication (TILE_M x TILE_N x TILE_K)
// Uses +1 column padding on shared memory tiles to eliminate 32-way bank conflicts
// during transpose / column-major accesses.
// -----------------------------------------------------------------------------
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_M + ty;
    const int col = blockIdx.x * TILE_N + tx;

    float acc = 0.0f;
    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE_K + tx;
        const int b_row = t * TILE_K + ty;

        As[ty][tx] = (row < M && a_col < K)
                         ? A[static_cast<unsigned long long>(row) * K + a_col]
                         : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N)
                         ? B[static_cast<unsigned long long>(b_row) * N + col]
                         : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k_idx = 0; k_idx < TILE_K; ++k_idx) {
            acc += As[ty][k_idx] * Bs[k_idx][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        const unsigned long long idx = static_cast<unsigned long long>(row) * N + col;
        const float prev = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * acc + beta * prev;
    }
}

// -----------------------------------------------------------------------------
// 2. warp_rowmax_sum_kernel
// Warp-level shuffle reductions (__shfl_xor_sync) for numerically stable
// row-wise softmax normalization. One warp (32 threads) processes each row.
// -----------------------------------------------------------------------------
__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, offset));
    }
    return val;
}

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__global__ void warp_rowmax_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = input + static_cast<unsigned long long>(warp_id) * cols;
    float* row_out = output + static_cast<unsigned long long>(warp_id) * cols;

    float local_max = -INFINITY;
    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        local_max = fmaxf(local_max, row_in[c]);
    }
    const float row_max = warp_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        local_sum += __expf(row_in[c] - row_max);
    }
    const float row_sum = warp_reduce_sum(local_sum);
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    for (int c = lane_id; c < cols; c += WARP_SIZE) {
        row_out[c] = (row_sum > 0.0f) ? (__expf(row_in[c] - row_max) * inv_sum) : 0.0f;
    }
}

// -----------------------------------------------------------------------------
// 3. causal_attention_tiled_kernel
// Tile-wise online softmax / block-wise attention scoring with causal masking
// (j <= i) to avoid materializing an N x N attention matrix in global memory.
// Shared memory stores K_tile and V_tile with +1 bank-conflict padding.
// -----------------------------------------------------------------------------
__global__ void causal_attention_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int seq_len,
    int head_dim,
    float scale,
    bool causal
) {
    // Shared memory for K and V tiles: [TILE_N][MAX_HEAD_DIM + 1]
    __shared__ float s_K[TILE_N][MAX_HEAD_DIM + 1];
    __shared__ float s_V[TILE_N][MAX_HEAD_DIM + 1];

    const int bh = blockIdx.y; // Batch * Head index
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    const unsigned long long head_offset = static_cast<unsigned long long>(bh) * seq_len * head_dim;
    const float* q_bh = Q + head_offset;
    const float* k_bh = K + head_offset;
    const float* v_bh = V + head_offset;
    float* o_bh = Out + head_offset;

    float q_reg[MAX_HEAD_DIM];
    float o_acc[MAX_HEAD_DIM];

    if (q_idx < seq_len) {
        for (int d = 0; d < head_dim; ++d) {
            q_reg[d] = q_bh[q_idx * head_dim + d];
            o_acc[d] = 0.0f;
        }
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const int num_kv_tiles = (seq_len + TILE_N - 1) / TILE_N;

    for (int tile_kv = 0; tile_kv < num_kv_tiles; ++tile_kv) {
        const int kv_start = tile_kv * TILE_N;

        // Cooperative load of K and V tile into padded shared memory
        if (tid < TILE_N) {
            const int kv_row = kv_start + tid;
            for (int d = 0; d < head_dim; ++d) {
                if (kv_row < seq_len) {
                    s_K[tid][d] = k_bh[kv_row * head_dim + d];
                    s_V[tid][d] = v_bh[kv_row * head_dim + d];
                } else {
                    s_K[tid][d] = 0.0f;
                    s_V[tid][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (q_idx < seq_len && (!causal || kv_start <= q_idx)) {
            const int kv_count = min(TILE_N, seq_len - kv_start);
            float tile_max = -INFINITY;
            float s_tile[TILE_N];

            for (int t = 0; t < kv_count; ++t) {
                const int j = kv_start + t;
                if (causal && j > q_idx) {
                    s_tile[t] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_reg[d] * s_K[t][d];
                    }
                    s_tile[t] = dot * scale;
                    tile_max = fmaxf(tile_max, s_tile[t]);
                }
            }

            const float m_new = fmaxf(m_i, tile_max);
            const float alpha_prev = (m_i == -INFINITY) ? 0.0f : __expf(m_i - m_new);

            for (int d = 0; d < head_dim; ++d) {
                o_acc[d] *= alpha_prev;
            }
            l_i *= alpha_prev;

            for (int t = 0; t < kv_count; ++t) {
                const int j = kv_start + t;
                if (causal && j > q_idx) continue;
                const float p_ij = __expf(s_tile[t] - m_new);
                l_i += p_ij;
                for (int d = 0; d < head_dim; ++d) {
                    o_acc[d] += p_ij * s_V[t][d];
                }
            }

            m_i = m_new;
        }
        __syncthreads();
    }

    if (q_idx < seq_len) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            o_bh[q_idx * head_dim + d] = o_acc[d] * inv_l;
        }
    }
}

} // extern "C"

#if !defined(ATTN_FORGE_CLANG_PTX_ONLY)

bool query_cuda_device_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear sticky driver/runtime error state on GPU-less hosts
        return false;
    }
    return (count > 0);
}

std::string query_cuda_device_name() {
    if (!query_cuda_device_available()) {
        return "CPU Host Fallback (No CUDA Device Detected at Runtime)";
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    char buf[256];
    std::snprintf(
        buf, sizeof(buf), "CUDA GPU (Live: %s, SM %d.%d, %d SMs)", prop.name, prop.major,
        prop.minor, prop.multiProcessorCount);
    return std::string(buf);
}

double launch_gemm_tiled_cuda(
    const float* h_A,
    const float* h_B,
    float* h_C,
    const GemmConfig& cfg
) {
    const std::size_t bytes_A = static_cast<std::size_t>(cfg.m) * cfg.k * sizeof(float);
    const std::size_t bytes_B = static_cast<std::size_t>(cfg.k) * cfg.n * sizeof(float);
    const std::size_t bytes_C = static_cast<std::size_t>(cfg.m) * cfg.n * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    dim3 block(TILE_N, TILE_M);
    dim3 grid((cfg.n + TILE_N - 1) / TILE_N, (cfg.m + TILE_M - 1) / TILE_M);

    // Warmup
    gemm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, cfg.m, cfg.n, cfg.k, cfg.alpha, cfg.beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (cfg.beta != 0.0f) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = (cfg.iters > 0) ? cfg.iters : 1;
    double elapsed_ms = 0.0;
    if (cfg.beta == 0.0f) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            gemm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, cfg.m, cfg.n, cfg.k, cfg.alpha, cfg.beta);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        elapsed_ms = static_cast<double>(total_ms) / iters;
    } else {
        for (int i = 0; i < iters; ++i) {
            CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(start));
            gemm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, cfg.m, cfg.n, cfg.k, cfg.alpha, cfg.beta);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float step_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&step_ms, start, stop));
            elapsed_ms += static_cast<double>(step_ms);
        }
        elapsed_ms /= iters;
    }

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return elapsed_ms;
}

double launch_warp_rowmax_sum_cuda(
    const float* h_input,
    float* h_output,
    int rows,
    int cols,
    int iters
) {
    const std::size_t bytes = static_cast<std::size_t>(rows) * cols * sizeof(float);
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_input, bytes, cudaMemcpyHostToDevice));

    constexpr int WARPS_PER_BLOCK = 4;
    const int threads = WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    warp_rowmax_sum_kernel<<<blocks, threads>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int run_iters = (iters > 0) ? iters : 1;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < run_iters; ++i) {
        warp_rowmax_sum_kernel<<<blocks, threads>>>(d_in, d_out, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_output, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return static_cast<double>(total_ms) / run_iters;
}

double launch_causal_attention_tiled_cuda(
    const float* h_Q,
    const float* h_K,
    const float* h_V,
    float* h_Out,
    const AttentionConfig& cfg
) {
    if (cfg.head_dim > MAX_HEAD_DIM) {
        throw std::invalid_argument("head_dim exceeds MAX_HEAD_DIM (128)");
    }

    const int total_heads = cfg.batch_size * cfg.num_heads;
    const std::size_t count = static_cast<std::size_t>(total_heads) * cfg.seq_len * cfg.head_dim;
    const std::size_t bytes = count * sizeof(float);

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_Out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_Out, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_M);
    dim3 grid((cfg.seq_len + TILE_M - 1) / TILE_M, total_heads);
    const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

    causal_attention_tiled_kernel<<<grid, block>>>(
        d_Q, d_K, d_V, d_Out, cfg.seq_len, cfg.head_dim, scale, cfg.causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = (cfg.iters > 0) ? cfg.iters : 1;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        causal_attention_tiled_kernel<<<grid, block>>>(
            d_Q, d_K, d_V, d_Out, cfg.seq_len, cfg.head_dim, scale, cfg.causal);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_Out, d_Out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_Out));

    return static_cast<double>(total_ms) / iters;
}

#endif // !defined(ATTN_FORGE_CLANG_PTX_ONLY)

} // namespace attn_forge
