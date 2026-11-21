#pragma once

#include "attn_forge/types.hpp"
#include <vector>

namespace attn_forge {

// Exact double-accumulated CPU reference for C = alpha * (A @ B) + beta * C
// A: [M x K], B: [K x N], C: [M x N]
void cpu_gemm_reference(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha = 1.0f,
    float beta = 0.0f
);

// CPU emulation of tiled GEMM with shared-memory padding (+1 stride)
void cpu_gemm_tiled_emulation(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha = 1.0f,
    float beta = 0.0f
);

// Numerically stable row-wise softmax reference (float64 accumulation)
// Input/Output: [rows x cols]
void cpu_row_softmax_reference(
    const float* input,
    float* output,
    int rows,
    int cols
);

// Exact CPU reference for Batched Multi-Head Causal Scaled Dot-Product Attention
// Q, K, V, Out: [batch_size * num_heads, seq_len, head_dim]
void cpu_causal_attention_reference(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool causal = true
);

// CPU emulation of Flash-style Tile-Wise Online Softmax Causal Attention
// Matches the exact block-wise online rescaling recurrences of causal_attention_tiled_kernel
void cpu_causal_attention_tiled_online(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool causal = true
);

// Compute max absolute error, mean absolute error, and cosine similarity
ParityMetrics compute_parity_metrics(
    const float* ref,
    const float* actual,
    std::size_t count,
    double abs_tol = 1e-4,
    double min_cosine = 0.9999
);

} // namespace attn_forge
