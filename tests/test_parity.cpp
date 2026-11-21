#include "attn_forge/cpu_reference.hpp"
#include "attn_forge/kernels.cuh"
#include "attn_forge/types.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_tests_run = 0;
int g_tests_failed = 0;

void report_test(const std::string& name, const attn_forge::ParityMetrics& m) {
    ++g_tests_run;
    std::cout << "[" << (m.passed ? "PASS" : "FAIL") << "] " << std::left << std::setw(48) << name
              << " | max_abs_err=" << std::scientific << std::setprecision(3) << m.max_abs_err
              << " | cosine=" << std::fixed << std::setprecision(7) << m.cosine_similarity << "\n";
    if (!m.passed) {
        ++g_tests_failed;
    }
}

void test_gemm_shape(const std::string& label, int m, int n, int k, float alpha = 1.0f, float beta = 0.0f) {
    attn_forge::GemmConfig cfg;
    cfg.m = m;
    cfg.n = n;
    cfg.k = k;
    cfg.alpha = alpha;
    cfg.beta = beta;
    cfg.iters = 3; // Verify multi-iteration idempotency even when beta != 0.0f

    const std::size_t size_A = static_cast<std::size_t>(m) * k;
    const std::size_t size_B = static_cast<std::size_t>(k) * n;
    const std::size_t size_C = static_cast<std::size_t>(m) * n;

    std::mt19937 rng(static_cast<unsigned int>(m * 31 + n * 17 + k));
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    std::vector<float> A(size_A), B(size_B), C_ref(size_C), C_act(size_C);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (std::size_t i = 0; i < size_C; ++i) {
        float init_c = dist(rng);
        C_ref[i] = init_c;
        C_act[i] = init_c;
    }

    attn_forge::cpu_gemm_reference(A.data(), B.data(), C_ref.data(), m, n, k, alpha, beta);
    attn_forge::launch_gemm_tiled(A.data(), B.data(), C_act.data(), cfg);

    auto metrics = attn_forge::compute_parity_metrics(C_ref.data(), C_act.data(), size_C, 1e-3, 0.99999);
    report_test(label, metrics);
}

void test_warp_softmax_shape(const std::string& label, int rows, int cols, float offset = 0.0f) {
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    std::mt19937 rng(static_cast<unsigned int>(rows * 19 + cols));
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    std::vector<float> input(count), ref_out(count, 0.0f), act_out(count, 0.0f);
    for (auto& v : input) v = dist(rng) + offset;

    attn_forge::cpu_row_softmax_reference(input.data(), ref_out.data(), rows, cols);
    attn_forge::launch_warp_rowmax_sum_softmax(input.data(), act_out.data(), rows, cols, 1);

    auto metrics = attn_forge::compute_parity_metrics(ref_out.data(), act_out.data(), count, 1e-5, 0.999999);

    // Also verify each row sums to 1.0
    for (int r = 0; r < rows; ++r) {
        double row_sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            row_sum += act_out[static_cast<std::size_t>(r) * cols + c];
        }
        if (std::abs(row_sum - 1.0) > 1e-3) {
            metrics.passed = false;
        }
    }
    report_test(label, metrics);
}

void test_causal_attention_shape(
    const std::string& label,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool causal = true
) {
    attn_forge::AttentionConfig cfg;
    cfg.batch_size = batch_size;
    cfg.num_heads = num_heads;
    cfg.seq_len = seq_len;
    cfg.head_dim = head_dim;
    cfg.causal = causal;
    cfg.iters = 3; // Verify multi-iteration idempotency even when beta != 0.0f

    const std::size_t count =
        static_cast<std::size_t>(batch_size) * num_heads * seq_len * head_dim;

    std::mt19937 rng(static_cast<unsigned int>(batch_size * 101 + num_heads * 37 + seq_len * 13 + head_dim));
    std::uniform_real_distribution<float> dist(-0.75f, 0.75f);

    std::vector<float> Q(count), K(count), V(count), Out_ref(count, 0.0f), Out_act(count, 0.0f);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    attn_forge::cpu_causal_attention_reference(
        Q.data(), K.data(), V.data(), Out_ref.data(),
        batch_size, num_heads, seq_len, head_dim, causal);

    attn_forge::launch_causal_attention_tiled(
        Q.data(), K.data(), V.data(), Out_act.data(), cfg);

    auto metrics = attn_forge::compute_parity_metrics(Out_ref.data(), Out_act.data(), count, 1e-4, 0.99999);
    report_test(label, metrics);
}

void test_causal_future_invariance() {
    // Verify causal mask invariance: perturbing future tokens (j > i) must not alter output at position i = 0
    const int B = 1, H = 2, N = 23, D = 32;
    const std::size_t count = static_cast<std::size_t>(B) * H * N * D;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> Q(count), K1(count), V1(count), K2(count), V2(count);
    for (std::size_t i = 0; i < count; ++i) {
        Q[i] = dist(rng);
        K1[i] = dist(rng);
        V1[i] = dist(rng);
        K2[i] = K1[i];
        V2[i] = V1[i];
    }

    // Massively perturb tokens j >= 1 in K2 and V2
    for (int h = 0; h < B * H; ++h) {
        for (int j = 1; j < N; ++j) {
            for (int d = 0; d < D; ++d) {
                std::size_t idx = (static_cast<std::size_t>(h) * N + j) * D + d;
                K2[idx] += 50.0f;
                V2[idx] -= 100.0f;
            }
        }
    }

    attn_forge::AttentionConfig cfg{B, H, N, D, true, 1};
    std::vector<float> Out1(count, 0.0f), Out2(count, 0.0f);
    attn_forge::launch_causal_attention_tiled(Q.data(), K1.data(), V1.data(), Out1.data(), cfg);
    attn_forge::launch_causal_attention_tiled(Q.data(), K2.data(), V2.data(), Out2.data(), cfg);

    // Extract position i = 0 across all heads and verify exact agreement
    std::vector<float> row0_a, row0_b;
    for (int h = 0; h < B * H; ++h) {
        for (int d = 0; d < D; ++d) {
            row0_a.push_back(Out1[static_cast<std::size_t>(h) * N * D + d]);
            row0_b.push_back(Out2[static_cast<std::size_t>(h) * N * D + d]);
        }
    }

    auto metrics = attn_forge::compute_parity_metrics(
        row0_a.data(), row0_b.data(), row0_a.size(), 1e-6, 0.9999999);
    report_test("Causal Mask Future-Token Invariance (j > 0)", metrics);
}

void test_head_dim_bounds_guard() {
    attn_forge::AttentionConfig bad_cfg{1, 1, 16, attn_forge::MAX_HEAD_DIM + 1, true, 1};
    std::vector<float> dummy(16 * (attn_forge::MAX_HEAD_DIM + 1), 0.0f);
    bool caught = false;
    try {
        attn_forge::launch_causal_attention_tiled(
            dummy.data(), dummy.data(), dummy.data(), dummy.data(), bad_cfg);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    attn_forge::ParityMetrics m;
    m.max_abs_err = 0.0;
    m.cosine_similarity = caught ? 1.0 : 0.0;
    m.passed = caught;
    report_test("Head Dimension Guard (head_dim > MAX_HEAD_DIM)", m);
}

} // namespace

int main() {
    std::cout << "================================================================================\n";
    std::cout << "  cuda-flash-attn-forge | Numerical Parity & Edge-Shape Verification Suite\n";
    std::cout << "  Backend: " << attn_forge::get_device_description() << "\n";
    std::cout << "================================================================================\n";

    // 1. Tiled GEMM parity across aligned, unaligned, rectangular, and alpha/beta shapes
    test_gemm_shape("GEMM Aligned Tile Multiple (64x64x64)", 64, 64, 64);
    test_gemm_shape("GEMM Non-Multiple Edge Shape (37x53x29)", 37, 53, 29);
    test_gemm_shape("GEMM Tall & Skinny (129x17x65)", 129, 17, 65);
    test_gemm_shape("GEMM Single Row/Col Boundary (1x19x33)", 1, 19, 33);
    test_gemm_shape("GEMM Alpha/Beta Accumulation (45x45x45)", 45, 45, 45, 0.75f, -0.25f);

    // 2. Warp-level rowmax + sum softmax reductions
    test_warp_softmax_shape("Warp Softmax Exact Warp Size (32x32)", 32, 32);
    test_warp_softmax_shape("Warp Softmax Non-Multiple Cols (19x73)", 19, 73);
    test_warp_softmax_shape("Warp Softmax Large Logit Shift (+500.0)", 16, 128, 500.0f);

    // 3. Tile-wise online softmax causal & full attention
    test_causal_attention_shape("Causal Attention Tile-Aligned (B=2,H=2,N=64,D=32)", 2, 2, 64, 32, true);
    test_causal_attention_shape("Causal Attention Non-Multiple (B=2,H=3,N=37,D=48)", 2, 3, 37, 48, true);
    test_causal_attention_shape("Causal Attention Single-Tile Boundary (B=1,H=4,N=15,D=64)", 1, 4, 15, 64, true);
    test_causal_attention_shape("Causal Attention Cross-Tile Boundary (B=1,H=1,N=17,D=16)", 1, 1, 17, 16, true);
    test_causal_attention_shape("Full Non-Causal Online Attention (B=2,H=2,N=35,D=32)", 2, 2, 35, 32, false);
    test_causal_future_invariance();
    test_head_dim_bounds_guard();

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Summary: " << (g_tests_run - g_tests_failed) << "/" << g_tests_run << " parity checks passed.\n";
    return (g_tests_failed == 0) ? 0 : 1;
}
