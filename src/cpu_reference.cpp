#include "attn_forge/cpu_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace attn_forge {

void cpu_gemm_reference(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double acc = 0.0;
            for (int p = 0; p < K; ++p) {
                acc += static_cast<double>(A[static_cast<std::size_t>(row) * K + p]) *
                       static_cast<double>(B[static_cast<std::size_t>(p) * N + col]);
            }
            const std::size_t idx = static_cast<std::size_t>(row) * N + col;
            double prev = (beta != 0.0f) ? static_cast<double>(C[idx]) : 0.0;
            C[idx] = static_cast<float>(static_cast<double>(alpha) * acc + static_cast<double>(beta) * prev);
        }
    }
}

void cpu_gemm_tiled_emulation(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    // Emulates the exact shared-memory tiled loop structure with +1 padded stride:
    // float As[TILE_M][TILE_K + 1], Bs[TILE_K][TILE_N + 1]
    const int num_tiles_m = (M + TILE_M - 1) / TILE_M;
    const int num_tiles_n = (N + TILE_N - 1) / TILE_N;
    const int num_tiles_k = (K + TILE_K - 1) / TILE_K;

    float As[TILE_M][TILE_K + 1];
    float Bs[TILE_K][TILE_N + 1];

    for (int bm = 0; bm < num_tiles_m; ++bm) {
        for (int bn = 0; bn < num_tiles_n; ++bn) {
            float acc[TILE_M][TILE_N] = {{0.0f}};

            for (int bk = 0; bk < num_tiles_k; ++bk) {
            // Cooperative load into padded shared memory tile
            for (int ty = 0; ty < TILE_M; ++ty) {
                int row = bm * TILE_M + ty;
                for (int tx = 0; tx < TILE_K; ++tx) {
                    int a_col = bk * TILE_K + tx;
                    As[ty][tx] = (row < M && a_col < K)
                                     ? A[static_cast<std::size_t>(row) * K + a_col]
                                     : 0.0f;
                }
            }
            for (int ty = 0; ty < TILE_K; ++ty) {
                int b_row = bk * TILE_K + ty;
                for (int tx = 0; tx < TILE_N; ++tx) {
                    int col = bn * TILE_N + tx;
                    Bs[ty][tx] = (b_row < K && col < N)
                                     ? B[static_cast<std::size_t>(b_row) * N + col]
                                     : 0.0f;
                }
            }

                // Tile multiply-accumulate
                for (int ty = 0; ty < TILE_M; ++ty) {
                    for (int tx = 0; tx < TILE_N; ++tx) {
                        for (int k_idx = 0; k_idx < TILE_K; ++k_idx) {
                            acc[ty][tx] += As[ty][k_idx] * Bs[k_idx][tx];
                        }
                    }
                }
            }

            // Write back tile to global C
            for (int ty = 0; ty < TILE_M; ++ty) {
                int row = bm * TILE_M + ty;
                if (row >= M) continue;
                for (int tx = 0; tx < TILE_N; ++tx) {
                    int col = bn * TILE_N + tx;
                    if (col >= N) continue;
                    std::size_t idx = static_cast<std::size_t>(row) * N + col;
                    float prev = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * acc[ty][tx] + beta * prev;
                }
            }
        }
    }
}

void cpu_row_softmax_reference(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    for (int r = 0; r < rows; ++r) {
        const float* row_in = input + static_cast<std::size_t>(r) * cols;
        float* row_out = output + static_cast<std::size_t>(r) * cols;

        double max_val = -std::numeric_limits<double>::infinity();
        for (int c = 0; c < cols; ++c) {
            max_val = std::max(max_val, static_cast<double>(row_in[c]));
        }

        double sum_exp = 0.0;
        if (std::isfinite(max_val)) {
            for (int c = 0; c < cols; ++c) {
                sum_exp += std::exp(static_cast<double>(row_in[c]) - max_val);
            }
        }

        const double inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : 0.0;
        for (int c = 0; c < cols; ++c) {
            row_out[c] = (sum_exp > 0.0)
                             ? static_cast<float>(std::exp(static_cast<double>(row_in[c]) - max_val) * inv_sum)
                             : 0.0f;
        }
    }
}

void cpu_causal_attention_reference(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool causal
) {
    const int total_heads = batch_size * num_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const std::size_t head_stride = static_cast<std::size_t>(seq_len) * head_dim;

    for (int bh = 0; bh < total_heads; ++bh) {
        const float* q_bh = Q + bh * head_stride;
        const float* k_bh = K + bh * head_stride;
        const float* v_bh = V + bh * head_stride;
        float* o_bh = Out + bh * head_stride;

        for (int i = 0; i < seq_len; ++i) {
            const int max_j = causal ? (i + 1) : seq_len;
            std::vector<double> scores(max_j, 0.0);
            double row_max = -std::numeric_limits<double>::infinity();

            for (int j = 0; j < max_j; ++j) {
                double dot = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(q_bh[i * head_dim + d]) *
                           static_cast<double>(k_bh[j * head_dim + d]);
                }
                scores[j] = dot * scale;
                row_max = std::max(row_max, scores[j]);
            }

            double row_sum = 0.0;
            for (int j = 0; j < max_j; ++j) {
                scores[j] = std::exp(scores[j] - row_max);
                row_sum += scores[j];
            }

            const double inv_sum = 1.0 / row_sum;
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int j = 0; j < max_j; ++j) {
                    acc += (scores[j] * inv_sum) * static_cast<double>(v_bh[j * head_dim + d]);
                }
                o_bh[i * head_dim + d] = static_cast<float>(acc);
            }
        }
    }
}

void cpu_causal_attention_tiled_online(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool causal
) {
    // Tile-wise online softmax recurrence (FlashAttention algorithm)
    // Avoids materializing the full N x N attention score matrix.
    const int total_heads = batch_size * num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::size_t head_stride = static_cast<std::size_t>(seq_len) * head_dim;
    const int num_kv_tiles = (seq_len + TILE_N - 1) / TILE_N;

    for (int bh = 0; bh < total_heads; ++bh) {
        const float* q_bh = Q + bh * head_stride;
        const float* k_bh = K + bh * head_stride;
        const float* v_bh = V + bh * head_stride;
        float* o_bh = Out + bh * head_stride;

        for (int i = 0; i < seq_len; ++i) {
            float m_i = -std::numeric_limits<float>::infinity();
            float l_i = 0.0f;
            std::vector<float> acc_o(head_dim, 0.0f);

            for (int tile_kv = 0; tile_kv < num_kv_tiles; ++tile_kv) {
                const int kv_start = tile_kv * TILE_N;
                if (causal && kv_start > i) {
                    break; // Entire KV tile is strictly in the future (j > i)
                }
                const int kv_end = std::min(seq_len, kv_start + TILE_N);

                // Compute tile scores S_ij and tile max m_ij
                float tile_max = -std::numeric_limits<float>::infinity();
                float s_tile[TILE_N];
                for (int t = 0; t < (kv_end - kv_start); ++t) {
                    const int j = kv_start + t;
                    if (causal && j > i) {
                        s_tile[t] = -std::numeric_limits<float>::infinity();
                    } else {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d) {
                            dot += q_bh[i * head_dim + d] * k_bh[j * head_dim + d];
                        }
                        s_tile[t] = dot * scale;
                        tile_max = std::max(tile_max, s_tile[t]);
                    }
                }

                // Online softmax update: new running maximum m_new = max(m_i, tile_max)
                const float m_new = std::max(m_i, tile_max);
                const float alpha_prev = (m_i == -std::numeric_limits<float>::infinity())
                                             ? 0.0f
                                             : std::exp(m_i - m_new);

                // Rescale previous accumulator and running denominator
                for (int d = 0; d < head_dim; ++d) {
                    acc_o[d] *= alpha_prev;
                }
                l_i *= alpha_prev;

                // Accumulate current tile contributions
                for (int t = 0; t < (kv_end - kv_start); ++t) {
                    const int j = kv_start + t;
                    if (causal && j > i) continue;
                    const float p_ij = std::exp(s_tile[t] - m_new);
                    l_i += p_ij;
                    for (int d = 0; d < head_dim; ++d) {
                        acc_o[d] += p_ij * v_bh[j * head_dim + d];
                    }
                }
                m_i = m_new;
            }

            // Final normalization by running sum l_i
            const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                o_bh[i * head_dim + d] = acc_o[d] * inv_l;
            }
        }
    }
}

ParityMetrics compute_parity_metrics(
    const float* ref,
    const float* actual,
    std::size_t count,
    double abs_tol,
    double min_cosine
) {
    ParityMetrics m;
    if (count == 0) {
        m.passed = true;
        return m;
    }

    double max_err = 0.0;
    double sum_err = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_act = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        const double r = static_cast<double>(ref[i]);
        const double a = static_cast<double>(actual[i]);
        if (std::isnan(r) || std::isnan(a) || std::isinf(r) || std::isinf(a)) {
            m.max_abs_err = std::numeric_limits<double>::infinity();
            m.cosine_similarity = 0.0;
            m.passed = false;
            return m;
        }
        const double err = std::abs(r - a);
        max_err = std::max(max_err, err);
        sum_err += err;
        dot += r * a;
        norm_ref += r * r;
        norm_act += a * a;
    }

    m.max_abs_err = max_err;
    m.mean_abs_err = sum_err / static_cast<double>(count);

    const double denom = std::sqrt(norm_ref) * std::sqrt(norm_act);
    if (denom > 1e-15) {
        m.cosine_similarity = dot / denom;
    } else if (max_err <= abs_tol) {
        m.cosine_similarity = 1.0;
    } else {
        m.cosine_similarity = 0.0;
    }

    m.passed = (m.max_abs_err <= abs_tol) && (m.cosine_similarity >= min_cosine);
    return m;
}

} // namespace attn_forge
