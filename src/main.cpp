#include "attn_forge/cpu_reference.hpp"
#include "attn_forge/kernels.cuh"
#include "attn_forge/types.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <gemm|attention|all>   Kernel suite to run (default: all)\n"
              << "  --m <int>                     GEMM M dimension (default: 256)\n"
              << "  --n <int>                     GEMM N dimension (default: 256)\n"
              << "  --k <int>                     GEMM K dimension (default: 256)\n"
              << "  --batch <int>                 Attention batch size (default: 2)\n"
              << "  --heads <int>                 Attention num heads (default: 4)\n"
              << "  --seq-len <int>               Attention sequence length (default: 128)\n"
              << "  --head-dim <int>              Attention head dimension (default: 64)\n"
              << "  --iters <int>                 Benchmark timing iterations (default: 10)\n"
              << "  --json                        Emit JSON summary block\n"
              << "  --help                        Show this help message\n";
}

void print_markdown_report(const std::vector<attn_forge::BenchmarkResult>& results) {
    if (results.empty()) return;

    const bool is_gpu = results.front().is_live_gpu;
    std::cout << "================================================================================\n";
    std::cout << "  cuda-flash-attn-forge | Tiled GEMM, Warp-Primitive & Causal Attention Engine\n";
    std::cout << "================================================================================\n";
    std::cout << "Execution Backend : " << results.front().backend_name << "\n";
    std::cout << "Measurement Mode  : "
              << (is_gpu ? "LIVE GPU HARDWARE TIMING (cudaEvent_t)"
                         : "CPU HOST FALLBACK TIMING + THEORETICAL CUDA SM OCCUPANCY MODEL")
              << "\n\n";

    std::cout << "### 1. Throughput & Numerical Parity Summary\n\n";
    std::cout << "| Kernel | Shape | Avg Latency (ms) | Throughput (GFLOP/s) | Eff. BW (GB/s) | Max Abs Err | Cosine Sim | Parity |\n";
    std::cout << "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |\n";

    for (const auto& r : results) {
        std::cout << "| `" << r.kernel_name << "` "
                  << "| " << r.shape_desc << " "
                  << "| " << std::fixed << std::setprecision(4) << r.avg_latency_ms << " "
                  << "| " << std::fixed << std::setprecision(2) << r.achieved_gflops << " "
                  << "| " << std::fixed << std::setprecision(2) << r.effective_bandwidth_gbps << " "
                  << "| " << std::scientific << std::setprecision(2) << r.parity.max_abs_err << " "
                  << "| " << std::fixed << std::setprecision(6) << r.parity.cosine_similarity << " "
                  << "| " << (r.parity.passed ? "PASS" : "FAIL") << " |\n";
    }

    std::cout << "\n### 2. Theoretical CUDA Occupancy & Roofline Profile\n\n";
    std::cout << "| Kernel | Threads/Block | Warps/Block | Shared Mem (B) | Active Blocks/SM | Est. Occupancy (%) | Arith. Intensity (FLOP/B) |\n";
    std::cout << "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& r : results) {
        std::cout << "| `" << r.kernel_name << "` "
                  << "| " << r.occupancy.threads_per_block << " "
                  << "| " << r.occupancy.warps_per_block << " "
                  << "| " << r.occupancy.shared_mem_bytes << " "
                  << "| " << r.occupancy.active_blocks_per_sm << " "
                  << "| " << std::fixed << std::setprecision(1) << r.occupancy.theoretical_occupancy_pct << "% "
                  << "| " << std::fixed << std::setprecision(2) << r.arithmetic_intensity << " |\n";
    }
    std::cout << "\n";
}

void print_json_summary(const std::vector<attn_forge::BenchmarkResult>& results) {
    std::cout << "{\n";
    std::cout << "  \"engine\": \"cuda-flash-attn-forge\",\n";
    std::cout << "  \"live_gpu_execution\": " << (!results.empty() && results.front().is_live_gpu ? "true" : "false") << ",\n";
    std::cout << "  \"backend\": \"" << (!results.empty() ? results.front().backend_name : "none") << "\",\n";
    std::cout << "  \"benchmarks\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "    {\n"
                  << "      \"kernel\": \"" << r.kernel_name << "\",\n"
                  << "      \"shape\": \"" << r.shape_desc << "\",\n"
                  << "      \"iterations\": " << r.iterations << ",\n"
                  << "      \"avg_latency_ms\": " << std::fixed << std::setprecision(6) << r.avg_latency_ms << ",\n"
                  << "      \"achieved_gflops\": " << std::fixed << std::setprecision(3) << r.achieved_gflops << ",\n"
                  << "      \"effective_bandwidth_gbps\": " << std::fixed << std::setprecision(3) << r.effective_bandwidth_gbps << ",\n"
                  << "      \"arithmetic_intensity_flop_per_byte\": " << std::fixed << std::setprecision(3) << r.arithmetic_intensity << ",\n"
                  << "      \"shared_mem_bytes\": " << r.occupancy.shared_mem_bytes << ",\n"
                  << "      \"theoretical_occupancy_pct\": " << std::fixed << std::setprecision(1) << r.occupancy.theoretical_occupancy_pct << ",\n"
                  << "      \"max_abs_err\": " << std::scientific << std::setprecision(4) << r.parity.max_abs_err << ",\n"
                  << "      \"cosine_similarity\": " << std::fixed << std::setprecision(7) << r.parity.cosine_similarity << ",\n"
                  << "      \"parity_passed\": " << (r.parity.passed ? "true" : "false") << "\n"
                  << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = "all";
    attn_forge::GemmConfig gemm_cfg;
    attn_forge::AttentionConfig attn_cfg;
    int iters = 10;
    bool emit_json = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_int = [&](int& target, const char* flag_name) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing integer value after " << flag_name << "\n";
                return false;
            }
            const char* val_str = argv[++i];
            char* endptr = nullptr;
            long parsed = std::strtol(val_str, &endptr, 10);
            if (endptr == val_str || *endptr != '\0') {
                std::cerr << "Invalid integer value '" << val_str << "' for " << flag_name << "\n";
                return false;
            }
            target = static_cast<int>(parsed);
            return true;
        };

        if (arg == "--mode") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --mode\n";
                return 1;
            }
            mode = argv[++i];
        } else if (arg == "--m") {
            if (!next_int(gemm_cfg.m, "--m")) return 1;
        } else if (arg == "--n") {
            if (!next_int(gemm_cfg.n, "--n")) return 1;
        } else if (arg == "--k") {
            if (!next_int(gemm_cfg.k, "--k")) return 1;
        } else if (arg == "--batch") {
            if (!next_int(attn_cfg.batch_size, "--batch")) return 1;
        } else if (arg == "--heads") {
            if (!next_int(attn_cfg.num_heads, "--heads")) return 1;
        } else if (arg == "--seq-len") {
            if (!next_int(attn_cfg.seq_len, "--seq-len")) return 1;
        } else if (arg == "--head-dim") {
            if (!next_int(attn_cfg.head_dim, "--head-dim")) return 1;
        } else if (arg == "--iters") {
            if (!next_int(iters, "--iters")) return 1;
        } else if (arg == "--json") {
            emit_json = true;
        } else if (arg == "--no-json") {
            emit_json = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (gemm_cfg.m <= 0 || gemm_cfg.n <= 0 || gemm_cfg.k <= 0) {
        std::cerr << "Invalid GEMM dimensions: M, N, K must be positive integers.\n";
        return 1;
    }
    if (attn_cfg.batch_size <= 0 || attn_cfg.num_heads <= 0 || attn_cfg.seq_len <= 0 ||
        attn_cfg.head_dim <= 0 || attn_cfg.head_dim > attn_forge::MAX_HEAD_DIM) {
        std::cerr << "Invalid Attention dimensions: batch, heads, seq-len must be > 0 and head-dim in [1, "
                  << attn_forge::MAX_HEAD_DIM << "].\n";
        return 1;
    }
    if (iters <= 0) {
        std::cerr << "Invalid --iters: must be >= 1.\n";
        return 1;
    }

    gemm_cfg.iters = iters;
    attn_cfg.iters = iters;

    std::vector<attn_forge::BenchmarkResult> results;

    if (mode == "gemm" || mode == "all") {
        results.push_back(attn_forge::run_gemm_benchmark(gemm_cfg));
    }
    if (mode == "attention" || mode == "all") {
        results.push_back(attn_forge::run_warp_softmax_benchmark(attn_cfg.seq_len, attn_cfg.seq_len, iters));
        results.push_back(attn_forge::run_attention_benchmark(attn_cfg));
    }

    if (results.empty()) {
        std::cerr << "Invalid --mode '" << mode << "'. Expected gemm, attention, or all.\n";
        return 1;
    }

    print_markdown_report(results);
    if (emit_json) {
        std::cout << "### 3. Structured JSON Telemetry\n\n```json\n";
        print_json_summary(results);
        std::cout << "```\n";
    }

    bool all_passed = true;
    for (const auto& r : results) {
        if (!r.parity.passed) {
            all_passed = false;
        }
    }
    return all_passed ? 0 : 2;
}
