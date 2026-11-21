# cuda-flash-attn-forge

**Tiled GEMM, Warp-Primitive & Causal Attention CUDA Kernel Engine** with a C++17 host harness, exact CPU reference parity verifier, and theoretical/live occupancy and roofline profiler.

---

## Architecture & Kernel Design

### 1. `gemm_tiled_kernel` — Shared-Memory Tiled GEMM with Bank-Conflict Padding
Computes $C = \alpha (A \times B) + \beta C$ for row-major matrices $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and $C \in \mathbb{R}^{M \times N}$.
- **Tiling Strategy:** Thread blocks of size `TILE_N x TILE_M` (`16 x 16 = 256` threads, 8 warps) cooperatively load $16 \times 16$ tiles of $A$ and $B$ from global memory into fast on-chip shared memory (`As` and `Bs`).
- **Bank-Conflict Mitigation (`+1` Stride Padding):** NVIDIA GPUs partition shared memory into 32 banks (4-byte width per bank). With a power-of-two tile width (`TILE_K = 16` or `32`), column-wise or strided accesses across rows map multiple threads in a warp to the same bank. Declaring `__shared__ float As[TILE_M][TILE_K + 1]` and `__shared__ float Bs[TILE_K][TILE_N + 1]` shifts each successive row by one 4-byte word, skewing column indices across distinct banks and eliminating shared-memory bank conflicts.

### 2. `warp_rowmax_sum_kernel` — Warp-Level Butterfly Reductions (`__shfl_xor_sync`)
Computes numerically stable row-wise softmax:
$$\text{softmax}(x)_i = \frac{\exp(x_i - \max_k x_k)}{\sum_j \exp(x_j - \max_k x_k)}$$
- Uses register-to-register butterfly warp reductions via `__shfl_xor_sync(0xffffffffu, val, offset)` with offsets `16, 8, 4, 2, 1` to compute both the row maximum and the exponentiated partition sum without shared memory synchronization barriers.

### 3. `causal_attention_tiled_kernel` — Flash-Style Online Softmax Causal Attention
Computes batched multi-head causal scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M_{\text{causal}}\right) V$$
where $M_{\text{causal}}(i, j) = 0$ for $j \le i$ and $-\infty$ for $j > i$.
- **Zero $N \times N$ Intermediate Materialization:** Instead of allocating an $O(N^2)$ attention matrix in global memory, each query row $i$ maintains running statistics $(m_i, \ell_i, \tilde{O}_i)$ in registers while streaming `(K, V)` tiles of width `TILE_N = 16` through padded shared memory (`s_K[TILE_N][MAX_HEAD_DIM + 1]` and `s_V[TILE_N][MAX_HEAD_DIM + 1]`):
  $$m_i^{\text{new}} = \max(m_i, \max_{j \in \text{tile}} S_{ij}), \qquad \alpha = \exp(m_i - m_i^{\text{new}})$$
  $$\ell_i \leftarrow \alpha \ell_i + \sum_{j \in \text{tile}} \exp(S_{ij} - m_i^{\text{new}}), \qquad \tilde{O}_i \leftarrow \alpha \tilde{O}_i + \sum_{j \in \text{tile}} \exp(S_{ij} - m_i^{\text{new}}) V_j$$
- **Causal Early-Exit:** Tiles where `kv_start > q_idx` lie strictly in the future ($j > i$) and are skipped automatically.

---

## Repository Layout

```text
cuda-flash-attn-forge/
├── CMakeLists.txt
├── README.md
├── .github/workflows/ci.yml
├── include/attn_forge/
│   ├── types.hpp          # Tensor views, kernel launch configs, occupancy & benchmark structs
│   ├── cpu_reference.hpp  # FP64-accumulated CPU reference & online-softmax tiled emulators
│   └── kernels.cuh        # Unified host/CUDA kernel dispatch declarations
├── src/
│   ├── cpu_reference.cpp  # Exact CPU reference implementations & parity verifiers
│   ├── kernels.cu         # CUDA kernels (gemm_tiled, warp_rowmax_sum, causal_attention_tiled)
│   ├── host_runner.cpp    # Host orchestration, cudaEvent_t / steady_clock timing, roofline metrics
│   └── main.cpp           # CLI runner (--mode gemm|attention|all) with Markdown & JSON output
└── tests/
    └── test_parity.cpp    # Edge-shape parity & causal mask invariance test suite
```

---

## Build & Verification Instructions

### 1. Build with CUDA (`nvcc` Available)
When `nvcc` is present on `PATH`, CMake automatically compiles `src/kernels.cu` and enables live GPU timing via `cudaEvent_t`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_KERNELS=ON
cmake --build build --config Release
```

### 2. Build in CPU/Host Fallback Mode (No NVIDIA GPU / `nvcc` Unavailable)
If `nvcc` is not installed or `-DBUILD_CUDA_KERNELS=OFF` is passed, the project compiles cleanly with any C++17 compiler (`g++`, `clang++`, `MSVC`) and runs the exact tiled C++ fallback against the FP64 CPU reference while clearly labeling output as CPU fallback mode:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_KERNELS=OFF
cmake --build build --config Release
```

### 3. Run Parity Test Suite & CLI Profiler

```bash
# Execute unit & edge-shape parity tests
./build/test_parity

# Run full GEMM + Warp Softmax + Causal Attention benchmark & parity report
./build/attn_forge --mode all --m 256 --n 256 --k 256 --seq-len 128 --head-dim 64 --iters 10
```
