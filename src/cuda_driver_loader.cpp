#include "attn_forge/kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace attn_forge {

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUevent = void*;
using CUdeviceptr = unsigned long long;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

struct CudaDriverApi {
    void* lib_handle = nullptr;
    bool initialized = false;
    bool available = false;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction fn_gemm = nullptr;
    CUfunction fn_softmax = nullptr;
    CUfunction fn_attention = nullptr;
    std::string device_desc;

    CUresult (*cuInit)(unsigned int) = nullptr;
    CUresult (*cuDeviceGetCount)(int*) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuCtxSynchronize)() = nullptr;
    CUresult (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr*, std::size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, std::size_t) = nullptr;
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, std::size_t) = nullptr;
    CUresult (*cuLaunchKernel)(
        CUfunction,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, CUstream, void**, void**) = nullptr;
    CUresult (*cuEventCreate)(CUevent*, unsigned int) = nullptr;
    CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*cuEventSynchronize)(CUevent) = nullptr;
    CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent) = nullptr;
    CUresult (*cuEventDestroy)(CUevent) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
};

void* load_symbol(void* handle, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

std::string load_ptx_source() {
#ifdef ATTN_FORGE_PTX_PATH
    std::ifstream ifs(ATTN_FORGE_PTX_PATH, std::ios::binary);
    if (ifs.is_open()) {
        std::ostringstream oss;
        oss << ifs.rdbuf();
        return oss.str();
    }
#endif
    std::ifstream local_ifs("kernels_sm75.ptx", std::ios::binary);
    if (local_ifs.is_open()) {
        std::ostringstream oss;
        oss << local_ifs.rdbuf();
        return oss.str();
    }
    return {};
}

CudaDriverApi& get_driver() {
    static CudaDriverApi api;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    if (api.initialized) {
        return api;
    }
    api.initialized = true;

#ifdef _WIN32
    api.lib_handle = static_cast<void*>(LoadLibraryA("nvcuda.dll"));
#else
    api.lib_handle = dlopen("libcuda.so.1", RTLD_LAZY);
#endif
    if (!api.lib_handle) {
        return api;
    }

#define LOAD_CU_SYM(field, sym_name)                                                   \
    api.field = reinterpret_cast<decltype(api.field)>(load_symbol(api.lib_handle, sym_name)); \
    if (!api.field) return api

    LOAD_CU_SYM(cuInit, "cuInit");
    LOAD_CU_SYM(cuDeviceGetCount, "cuDeviceGetCount");
    LOAD_CU_SYM(cuDeviceGet, "cuDeviceGet");
    LOAD_CU_SYM(cuDeviceGetName, "cuDeviceGetName");
    LOAD_CU_SYM(cuDeviceGetAttribute, "cuDeviceGetAttribute");
    LOAD_CU_SYM(cuCtxCreate, "cuCtxCreate_v2");
    LOAD_CU_SYM(cuCtxSetCurrent, "cuCtxSetCurrent");
    LOAD_CU_SYM(cuCtxSynchronize, "cuCtxSynchronize");
    LOAD_CU_SYM(cuModuleLoadData, "cuModuleLoadData");
    LOAD_CU_SYM(cuModuleGetFunction, "cuModuleGetFunction");
    LOAD_CU_SYM(cuMemAlloc, "cuMemAlloc_v2");
    LOAD_CU_SYM(cuMemFree, "cuMemFree_v2");
    LOAD_CU_SYM(cuMemcpyHtoD, "cuMemcpyHtoD_v2");
    LOAD_CU_SYM(cuMemcpyDtoH, "cuMemcpyDtoH_v2");
    LOAD_CU_SYM(cuLaunchKernel, "cuLaunchKernel");
    LOAD_CU_SYM(cuEventCreate, "cuEventCreate");
    LOAD_CU_SYM(cuEventRecord, "cuEventRecord");
    LOAD_CU_SYM(cuEventSynchronize, "cuEventSynchronize");
    LOAD_CU_SYM(cuEventElapsedTime, "cuEventElapsedTime");
    LOAD_CU_SYM(cuEventDestroy, "cuEventDestroy_v2");
    LOAD_CU_SYM(cuGetErrorString, "cuGetErrorString");
#undef LOAD_CU_SYM

    if (api.cuInit(0) != CUDA_SUCCESS) {
        return api;
    }
    int count = 0;
    if (api.cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
        return api;
    }
    if (api.cuDeviceGet(&api.device, 0) != CUDA_SUCCESS) {
        return api;
    }
    if (api.cuCtxCreate(&api.context, 0, api.device) != CUDA_SUCCESS) {
        return api;
    }

    const std::string ptx = load_ptx_source();
    if (ptx.empty()) {
        return api;
    }
    if (api.cuModuleLoadData(&api.module, ptx.c_str()) != CUDA_SUCCESS) {
        return api;
    }
    if (api.cuModuleGetFunction(&api.fn_gemm, api.module, "gemm_tiled_kernel") != CUDA_SUCCESS ||
        api.cuModuleGetFunction(&api.fn_softmax, api.module, "warp_rowmax_sum_kernel") != CUDA_SUCCESS ||
        api.cuModuleGetFunction(&api.fn_attention, api.module, "causal_attention_tiled_kernel") != CUDA_SUCCESS) {
        return api;
    }

    char name[256] = {0};
    api.cuDeviceGetName(name, sizeof(name), api.device);
    int major = 0, minor = 0, sms = 0;
    api.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, api.device);
    api.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, api.device);
    api.cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, api.device);

    char desc_buf[320];
    std::snprintf(
        desc_buf, sizeof(desc_buf),
        "CUDA GPU (Live: %s, SM %d.%d, %d SMs)",
        name, major, minor, sms);
    api.device_desc = desc_buf;
    api.available = true;
    return api;
}

void check_cu(CudaDriverApi& api, CUresult err, const char* call_str, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        const char* err_str = "Unknown Driver Error";
        if (api.cuGetErrorString) {
            api.cuGetErrorString(err, &err_str);
        }
        char buf[512];
        std::snprintf(buf, sizeof(buf), "CUDA Driver error at %s:%d in %s: %s (%d)",
                      file, line, call_str, err_str ? err_str : "unknown", err);
        throw std::runtime_error(buf);
    }
}

// RAII guard: frees device buffers and events even if a CU_DRV_CHECK throws.
struct DeviceScope {
    CudaDriverApi& api;
    std::vector<CUdeviceptr> ptrs;
    std::vector<CUevent> events;
    explicit DeviceScope(CudaDriverApi& a) : api(a) {}
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    void track(CUdeviceptr p) { ptrs.push_back(p); }
    void track_event(CUevent e) { events.push_back(e); }
    ~DeviceScope() {
        for (CUevent e : events) { if (e) api.cuEventDestroy(e); }
        for (CUdeviceptr p : ptrs) { if (p) api.cuMemFree(p); }
    }
};

#define CU_DRV_CHECK(api, call) check_cu((api), (call), #call, __FILE__, __LINE__)

} // namespace

bool query_cuda_device_available() {
    return get_driver().available;
}

std::string query_cuda_device_name() {
    auto& api = get_driver();
    if (!api.available) {
        return "CPU Host Fallback (No CUDA Device Detected at Runtime)";
    }
    return api.device_desc;
}

double launch_gemm_tiled_cuda(
    const float* h_A,
    const float* h_B,
    float* h_C,
    const GemmConfig& cfg
) {
    auto& api = get_driver();
    CU_DRV_CHECK(api, api.cuCtxSetCurrent(api.context));
    DeviceScope scope(api);

    const std::size_t bytes_A = static_cast<std::size_t>(cfg.m) * cfg.k * sizeof(float);
    const std::size_t bytes_B = static_cast<std::size_t>(cfg.k) * cfg.n * sizeof(float);
    const std::size_t bytes_C = static_cast<std::size_t>(cfg.m) * cfg.n * sizeof(float);

    CUdeviceptr d_A = 0, d_B = 0, d_C = 0;
    CU_DRV_CHECK(api, api.cuMemAlloc(&d_A, bytes_A)); scope.track(d_A);
    CU_DRV_CHECK(api, api.cuMemAlloc(&d_B, bytes_B)); scope.track(d_B);
    CU_DRV_CHECK(api, api.cuMemAlloc(&d_C, bytes_C)); scope.track(d_C);

    CU_DRV_CHECK(api, api.cuMemcpyHtoD(d_A, h_A, bytes_A));
    CU_DRV_CHECK(api, api.cuMemcpyHtoD(d_B, h_B, bytes_B));
    CU_DRV_CHECK(api, api.cuMemcpyHtoD(d_C, h_C, bytes_C));

    const unsigned int grid_x = static_cast<unsigned int>((cfg.n + TILE_N - 1) / TILE_N);
    const unsigned int grid_y = static_cast<unsigned int>((cfg.m + TILE_M - 1) / TILE_M);
    const unsigned int block_x = static_cast<unsigned int>(TILE_N);
    const unsigned int block_y = static_cast<unsigned int>(TILE_M);

    int M = cfg.m, N = cfg.n, K = cfg.k;
    float alpha = cfg.alpha, beta = cfg.beta;
    void* args[] = { &d_A, &d_B, &d_C, &M, &N, &K, &alpha, &beta };

    // Warmup pass on live GPU
    CU_DRV_CHECK(api, api.cuLaunchKernel(
        api.fn_gemm, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr));
    CU_DRV_CHECK(api, api.cuCtxSynchronize());
    if (cfg.beta != 0.0f) {
        CU_DRV_CHECK(api, api.cuMemcpyHtoD(d_C, h_C, bytes_C));
    }

    CUevent start = nullptr, stop = nullptr;
    CU_DRV_CHECK(api, api.cuEventCreate(&start, 0)); scope.track_event(start);
    CU_DRV_CHECK(api, api.cuEventCreate(&stop, 0)); scope.track_event(stop);

    const int iters = (cfg.iters > 0) ? cfg.iters : 1;
    double elapsed_ms = 0.0;
    if (cfg.beta == 0.0f) {
        CU_DRV_CHECK(api, api.cuEventRecord(start, nullptr));
        for (int i = 0; i < iters; ++i) {
            CU_DRV_CHECK(api, api.cuLaunchKernel(
                api.fn_gemm, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr));
        }
        CU_DRV_CHECK(api, api.cuEventRecord(stop, nullptr));
        CU_DRV_CHECK(api, api.cuEventSynchronize(stop));
        float total_ms = 0.0f;
        CU_DRV_CHECK(api, api.cuEventElapsedTime(&total_ms, start, stop));
        elapsed_ms = static_cast<double>(total_ms) / iters;
    } else {
        for (int i = 0; i < iters; ++i) {
            CU_DRV_CHECK(api, api.cuMemcpyHtoD(d_C, h_C, bytes_C));
            CU_DRV_CHECK(api, api.cuEventRecord(start, nullptr));
            CU_DRV_CHECK(api, api.cuLaunchKernel(
                api.fn_gemm, grid_x, grid_y, 1, block_x, block_y, 1, 0, nullptr, args, nullptr));
            CU_DRV_CHECK(api, api.cuEventRecord(stop, nullptr));
            CU_DRV_CHECK(api, api.cuEventSynchronize(stop));
            float step_ms = 0.0f;
            CU_DRV_CHECK(api, api.cuEventElapsedTime(&step_ms, start, stop));
            elapsed_ms += static_cast<double>(step_ms);
        }
        elapsed_ms /= iters;
    }

    CU_DRV_CHECK(api, api.cuMemcpyDtoH(h_C, d_C, bytes_C));


    return elapsed_ms;
}

double launch_warp_rowmax_sum_cuda(
    const float* h_input,
    float* h_output,
    int rows,
    int cols,
    int iters
) {
    auto& api = get_driver();
    CU_DRV_CHECK(api, api.cuCtxSetCurrent(api.context));
    DeviceScope scope(api);

    const std::size_t bytes = static_cast<std::size_t>(rows) * cols * sizeof(float);
    CUdeviceptr d_in = 0, d_out = 0;
    CU_DRV_CHECK(api, api.cuMemAlloc(&d_in, bytes)); scope.track(d_in);
    CU_DRV_CHECK(api, api.cuMemAlloc(&d_out, bytes)); scope.track(d_out);
    CU_DRV_CHECK(api, api.cuMemcpyHtoD(d_in, h_input, bytes));

    constexpr int WARPS_PER_BLOCK = 4;
    const unsigned int threads = static_cast<unsigned int>(WARPS_PER_BLOCK * WARP_SIZE);
    const unsigned int blocks = static_cast<unsigned int>((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    int r_arg = rows, c_arg = cols;
    void* args[] = { &d_in, &d_out, &r_arg, &c_arg };

    // Warmup
    CU_DRV_CHECK(api, api.cuLaunchKernel(
        api.fn_softmax, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
    CU_DRV_CHECK(api, api.cuCtxSynchronize());

    CUevent start = nullptr, stop = nullptr;
    CU_DRV_CHECK(api, api.cuEventCreate(&start, 0)); scope.track_event(start);
    CU_DRV_CHECK(api, api.cuEventCreate(&stop, 0)); scope.track_event(stop);

    const int run_iters = (iters > 0) ? iters : 1;
    CU_DRV_CHECK(api, api.cuEventRecord(start, nullptr));
    for (int i = 0; i < run_iters; ++i) {
        CU_DRV_CHECK(api, api.cuLaunchKernel(
            api.fn_softmax, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
    }
    CU_DRV_CHECK(api, api.cuEventRecord(stop, nullptr));
    CU_DRV_CHECK(api, api.cuEventSynchronize(stop));

    float total_ms = 0.0f;
    CU_DRV_CHECK(api, api.cuEventElapsedTime(&total_ms, start, stop));

    CU_DRV_CHECK(api, api.cuMemcpyDtoH(h_output, d_out, bytes));


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

    auto& api = get_driver();
    CU_DRV_CHECK(api, api.cuCtxSetCurrent(api.context));
    DeviceScope scope(api);

    const int total_heads = cfg.batch_size * cfg.num_heads;
    const std::size_t count = static_cast<std::size_t>(total_heads) * cfg.seq_len * cfg.head_dim;
    const std::size_t bytes = count * sizeof(float);

    CUdeviceptr d_Q = 0, d_K = 0, d_V = 0, d_Out = 0;
    CU_DRV_CHECK(api, api.cuMemAlloc(&d_Q, bytes)); scope.track(d_Q);
    CU_DRV_CHECK(api, api.cuMemAlloc(&d_K, bytes)); scope.track(d_K);
    CU_DRV_CHECK(api, api.cuMemAlloc(&d_V, bytes)); scope.track(d_V);
    CU_DRV_CHECK(api, api.cuMemAlloc(&d_Out, bytes)); scope.track(d_Out);

    CU_DRV_CHECK(api, api.cuMemcpyHtoD(d_Q, h_Q, bytes));
    CU_DRV_CHECK(api, api.cuMemcpyHtoD(d_K, h_K, bytes));
    CU_DRV_CHECK(api, api.cuMemcpyHtoD(d_V, h_V, bytes));

    const unsigned int block_x = static_cast<unsigned int>(TILE_M);
    const unsigned int grid_x = static_cast<unsigned int>((cfg.seq_len + TILE_M - 1) / TILE_M);
    const unsigned int grid_y = static_cast<unsigned int>(total_heads);

    int seq_arg = cfg.seq_len;
    int dim_arg = cfg.head_dim;
    float scale_arg = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    bool causal_arg = cfg.causal;
    void* args[] = { &d_Q, &d_K, &d_V, &d_Out, &seq_arg, &dim_arg, &scale_arg, &causal_arg };

    // Warmup
    CU_DRV_CHECK(api, api.cuLaunchKernel(
        api.fn_attention, grid_x, grid_y, 1, block_x, 1, 1, 0, nullptr, args, nullptr));
    CU_DRV_CHECK(api, api.cuCtxSynchronize());

    CUevent start = nullptr, stop = nullptr;
    CU_DRV_CHECK(api, api.cuEventCreate(&start, 0)); scope.track_event(start);
    CU_DRV_CHECK(api, api.cuEventCreate(&stop, 0)); scope.track_event(stop);

    const int iters = (cfg.iters > 0) ? cfg.iters : 1;
    CU_DRV_CHECK(api, api.cuEventRecord(start, nullptr));
    for (int i = 0; i < iters; ++i) {
        CU_DRV_CHECK(api, api.cuLaunchKernel(
            api.fn_attention, grid_x, grid_y, 1, block_x, 1, 1, 0, nullptr, args, nullptr));
    }
    CU_DRV_CHECK(api, api.cuEventRecord(stop, nullptr));
    CU_DRV_CHECK(api, api.cuEventSynchronize(stop));

    float total_ms = 0.0f;
    CU_DRV_CHECK(api, api.cuEventElapsedTime(&total_ms, start, stop));

    CU_DRV_CHECK(api, api.cuMemcpyDtoH(h_Out, d_Out, bytes));


    return static_cast<double>(total_ms) / iters;
}

} // namespace attn_forge
