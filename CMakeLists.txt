cmake_minimum_required(VERSION 3.18)
project(cuda-flash-attn-forge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_CUDA_KERNELS "Enable NVIDIA CUDA kernel compilation (via nvcc or Clang NVPTX + CUDA Driver API)" ON)

include(CheckLanguage)
set(ATTN_FORGE_HAS_NVCC FALSE)
set(ATTN_FORGE_HAS_CLANG_PTX FALSE)

if(BUILD_CUDA_KERNELS)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
        endif()
        set(ATTN_FORGE_HAS_NVCC TRUE)
        message(STATUS "[attn_forge] CUDA nvcc compiler found: ${CMAKE_CUDA_COMPILER} (Building live GPU kernels via cudart)")
    else()
        find_program(CLANGXX_NVPTX_EXECUTABLE NAMES clang++ clang)
        set(_attn_clang_has_nvptx FALSE)
        if(CLANGXX_NVPTX_EXECUTABLE)
            execute_process(COMMAND "${CLANGXX_NVPTX_EXECUTABLE}" --print-targets
                OUTPUT_VARIABLE _attn_clang_targets ERROR_QUIET)
            if(_attn_clang_targets MATCHES "nvptx64")
                set(_attn_clang_has_nvptx TRUE)
            endif()
        endif()
        if(_attn_clang_has_nvptx)
            set(ATTN_FORGE_HAS_CLANG_PTX TRUE)
            message(STATUS "[attn_forge] nvcc not found; using Clang NVPTX backend (${CLANGXX_NVPTX_EXECUTABLE}) + CUDA Driver API (nvcuda) for live SM 7.5 GPU execution")
        else()
            message(STATUS "[attn_forge] Neither nvcc nor clang++ NVPTX found; falling back to CPU/host execution mode")
        endif()
    endif()
else()
    message(STATUS "[attn_forge] BUILD_CUDA_KERNELS=OFF; building in CPU/host fallback mode")
endif()

set(CORE_SOURCES
    src/cpu_reference.cpp
    src/host_runner.cpp
)

if(ATTN_FORGE_HAS_NVCC)
    list(APPEND CORE_SOURCES src/kernels.cu)
elseif(ATTN_FORGE_HAS_CLANG_PTX)
    list(APPEND CORE_SOURCES src/cuda_driver_loader.cpp)
    set(GENERATED_PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/kernels_sm75.ptx")
    add_custom_command(
        OUTPUT "${GENERATED_PTX_FILE}"
        COMMAND "${CLANGXX_NVPTX_EXECUTABLE}"
            -x cuda
            --cuda-device-only
            --cuda-gpu-arch=sm_75
            -nocudainc
            -nocudalib
            -fms-extensions
            -DATTN_FORGE_CLANG_PTX_ONLY
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -O3
            -S "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cu"
            -o "${GENERATED_PTX_FILE}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/attn_forge/cuda_clang_device.cuh"
        COMMENT "Compiling src/kernels.cu to SM 7.5 PTX via Clang NVPTX backend"
        VERBATIM
    )
    add_custom_target(attn_forge_ptx ALL DEPENDS "${GENERATED_PTX_FILE}")
endif()

add_library(attn_forge_core STATIC ${CORE_SOURCES})
target_include_directories(attn_forge_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ATTN_FORGE_HAS_NVCC)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(attn_forge_core PUBLIC CUDA::cudart)
    target_compile_definitions(attn_forge_core PUBLIC ATTN_FORGE_ENABLE_CUDA=1 ATTN_FORGE_USE_CUDART=1)
    set_target_properties(attn_forge_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
elseif(ATTN_FORGE_HAS_CLANG_PTX)
    add_dependencies(attn_forge_core attn_forge_ptx)
    target_compile_definitions(attn_forge_core PUBLIC
        ATTN_FORGE_ENABLE_CUDA=1
        ATTN_FORGE_USE_CUDART=0
        ATTN_FORGE_PTX_PATH="${GENERATED_PTX_FILE}"
    )
    if(UNIX AND NOT APPLE)
        target_link_libraries(attn_forge_core PUBLIC dl)
    endif()
else()
    target_compile_definitions(attn_forge_core PUBLIC ATTN_FORGE_ENABLE_CUDA=0 ATTN_FORGE_USE_CUDART=0)
endif()

add_executable(attn_forge src/main.cpp)
target_link_libraries(attn_forge PRIVATE attn_forge_core)

enable_testing()
add_executable(test_parity tests/test_parity.cpp)
target_link_libraries(test_parity PRIVATE attn_forge_core)
add_test(NAME ParityTestSuite COMMAND test_parity)

if(MSVC)
    target_compile_options(attn_forge_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>)
    target_compile_options(attn_forge PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>)
    target_compile_options(test_parity PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>)
else()
    target_compile_options(attn_forge_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>)
    target_compile_options(attn_forge PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>)
    target_compile_options(test_parity PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>)
endif()
