cmake_minimum_required(VERSION 3.18)
project(cuda-flash-attn-forge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_CUDA_KERNELS "Enable NVIDIA CUDA kernel compilation via nvcc" ON)

include(CheckLanguage)
if(BUILD_CUDA_KERNELS)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
        endif()
        set(ATTN_FORGE_HAS_CUDA TRUE)
        message(STATUS "[attn_forge] CUDA compiler found: ${CMAKE_CUDA_COMPILER} (Building live GPU kernels)")
    else()
        set(ATTN_FORGE_HAS_CUDA FALSE)
        message(STATUS "[attn_forge] nvcc not found; falling back to CPU/host execution mode (BUILD_CUDA_KERNELS=OFF)")
    endif()
else()
    set(ATTN_FORGE_HAS_CUDA FALSE)
    message(STATUS "[attn_forge] BUILD_CUDA_KERNELS=OFF; building in CPU/host fallback mode")
endif()

set(CORE_SOURCES
    src/cpu_reference.cpp
    src/host_runner.cpp
)

if(ATTN_FORGE_HAS_CUDA)
    list(APPEND CORE_SOURCES src/kernels.cu)
endif()

add_library(attn_forge_core STATIC ${CORE_SOURCES})
target_include_directories(attn_forge_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ATTN_FORGE_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(attn_forge_core PUBLIC CUDA::cudart)
    target_compile_definitions(attn_forge_core PUBLIC ATTN_FORGE_ENABLE_CUDA=1)
    set_target_properties(attn_forge_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    target_compile_definitions(attn_forge_core PUBLIC ATTN_FORGE_ENABLE_CUDA=0)
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
