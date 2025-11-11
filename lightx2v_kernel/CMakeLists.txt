cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(lightx2v-kernel LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Python
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Torch
find_package(Torch REQUIRED)
# clean Torch Flag
clear_cuda_arches(CMAKE_FLAG)


# cutlass
if(CUTLASS_PATH)
    set(repo-cutlass_SOURCE_DIR ${CUTLASS_PATH})
    message(STATUS "Using local CUTLASS from: ${CUTLASS_PATH}")
else()
    message(FATAL_ERROR "CUTLASS_PATH is not set. Please manually download CUTLASS first.")
endif()


# ccache option
option(ENABLE_CCACHE "Whether to use ccache" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
    message(STATUS "Building with CCACHE enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
endif()


include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
)

set(LIGHTX2V_KERNEL_CUDA_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=lightx2v-kernel"
    "-O3"
    "-Xcompiler"
    "-fPIC"
    "-std=c++17"
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTLASS_TEST_LEVEL=0"
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--threads=32"

    # Suppress warnings
    "-Xcompiler=-Wconversion"
    "-Xcompiler=-fno-strict-aliasing"

)


list(APPEND LIGHTX2V_KERNEL_CUDA_FLAGS
    # "-gencode=arch=compute_90,code=sm_90"
    # "-gencode=arch=compute_90a,code=sm_90a"
    # "-gencode=arch=compute_100,code=sm_100"
    # "-gencode=arch=compute_100a,code=sm_100a"
    # "-gencode=arch=compute_120,code=sm_120"
    "-gencode=arch=compute_120a,code=sm_120a"
)


set(SOURCES
    "csrc/gemm/nvfp4_scaled_mm_kernels_sm120.cu"
    "csrc/gemm/nvfp4_quant_kernels_sm120.cu"
    "csrc/gemm/mxfp4_quant_kernels_sm120.cu"
    "csrc/gemm/mxfp8_quant_kernels_sm120.cu"
    "csrc/gemm/mxfp6_quant_kernels_sm120.cu"
    "csrc/gemm/mxfp4_scaled_mm_kernels_sm120.cu"
    "csrc/gemm/mxfp6_mxfp8_scaled_mm_kernels_sm120.cu"
    "csrc/gemm/mxfp8_scaled_mm_kernels_sm120.cu"
    "csrc/common_extension.cc"
)

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

message(STATUS "LIGHTX2V_KERNEL_CUDA_FLAGS: ${LIGHTX2V_KERNEL_CUDA_FLAGS}")

target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${LIGHTX2V_KERNEL_CUDA_FLAGS}>)
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)

install(TARGETS common_ops LIBRARY DESTINATION lightx2v_kernel)
