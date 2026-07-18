cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(lightx2v-kernel LANGUAGES CXX CUDA)

option(LIGHTX2V_THOR_NVFP4_ONLY "Build only NVFP4 operators for NVIDIA Thor" OFF)
option(
    LIGHTX2V_ENABLE_CUTEDSL_VIT_FMHA
    "Build the TensorRT-Edge-LLM CuTe DSL ViT FMHA bridge for NVIDIA Thor"
    OFF
)
set(
    TENSORRT_EDGE_LLM_ROOT
    ""
    CACHE PATH
    "Path to a TensorRT-Edge-LLM checkout containing the SM110 CuTe DSL prebuilt artifact"
)

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


if(LIGHTX2V_THOR_NVFP4_ONLY)
    list(APPEND LIGHTX2V_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_110a,code=sm_110a"
    )
    set(SOURCES
        "csrc/gemm/nvfp4_scaled_mm_kernels_sm120.cu"
        "csrc/gemm/nvfp4_quant_kernels_sm120.cu"
        "csrc/common_extension.cc"
    )
else()
    list(APPEND LIGHTX2V_KERNEL_CUDA_FLAGS
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
        "csrc/kvcache/kv_dequant_cuda.cu"
        "csrc/common_extension.cc"
    )
endif()

if(LIGHTX2V_ENABLE_CUTEDSL_VIT_FMHA)
    if(NOT LIGHTX2V_THOR_NVFP4_ONLY)
        message(FATAL_ERROR "LIGHTX2V_ENABLE_CUTEDSL_VIT_FMHA requires LIGHTX2V_THOR_NVFP4_ONLY=ON.")
    endif()
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        message(FATAL_ERROR "The SM110 CuTe DSL ViT FMHA artifact is available only for AArch64 Thor builds.")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
        message(FATAL_ERROR "The bundled SM110 CuTe DSL artifact requires CUDA Toolkit 13.0 or newer.")
    endif()
    if(NOT TENSORRT_EDGE_LLM_ROOT)
        message(FATAL_ERROR "Set TENSORRT_EDGE_LLM_ROOT when LIGHTX2V_ENABLE_CUTEDSL_VIT_FMHA=ON.")
    endif()

    set(
        _cutedsl_artifact_dir
        "${TENSORRT_EDGE_LLM_ROOT}/cpp/kernels/cuteDSLArtifact/aarch64/sm_110"
    )
    if(NOT EXISTS "${_cutedsl_artifact_dir}/metadata.json")
        set(
            _cutedsl_legacy_artifact_dir
            "${TENSORRT_EDGE_LLM_ROOT}/cpp/kernels/cuteDSLArtifact/aarch64"
        )
        if(EXISTS "${_cutedsl_legacy_artifact_dir}/metadata.json")
            set(_cutedsl_artifact_dir "${_cutedsl_legacy_artifact_dir}")
        else()
            set(
                _cutedsl_prebuilt
                "${TENSORRT_EDGE_LLM_ROOT}/kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_110_cuda13.tar.gz"
            )
            if(NOT EXISTS "${_cutedsl_prebuilt}")
                message(FATAL_ERROR "SM110 CuTe DSL prebuilt artifact not found: ${_cutedsl_prebuilt}")
            endif()

            set(_cutedsl_extract_root "${CMAKE_CURRENT_BINARY_DIR}/cutedsl_artifact/aarch64")
            set(_cutedsl_artifact_dir "${_cutedsl_extract_root}/sm_110")
            if(NOT EXISTS "${_cutedsl_artifact_dir}/metadata.json")
                file(MAKE_DIRECTORY "${_cutedsl_extract_root}")
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E tar xzf "${_cutedsl_prebuilt}"
                    WORKING_DIRECTORY "${_cutedsl_extract_root}"
                    RESULT_VARIABLE _cutedsl_extract_result
                )
                if(NOT _cutedsl_extract_result EQUAL 0)
                    message(FATAL_ERROR "Failed to extract ${_cutedsl_prebuilt}.")
                endif()
            endif()
        endif()
    endif()

    set(_cutedsl_metadata "${_cutedsl_artifact_dir}/metadata.json")
    set(_cutedsl_include_dir "${_cutedsl_artifact_dir}/include")
    set(_cutedsl_static_lib "${_cutedsl_artifact_dir}/libcutedsl_aarch64.a")
    foreach(_required_file IN ITEMS
        "${_cutedsl_metadata}"
        "${_cutedsl_include_dir}/vit_fmha_d64.h"
        "${_cutedsl_include_dir}/vit_fmha_d72.h"
        "${_cutedsl_include_dir}/vit_fmha_d80.h"
        "${_cutedsl_include_dir}/vit_fmha_d128.h"
        "${_cutedsl_static_lib}"
    )
        if(NOT EXISTS "${_required_file}")
            message(FATAL_ERROR "Incomplete SM110 CuTe DSL artifact; missing ${_required_file}.")
        endif()
    endforeach()

    file(READ "${_cutedsl_metadata}" _cutedsl_metadata_json)
    string(JSON _cutedsl_gpu_arch GET "${_cutedsl_metadata_json}" "gpu_arch")
    string(JSON _cutedsl_cuda_version GET "${_cutedsl_metadata_json}" "cuda_version")
    if(NOT _cutedsl_gpu_arch STREQUAL "sm_110")
        message(FATAL_ERROR "Expected an sm_110 CuTe DSL artifact, got ${_cutedsl_gpu_arch}.")
    endif()
    if(NOT _cutedsl_cuda_version MATCHES "^13\\.")
        message(FATAL_ERROR "Expected a CUDA 13 CuTe DSL artifact, got ${_cutedsl_cuda_version}.")
    endif()

    string(JSON _cutedsl_variant_count LENGTH "${_cutedsl_metadata_json}" "variants")
    if(_cutedsl_variant_count LESS 1)
        message(FATAL_ERROR "SM110 CuTe DSL artifact has no compiled variants.")
    endif()
    math(EXPR _cutedsl_last_variant "${_cutedsl_variant_count} - 1")
    set(_cutedsl_variants)
    foreach(_variant_index RANGE ${_cutedsl_last_variant})
        string(JSON _variant GET "${_cutedsl_metadata_json}" "variants" ${_variant_index})
        list(APPEND _cutedsl_variants "${_variant}")
    endforeach()
    foreach(_required_variant IN ITEMS vit_fmha_d64 vit_fmha_d72 vit_fmha_d80 vit_fmha_d128)
        if(NOT _required_variant IN_LIST _cutedsl_variants)
            message(FATAL_ERROR "SM110 CuTe DSL artifact is missing ${_required_variant}.")
        endif()
    endforeach()

    list(APPEND SOURCES "csrc/attention/cute_dsl_vit_fmha_sm110.cpp")
endif()

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

if(LIGHTX2V_THOR_NVFP4_ONLY)
    target_compile_definitions(common_ops PRIVATE LIGHTX2V_THOR_NVFP4_ONLY=1)
endif()

message(STATUS "LIGHTX2V_KERNEL_CUDA_FLAGS: ${LIGHTX2V_KERNEL_CUDA_FLAGS}")

target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${LIGHTX2V_KERNEL_CUDA_FLAGS}>)
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)

if(LIGHTX2V_ENABLE_CUTEDSL_VIT_FMHA)
    target_compile_definitions(common_ops PRIVATE LIGHTX2V_CUTEDSL_VIT_FMHA_SM110=1)
    target_include_directories(common_ops SYSTEM PRIVATE "${_cutedsl_include_dir}")
    target_link_libraries(common_ops PRIVATE "${_cutedsl_static_lib}" CUDA::cuda_driver CUDA::cudart)
    message(STATUS "CuTe DSL ViT FMHA SM110 artifact: ${_cutedsl_artifact_dir}")
endif()

install(TARGETS common_ops LIBRARY DESTINATION lightx2v_kernel)
