# Copied and adapted from https://github.com/triple-mu/fast-ulysses
cmake_minimum_required(VERSION 3.18)
project(lightx2v_fast_ulysses LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "80;90;100")
endif ()

# ccache: wrap CXX/CUDA compilers to cache objects and skip recompiling unchanged TUs (skipped if not installed)
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
endif ()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}" CACHE STRING "Torch prefix" FORCE)
find_package(Torch REQUIRED)

if (NOT DEFINED NVSHMEM_HOME OR NOT EXISTS "${NVSHMEM_HOME}/include/nvshmem.h")
    message(FATAL_ERROR "NVSHMEM_HOME missing: setup.py must pass -DNVSHMEM_HOME=<absolute>.")
endif ()
find_package(NVSHMEM REQUIRED HINTS "${NVSHMEM_HOME}/lib/cmake/nvshmem")

if (NOT DEFINED EXT_SUFFIX)
    set(EXT_SUFFIX ".so")
endif ()

set(SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/bindings.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/all_to_all.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/all_to_all_tma.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/symmetric_pool.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/ulysses_group.cu")
add_library(_C SHARED ${SOURCES})
set_target_properties(_C PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}"
        CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(_C PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc" "${NVSHMEM_HOME}/include")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python::Module
        nvshmem::nvshmem_host cuda)  # cuda: CUDA driver, for cuTensorMapEncodeTiled (TMA)
target_compile_options(_C PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr
        -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__>)
