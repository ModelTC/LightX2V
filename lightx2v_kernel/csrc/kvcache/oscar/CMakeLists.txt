cmake_minimum_required(VERSION 3.22.1)
project(oscar CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

if(CUTLASS_PATH)
    set(INCLUDE_DIR ${CUTLASS_PATH}/include)
elseif(DEFINED ENV{CUTLASS_PATH})
    set(INCLUDE_DIR $ENV{CUTLASS_PATH}/include)
else()
    message(FATAL_ERROR "CUTLASS_PATH is not set.")
endif()

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# message(STATUS "Compile testing packdecode kernel.")
# add_executable(test_single_packdecode
#     ${PROJECT_SOURCE_DIR}/src/test_single_packdecode.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_hdim128_fp16_sm80.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_2bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_4bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_2bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_4bit.cu
# )
# target_link_libraries(test_single_packdecode "${TORCH_LIBRARIES}")
# target_include_directories(test_single_packdecode PRIVATE ${INCLUDE_DIR})
# target_compile_options(test_single_packdecode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_80,code=sm_80 -w>)

message(STATUS "Compile testing residual kernel.")
add_executable(test_single_residual
    ${PROJECT_SOURCE_DIR}/src/test_single_residual.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_hdim128_fp16_sm80.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_2bit.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_4bit.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_2bit.cu
    ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_4bit.cu
)
target_link_libraries(test_single_residual "${TORCH_LIBRARIES}")
target_include_directories(test_single_residual PRIVATE ${INCLUDE_DIR})
target_compile_options(test_single_residual PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_80,code=sm_80 -w>)

# message(STATUS "Compile testing packdecode kernel.")
# add_executable(test_batch_packdecode
#     ${PROJECT_SOURCE_DIR}/src/test_batch_packdecode.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_hdim128_fp16_sm80.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_2bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_4bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_2bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_4bit.cu
# )
# target_link_libraries(test_batch_packdecode "${TORCH_LIBRARIES}")
# target_include_directories(test_batch_packdecode PRIVATE ${INCLUDE_DIR})
# target_compile_options(test_batch_packdecode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_80,code=sm_80 -w>)

# message(STATUS "Compile benchmarking kernel.")
# add_executable(bench_single_packdecode
#     ${PROJECT_SOURCE_DIR}/src/bench_single_packdecode.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_hdim128_fp16_sm80.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_2bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_4bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_2bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_4bit.cu
# )
# target_link_libraries(bench_single_packdecode "${TORCH_LIBRARIES}")
# target_include_directories(bench_single_packdecode PRIVATE ${INCLUDE_DIR})
# target_compile_options(bench_single_packdecode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_80,code=sm_80 -w>)

# message(STATUS "Compile benchmarking kernel.")
# add_executable(bench_single_residual
#     ${PROJECT_SOURCE_DIR}/src/bench_single_residual.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_hdim128_fp16_sm80.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_2bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_qpack_hdim128_fp16_sm80_4bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_2bit.cu
#     ${PROJECT_SOURCE_DIR}/src/genfile/flash_fwd_split_hdim128_fp16_sm80_4bit.cu
# )
# target_link_libraries(bench_single_residual "${TORCH_LIBRARIES}")
# target_include_directories(bench_single_residual PRIVATE ${INCLUDE_DIR})
# target_compile_options(bench_single_residual PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=255 -gencode arch=compute_80,code=sm_80 -w>)
