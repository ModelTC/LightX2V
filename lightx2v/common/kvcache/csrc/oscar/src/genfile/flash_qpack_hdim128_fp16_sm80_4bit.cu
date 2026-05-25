// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

template<>
void run_kvcache_qpack_<cutlass::half_t, 128, 1, 4, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_kvcache_qpack_hdim128<cutlass::half_t, 1, 4, 32>(params, stream);
}
// template<>
// void run_kvcache_qpack_<cutlass::half_t, 128, 1, 4, 64>(Flash_fwd_params &params, cudaStream_t stream) {
//     run_kvcache_qpack_hdim128<cutlass::half_t, 1, 4, 64>(params, stream);
// }
template<>
void run_kvcache_qpack_<cutlass::half_t, 128, 1, 4, 128>(Flash_fwd_params &params, cudaStream_t stream) {
    run_kvcache_qpack_hdim128<cutlass::half_t, 1, 4, 128>(params, stream);
}

template<>
void run_kvcache_qpack_<cutlass::bfloat16_t, 128, 1, 4, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_kvcache_qpack_hdim128<cutlass::bfloat16_t, 1, 4, 32>(params, stream);
}

template<>
void run_kvcache_qpack_<cutlass::bfloat16_t, 128, 1, 4, 128>(Flash_fwd_params &params, cudaStream_t stream) {
    run_kvcache_qpack_hdim128<cutlass::bfloat16_t, 1, 4, 128>(params, stream);
}


// template<>
// void run_kvcache_qpack_<cutlass::half_t, 128, 0, 4, 32>(Flash_fwd_params &params, cudaStream_t stream) {
//     run_kvcache_qpack_hdim128<cutlass::half_t, 0, 4, 32>(params, stream);
// }
// template<>
// void run_kvcache_qpack_<cutlass::half_t, 128, 0, 4, 64>(Flash_fwd_params &params, cudaStream_t stream) {
//     run_kvcache_qpack_hdim128<cutlass::half_t, 0, 4, 64>(params, stream);
// }
// template<>
// void run_kvcache_qpack_<cutlass::half_t, 128, 0, 4, 128>(Flash_fwd_params &params, cudaStream_t stream) {
//     run_kvcache_qpack_hdim128<cutlass::half_t, 0, 4, 128>(params, stream);
// }
