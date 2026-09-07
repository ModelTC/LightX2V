# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
import os
import sys
import time
from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import numpy as np
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Int64

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, ".."))

from . import cutedsl_fmha_helpers as fmha_utils  # isort: skip

"""
A fused multi-head attention (FMHA) example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of fused multi-head attention using a TMA + Blackwell SM100
TensorCore warp-specialized persistent kernel. The implementation integrates the Q*K^T matrix multiplication,
softmax normalization, and softmax(Q*K^T)*V into a single kernel, avoiding intermediate data movement between
global memory and shared memory, thus improving computational efficiency.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, softmax, correction, epilogue)
- Pipeline stages between different warps for overlapping computation and memory access
- Support for different precision data types
- Optional causal masking for autoregressive models

To run this example:

.. code-block:: bash

    python examples/blackwell/fmha.py                                     \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent

The above example runs FMHA with batch size 4, sequence length 1024, 8 attention heads, and head
dimension 64. The Blackwell tcgen05 MMA tile shape is (128, 128), and the kernel uses fp16 for input/output
with fp32 for accumulation.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/fmha.py                                 \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent --warmup_iterations 10                              \
      --iterations 10 --skip_ref_check

Constraints for this example:
* Supported head dimensions: 32, 64, and 128
* Number of heads in Q must be divisible by number of heads in K
* mma_tiler_mn must be 128,128
* Batch size must be the same for Q, K, and V tensors
* For causal masking, use --is_causal (note: specify without =True/False)
* For persistent scheduling, use --is_persistent (note: specify without =True/False)
"""


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class BlackwellFusedMultiHeadAttentionForward:
    WINDOW_NO_LIMIT = 1 << 30

    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_persistent: bool,
        mask_type: fmha_utils.MaskEnum,
        is_causal: bool = False,
        use_sliding_window: bool = False,
        actual_head_dim: Optional[int] = None,
    ):
        """Initializes the configuration for a Blackwell Fused Multi-Head Attention (FMHA) kernel.

        This configuration includes several key aspects:

        1.  Data Type Settings:
            - qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
            - pv_acc_dtype: Data type for P*V matrix multiplication accumulator

        2.  MMA Instruction Settings:
            - mma_tiler: The (M, N, K) shape of the MMA instruction unit
            - qk_mma_tiler: MMA shape for Q*K^T computation
            - pv_mma_tiler: MMA shape for P*V computation

        3.  Kernel Execution Mode:
            - is_persistent: Boolean indicating whether to use persistent kernel mode
            - mask_type: Specifies the type of mask to use (no mask, residual mask, or causal mask)
            - is_causal: Whether to apply causal masking (window_size_right = 0)
            - use_sliding_window: Whether to compile with sliding window support

        :param qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
        :type qk_acc_dtype: Type[cutlass.Numeric]
        :param pv_acc_dtype: Data type for P*V matrix multiplication accumulator
        :type pv_acc_dtype: Type[cutlass.Numeric]
        :param mma_tiler: The (M, N, K) shape of the MMA instruction
        :type mma_tiler: Tuple[int, int, int]
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param mask_type: Type of mask to use
        :type mask_type: fmha_utils.MaskEnum
        :param is_causal: Whether to apply causal masking. When True, window_size_right
            is set to Int32(0) as a compile-time constant so tokens cannot attend
            to future positions. When False, window_size_right is None (bidirectional).
        :type is_causal: bool
        :param use_sliding_window: If True, compile with sliding window masking code.
            If False, window_size_left is treated as None at compile time,
            eliminating left-side window masking code for better performance.
        :type use_sliding_window: bool
        """

        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        # head_dim = actual tensor dimension (e.g. 72).
        # MMA/SMEM/CTA tilers below use mma_tiler[2] (padded, e.g. 80).
        # TMA ZFILL bridges the gap on loads; OOB drop on stores.
        self.head_dim = actual_head_dim if actual_head_dim is not None else mma_tiler[2]
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)
        self.log2_e = math.log2(math.e)
        self.cta_tiler = (
            2 * mma_tiler[0],  # 2 Q tile per CTA
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = mma_tiler
        self.pv_mma_tiler = (
            mma_tiler[0],
            mma_tiler[2],
            mma_tiler[1],
        )
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        self.is_causal = is_causal
        self.use_sliding_window = use_sliding_window

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epilogue_warp_id = 14
        self.empty_warp_id = 15
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.empty_warp_id,
            )
        )

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp,
        )

        self.tmem_s0_offset = 0
        self.tmem_s1_offset = 128
        self.tmem_o0_offset = 256
        self.tmem_o1_offset = 384
        self.tmem_p0_offset = 32
        self.tmem_p1_offset = 160

        # vec buffer for row_max & row_sum
        self.tmem_vec0_offset = 0
        self.tmem_vec1_offset = 128

        self.num_regs_softmax = int(os.environ.get("SPARSE_FMHA_REGS_SOFTMAX", "192"))
        self.num_regs_correction = int(os.environ.get("SPARSE_FMHA_REGS_CORRECTION", "96"))
        self.num_regs_other = int(os.environ.get("SPARSE_FMHA_REGS_OTHER", "32"))
        self.min_blocks_per_mp = int(os.environ.get("SPARSE_FMHA_MIN_BLOCKS", "1"))
        self.fixed_single_batch = os.environ.get("SPARSE_FMHA_FIXED_B1", "0") == "1"
        # Delayed/zero-reference max modes are approximate experiments. Keep the
        # default path equivalent to the online-softmax reference semantics.
        self.max_update_interval = int(os.environ.get("SPARSE_FMHA_MAX_UPDATE_INTERVAL", "1"))
        if self.max_update_interval not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("SPARSE_FMHA_MAX_UPDATE_INTERVAL must be a power of two from 1 to 64")
        self.zero_reference_max = os.environ.get("SPARSE_FMHA_ZERO_REFERENCE_MAX", "0") == "1"

        self.buffer_align_bytes = 1024

        num_warps_per_warpgroup = 4
        self.softmax_warpgroup_count = len((*self.softmax0_warp_ids, *self.softmax1_warp_ids)) // num_warps_per_warpgroup

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.q_stage = int(os.environ.get("SPARSE_FMHA_Q_STAGE", "2"))
        # SM110 tuning: BF16 K/V staging is faster at two stages while retaining
        # the original four-stage default for the separately tuned FP8 path.
        default_kv_stage = 4 if self.q_dtype.width == 8 else 2
        self.kv_stage = int(os.environ.get("SPARSE_FMHA_KV_STAGE", str(default_kv_stage)))
        self.acc_stage = 1
        self.softmax_corr_stage = 1
        self.mma_corr_stage = int(os.environ.get("SPARSE_FMHA_MMA_CORR_STAGE", "2"))
        self.mma_softmax_stage = 1
        self.epi_stage = int(os.environ.get("SPARSE_FMHA_EPI_STAGE", "2"))

        # Pre-scale softmax output by 2^FP8_E4M3_PRESCALE_LOG2=256 for FP8 to maximize
        # E4M3 dynamic range.
        # P ∈ [0,1] → P*256 ∈ [0,256], utilizing more of the [0,448] FP8 range.
        # Fused into exp2: exp2(x + 8) = exp2(x) * 256, zero extra per-element ops.
        FP8_E4M3_PRESCALE_LOG2 = 8.0  # log2(256) — tuned for FP8 E4M3 range [0, 448]
        self.softmax_prescale_log2 = FP8_E4M3_PRESCALE_LOG2 if self.q_dtype.width == 8 else 0.0
        self.softmax_prescale_ln = self.softmax_prescale_log2 * 0.6931471805599453

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,  # (B, S_q, H_q, D) — B,S_q,H_q dynamic; D static
        kv_cache: cute.Tensor,  # (B, 2, H_kv, S_k, D) — B,H_kv,S_k dynamic; D static
        o_tensor: cute.Tensor,  # (B, S_q, H_q, D) — same layout as Q
        cum_seqlen_k: cute.Tensor,  # (B+1,) Int32 — cumulative KV sequence lengths
        window_size_left: Int32,
        scale_q: Float32,
        scale_k: Float32,
        scale_v: Float32,
        inv_scale_o: Float32,
        stream: cuda.CUstream,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters

        The softmax scale is computed as:
            scale_softmax = scale_q * scale_k * (1 / sqrt(head_dim))
        For FP16 (no quantization), pass scale_q = scale_k = scale_v = inv_scale_o = 1.0.
        For FP8, pass the dequant scales so the kernel folds them into softmax/output scaling.

        :param q_tensor: The query tensor (B, S_q, H_q, D) with dynamic B, S_q, H_q
        :type q_tensor: cute.Tensor
        :param kv_cache: The KV cache tensor (B, 2, H_kv, S_k, D) with dynamic B, H_kv, S_k
        :type kv_cache: cute.Tensor
        :param o_tensor: The output tensor (B, S_q, H_q, D)
        :type o_tensor: cute.Tensor
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Int32
        :param scale_q: Dequantization scale for Q (quant→orig). 1.0 for FP16.
        :type scale_q: Float32
        :param scale_k: Dequantization scale for K (quant→orig). 1.0 for FP16.
        :type scale_k: Float32
        :param scale_v: Dequantization scale for V (quant→orig). 1.0 for FP16.
        :type scale_v: Float32
        :param inv_scale_o: Inverse output quantization scale. 1.0 for FP16 output.
        :type inv_scale_o: Float32
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream
        :raises TypeError: If tensor data types don't match or aren't supported
        :raises RuntimeError: If tensor layouts aren't in supported formats
        """
        scale_softmax = scale_q * scale_k * self.inv_sqrt_head_dim
        scale_softmax_log2 = scale_softmax * self.log2_e
        scale_output = scale_v * inv_scale_o
        b = q_tensor.layout.shape[0]
        s_q = q_tensor.layout.shape[1]
        h_q = q_tensor.layout.shape[2]
        h_k = kv_cache.layout.shape[2]
        cap = kv_cache.layout.shape[3]
        s_lse = s_q
        d = self.head_dim

        q_iter = q_tensor.iterator
        # KV cache shape[3] = cap (physical capacity, for stride computation).
        # Per-batch actual KV lengths come from cum_seqlen_k at runtime.
        kv_base = kv_cache.iterator
        stride_kv_head = cap * d
        stride_kv_select = h_k * stride_kv_head
        k_iter = kv_base
        v_iter = kv_base + stride_kv_select
        o_iter = o_tensor.iterator

        cum_seqlen_q = None
        lse_iter = None
        h_r = h_q // h_k
        # Always use batch-strided layout (not packed varlen).
        # cum_seqlen_k is only used inside the kernel for per-batch seqlen_k override.
        qo_offset = 0
        kv_offset = 0
        b_qo = b
        b_kv = b
        stride_b_qo = h_r * h_k * s_q * d
        stride_b_kv = 2 * stride_kv_select
        b_lse = b
        stride_b_lse = h_r * h_k * s_lse

        # (s, d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b_qo)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
        )
        q = cute.make_tensor(q_iter + qo_offset, q_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (cap, d, ((h_r, h_k), b_kv)),
            stride=(d, 1, ((0, stride_kv_head), stride_b_kv)),
        )
        k = cute.make_tensor(k_iter + kv_offset, k_layout)
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v_layout = cute.make_layout(
            (d, cap, ((h_r, h_k), b_kv)),
            stride=(1, d, ((0, stride_kv_head), stride_b_kv)),
        )
        v = cute.make_tensor(v_iter + kv_offset, v_layout)
        # (s, d, ((h_r, h_k), b))
        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b_qo)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
        )
        o = cute.make_tensor(o_iter + qo_offset, o_layout)
        if cutlass.const_expr(lse_iter is not None):
            # (s, ((h_r, h_k), b))
            lse_layout = cute.make_layout(
                (s_lse, ((h_r, h_k), b_lse)),
                stride=(1, ((s_lse, h_r * s_lse), stride_b_lse)),
            )
            lse = cute.make_tensor(lse_iter, lse_layout)
        else:
            lse = None

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            cute.shape((s_q, d, ((h_r, h_k), b))),
            self.cta_tiler,
            self.is_persistent,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE
        # the intermediate tensor p is from tmem & k-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.pv_mma_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.acc_stage,
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )

        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])

        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            o,
            o_smem_layout,
            self.epi_tile,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kv_bytes = k_copy_size

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s0_s1_sequence_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_warpgroup_count]
            corr_epi_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Compile-time dispatch: when use_sliding_window is False, pass None for
        # window_size_left to eliminate left-side window masking code in the kernel.
        if cutlass.const_expr(self.use_sliding_window):
            _wsl = window_size_left
        else:
            _wsl = None

        _wsr = Int32(0) if cutlass.const_expr(self.is_causal) else None

        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            o,
            cum_seqlen_q,
            cum_seqlen_k,
            lse,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            _wsl,
            _wsr,
            None,
            None,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout_staged,
            v_smem_layout_staged,
            o_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def __call_vit__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        o_tensor: cute.Tensor,
        cu_seqlens: cute.Tensor,
        max_seqlen: Int32,
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        stream: cuda.CUstream,
    ):
        """Compile-time dense entry point for standalone packed ViT attention."""
        self._call_vit_impl(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            cu_seqlens,
            max_seqlen,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            None,
            None,
            stream,
        )

    @cute.jit
    def __call_vit_sparse__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        o_tensor: cute.Tensor,
        cu_seqlens: cute.Tensor,
        max_seqlen: Int32,
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        sparse_block_count: cute.Tensor,
        sparse_block_indices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Block-sparse packed ViT attention.

        Metadata uses one non-empty KV block list per 256-row Q tile:
        ``count[B, Hq, ceil(max_seqlen / 256)]`` and
        ``indices[B, Hq, ceil(max_seqlen / 256), max_blocks]``. Each index
        addresses a 128-token, sequence-local KV tile.
        """
        self._call_vit_impl(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            cu_seqlens,
            max_seqlen,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            sparse_block_count,
            sparse_block_indices,
            stream,
        )

    @cute.jit
    def _call_vit_impl(
        self,
        q_tensor: cute.Tensor,  # (total_S, H_q, D) — packed varlen
        k_tensor: cute.Tensor,  # (total_S, H_kv, D) — packed varlen
        v_tensor: cute.Tensor,  # (total_S, H_kv, D) — packed varlen
        o_tensor: cute.Tensor,  # (total_S, H_q, D)
        cu_seqlens: cute.Tensor,  # (B+1,) Int32
        max_seqlen: Int32,
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        sparse_block_count: Optional[cute.Tensor],
        sparse_block_indices: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        """ViT FMHA: packed varlen, separate Q/K/V, bidirectional (no causal mask).

        All sequences are packed into flat [total_S, H, D] tensors with boundaries
        defined by cu_seqlens.  Each sequence attends to all tokens in that sequence
        (PADDING / RESIDUAL_MASK — no causal ordering).

        max_seqlen is the longest individual sequence length (not total_S).
        It controls the grid size and per-sequence tile math. TMA descriptors
        stay bounded by the compact total_S tensor extent.
        """
        total_s = q_tensor.layout.shape[0]
        s_q = max_seqlen
        h_q = q_tensor.layout.shape[1]
        h_k = k_tensor.layout.shape[1]
        d = self.head_dim

        b = cu_seqlens.layout.shape[0] - 1
        h_r = h_q // h_k

        q_iter = q_tensor.iterator
        k_iter = k_tensor.iterator
        v_iter = v_tensor.iterator
        o_iter = o_tensor.iterator

        cum_seqlen_q = None if self.fixed_single_batch else cu_seqlens
        cum_seqlen_k = None if self.fixed_single_batch else cu_seqlens

        # The generic packed-varlen path drops the batch coordinate after applying
        # cu_seqlens. The B=1 specialization keeps that coordinate in the tensor
        # descriptor so the kernel can use its normal non-varlen coordinate path.
        if cutlass.const_expr(self.fixed_single_batch):
            q_layout = cute.make_layout(
                (total_s, d, ((h_r, h_k), 1)),
                stride=(
                    d * h_r * h_k,
                    1,
                    ((d, d * h_r), total_s * d * h_r * h_k),
                ),
            )
            k_layout = cute.make_layout(
                (total_s, d, ((h_r, h_k), 1)),
                stride=(d * h_k, 1, ((0, d), total_s * d * h_k)),
            )
            v_layout = cute.make_layout(
                (d, total_s, ((h_r, h_k), 1)),
                stride=(1, d * h_k, ((0, d), total_s * d * h_k)),
            )
            o_layout = q_layout
        else:
            # Descriptor OOB is based on total_s, not on the synthetic
            # max_seqlen * batch envelope.
            q_layout = cute.make_layout(
                (total_s, d, (h_r, h_k)),
                stride=(d * h_r * h_k, 1, (d, d * h_r)),
            )
            k_layout = cute.make_layout(
                (total_s, d, (h_r, h_k)),
                stride=(d * h_k, 1, (0, d)),
            )
            v_layout = cute.make_layout(
                (d, total_s, (h_r, h_k)),
                stride=(1, d * h_k, (0, d)),
            )
            o_layout = q_layout
        q = cute.make_tensor(q_iter, q_layout)
        k = cute.make_tensor(k_iter, k_layout)
        v = cute.make_tensor(v_iter, v_layout)
        o = cute.make_tensor(o_iter, o_layout)
        lse = None

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            cute.shape((s_q, d, ((h_r, h_k), b))),
            self.cta_tiler,
            self.is_persistent,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.pv_mma_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage)
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(qk_tiled_mma, self.qk_mma_tiler, self.k_dtype, self.kv_stage)
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, self.acc_stage)
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.kv_stage)
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.o_dtype, self.o_layout, self.epi_tile, self.epi_stage)

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, q, q_smem_layout, self.qk_mma_tiler, qk_tiled_mma, self.cluster_layout_vmnk.shape)

        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, k, k_smem_layout, self.qk_mma_tiler, qk_tiled_mma, self.cluster_layout_vmnk.shape)

        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, v, v_smem_layout, self.pv_mma_tiler, pv_tiled_mma, self.cluster_layout_vmnk.shape)

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(tma_store_op, o, o_smem_layout, self.epi_tile)

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kv_bytes = k_copy_size

        @cute.struct
        class SharedStorage:
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s0_s1_sequence_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_warpgroup_count]
            corr_epi_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
            tmem_holding_buf: Int32
            sO: cute.struct.Align[cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)], self.buffer_align_bytes]
            sQ: cute.struct.Align[cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)], self.buffer_align_bytes]
            sK: cute.struct.Align[cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage

        # ViT: bidirectional — no sliding window, no causal masking.
        # Both window sizes are None (is_causal=False) so apply_mask only
        # applies RESIDUAL_MASK for out-of-bounds elements on partial tiles.
        _wsl = None
        _wsr = Int32(0) if cutlass.const_expr(self.is_causal) else None

        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            o,
            cum_seqlen_q,
            cum_seqlen_k,
            lse,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            _wsl,
            _wsr,
            sparse_block_count,
            sparse_block_indices,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout_staged,
            v_smem_layout_staged,
            o_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        mO_gmem: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sparse_block_count: Optional[cute.Tensor],
        sparse_block_indices: Optional[cute.Tensor],
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.

        :param qk_tiled_mma: Tiled MMA for Q*K^T
        :type qk_tiled_mma: cute.TiledMma
        :param pv_tiled_mma: Tiled MMA for P*V
        :type pv_tiled_mma: cute.TiledMma
        :param tma_atom_q: TMA copy atom for query tensor
        :type tma_atom_q: cute.CopyAtom
        :param mQ_qdl: Partitioned query tensor
        :type mQ_qdl: cute.Tensor
        :param tma_atom_k: TMA copy atom for key tensor
        :type tma_atom_k: cute.CopyAtom
        :param mK_kdl: Partitioned key tensor
        :type mK_kdl: cute.Tensor
        :param tma_atom_v: TMA copy atom for value tensor
        :type tma_atom_v: cute.CopyAtom
        :param mV_dkl: Partitioned value tensor
        :type mV_dkl: cute.Tensor
        :param tma_atom_o: TMA copy atom for output tensor
        :type tma_atom_o: cute.CopyAtom
        :param mO_qdl: Partitioned output tensor
        :type mO_qdl: cute.Tensor
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_softmax: The scale factor for softmax
        :type scale_softmax: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param q_smem_layout_staged: Shared memory layout for query tensor
        :type q_smem_layout_staged: cute.ComposedLayout
        :param k_smem_layout_staged: Shared memory layout for key tensor
        :type k_smem_layout_staged: cute.ComposedLayout
        :param p_tmem_layout_staged: Tensor memory layout for probability matrix
        :type p_tmem_layout_staged: cute.ComposedLayout
        :param v_smem_layout_staged: Shared memory layout for value tensor
        :type v_smem_layout_staged: cute.ComposedLayout
        :param o_smem_layout_staged: Shared memory layout for output tensor
        :type o_smem_layout_staged: cute.ComposedLayout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
        ).make_participants()
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.softmax0_warp_ids)),
            barrier_storage=storage.mma_s0_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_s1_producer, mma_s1_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.softmax1_warp_ids)),
            barrier_storage=storage.mma_s1_mbar_ptr.data_ptr(),
        ).make_participants()
        s0_corr_producer, s0_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.softmax0_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.correction_warp_ids)),
            barrier_storage=storage.s0_corr_mbar_ptr.data_ptr(),
        ).make_participants()
        s1_corr_producer, s1_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.softmax1_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.correction_warp_ids)),
            barrier_storage=storage.s1_corr_mbar_ptr.data_ptr(),
        ).make_participants()
        corr_epi_producer, corr_epi_consumer = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.correction_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len([self.epilogue_warp_id])),
            barrier_storage=storage.corr_epi_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.correction_warp_ids)),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
        ).make_participants()
        s0_s1_sequence_producer, s0_s1_sequence_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.softmax0_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.softmax1_warp_ids)),
            barrier_storage=storage.s0_s1_sequence_mbar_ptr.data_ptr(),
        ).make_participants()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()

        #  Correction & Epilogue & tmem barrier init
        if warp_idx == self.empty_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.threads_per_warp
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        cute.arch.mbarrier_init_fence()

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)
        sO = storage.sO.get_tensor(o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner)
        qk_thr_mma = qk_tiled_mma.get_slice(0)  # default 1sm
        pv_thr_mma = pv_tiled_mma.get_slice(0)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(qk_acc_shape)
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)

        tStS0 = cute.make_tensor(tStS.iterator + self.tmem_s0_offset, tStS.layout)
        tStS1 = cute.make_tensor(tStS.iterator + self.tmem_s1_offset, tStS.layout)
        tOtO0 = cute.make_tensor(tOtO.iterator + self.tmem_o0_offset, tOtO.layout)
        tOtO1 = cute.make_tensor(tOtO.iterator + self.tmem_o1_offset, tOtO.layout)

        tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
        tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
        tOrP0 = cute.make_tensor(
            tOrP.iterator + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p0_offset,
            tOrP.layout,
        )
        tOrP1 = cute.make_tensor(
            tOrP.iterator + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p1_offset,
            tOrP.layout,
        )
        self.cta_sync_barrier.arrive_and_wait()
        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )
                if not continue_cond:
                    mQ_qdl_ = mQ_qdl
                    mK_kdl_ = mK_kdl
                    mV_dkl_ = mV_dkl
                    seqlen_k = mK_kdl.shape[0]
                    curr_block_coord_q = curr_block_coord
                    curr_block_coord_kv = curr_block_coord

                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mQ = (
                            cuseqlen_q,
                            0,
                            (0, 0),
                        )
                        mQ_qdl_ = cute.domain_offset(logical_offset_mQ, mQ_qdl)
                        curr_block_coord_q = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            curr_block_coord[2][0],
                        )

                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                        if cutlass.const_expr(cum_seqlen_q is not None):
                            logical_offset_mK = (
                                cuseqlen_k,
                                0,
                                (0, 0),
                            )
                            logical_offset_mV = (
                                0,
                                cuseqlen_k,
                                (0, 0),
                            )
                            mK_kdl_ = cute.domain_offset(logical_offset_mK, mK_kdl)
                            mV_dkl_ = cute.domain_offset(logical_offset_mV, mV_dkl)
                            curr_block_coord_kv = (
                                curr_block_coord[0],
                                curr_block_coord[1],
                                curr_block_coord[2][0],
                            )

                    # Local tile partition global tensors
                    # (bM, bK, loopM, loopK, loopL)
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord_q[2]]

                    gK_kdl = cute.flat_divide(mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2]))
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    tKgK = tKgK_kdl[None, None, 0, curr_block_coord_kv[2]]

                    gV_dkl = cute.flat_divide(mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2]))
                    tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tSgV_dkl, 0, 3),
                    )
                    tVgV = tVgV_dkl[None, 0, None, curr_block_coord_kv[2]]

                    # Q0
                    q0_coord = 2 * curr_block_coord_q[0]
                    q0_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q0_coord],
                        tQsQ[None, q0_handle.index],
                        tma_bar_ptr=q0_handle.barrier,
                    )
                    # K0
                    if cutlass.const_expr(sparse_block_count is not None):
                        head_coord = curr_block_coord[2][0]
                        curr_sparse_block_count = sparse_block_count[batch_coord, head_coord, curr_block_coord[0]]
                        curr_sparse_block_indices = sparse_block_indices[batch_coord, head_coord, curr_block_coord[0], None]
                        kv_coord = curr_sparse_block_indices[0]
                    else:
                        seqlen_kv_loop_start = fmha_utils.FusedMask.get_trip_start(
                            self.mask_type,
                            curr_block_coord,
                            self.cta_tiler,
                            seqlen_q,
                            seqlen_k,
                            window_size_left,
                        )
                        kv_coord = seqlen_kv_loop_start
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, kv_coord],
                        tKsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    # Q1
                    q1_coord = q0_coord + 1
                    q1_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q1_coord],
                        tQsQ[None, q1_handle.index],
                        tma_bar_ptr=q1_handle.barrier,
                    )
                    # V0
                    v_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, kv_coord],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=v_handle.barrier,
                    )
                    if cutlass.const_expr(sparse_block_count is None):
                        kv_coord += 1

                    if cutlass.const_expr(sparse_block_count is not None):
                        seqlen_kv_loop_steps = curr_sparse_block_count - 1
                    else:
                        seqlen_kv_loop_steps = (
                            fmha_utils.FusedMask.get_trip_count(
                                self.mask_type,
                                curr_block_coord,
                                self.cta_tiler,
                                seqlen_q,
                                seqlen_k,
                                window_size_left,
                                window_size_right,
                            )
                            - 1
                        )
                    for i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # Ki
                        if cutlass.const_expr(sparse_block_count is not None):
                            kv_coord = curr_sparse_block_indices[i + 1]
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_coord],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        # Vi
                        v_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, kv_coord],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=v_handle.barrier,
                        )
                        if cutlass.const_expr(sparse_block_count is None):
                            kv_coord += 1
                    # End of seqlen_kv loop

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                # End of persistent scheduler loop

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            self.tmem_alloc_barrier.arrive_and_wait()
            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                seqlen_q = mQ_qdl.shape[0]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )

                if not continue_cond:
                    seqlen_k = mK_kdl.shape[0]
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    # GEMM_QK00 (Q0 * K0 -> S0)
                    # 1. wait for Q0
                    q0_handle = load_q_consumer.wait_and_advance()
                    tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                    # 2. wait for K0
                    k_handle = load_kv_consumer.wait_and_advance()
                    tSrK0 = tSrK[None, None, None, k_handle.index]
                    # 3. acquire empty S0 buffer
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    # 4. gemm
                    num_kphases = cute.size(tSrQ0, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord = (None, None, kphase_idx)
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            qk_tiled_mma,
                            tStS0,
                            tSrQ0[kphase_coord],
                            tSrK0[kphase_coord],
                            tStS0,
                        )
                    # 5. release S0
                    s0_handle.commit()
                    # End of GEMM (Q0 * K0 -> S0)

                    # GEMM_QK10 (Q1 * K0 -> S1), K0 is ready in GEMM_QK00
                    # 1. wait for Q1
                    q1_handle = load_q_consumer.wait_and_advance()
                    tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                    # 2. acquire empty S1
                    s1_handle = mma_s1_producer.acquire_and_advance()
                    # 3. gemm
                    num_kphases = cute.size(tSrQ1, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord = (None, None, kphase_idx)
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            qk_tiled_mma,
                            tStS1,
                            tSrQ1[kphase_coord],
                            tSrK0[kphase_coord],
                            tStS1,
                        )
                    # 4. release S1
                    s1_handle.commit()
                    # 5. release K0
                    k_handle.release()
                    # End of GEMM (Q1 * K0 -> S1)
                    # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                    # so we need to release them after the seqlen_kv loop

                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    v_handle = load_kv_consumer.wait_and_advance()
                    tOrVi = tOrV[None, None, None, v_handle.index]
                    # 2. acquire corrected O0_partial
                    # Note: acquire corr first to take it out of the critical
                    # path since softmax takes longer
                    o0_handle = mma_corr_producer.acquire_and_advance()
                    # 3. acquire P0
                    # this acquire returns the ownership of all of S0 to the mma warp
                    # including the P0 part (inplaced in S0)
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    # 4. gemm
                    num_kphases = cute.size(tOrP0, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord = (None, None, kphase_idx)
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO0,
                            tOrP0[kphase_coord],
                            tOrVi[kphase_coord],
                            tOtO0,
                        )
                    # 5. release accumulated O0_partial
                    o0_handle.commit()
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                    if cutlass.const_expr(sparse_block_count is not None):
                        head_coord = curr_block_coord[2][0]
                        seqlen_kv_loop_steps = sparse_block_count[batch_coord, head_coord, curr_block_coord[0]] - 1
                    else:
                        seqlen_kv_loop_steps = (
                            fmha_utils.FusedMask.get_trip_count(
                                self.mask_type,
                                curr_block_coord,
                                self.cta_tiler,
                                seqlen_q,
                                seqlen_k,
                                window_size_left,
                                window_size_right,
                            )
                            - 1
                        )

                    # O1 hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                    pv_whether_acc = False
                    for i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        k_handle = load_kv_consumer.wait_and_advance()
                        tSrKi = tSrK[None, None, None, k_handle.index]
                        # 2. gemm
                        inner_num_kphases = cute.size(tSrQ0, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                            cute.gemm(
                                qk_tiled_mma,
                                tStS0,
                                tSrQ0[kphase_coord],
                                tSrKi[kphase_coord],
                                tStS0,
                            )
                        # 3. release S0
                        s0_handle.commit()
                        # End of GEMM_QK0i (Q0 * Ki -> S0)

                        # GEMM_PV1(i-1) (P1 * V(i-1) -> O1_partial), V(i-1) is ready in GEMM_PV0(i-1)
                        # 1. acquire corrected O1_partial
                        o1_handle = mma_corr_producer.acquire_and_advance()
                        # 2. acquire P1
                        s1_handle = mma_s1_producer.acquire_and_advance()
                        # 3. gemm
                        inner_num_kphases = cute.size(tOrP0, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            cute.gemm(
                                pv_tiled_mma,
                                tOtO1,
                                tOrP1[kphase_coord],
                                tOrVi[kphase_coord],
                                tOtO1,
                            )
                            pv_whether_acc = True
                        # 4. release accumulated O1_partial
                        o1_handle.commit()
                        # 5. release V(i-1)
                        v_handle.release()
                        # End of GEMM_PV1(i-1) (P1 * V(i-1) -> O1_partial)

                        # GEMM_QK1i (Q1 * Ki -> S1), Q1 is ready in GEMM_QK10; Ki is ready in GEMM_QK0i
                        # 1. gemm
                        inner_num_kphases = cute.size(tSrQ1, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                            cute.gemm(
                                qk_tiled_mma,
                                tStS1,
                                tSrQ1[kphase_coord],
                                tSrKi[kphase_coord],
                                tStS1,
                            )
                        s1_handle.commit()
                        # 2. release Ki
                        k_handle.release()
                        # End of GEMM_QK1i (Q1 * Ki -> S1)

                        # GEMM_PV0i (P0 * Vi -> O0_partial)
                        # 1. wait for Vi
                        v_handle = load_kv_consumer.wait_and_advance()
                        tOrVi = tOrV[None, None, None, v_handle.index]
                        # 2. acquire corrected O0_partial
                        o0_handle = mma_corr_producer.acquire_and_advance()
                        # 3. acquire P0
                        s0_handle = mma_s0_producer.acquire_and_advance()
                        # 4. gemm
                        inner_num_kphases = cute.size(tOrP0, mode=[2])
                        for kphase_idx in cutlass.range(inner_num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(
                                pv_tiled_mma,
                                tOtO0,
                                tOrP0[kphase_coord],
                                tOrVi[kphase_coord],
                                tOtO0,
                            )
                        # 5. release accumulated O0_partial
                        o0_handle.commit()
                        # End of GEMM_PV0i (P0 * Vi -> O0_partial)
                    # End of seqlen_kv loop

                    # release Q0 & Q1
                    q0_handle.release()
                    q1_handle.release()

                    # GEMM_PV1(i_end) (P1 * Vi_end -> O1)
                    # 1. acquire corrected O1_partial
                    o1_handle = mma_corr_producer.acquire_and_advance()
                    # 2. acquire P1
                    s1_handle = mma_s1_producer.acquire_and_advance()
                    # 3. gemm
                    num_kphases = cute.size(tOrP1, mode=[2])
                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                        kphase_coord = (None, None, kphase_idx)
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO1,
                            tOrP1[kphase_coord],
                            tOrVi[kphase_coord],
                            tOtO1,
                        )
                        pv_whether_acc = True
                    # 4. commit accumulated O1
                    o1_handle.commit()
                    # 5. release Vi_end
                    v_handle.release()
                    # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

                    # Commit S0 and S1
                    s0_handle.commit()
                    s1_handle.commit()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop

            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )
                if not continue_cond:
                    curr_block_coord_o = curr_block_coord
                    mO_qdl_ = mO_qdl
                    mO_gmem_ = mO_gmem
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        logical_offset_mO = (
                            cuseqlen_q,
                            0,
                            (0, 0),
                        )
                        mO_qdl_ = cute.domain_offset(logical_offset_mO, mO_qdl_)
                        mO_gmem_ = cute.domain_offset(logical_offset_mO, mO_gmem_)
                        curr_block_coord_o = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            curr_block_coord[2][0],
                        )

                    o0_coord = 2 * curr_block_coord_o[0]
                    o1_coord = o0_coord + 1
                    gO_qdl = cute.flat_divide(mO_qdl_, cute.select(self.pv_mma_tiler, mode=[0, 1]))
                    gO = gO_qdl[None, None, None, 0, curr_block_coord_o[2]]
                    tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_o,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sO, 0, 2),
                        cute.group_modes(gO, 0, 2),
                    )

                    # O0 O1 using the same pipeline
                    # wait from corr, issue tma store on smem
                    # O0
                    # 1. wait for O0 final
                    o0_handle = corr_epi_consumer.wait_and_advance()
                    # 2. copy O0 to gmem
                    o0_row = o0_coord * self.epi_tile[0]
                    o0_use_tma = o0_row + self.epi_tile[0] <= seqlen_q
                    if o0_use_tma:
                        cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])
                        cute.arch.cp_async_bulk_commit_group()
                    else:
                        self.store_o_tail(sO, mO_gmem_, o0_row, seqlen_q, curr_block_coord_o[2], 0)
                    # O1
                    # 1. wait for O1 final
                    o1_handle = corr_epi_consumer.wait_and_advance()
                    # 2. copy O1 to gmem
                    o1_row = o1_coord * self.epi_tile[0]
                    o1_use_tma = o1_row + self.epi_tile[0] <= seqlen_q
                    if o1_use_tma:
                        cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])
                        cute.arch.cp_async_bulk_commit_group()
                    else:
                        self.store_o_tail(sO, mO_gmem_, o1_row, seqlen_q, curr_block_coord_o[2], 1)

                    # Ensure O0 buffer is ready to be released
                    if o0_use_tma:
                        if o1_use_tma:
                            cute.arch.cp_async_bulk_wait_group(1, read=True)
                        else:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                    o0_handle.release()
                    # Ensure O1 buffer is ready to be released
                    if o1_use_tma:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    o1_handle.release()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax0
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.softmax1_warp_ids[0]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)

            self.softmax(
                stage=0,
                seqlen_k=mK_kdl.shape[0],
                seqlen_q=mQ_qdl.shape[0],
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
                scale_softmax_log2=scale_softmax_log2,
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS0,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                sparse_block_count=sparse_block_count,
                sparse_block_indices=sparse_block_indices,
                mma_si_consumer=mma_s0_consumer,
                si_corr_producer=s0_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax1
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)

            self.softmax(
                stage=1,
                seqlen_k=mK_kdl.shape[0],
                seqlen_q=mQ_qdl.shape[0],
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
                scale_softmax_log2=scale_softmax_log2,
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS1,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                sparse_block_count=sparse_block_count,
                sparse_block_indices=sparse_block_indices,
                mma_si_consumer=mma_s1_consumer,
                si_corr_producer=s1_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)

            cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
            tScS = qk_thr_mma.partition_C(cS)

            tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))

            tStS_vec0 = cute.make_tensor(tStS.iterator + self.tmem_vec0_offset, tStS_vec_layout)
            tStS_vec1 = cute.make_tensor(tStS.iterator + self.tmem_vec1_offset, tStS_vec_layout)

            tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
            tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

            tmem_load_v_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
                self.qk_acc_dtype,
            )

            tiled_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStS_vec0)
            thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
            thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(thread_idx)

            tTMEM_LOAD_VECtS0 = thr_tmem_load_vec.partition_S(tStS_vec0)
            tTMEM_LOAD_VECtS1 = thr_tmem_load_vec.partition_S(tStS_vec1)
            tTMEM_LOAD_VECcS = thr_tmem_load_vec.partition_D(tScS_vec)

            tile_sched = fmha_utils.create_fmha_static_tile_scheduler(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                curr_block_coord_lse = curr_block_coord
                batch_coord = curr_block_coord[2][1]
                seqlen_k = mK_kdl.shape[0]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    # for varlen LSE, batch == 1
                    curr_block_coord_lse = (
                        curr_block_coord[0],
                        curr_block_coord[1],
                        (curr_block_coord[2][0], 0),
                    )
                    continue_cond = not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )

                if not continue_cond:
                    row_idx = curr_block_coord[0] * self.cta_tiler[0] + tTMEM_LOAD_VECcS[0][0]
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    # Ignore first signal: the first O partial needs no correction.
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    vec0_handle.release()
                    vec1_handle = s1_corr_consumer.wait_and_advance()

                    if cutlass.const_expr(sparse_block_count is not None):
                        head_coord = curr_block_coord[2][0]
                        seqlen_kv_loop_steps = sparse_block_count[batch_coord, head_coord, curr_block_coord[0]] - 1
                    else:
                        seqlen_kv_loop_steps = (
                            fmha_utils.FusedMask.get_trip_count(
                                self.mask_type,
                                curr_block_coord,
                                self.cta_tiler,
                                seqlen_q,
                                seqlen_k,
                                window_size_left,
                                window_size_right,
                            )
                            - 1
                        )
                    for i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # wait for vec0 (row_wise current max & previous max)
                        vec0_handle = s0_corr_consumer.wait_and_advance()
                        # wait for o0
                        o0_handle = mma_corr_consumer.wait_and_advance()
                        if cutlass.const_expr(not self.zero_reference_max):
                            tTMEM_LOAD_VECrS = cute.make_rmem_tensor(tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype)
                            cute.copy(
                                tiled_tmem_load_vec,
                                tTMEM_LOAD_VECtS0,
                                tTMEM_LOAD_VECrS,
                            )
                            scale_ = Float32(0.0)
                            scale = Float32(1.0)
                            warp_needs_rescale = cute.arch.vote_any_sync(tTMEM_LOAD_VECrS[0] != tTMEM_LOAD_VECrS[1])
                            if warp_needs_rescale:
                                scale_ = scale_softmax_log2 * (tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1])
                                scale = cute.math.exp2(scale_, fastmath=True)
                                self.correction_rescale(pv_thr_mma, tOtO0, scale)
                        # release vec1 & o0
                        vec1_handle.release()
                        if cutlass.const_expr(not self.zero_reference_max):
                            cute.arch.fence_view_async_tmem_store()
                        o0_handle.release()

                        # wait for vec1 (row_wise current max & previous max)
                        vec1_handle = s1_corr_consumer.wait_and_advance()
                        o1_handle = mma_corr_consumer.wait_and_advance()
                        if cutlass.const_expr(not self.zero_reference_max):
                            cute.copy(
                                tiled_tmem_load_vec,
                                tTMEM_LOAD_VECtS1,
                                tTMEM_LOAD_VECrS,
                            )
                            scale_ = Float32(0.0)
                            scale = Float32(1.0)
                            warp_needs_rescale = cute.arch.vote_any_sync(tTMEM_LOAD_VECrS[0] != tTMEM_LOAD_VECrS[1])
                            if warp_needs_rescale:
                                scale_ = scale_softmax_log2 * (tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1])
                                scale = cute.math.exp2(scale_, fastmath=True)
                                self.correction_rescale(pv_thr_mma, tOtO1, scale)
                        vec0_handle.release()
                        if cutlass.const_expr(not self.zero_reference_max):
                            cute.arch.fence_view_async_tmem_store()
                        o1_handle.release()
                    # End of seqlen_corr_loop_steps
                    vec1_handle.release()

                    # wait for vec0 (row_wise global sum)
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    tTMEM_LOAD_VECrS = cute.make_rmem_tensor(tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype)
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECrS)
                    cute.arch.fence_view_async_tmem_load()
                    vec0_handle.release()
                    # wait for o0
                    o0_handle = mma_corr_consumer.wait_and_advance()
                    o0_final_handle = corr_epi_producer.acquire_and_advance()
                    self.correction_epilog(
                        pv_thr_mma,
                        tOtO0,
                        mLSE,
                        tTMEM_LOAD_VECrS,
                        row_idx,
                        cuseqlen_q,
                        seqlen_q,
                        curr_block_coord_lse,
                        scale_softmax,
                        scale_output / tTMEM_LOAD_VECrS[0],
                        sO[None, None, 0],
                    )
                    o0_handle.release()
                    o0_final_handle.commit()

                    # wait for vec1 (row_wise global sum)
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECrS)
                    cute.arch.fence_view_async_tmem_load()
                    vec1_handle.release()
                    # wait for o1
                    o1_handle = mma_corr_consumer.wait_and_advance()
                    o1_final_handle = corr_epi_producer.acquire_and_advance()
                    row_idx += self.qk_mma_tiler[0]
                    self.correction_epilog(
                        pv_thr_mma,
                        tOtO1,
                        mLSE,
                        tTMEM_LOAD_VECrS,
                        row_idx,
                        cuseqlen_q,
                        seqlen_q,
                        curr_block_coord_lse,
                        scale_softmax,
                        scale_output / tTMEM_LOAD_VECrS[0],
                        sO[None, None, 1],
                    )
                    o1_handle.release()
                    o1_final_handle.commit()
                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
        return

    @cute.jit
    def store_o_tail(
        self,
        sO: cute.Tensor,
        mO_gmem: cute.Tensor,
        row_start: Int32,
        seqlen_q: Int32,
        head_coord: cute.Coord,
        stage: cutlass.Constexpr,
    ):
        """Predicated O tile store for packed-varlen sequence tails."""
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = tidx % self.threads_per_warp
        valid_rows = seqlen_q - row_start
        if valid_rows > self.epi_tile[0]:
            valid_rows = self.epi_tile[0]
        if valid_rows > 0:
            valid_elems = valid_rows * self.head_dim
            for elem_idx in cutlass.range(lane_idx, valid_elems, self.threads_per_warp, unroll=1):
                row = elem_idx // self.head_dim
                col = elem_idx - row * self.head_dim
                mO_gmem[row_start + row, col, head_coord] = sO[row, col, stage]

    @cute.jit
    def softmax_step(
        self,
        stage: int,
        need_apply_mask: bool,
        iter_args: tuple,
        value_args: tuple,
        pipeline_args: tuple,
        atom_args: tuple,
        tensor_args: tuple,
    ) -> Tuple[
        Float32,
        Float32,
        Float32,
        pipeline.PipelineProducer.ImmutableResourceHandle,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
    ]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages

        :param stage: Processing stage (0 for first half, 1 for second half)
        :type stage: int
        :param need_apply_mask: Whether to apply attention masking
        :type need_apply_mask: bool
        :param iter_args: Tuple containing the counting tensor, row_max, row_sum, and vector buffer's handle for current iteration
        :type iter_args: tuple
        :param value_args: Tuple containing seqlen_k, seqlen_q, and scale_softmax_log2
        :type value_args: tuple
        :param pipeline_args: Tuple containing pipeline related arguments for MMA, correction, and sequence synchronization
        :type pipeline_args: tuple
        :param atom_args: Tuple containing mma & copy atoms
        :type atom_args: tuple
        :param tensor_args: Tuple containing softmax related tensors
        :type tensor_args: tuple
        :param fused_mask: Compute trip counts and apply masking for attention blocks
        :type fused_mask: fmha_utils.FusedMask
        :return: Updated state values (row_max, row_sum, and pipeline related arguments)
        :rtype: tuple
        """
        cS, row_max, observed_row_max, row_sum, update_row_max, vec_i_handle = iter_args
        seqlen_k, seqlen_q, scale_softmax_log2, window_size_left, window_size_right = value_args
        (
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        ) = pipeline_args
        (
            qk_thr_mma,
            tiled_tmem_load,
            tiled_tmem_store,
            tiled_tmem_store_vec,
            thr_tmem_load,
            thr_tmem_store,
            thr_tmem_store_vec,
        ) = atom_args
        (
            tTMEM_LOADtS,
            tTMEM_STORE_VECtS,
            tTMEM_STOREtS_x4,
        ) = tensor_args

        tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.o_dtype.width
        tScS = qk_thr_mma.partition_C(cS)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

        tScS_P_layout = cute.composition(tScS.layout, cute.make_layout((128, tilePlikeFP32)))
        tScS_P = cute.make_tensor(tScS.iterator, tScS_P_layout)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P)

        # Wait for Si
        si_handle = mma_si_consumer.wait_and_advance()
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)
        if need_apply_mask:
            fmha_utils.FusedMask.apply_mask(
                self.mask_type,
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )

        old_row_max = row_max
        if cutlass.const_expr(self.zero_reference_max):
            observed_row_max = row_max
            row_max_safe = row_max
        else:
            observed_row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, observed_row_max, 0)
            if cutlass.const_expr(self.max_update_interval == 1):
                row_max = observed_row_max
            else:
                if update_row_max or row_max == -cutlass.Float32.inf:
                    row_max = observed_row_max
            row_max_safe = row_max
            if row_max == -cutlass.Float32.inf:
                row_max_safe = 0.0
            tTMEM_STORE_VECrS = cute.make_rmem_tensor(tTMEM_STORE_VECcS.shape, self.qk_acc_dtype)
            tTMEM_STORE_VECrS[0] = old_row_max
            tTMEM_STORE_VECrS[1] = row_max_safe
            cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
            cute.arch.fence_view_async_tmem_store()
        # Notify correction wg that row_max is ready.
        vec_i_handle.commit()

        tTMEM_STORErS_x4 = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.qk_acc_dtype)
        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS_x4.iterator, dtype=self.q_dtype),
            tTMEM_LOADrS.layout,
        )

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale + self.softmax_prescale_log2

        # Sequence barrier wait
        if cutlass.const_expr(stage == 0):
            sequence_producer_handle = s0_s1_sequence_producer.acquire_and_advance()
        else:
            sequence_consumer_handle = s0_s1_sequence_consumer.wait_and_advance()
        frg_cnt = 4
        frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        tTMEM_STORErS_x4_e_frg = cute.logical_divide(tTMEM_STORErS_x4_e, cute.make_layout(frg_tile))
        if cutlass.const_expr(not self.zero_reference_max):
            acc_scale_ = scale * (old_row_max - row_max_safe)
            acc_scale = 1.0
            if old_row_max != row_max_safe:
                acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            row_sum *= acc_scale
        for j in range(frg_cnt):
            for k in cutlass.range(cute.size(tTMEM_LOADrS_frg, mode=[0]), vectorize=True):
                tTMEM_LOADrS_frg[k, j] = tTMEM_LOADrS_frg[k, j] * scale + minus_row_max_scale
                tTMEM_LOADrS_frg[k, j] = cute.math.exp2(tTMEM_LOADrS_frg[k, j], fastmath=True)

            s_vec = tTMEM_LOADrS_frg[None, j].load()
            row_sum = s_vec.reduce(cute.ReductionOp.ADD, row_sum, 0)
            tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))
        # Sequence barrier arrive
        if cutlass.const_expr(stage == 0):
            sequence_producer_handle.commit()
        else:
            sequence_consumer_handle.release()
        cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4)
        cute.arch.fence_view_async_tmem_store()
        # Notify tensor core warp that softmax(S->P) is ready
        si_handle.release()

        vec_i_handle = si_corr_producer.acquire_and_advance()

        return (
            row_max,
            observed_row_max,
            row_sum,
            vec_i_handle,
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        )

    # For both softmax0 and softmax1 warp group
    @cute.jit
    def softmax(
        self,
        stage: int,
        seqlen_k: Int32,
        seqlen_q: Int32,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        qk_thr_mma: cute.ThrMma,
        tStS: cute.Tensor,
        tStSi: cute.Tensor,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sparse_block_count: Optional[cute.Tensor],
        sparse_block_indices: Optional[cute.Tensor],
        mma_si_consumer: pipeline.PipelineConsumer,
        si_corr_producer: pipeline.PipelineProducer,
        s0_s1_sequence_consumer: pipeline.PipelineConsumer,
        s0_s1_sequence_producer: pipeline.PipelineProducer,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.

        :param stage: Processing stage (0 for first half, 1 for second half of attention matrix)
        :type stage: int
        :param seqlen_k: Length of the key sequence
        :type seqlen_k: Int32
        :param seqlen_q: Length of the query sequence
        :type seqlen_q: Int32
        :param cum_seqlen_q: Cumulative sequence lengths for queries
        :type cum_seqlen_q: cute.Tensor | None
        :param cum_seqlen_k: Cumulative sequence lengths for keys
        :type cum_seqlen_k: cute.Tensor | None
        :param scale_softmax_log2: Log2 scale factor for softmax operation
        :type scale_softmax_log2: Float32
        :param qk_thr_mma: Thread MMA operation for QK matrix multiplication
        :type qk_thr_mma: cute.ThrMma
        :param tStS: Shared tensor for softmax input/output
        :type tStS: cute.Tensor
        :param tStSi: Input tensor containing attention scores
        :type tStSi: cute.Tensor
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param mma_si_pipeline: Pipeline for synchronizing with MMA operations
        :type mma_si_pipeline: pipeline.PipelineAsync
        :param si_corr_pipeline: Pipeline for synchronizing with correction operations
        :type si_corr_pipeline: pipeline.PipelineAsync
        :param s0_s1_sequence_pipeline: Pipeline for synchronizing between stage 0 and 1
        :type s0_s1_sequence_pipeline: pipeline.PipelineAsync
        :param tile_sched_params: Parameters for tile scheduling
        :type tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams
        :param fused_mask: Compute trip counts and apply masking for attention blocks
        :type fused_mask: fmha_utils.FusedMask
        """
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)))

        cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.o_dtype.width
        tScS = qk_thr_mma.partition_C(cS_base)
        tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
        tmem_vec_offset = self.tmem_vec0_offset if stage == 0 else self.tmem_vec1_offset
        tStS_vec = cute.make_tensor(tStS.iterator + tmem_vec_offset, tStS_vec_layout)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        tStS_P_layout = cute.composition(tStS.layout, cute.make_layout((128, tilePlikeFP32)))
        tmem_p_offset = self.tmem_p0_offset if stage == 0 else self.tmem_p1_offset
        tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi)
        thread_idx = tidx % (self.threads_per_warp * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)))
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStSi)
        tmem_store_vec_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_vec = tcgen05.make_tmem_copy(tmem_store_vec_atom, tStS_vec)
        thr_tmem_store_vec = tiled_tmem_store_vec.get_slice(thread_idx)
        tTMEM_STORE_VECtS = thr_tmem_store_vec.partition_D(tStS_vec)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P)

        tile_sched = fmha_utils.create_fmha_static_tile_scheduler(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            seqlen_k_ = seqlen_k
            seqlen_q_ = seqlen_q
            continue_cond = False
            cuseqlen_q = Int32(0)
            seqlen_q_ = seqlen_q
            if cutlass.const_expr(cum_seqlen_q is not None):
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q_ = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                    self.cta_tiler[0],
                    curr_block_coord[0],
                    seqlen_q_,
                )

            if not continue_cond:
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k_ = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                if cutlass.const_expr(self.zero_reference_max):
                    row_max = Float32(0.0)
                    observed_row_max = Float32(0.0)
                else:
                    row_max = -Float32.inf
                    observed_row_max = -Float32.inf
                row_sum = 0.0
                value_args = (
                    seqlen_k_,
                    seqlen_q_,
                    scale_softmax_log2,
                    window_size_left,
                    window_size_right,
                )
                atom_args = (
                    qk_thr_mma,
                    tiled_tmem_load,
                    tiled_tmem_store,
                    tiled_tmem_store_vec,
                    thr_tmem_load,
                    thr_tmem_store,
                    thr_tmem_store_vec,
                )
                tensor_args = (
                    tTMEM_LOADtS,
                    tTMEM_STORE_VECtS,
                    tTMEM_STOREtS_x4,
                )

                logical_offset = (
                    curr_block_coord[0] * self.cta_tiler[0] + stage * self.qk_mma_tiler[0],
                    0,
                )
                cS = cute.domain_offset(logical_offset, cS_base)
                vec_i_handle = si_corr_producer.acquire_and_advance()

                sparse_trailing_mask_count = Int32(0)
                if cutlass.const_expr(sparse_block_count is not None):
                    head_coord = curr_block_coord[2][0]
                    curr_sparse_block_indices = sparse_block_indices[batch_coord, head_coord, curr_block_coord[0], None]
                    start_count = Int32(0)
                    sparse_count = sparse_block_count[batch_coord, head_coord, curr_block_coord[0]]
                    leading_mask_count = Int32(0)
                    if seqlen_k_ % self.qk_mma_tiler[1] != 0:
                        last_sparse_block = curr_sparse_block_indices[sparse_count - 1]
                        if last_sparse_block == (cute.ceil_div(seqlen_k_, self.qk_mma_tiler[1]) - 1):
                            sparse_trailing_mask_count = Int32(1)
                else:
                    start_count = fmha_utils.FusedMask.get_trip_start(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q_,
                        seqlen_k_,
                        window_size_left,
                    )

                    leading_mask_count = fmha_utils.FusedMask.get_masked_leading_count(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q_,
                        seqlen_k_,
                        window_size_left,
                        window_size_right,
                    )
                for i in cutlass.range(start_count, start_count + leading_mask_count, 1, unroll=1):
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    update_row_max = True
                    if cutlass.const_expr(not self.zero_reference_max and self.max_update_interval > 1):
                        update_row_max = (i & (self.max_update_interval - 1)) == 0
                    iter_args = (
                        cS_iter,
                        row_max,
                        observed_row_max,
                        row_sum,
                        update_row_max,
                        vec_i_handle,
                    )
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        observed_row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        True,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                    )
                if cutlass.const_expr(sparse_block_count is not None):
                    unmask_count = sparse_count - sparse_trailing_mask_count
                else:
                    unmask_count = fmha_utils.FusedMask.get_unmasked_trip_count(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q_,
                        seqlen_k_,
                        window_size_left,
                        window_size_right,
                    )
                for i in cutlass.range(
                    start_count + leading_mask_count,
                    start_count + leading_mask_count + unmask_count,
                    1,
                    unroll=1,
                ):
                    n_block = i
                    if cutlass.const_expr(sparse_block_count is not None):
                        n_block = curr_sparse_block_indices[i]
                    cS_iter = cute.domain_offset((0, n_block * self.qk_mma_tiler[1]), cS)
                    update_row_max = True
                    if cutlass.const_expr(not self.zero_reference_max and self.max_update_interval > 1):
                        update_row_max = (i & (self.max_update_interval - 1)) == 0
                    iter_args = (
                        cS_iter,
                        row_max,
                        observed_row_max,
                        row_sum,
                        update_row_max,
                        vec_i_handle,
                    )
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        observed_row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        False,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                    )
                if cutlass.const_expr(sparse_block_count is not None):
                    trailing_mask_count = sparse_trailing_mask_count
                else:
                    trailing_mask_count = fmha_utils.FusedMask.get_masked_trailing_count(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q_,
                        seqlen_k_,
                        window_size_left,
                        window_size_right,
                    )

                for i in cutlass.range(
                    start_count + leading_mask_count + unmask_count,
                    start_count + leading_mask_count + unmask_count + trailing_mask_count,
                    1,
                    unroll=1,
                ):
                    n_block = i
                    if cutlass.const_expr(sparse_block_count is not None):
                        n_block = curr_sparse_block_indices[i]
                    cS_iter = cute.domain_offset((0, n_block * self.qk_mma_tiler[1]), cS)
                    update_row_max = True
                    if cutlass.const_expr(not self.zero_reference_max and self.max_update_interval > 1):
                        update_row_max = (i & (self.max_update_interval - 1)) == 0
                    iter_args = (
                        cS_iter,
                        row_max,
                        observed_row_max,
                        row_sum,
                        update_row_max,
                        vec_i_handle,
                    )
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        observed_row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        True,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                    )
                si_handle = mma_si_consumer.wait_and_advance()
                tTMEM_STORE_VECrS = cute.make_rmem_tensor(tTMEM_STORE_VECcS.shape, self.qk_acc_dtype)
                tTMEM_STORE_VECrS[0] = row_sum
                tTMEM_STORE_VECrS[1] = row_max
                cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
                cute.arch.fence_view_async_tmem_store()
                vec_i_handle.commit()
                si_corr_producer.acquire()
                # Empty step to sync against pipe s
                si_handle.release()

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor representing partial attention output to be rescaled
        :type tOtO: cute.Tensor
        :param scale: Scaling factor to apply to the partial results
        :type scale: Float32
        """
        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)
        tOcO = thr_mma.partition_C(cO)

        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )

        tOtO_i_layout = cute.composition(tOtO.layout, cute.make_layout((128, corr_tile_size)))
        tOcO_i_layout = cute.composition(tOcO.layout, cute.make_layout((128, corr_tile_size)))

        tOtO_i = cute.make_tensor(tOtO.iterator, tOtO_i_layout)
        tOcO_i = cute.make_tensor(tOcO.iterator, tOcO_i_layout)

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i)

        tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i)

        tTMrO = cute.make_rmem_tensor((tTMEM_LOADcO.shape, 128 // corr_tile_size), self.pv_acc_dtype)
        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMrO_i_ = tTMrO[None, i]
            tTMrO_i_layout = cute.composition(tTMrO_i_.layout, cute.make_layout(tTMrO.shape[0]))
            tTMrO_i = cute.make_tensor(tTMrO_i_.iterator, tTMrO_i_layout)
            tTMEM_LOADtO_i = cute.make_tensor(tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout)
            tTMEM_STOREtO_i = cute.make_tensor(tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout)

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i)
            for j in cutlass.range(cute.size(tTMrO_i), vectorize=True):
                tTMrO_i[j] = tTMrO_i[j] * scale
            cute.copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i)

    @cute.jit
    def correction_epilog(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tTMEM_LOAD_VECrS: cute.Tensor,
        row_idx: Int32,
        cuseqlen_q: Int32,
        seqlen_q: Int32,
        blk_coord: Int32,
        scale_softmax: Float32,
        scale: Float32,
        sO: cute.Tensor,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilog function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param mLSE: Tensor containing log-sum-exp values for LSE calculation
        :type mLSE: cute.Tensor | None
        :param tTMEM_LOAD_VECrS: Tensor containing row sum and max values for softmax calculation
        :type tTMEM_LOAD_VECrS: cute.Tensor
        :param row_idx: Index of the current row being processed
        :type row_idx: Int32
        :param cuseqlen_q: Cumulative sequence length of the current query
        :type cuseqlen_q: Int32
        :param seqlen_q: Sequence length of the current query
        :type seqlen_q: Int32
        :param blk_coord: Coordinate of the current block being processed
        :type blk_coord: Int32
        :param scale_softmax: Scaling factor for softmax calculation
        :type scale_softmax: Float32
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cO)

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((128, corr_tile_size)))
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        smem_copy_atom = sm100_utils.get_smem_store_op(self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load)
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tTMEM_LOADoO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
            tTMEM_LOADsO_i = tTMEM_LOADsO[None, 0, 0, i]
            tTMrO = cute.make_rmem_tensor(tTMEM_LOADoO[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
            for j in range(cute.size(tTMrO), vectorize=True):
                tTMrO[j] = tTMrO[j] * scale
            tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
            o_vec = tTMrO.load()
            tSMrO.store(o_vec.to(self.o_dtype))
            cute.copy(tiled_smem_store, tSMrO, tTMEM_LOADsO_i)

        if cutlass.const_expr(mLSE is not None):
            scaled_tmp = scale_softmax * tTMEM_LOAD_VECrS[1]
            lse = cute.math.log(tTMEM_LOAD_VECrS[0], fastmath=True) + scaled_tmp - self.softmax_prescale_ln
            if row_idx < seqlen_q:
                mLSE[row_idx + cuseqlen_q, blk_coord[2]] = lse

        # fence view async shared
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )


def run(
    q_shape: Union[Tuple[int, int, int, int], Tuple[int, Tuple[int, ...], int, int]],
    k_shape: Union[Tuple[int, int, int, int], Tuple[int, Tuple[int, ...], int, int]],
    in_dtype: Type[cutlass.Numeric],
    out_dtype: Type[cutlass.Numeric],
    qk_acc_dtype: Type[cutlass.Numeric],
    pv_acc_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    is_persistent: bool,
    is_causal: bool,
    bottom_right_align: bool,
    lse_calculation: bool,
    window_size: Tuple[int, int],
    scale_q: float,
    scale_k: float,
    scale_v: float,
    inv_scale_o: float,
    scale_softmax: float,
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    output_dir: str = "./fmha_aot_artifacts",
    export_only: bool = False,
    file_name: str = "fmha",
    function_prefix: str = "fmha",
    vit_mode: bool = False,
    **kwargs,
):
    """Execute Fused Multi-Head Attention (FMHA) on Blackwell architecture and validate results.

    This function creates random input tensors for query, key, and value, then performs the
    complete FMHA computation pipeline. It supports configurable data types, tiling parameters,
    and various attention masking options. Results can be validated against a PyTorch reference
    implementation or run multiple times for performance measurement.

    The implementation leverages specialized tensor memory operations and efficient math
    operations optimized for Blackwell architecture, including pipelined computation stages
    for maximum throughput.

    :param q_shape: Query tensor shape (B, S_q, H, D) where B=batch size, S_q=query sequence length,
                    H=number of heads, D=head dimension.
                    If S_q is a tuple, it is the variable sequence length.
    :type q_shape: Union[Tuple[int, int, int, int], Tuple[int, Tuple[int, ...], int, int]]
    :param k_shape: Key tensor shape (B, S_k, H_k, D) where B=batch size, S_k=key sequence length,
                    H_k=number of key heads (H must be divisible by H_k), D=head dimension.
                    If S_k is a tuple, it is the variable sequence length.
    :type k_shape: Union[Tuple[int, int, int, int], Tuple[int, Tuple[int, ...], int, int]]
    :param in_dtype: Input data type for query, key and value tensors
    :type in_dtype: Type[cutlass.Numeric]
    :param out_dtype: Output data type for attention output
    :type out_dtype: Type[cutlass.Numeric]
    :param qk_acc_dtype: Accumulator data type for query-key matrix multiplication
    :type qk_acc_dtype: Type[cutlass.Numeric]
    :param pv_acc_dtype: Accumulator data type for probability-value matrix multiplication
    :type pv_acc_dtype: Type[cutlass.Numeric]
    :param mma_tiler_mn: Matrix multiply accumulate tile shape (M, N)
    :type mma_tiler_mn: Tuple[int, int]
    :param is_persistent: Whether to use persistent kernel optimization
    :type is_persistent: bool
    :param is_causal: Whether to apply causal masking
    :type is_causal: bool
    :param lse_calculation: Whether to calculate lse
    :type lse_calculation: bool
    :param window_size: Sliding window size (left, right) for attention masking. Controls which positions each query can attend to.
    :type window_size: Tuple[int, int]
    :param scale_q: Scaling factor for query tensor
    :type scale_q: float
    :param scale_k: Scaling factor for key tensor
    :type scale_k: float
    :param scale_v: Scaling factor for value tensor
    :type scale_v: float
    :param inv_scale_o: Inverse scaling factor for output tensor
    :type inv_scale_o: float
    :param scale_softmax: Attention score scaling factor (defaults to 1/sqrt(D) if set to 0)
    :type scale_softmax: float
    :param tolerance: Maximum acceptable error for validation
    :type tolerance: float
    :param warmup_iterations: Number of warmup iterations
    :type warmup_iterations: int
    :param iterations: Number of iterations to run for performance testing
    :type iterations: int
    :param skip_ref_check: Skip validation against reference implementation
    :type skip_ref_check: bool
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache
    :type use_cold_l2: bool

    :raises ValueError: If input shapes are incompatible or head dimension is unsupported
    :raises RuntimeError: If GPU is unavailable for computation
    :return: Execution time of the FMHA kernel in microseconds
    :rtype: float
    """

    # Use file_name as tag so parallel process output is identifiable
    _tag = f"[{file_name}]"

    if export_only:
        print(
            f"{_tag} Compiling: head_dim={q_shape[-1]}, in_dtype={in_dtype}, "
            f"causal={is_causal}, window={window_size}, "
            f"mma_tiler_mn={mma_tiler_mn}, persistent={is_persistent}, "
            f"bottom_right_align={bottom_right_align}, "
            f"sliding_window={window_size[0] != -1}"
        )
    else:
        print(f"{_tag} Running Blackwell SM100 FMHA test with:")
        print(f"{_tag}   q_shape={q_shape}, k_shape={k_shape}")
        print(f"{_tag}   in_dtype={in_dtype}, out_dtype={out_dtype}")
        print(f"{_tag}   qk_acc_dtype={qk_acc_dtype}, pv_acc_dtype={pv_acc_dtype}")
        print(f"{_tag}   mma_tiler_mn={mma_tiler_mn}, is_persistent={is_persistent}")
        print(f"{_tag}   is_causal={is_causal}, window_size={window_size}")
        print(f"{_tag}   tolerance={tolerance}, warmup={warmup_iterations}, iterations={iterations}")

    # ---- CuPy/NumPy helpers (replacing cutlass.torch) ----
    def _cutlass_to_cupy_dtype(cutlass_dtype):
        if cutlass_dtype == cutlass.Float16:
            return cp.float16
        elif cutlass_dtype in (cutlass.Float32, Float32):
            return cp.float32
        elif (cutlass_dtype.is_float and cutlass_dtype.width <= 8) or (cutlass_dtype.is_integer and cutlass_dtype.width == 4):
            return cp.uint8  # FP8/Int4 stored as bytes
        else:
            raise ValueError(f"Unsupported dtype for CuPy: {cutlass_dtype}")

    def _get_leading_dim(cp_array):
        for i, s in enumerate(cp_array.strides):
            if s == cp_array.itemsize:
                return i
        return len(cp_array.shape) - 1

    # Unpack parameters
    b, s_q, h_q, d = q_shape
    b_, s_k, h_k, d_ = k_shape
    window_size_left, window_size_right = window_size
    if window_size_left == -1:
        window_size_left = None
    if window_size_right == -1:
        window_size_right = None
    if is_causal:
        window_size_right = 0

    if b != b_:
        raise ValueError("q & k must have the same batch size")

    if d != d_:
        raise ValueError("q & k must have the same head dimension")

    # if d not in {32, 64, 128}:
    #     raise ValueError("head dimension must be 32, 64, or 128")

    if h_q % h_k != 0:
        raise ValueError("h_q must be divisible by h_k")

    if isinstance(s_q, tuple) and len(s_q) != b:
        raise ValueError("variable_seqlen s_q must have the length of batch size")
    if isinstance(s_k, tuple) and len(s_k) != b:
        raise ValueError("variable_seqlen s_k must have the length of batch size")

    if in_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16}:
        raise ValueError("in_dtype must be Float8E4M3FN or Float16")

    if out_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16}:
        raise ValueError("out_dtype must be Float8E4M3FN or Float16")

    if qk_acc_dtype not in {Float32}:
        raise ValueError("qk_acc_dtype must be Float32")

    if pv_acc_dtype not in {Float32}:
        raise ValueError("pv_acc_dtype must be Float32")

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    h_r = h_q // h_k

    # Prepare GPU tensors: Q, KV cache, O
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required to run this example!")

    if not export_only:
        cp.random.seed(1111)
    np.random.seed(1111)

    if isinstance(s_q, tuple) or isinstance(s_k, tuple):
        raise NotImplementedError("Variable-length sequences (nested tensors) require PyTorch. Use fmha_runtimeargs_kvcache.py for variable-length support.")

    def create_and_pad_tensor(shape, padding, dtype, is_dynamic_layout=True):
        shape_ = tuple(map(lambda x, y: x + y, shape, padding))

        if export_only:
            f32_gpu_full = cp.zeros(shape_, dtype=cp.float32)
        else:
            min_val = -2 if dtype.is_float or dtype.signed else 0
            f32_gpu_full = cp.random.randint(min_val, 2, shape_).astype(cp.float32)

        # Create dtype GPU buffer and initialize
        cp_dtype = _cutlass_to_cupy_dtype(dtype)
        is_narrow = (dtype.is_float and dtype.width <= 8) or (dtype.is_integer and dtype.width == 4)
        dtype_gpu_full = cp.empty(shape_, dtype=cp_dtype)

        if is_narrow:
            # FP8/Int4: use cute.testing.convert
            f32_cute = from_dlpack(f32_gpu_full)
            if is_dynamic_layout:
                f32_cute = f32_cute.mark_layout_dynamic(leading_dim=_get_leading_dim(f32_gpu_full))
            dtype_cute_full = from_dlpack(dtype_gpu_full, assumed_align=16)
            dtype_cute_full.element_type = dtype
            if is_dynamic_layout:
                dtype_cute_full = dtype_cute_full.mark_layout_dynamic(leading_dim=_get_leading_dim(dtype_gpu_full))
            cute.testing.convert(f32_cute, dtype_cute_full)
        else:
            dtype_gpu_full[:] = f32_gpu_full.astype(cp_dtype)

        # Offset the tensor (slice into padded region)
        slices = tuple(slice(s, e) for s, e in zip(padding, shape_))
        dtype_gpu = dtype_gpu_full[slices]
        f32_gpu = f32_gpu_full[slices]

        # Create cute tensor from sliced GPU buffer
        cute_tensor = from_dlpack(dtype_gpu, assumed_align=16)
        cute_tensor.element_type = dtype

        # f32 reference on CPU (numpy) for comparison
        f32_ref = f32_gpu.get()

        # Return full buffers too to prevent GC
        return (f32_ref, cute_tensor, dtype_gpu, dtype_gpu_full, f32_gpu_full)

    qo_shape = (b, s_q, h_r * h_k, d)
    kvcache_shape = (b, 2, h_k, s_k, d)
    lse_shape = (b, h_r * h_k, s_q)
    qo_padding = (0, 0, 0, 0, 0)
    kvcache_padding = (0, 0, 0, 0, 0, 0)
    lse_padding = (0, 0, 0, 0)

    q_ref, q_tensor, q_cp, *_q_keep = create_and_pad_tensor(
        qo_shape,
        qo_padding,
        in_dtype,
        is_dynamic_layout=True,
    )
    kvcache_ref, kvcache_tensor, kvcache_cp, *_kv_keep = create_and_pad_tensor(
        kvcache_shape,
        kvcache_padding,
        in_dtype,
        is_dynamic_layout=True,
    )
    _, o_tensor, o_cp, *_o_keep = create_and_pad_tensor(
        qo_shape,
        qo_padding,
        out_dtype,
        is_dynamic_layout=True,
    )
    if lse_calculation:
        _, lse_tensor, lse_cp, *_lse_keep = create_and_pad_tensor(
            lse_shape,
            lse_padding,
            cutlass.Float32,
            is_dynamic_layout=True,
        )
    else:
        lse_cp = None

    # SM100 tcgen05.mma atom K = 256 bits / element_bits.  For fp16: 16 elems.
    # The MMA tiler K must be a multiple of this atom.  Non-aligned dims like
    # 72 are padded up (→ 80); TMA ZFILL/OOB-drop bridge the gap at zero cost.
    _MMA_K_ATOM = 256 // 16  # 16 for fp16
    padded_d = ((d + _MMA_K_ATOM - 1) // _MMA_K_ATOM) * _MMA_K_ATOM
    actual_head_dim = d if padded_d != d else None
    if actual_head_dim is not None:
        print(f"[fmha] head_dim {d} not MMA-aligned; tiler K padded to {padded_d}, tensors stay at {d}")

    mma_tiler = (*mma_tiler_mn, padded_d)

    mask_type = fmha_utils.MaskEnum.WINDOW_MASK
    if bottom_right_align:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
    # Note: window_size_right is always 0 (causal), so window/causal masking is
    # always active. RESIDUAL_MASK fallback (no masking) is not reachable.

    s_q_list = s_q if isinstance(s_q, tuple) else [s_q] * b
    s_k_list = s_k if isinstance(s_k, tuple) else [s_k] * b

    # To avoid mask out the whole row which results in NaN in softmax
    def check_seqlen_valid(s_q, s_k, window_size_left, window_size_right, bottom_right_align):
        for i in range(s_q):
            offset = 0 if not bottom_right_align else s_k - s_q

            s_q_start = 0 if window_size_left is None else i + offset - window_size_left
            s_q_end = s_q if window_size_right is None else i + offset + window_size_right
            s_q_min = max(s_q_start, 0)
            s_q_max = min(s_q_end, s_k)

            if s_q_max - s_q_min == 0 and (i != 0 and i != s_q - 1):
                return False
        return True

    need_check_seqlen_valid = window_size_left is not None or window_size_right is not None
    for i in range(b):
        if need_check_seqlen_valid and not check_seqlen_valid(
            s_q_list[i],
            s_k_list[i],
            window_size_left,
            window_size_right,
            bottom_right_align,
        ):
            raise ValueError("sliding window doesn't support current setting")

    use_sliding_window = window_size_left is not None
    if vit_mode:
        mask_type = fmha_utils.MaskEnum.RESIDUAL_MASK
    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler,
        is_persistent,
        mask_type,
        is_causal=(is_causal and not vit_mode),
        use_sliding_window=(use_sliding_window and not vit_mode),
        actual_head_dim=actual_head_dim,
    )

    # Initialize Stream
    current_stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    # Compute folded scales for numpy reference and ViT path.
    # The LLM __call__ computes these internally from the raw per-tensor scales.
    if scale_softmax == 0.0:  # default to 1/sqrt(d)
        scale_softmax = 1.0 / math.sqrt(d)
    log2_e = math.log2(math.exp(1.0))

    ref_scale_softmax = scale_q * scale_k * scale_softmax
    ref_scale_softmax_log2 = ref_scale_softmax * log2_e
    ref_scale_output = scale_v * inv_scale_o

    def mark_bshd_dynamic(tensor):
        so = (0, 1, 2, 3)  # outermost-to-innermost for contiguous BSHD
        return (
            tensor.mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so)
        )

    def mark_shd_dynamic(tensor):
        so = (0, 1, 2)  # outermost-to-innermost for packed (total_S, H, D)
        return tensor.mark_layout_dynamic(leading_dim=2).mark_compact_shape_dynamic(mode=0, stride_order=so).mark_compact_shape_dynamic(mode=1, stride_order=so)

    def mark_kv_cache_dynamic(tensor):
        so = (0, 1, 2, 3, 4)  # outermost-to-innermost for contiguous (B,2,H,S,D)
        return (
            tensor.mark_layout_dynamic(leading_dim=4)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)  # B
            .mark_compact_shape_dynamic(mode=2, stride_order=so)  # H_kv
            .mark_compact_shape_dynamic(mode=3, stride_order=so)  # S
        )

    def mark_1d_dynamic(tensor):
        return tensor.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

    if vit_mode:
        # ViT: packed [total_S, H, D] with cu_seqlens for ragged batching.
        # For the reference test, total_S = b * s_q, uniform lengths.
        _s = s_q if not isinstance(s_q, tuple) else max(s_q)
        total_S = b * _s
        cu_seqlens_np = np.arange(b + 1, dtype=np.int32) * _s

        # Reshape Q, K, V from (B, S, H, D) → (total_S, H, D)
        q_vit_shape = (total_S, h_r * h_k, d)
        k_vit_shape = (total_S, h_k, d)
        q_vit_ref, q_vit_tensor, q_vit_cp, *_qv = create_and_pad_tensor(q_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True)
        k_vit_ref, k_vit_tensor, k_vit_cp, *_kv = create_and_pad_tensor(k_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True)
        v_vit_ref, v_vit_tensor, v_vit_cp, *_vv = create_and_pad_tensor(k_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True)
        _, o_vit_tensor, o_vit_cp, *_ov = create_and_pad_tensor(q_vit_shape, (0, 0, 0, 0), out_dtype, is_dynamic_layout=True)

        cu_seqlens_cp = cp.asarray(cu_seqlens_np)
        cu_seqlens = from_dlpack(cu_seqlens_cp, assumed_align=16)

        q_dyn = mark_shd_dynamic(q_vit_tensor)
        k_dyn = mark_shd_dynamic(k_vit_tensor)
        v_dyn = mark_shd_dynamic(v_vit_tensor)
        o_dyn = mark_shd_dynamic(o_vit_tensor)
        cu_dyn = mark_1d_dynamic(cu_seqlens)

        _max_seqlen = Int32(_s)

        start_time = time.time()
        compiled_fmha = cute.compile(
            fmha.__call_vit__,
            q_dyn,
            k_dyn,
            v_dyn,
            o_dyn,
            cu_dyn,
            _max_seqlen,
            ref_scale_softmax_log2,
            ref_scale_softmax,
            ref_scale_output,
            current_stream,
        )
    else:
        # LLM: batched Q [B,S,H,D] + combined KV cache [B,2,H,Cap,D]
        q_dyn = mark_bshd_dynamic(q_tensor)
        kv_dyn = mark_kv_cache_dynamic(kvcache_tensor)
        o_dyn = mark_bshd_dynamic(o_tensor)

        _wsl = Int32(window_size_left) if window_size_left is not None else Int32(0)

        _s_k = s_k if not isinstance(s_k, tuple) else max(s_k)
        cu_kv_seqlens_np = np.arange(b + 1, dtype=np.int32) * _s_k
        cu_kv_seqlens_cp = cp.asarray(cu_kv_seqlens_np)
        cu_kv_seqlens = from_dlpack(cu_kv_seqlens_cp, assumed_align=16)
        cu_kv_seqlens = mark_1d_dynamic(cu_kv_seqlens)

        start_time = time.time()
        compiled_fmha = cute.compile(
            fmha,
            q_dyn,
            kv_dyn,
            o_dyn,
            cu_kv_seqlens,
            _wsl,
            scale_q,
            scale_k,
            scale_v,
            inv_scale_o,
            current_stream,
        )

    compilation_time = time.time() - start_time
    print(f"{_tag} Compilation time: {compilation_time:.4f}s")

    if export_only:
        os.makedirs(output_dir, exist_ok=True)
        compiled_fmha.export_to_c(
            file_path=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
        )
        print(f"{_tag} Exported to {output_dir}/{file_name}.h and {file_name}.o")
        return None

    def _numpy_softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        # Handle rows that are all -inf
        x_max = np.where(np.isfinite(x_max), x_max, 0.0)
        e_x = np.exp(x - x_max)
        s = np.sum(e_x, axis=axis, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        return e_x / s

    def _numpy_logsumexp(x, axis=-1):
        x_max = np.max(x, axis=axis)
        x_max_safe = np.where(np.isfinite(x_max), x_max, 0.0)
        return np.log(np.sum(np.exp(x - x_max_safe[..., np.newaxis]), axis=axis)) + x_max

    def run_numpy_single_shot_reference_packed(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q,
        cu_seqlens_k,
        scale_softmax=1.0,
        scale_output=1.0,
        is_causal=False,
        bottom_right_align=False,
        lse_calculation=False,
        window_size_left=None,
        window_size_right=None,
    ):
        """Packed (ViT-style) numpy reference for single-shot attention.

        q_packed: [total_q, H_q, D]
        k_packed/v_packed: [total_k, H_k, D]
        cu_seqlens_q/cu_seqlens_k: cumulative offsets of length B+1.
        """
        h_q_local = q_packed.shape[1]
        h_k_local = k_packed.shape[1]
        if h_q_local % h_k_local != 0:
            raise ValueError("H_q must be divisible by H_k in packed reference")
        repeat_factor = h_q_local // h_k_local
        _wsr = 0 if is_causal else window_size_right

        ref_list = []
        lse_list = []
        batch_size = len(cu_seqlens_q) - 1
        for batch_idx in range(batch_size):
            q_start = cu_seqlens_q[batch_idx]
            q_end = cu_seqlens_q[batch_idx + 1]
            k_start = cu_seqlens_k[batch_idx]
            k_end = cu_seqlens_k[batch_idx + 1]

            q_i = q_packed[q_start:q_end].transpose(1, 0, 2)  # (H_q, S_q, D)
            k_i = k_packed[k_start:k_end].transpose(1, 0, 2)  # (H_k, S_k, D)
            v_i = v_packed[k_start:k_end].transpose(1, 0, 2)  # (H_k, S_k, D)

            if repeat_factor > 1:
                k_i = np.repeat(k_i, repeat_factor, axis=0)
                v_i = np.repeat(v_i, repeat_factor, axis=0)

            s_i = np.einsum("hqd,hkd->hqk", q_i, k_i) * scale_softmax
            s_q_local = q_i.shape[1]
            s_k_local = k_i.shape[1]

            if window_size_left is not None or _wsr is not None:
                q_coords = np.arange(s_q_local).reshape(-1, 1)
                k_coords = np.arange(s_k_local).reshape(1, -1)
                offset = 0 if not bottom_right_align else s_k_local - s_q_local
                if window_size_left is None:
                    _mask = k_coords > q_coords + offset + _wsr
                elif _wsr is None:
                    _mask = k_coords < q_coords + offset - window_size_left
                else:
                    _mask = (k_coords > q_coords + offset + _wsr) | (k_coords < q_coords + offset - window_size_left)
                s_i = np.where(_mask, -np.inf, s_i)

            if lse_calculation:
                lse_i = _numpy_logsumexp(s_i, axis=-1)
            else:
                lse_i = None

            p_i = _numpy_softmax(s_i, axis=-1)
            ref_i = np.einsum("hqk,hkd->hqd", p_i, v_i)
            ref_i = ref_i.transpose(1, 0, 2) * scale_output
            ref_list.append(ref_i)
            if lse_calculation:
                # (H_q, S_q) -> (S_q, H_q) to align packed output order.
                lse_list.append(lse_i.transpose(1, 0))

        ref = np.concatenate(ref_list, axis=0)
        lse = np.concatenate(lse_list, axis=0) if lse_calculation else None
        return ref, lse

    def _maybe_quantize_ref_for_narrow_out(o_ref_np):
        if not (out_dtype.is_float and out_dtype.width <= 8):
            return o_ref_np, tolerance
        ref_narrow_cp = cp.empty(o_ref_np.shape, dtype=cp.uint8)
        ref_narrow_cute = from_dlpack(ref_narrow_cp, assumed_align=16)
        ref_narrow_cute.element_type = out_dtype
        ref_narrow_cute = ref_narrow_cute.mark_layout_dynamic(leading_dim=_get_leading_dim(ref_narrow_cp))

        ref_o_f32_cp = cp.asarray(o_ref_np)
        ref_o_f32_cute = from_dlpack(ref_o_f32_cp, assumed_align=16)
        ref_o_f32_cute.element_type = cutlass.Float32
        ref_o_f32_cute = ref_o_f32_cute.mark_layout_dynamic(leading_dim=_get_leading_dim(ref_o_f32_cp))

        cute.testing.convert(ref_o_f32_cute, ref_narrow_cute)
        cute.testing.convert(ref_narrow_cute, ref_o_f32_cute)
        return ref_o_f32_cp.get(), 0.13

    if vit_mode:
        _vit_test_tag = "[vit_single_shot_test]"
        if not skip_ref_check:
            print(f"{_vit_test_tag} Running single-shot packed accuracy test:")
            print(f"{_vit_test_tag}   b={b}, seq_len={_s}, total_s={total_S}, h_q={h_q}, h_k={h_k}, d={d}, is_causal=False")
            print(f"{_vit_test_tag}   layout=[total_S,H,D], uniform cu_seqlens, max_seqlen={_s}")
            compiled_fmha(
                q_vit_tensor,
                k_vit_tensor,
                v_vit_tensor,
                o_vit_tensor,
                cu_seqlens,
                _max_seqlen,
                ref_scale_softmax_log2,
                ref_scale_softmax,
                ref_scale_output,
                current_stream,
            )

            o_fp32_cp = cp.empty(o_vit_cp.shape, dtype=cp.float32)
            o_fp32_cute = from_dlpack(o_fp32_cp, assumed_align=16)
            o_fp32_cute.element_type = Float32
            o_fp32_cute = o_fp32_cute.mark_layout_dynamic(leading_dim=2)
            cute.testing.convert(o_vit_tensor, o_fp32_cute)
            o_result = o_fp32_cp.get()

            o_ref, _ = run_numpy_single_shot_reference_packed(
                q_vit_ref,
                k_vit_ref,
                v_vit_ref,
                cu_seqlens_np,
                cu_seqlens_np,
                scale_softmax=ref_scale_softmax,
                scale_output=ref_scale_output,
                is_causal=False,
                bottom_right_align=False,
                lse_calculation=False,
                window_size_left=None,
                window_size_right=None,
            )
            o_ref, tol_for_check = _maybe_quantize_ref_for_narrow_out(o_ref)
            np.testing.assert_allclose(o_result, o_ref, atol=tol_for_check, rtol=1e-05)
            print(f"{_vit_test_tag} ViT single-shot accuracy check passed.")

        def generate_vit_tensors():
            _, q_ws, *_gq = create_and_pad_tensor(q_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True)
            _, k_ws, *_gk = create_and_pad_tensor(k_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True)
            _, v_ws, *_gv = create_and_pad_tensor(k_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True)
            _, o_ws, *_go = create_and_pad_tensor(q_vit_shape, (0, 0, 0, 0), out_dtype, is_dynamic_layout=True)
            return testing.JitArguments(
                mark_shd_dynamic(q_ws),
                mark_shd_dynamic(k_ws),
                mark_shd_dynamic(v_ws),
                mark_shd_dynamic(o_ws),
                cu_dyn,
                _max_seqlen,
                ref_scale_softmax_log2,
                ref_scale_softmax,
                ref_scale_output,
                current_stream,
            )

        exec_time = testing.benchmark(
            compiled_fmha,
            workspace_generator=generate_vit_tensors,
            workspace_count=1,
            stream=current_stream,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )
        return exec_time

    # LLM path only below: plugin-aligned multi-round prefill regression.
    if not skip_ref_check:
        # LLM-only regression that mirrors attention plugin unit test
        # (`test_plugin_vs_numpy_prefill`) and exercises cap != s_k stride.
        # Keep this as the only LLM correctness check.
        llm_prefill_tolerance = 0.13 if (out_dtype.is_float and out_dtype.width <= 8) else tolerance
        if not isinstance(s_q, tuple) and not isinstance(s_k, tuple):
            _num_rounds = 3
            _prefill_seq = min(s_q, s_k)
            _cap = _prefill_seq * _num_rounds
            print(f"{_tag} Running LLM multi-round prefill test (cap={_cap}, seq_len={_prefill_seq}, rounds={_num_rounds}) ...")
            run_llm_multi_round_prefill_test(
                batch_size=b,
                seq_len=_prefill_seq,
                num_rounds=_num_rounds,
                h_q=h_q,
                h_k=h_k,
                d=d,
                kv_cache_capacity=_cap,
                mma_tiler_mn=mma_tiler_mn,
                is_persistent=is_persistent,
                is_causal=is_causal,
                bottom_right_align=bottom_right_align,
                use_sliding_window=use_sliding_window,
                window_size_left_val=(window_size_left if window_size_left is not None else -1),
                tolerance=llm_prefill_tolerance,
            )
            print(f"{_tag} LLM multi-round prefill test passed.")

    def generate_tensors():
        _, q_tensor_workspace, *_gq = create_and_pad_tensor(
            qo_shape,
            qo_padding,
            in_dtype,
            is_dynamic_layout=True,
        )
        _, kvcache_tensor_workspace, *_gkv = create_and_pad_tensor(
            kvcache_shape,
            kvcache_padding,
            in_dtype,
            is_dynamic_layout=True,
        )
        _, o_tensor_workspace, *_go = create_and_pad_tensor(
            qo_shape,
            qo_padding,
            out_dtype,
            is_dynamic_layout=True,
        )
        if lse_calculation:
            _, lse_tensor, *_gl = create_and_pad_tensor(
                lse_shape,
                lse_padding,
                cutlass.Float32,
                is_dynamic_layout=True,
            )
        else:
            pass

        q_ws = mark_bshd_dynamic(q_tensor_workspace)
        kv_ws = mark_kv_cache_dynamic(kvcache_tensor_workspace)
        o_ws = mark_bshd_dynamic(o_tensor_workspace)

        args = testing.JitArguments(
            q_ws,
            kv_ws,
            o_ws,
            cu_kv_seqlens,
            _wsl,
            scale_q,
            scale_k,
            scale_v,
            inv_scale_o,
            current_stream,
        )
        return args

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = q_cp.size * q_cp.itemsize + kvcache_cp.size * kvcache_cp.itemsize + o_cp.size * o_cp.itemsize + (lse_cp.size * lse_cp.itemsize if lse_cp is not None else 0)
        workspace_count = testing.get_workspace_count(one_workspace_bytes, warmup_iterations, iterations)

    exec_time = testing.benchmark(
        compiled_fmha,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time  # Return execution time in microseconds


def run_llm_multi_round_prefill_test(
    batch_size: int = 4,
    seq_len: int = 8,
    num_rounds: int = 3,
    h_q: int = 8,
    h_k: int = 8,
    d: int = 128,
    kv_cache_capacity: int = 64,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    is_persistent: bool = True,
    is_causal: bool = True,
    bottom_right_align: bool = True,
    use_sliding_window: bool = False,
    window_size_left_val: int = -1,
    tolerance: float = 0.1,
):
    """LLM FMHA multi-round prefill accuracy test aligned with plugin unit test.

    Each round appends seq_len new tokens to a KV cache with physical capacity
    kv_cache_capacity >> effective_kv_len, exercising the cap vs s_k stride
    distinction.  After each round the CuTe DSL FMHA output is compared against
    a numpy reference.

    This helper is intentionally LLM-only: it validates the [B,S,H,D] query path
    with KV cache layout [B,2,H,cap,D], matching attention plugin prefill tests.

    :param batch_size: Number of batches.
    :param seq_len: Tokens per prefill round (same for Q and new K/V).
    :param num_rounds: How many rounds of prefill to run.
    :param h_q: Number of query heads.
    :param h_k: Number of KV heads.
    :param d: Head dimension (64 or 128).
    :param kv_cache_capacity: Physical KV cache capacity (cap).
    :param mma_tiler_mn: MMA tile shape.
    :param is_persistent: Use persistent kernel.
    :param is_causal: Enable causal masking (window_size_right = 0).
    :param bottom_right_align: bottom-right causal mask alignment.
    :param use_sliding_window: Enable sliding window masking.
    :param window_size_left_val: Left window size (-1 = disabled).
    :param tolerance: Max absolute error tolerance.
    """
    _tag = "[llm_prefill_test]"
    b = batch_size
    cap = kv_cache_capacity
    h_r = h_q // h_k
    window_size_left = window_size_left_val if use_sliding_window else None
    window_size_right = 0 if is_causal else None

    print(f"{_tag} Running multi-round prefill accuracy test:")
    print(f"{_tag}   b={b}, seq_len={seq_len}, rounds={num_rounds}, cap={cap}, h_q={h_q}, h_k={h_k}, d={d}, is_causal={is_causal}")

    _MMA_K_ATOM = 256 // 16
    padded_d = ((d + _MMA_K_ATOM - 1) // _MMA_K_ATOM) * _MMA_K_ATOM
    actual_head_dim = d if padded_d != d else None

    if h_q % h_k != 0:
        raise ValueError("h_q must be divisible by h_k")
    if num_rounds * seq_len > cap:
        raise ValueError(f"total tokens ({num_rounds * seq_len}) exceeds capacity ({cap})")

    cp.random.seed(42)
    np.random.seed(42)

    # FP16 test: all per-tensor scales are 1.0.
    # The kernel computes softmax_scale = scale_q * scale_k / sqrt(d) internally.
    _scale_q = 1.0
    _scale_k = 1.0
    _scale_v = 1.0
    _inv_scale_o = 1.0
    # Reference softmax scale for numpy validation
    ref_scale_softmax = 1.0 / math.sqrt(d)

    mask_type = fmha_utils.MaskEnum.WINDOW_MASK
    if bottom_right_align:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE

    fmha_op = BlackwellFusedMultiHeadAttentionForward(
        Float32,
        Float32,
        (*mma_tiler_mn, padded_d),
        is_persistent,
        mask_type,
        use_sliding_window=use_sliding_window,
        is_causal=is_causal,
        actual_head_dim=actual_head_dim,
    )
    current_stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    _wsl = Int32(window_size_left_val) if use_sliding_window else Int32(0)

    # ---- helpers ----
    def _to_cute(arr, element_type):
        t = from_dlpack(arr, assumed_align=16)
        t.element_type = element_type
        return t

    def mark_bshd_dynamic(tensor):
        so = (0, 1, 2, 3)
        return (
            tensor.mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so)
        )

    def mark_kv_cache_dynamic(tensor):
        so = (0, 1, 2, 3, 4)
        return (
            tensor.mark_layout_dynamic(leading_dim=4)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so)
            .mark_compact_shape_dynamic(mode=3, stride_order=so)
        )

    def _numpy_softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        x_max = np.where(np.isfinite(x_max), x_max, 0.0)
        e_x = np.exp(x - x_max)
        s = np.sum(e_x, axis=axis, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        return e_x / s

    # ---- state: KV cache (zero-initialized, filled progressively) ----
    kv_np = np.zeros((b, 2, h_k, cap, d), dtype=np.float32)
    compiled_fmha = None

    all_pass = True
    current_pos = 0

    for round_idx in range(num_rounds):
        effective_kv_len = current_pos + seq_len
        print(f"\n--- Round {round_idx + 1}/{num_rounds} (pos={current_pos}, s_k={effective_kv_len}, cap={cap}) ---")

        # ---- generate new Q and K/V for this round ----
        q_np = np.random.randint(-2, 2, (b, seq_len, h_q, d)).astype(np.float32)
        new_k_np = np.random.randint(-2, 2, (b, h_k, seq_len, d)).astype(np.float32)
        new_v_np = np.random.randint(-2, 2, (b, h_k, seq_len, d)).astype(np.float32)

        # write new K/V into cache at [current_pos : current_pos + seq_len]
        kv_np[:, 0, :, current_pos : current_pos + seq_len, :] = new_k_np
        kv_np[:, 1, :, current_pos : current_pos + seq_len, :] = new_v_np

        # ---- upload to GPU (FP16) ----
        q_cp = cp.asarray(q_np.astype(np.float16))
        kv_cp = cp.asarray(kv_np.astype(np.float16))
        o_cp = cp.zeros((b, seq_len, h_q, d), dtype=cp.float16)

        q_t = mark_bshd_dynamic(_to_cute(q_cp, cutlass.Float16))
        kv_t = mark_kv_cache_dynamic(_to_cute(kv_cp, cutlass.Float16))
        o_t = mark_bshd_dynamic(_to_cute(o_cp, cutlass.Float16))

        # cum_seqlen_k: uniform effective_kv_len across batches
        cu_kv_np = np.arange(b + 1, dtype=np.int32) * effective_kv_len
        cu_kv_cp = cp.asarray(cu_kv_np)
        cu_kv = from_dlpack(cu_kv_cp, assumed_align=16)
        cu_kv = cu_kv.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

        # ---- compile on first round ----
        if compiled_fmha is None:
            start_time = time.time()
            compiled_fmha = cute.compile(
                fmha_op,
                q_t,
                kv_t,
                o_t,
                cu_kv,
                _wsl,
                _scale_q,
                _scale_k,
                _scale_v,
                _inv_scale_o,
                current_stream,
            )
            print(f"{_tag} Compilation time: {time.time() - start_time:.4f}s")

        # ---- run kernel ----
        compiled_fmha(
            q_t,
            kv_t,
            o_t,
            cu_kv,
            _wsl,
            _scale_q,
            _scale_k,
            _scale_v,
            _inv_scale_o,
            current_stream,
        )

        # ---- read output ----
        o_f32_cp = cp.empty(o_cp.shape, dtype=cp.float32)
        o_f32_cute = from_dlpack(o_f32_cp, assumed_align=16)
        o_f32_cute.element_type = Float32
        o_f32_cute = o_f32_cute.mark_layout_dynamic(leading_dim=3)
        cute.testing.convert(o_t, o_f32_cute)
        o_result = o_f32_cp.get()

        # ---- numpy reference (only attend to valid tokens, not full cap) ----
        for bi in range(b):
            q_b = q_np[bi].transpose(1, 0, 2)  # (h_q, s_q, d)
            k_b = kv_np[bi, 0, :, :effective_kv_len]  # (h_k, s_k, d)
            v_b = kv_np[bi, 1, :, :effective_kv_len]  # (h_k, s_k, d)

            if h_q != h_k:
                k_b = np.repeat(k_b, h_r, axis=0)
                v_b = np.repeat(v_b, h_r, axis=0)

            scores = np.einsum("hqd,hkd->hqk", q_b, k_b) * ref_scale_softmax

            s_k = effective_kv_len
            if window_size_left is not None or window_size_right is not None:
                q_coords = np.arange(seq_len).reshape(-1, 1)
                k_coords = np.arange(s_k).reshape(1, -1)
                offset = (s_k - seq_len) if bottom_right_align else 0
                if window_size_left is None:
                    mask = k_coords > q_coords + offset + window_size_right
                elif window_size_right is None:
                    mask = k_coords < q_coords + offset - window_size_left
                else:
                    mask = (k_coords > q_coords + offset + window_size_right) | (k_coords < q_coords + offset - window_size_left)
                scores = np.where(mask, -np.inf, scores)
            probs = _numpy_softmax(scores, axis=-1)
            o_ref = np.einsum("hqk,hkd->hqd", probs, v_b)
            o_ref = o_ref.transpose(1, 0, 2) * _scale_v * _inv_scale_o

            o_actual = o_result[bi]
            max_diff = np.max(np.abs(o_actual - o_ref))
            mean_diff = np.mean(np.abs(o_actual - o_ref))

            if max_diff > tolerance:
                print(f"  batch {bi}: FAIL  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
                all_pass = False
            else:
                print(f"  batch {bi}: PASS  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

        current_pos += seq_len

    if all_pass:
        print(f"\n{_tag} All {num_rounds} rounds passed.")
    else:
        raise AssertionError(f"{_tag} Some rounds failed accuracy check!")

    return all_pass


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str):
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")

    def parse_nested_comma_separated_ints(s: str):
        try:
            s = s.strip()
            if "(" not in s:
                return tuple(int(x.strip()) for x in s.split(","))

            start = s.find("(")
            end = s.find(")")
            if start == -1 or end == -1:
                raise ValueError("Mismatched parentheses")

            before = s[:start].strip().rstrip(",")
            middle = s[start + 1 : end].strip()
            after = s[end + 1 :].strip().lstrip(",")

            result = []
            if before:
                result.extend(int(x.strip()) for x in before.split(","))

            if middle:
                nested_tuple = tuple(int(x.strip()) for x in middle.split(","))
                result.append(nested_tuple)

            if after:
                result.extend(int(x.strip()) for x in after.split(","))

            return tuple(result)

        except ValueError as e:
            if str(e) == "Mismatched parentheses":
                raise argparse.ArgumentTypeError("Mismatched parentheses in input")
            else:
                raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers with optional parentheses for nested tuple.")

    parser = argparse.ArgumentParser(description="Example of FMHA on Blackwell.")

    parser.add_argument(
        "--in_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
        help="Input data type",
    )

    parser.add_argument(
        "--out_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
        help="Output data type",
    )

    parser.add_argument(
        "--qk_acc_dtype",
        type=cutlass.dtype,
        default=Float32,
        help="QK accumulator data type",
    )

    parser.add_argument(
        "--pv_acc_dtype",
        type=cutlass.dtype,
        default=Float32,
        help="PV accumulator data type",
    )

    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="MMA tile shape (M, N)",
    )

    parser.add_argument(
        "--is_persistent",
        action="store_true",
        help="Is persistent",
    )

    parser.add_argument(
        "--is_causal",
        action="store_true",
        help="Whether to use casual mask",
    )

    parser.add_argument(
        "--bottom_right_align",
        action="store_true",
        help="Whether to use bottom right align, under this settion, the end of q is aligned with the end of k.",
    )

    parser.add_argument(
        "--lse_calculation",
        action="store_true",
        help="Whether to calculate lse",
    )

    parser.add_argument(
        "--window_size",
        type=parse_comma_separated_ints,
        default=(-1, -1),
        help="Sliding window size (left, right) for attention masking.",
    )

    parser.add_argument(
        "--q_shape",
        type=parse_nested_comma_separated_ints,
        default=(1, 256, 8, 128),
        help="Shape of Q (B, S_q, H, D)",
    )

    parser.add_argument(
        "--k_shape",
        type=parse_nested_comma_separated_ints,
        default=(1, 256, 8, 128),
        help="Shape of K (B, S_k, H_k, D)",
    )

    parser.add_argument(
        "--scale_q",
        type=float,
        default=1.0,
        help="Scaling factors to dequantize Q",
    )

    parser.add_argument(
        "--scale_k",
        type=float,
        default=1.0,
        help="Scaling factors to dequantize K",
    )

    parser.add_argument(
        "--scale_v",
        type=float,
        default=1.0,
        help="Scaling factors to dequantize V",
    )

    parser.add_argument(
        "--inv_scale_o",
        type=float,
        default=1.0,
        help="Scaling factor to quantize O",
    )

    parser.add_argument(
        "--scale_softmax",
        type=float,
        default=0.0,
        help="Scaling factor to scale S (i.e. Q*K); if zero, defaults to 1/sqrt(D)",
    )

    parser.add_argument("--tolerance", type=float, default=1e-01, help="Tolerance for validation")

    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of iterations for warmup",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations after warmup",
    )

    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference check",
    )

    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fmha_aot_artifacts",
        help="Output directory for AOT compiled artifacts (fmha.h and fmha.o)",
    )

    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Compile and export only; skip reference check and benchmark",
    )

    parser.add_argument(
        "--file_name",
        type=str,
        default="fmha",
        help="Base file name for exported artifacts (e.g., fmha_d64 -> fmha_d64.h, fmha_d64.o)",
    )

    parser.add_argument(
        "--function_prefix",
        type=str,
        default="fmha",
        help="Function prefix for exported C symbols (avoids conflicts when compiling multiple variants)",
    )

    parser.add_argument(
        "--vit_mode",
        action="store_true",
        help="Compile ViT FMHA variant: packed varlen with separate Q/K/V, bidirectional (no causal mask). Produces a different ABI.",
    )

    args = parser.parse_args()

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required to run this example!")

    if len(args.q_shape) != 4:
        parser.error("--q_shape must contain exactly 4 values")

    if len(args.k_shape) != 4:
        parser.error("--k_shape must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if args.vit_mode:
        assert args.k_shape == args.q_shape, f"vit_mode requires k_shape == q_shape; got k_shape={args.k_shape}, q_shape={args.q_shape}"
        assert not args.is_causal, "vit_mode is bidirectional; --is_causal must not be set"
        assert args.window_size == (-1, -1), f"vit_mode does not support sliding window; got --window_size={args.window_size}"

    latency = run(
        args.q_shape,
        args.k_shape,
        args.in_dtype,
        args.out_dtype,
        args.qk_acc_dtype,
        args.pv_acc_dtype,
        args.mma_tiler_mn,
        args.is_persistent,
        args.is_causal,
        args.bottom_right_align,
        args.lse_calculation,
        args.window_size,
        args.scale_q,
        args.scale_k,
        args.scale_v,
        args.inv_scale_o,
        args.scale_softmax,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        output_dir=args.output_dir,
        export_only=args.export_only,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
        vit_mode=args.vit_mode,
    )

    if latency is not None:
        print(f"Latency: {latency:.4f} us")
