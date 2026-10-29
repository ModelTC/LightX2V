# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Flash Attention v2 with JVP
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Taken from https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

Modified 2025/03; Author: Kaiwen Zheng (zkwthu@gmail.com)

(1) Simplified version, combining Triton forward and official backward

(2) Support Jacobian-vector-product (JVP) computation in the forward pass

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

Citation:

@article{zheng2025rcm,
  title={Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency},
  author={Zheng, Kaiwen and Wang, Yuji and Ma, Qianli and Chen, Huayu and Zhang, Jintao and Balaji, Yogesh and Chen, Jianfei and Liu, Ming-Yu and Zhu, Jun and Zhang, Qinsheng},
  journal={arXiv preprint arXiv:2510.08431},
  year={2025}
}
"""

# Adapted forward-only kernel from NVIDIA TurboDiffusion, Apache-2.0.
# Repository: https://github.com/thu-ml/TurboDiffusion
# Source: turbodiffusion/rcm/utils/flash_attention_jvp_triton.py
# Revision: e3d61360b0eeab60c5cd09ff7d6c1020cccc80b2
# Kept the exact streaming JVP algorithm; omitted backward and benchmarks.

# Wan FP32 specialization: explicit IEEE dot at every matmul. Compiler
# default_dot_input_precision alone does not override tl.dot defaults in Triton3.4.
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    tQ,
    tK,
    tV,
    sm_scale,  # fp32 scalar
    LSE,  # [B, H, SEQ_LEN_Q] fp32 (natural log)
    O,  # noqa: E741 - retain the upstream kernel's output-pointer name.
    tO,  # [B, H, SEQ_LEN_Q, HEAD_DIM_V]
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    B: tl.constexpr,
    H: tl.constexpr,
    SEQ_LEN_Q: tl.constexpr,
    SEQ_LEN_KV: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)  # query block id
    pid_bh = tl.program_id(1).to(tl.int64)  # fused batch-head

    off_b = pid_bh // H
    off_h = pid_bh % H

    # offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = tl.arange(0, BLOCK_N)  # [BN]
    offs_dq = tl.arange(0, HEAD_DIM_QK)  # [Dq]
    offs_dv = tl.arange(0, HEAD_DIM_V)  # [Dv]

    # base pointers for this (b, h)
    q_base = off_b * stride_qb + off_h * stride_qh
    k_base = off_b * stride_kb + off_h * stride_kh
    v_base = off_b * stride_vb + off_h * stride_vh
    o_base = off_b * stride_ob + off_h * stride_oh

    # Q / tQ: [BM, Dq]
    Q_ptrs = Q + q_base + offs_m[:, None] * stride_qm + offs_dq[None, :] * stride_qd
    tQ_ptrs = tQ + q_base + offs_m[:, None] * stride_qm + offs_dq[None, :] * stride_qd
    m_mask = offs_m < SEQ_LEN_Q
    q = tl.load(Q_ptrs, mask=m_mask[:, None], other=0.0)
    tq = tl.load(tQ_ptrs, mask=m_mask[:, None], other=0.0)

    # streaming softmax stats (base-2)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)  # row max in log2 domain
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # row sum of exp2
    # JVP-specific accumulators (Algorithm 2: O, A, r, B)
    o_i = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)  # O~ (unnormalized)
    A_i = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)  # A~ = P~ tV
    B_i = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)  # B~ = H~ V
    r_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # r~ = rowsum(H~)

    qk_scale_log2 = sm_scale * 1.4426950408889634  # scale for exp2

    # loop over K/V blocks
    for start_n in range(0, SEQ_LEN_KV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < SEQ_LEN_KV

        # K / tK in transposed tile layout: [Dq, BN]
        K_ptrs = K + k_base + n_idx[None, :] * stride_kn + offs_dq[:, None] * stride_kd
        tK_ptrs = tK + k_base + n_idx[None, :] * stride_kn + offs_dq[:, None] * stride_kd
        k = tl.load(K_ptrs, mask=n_mask[None, :], other=0.0)
        tk = tl.load(tK_ptrs, mask=n_mask[None, :], other=0.0)

        # scores in log2 domain: qk = (QK^T) * sm_scale / ln2
        qk = tl.dot(q, k, input_precision="ieee").to(tl.float32) * qk_scale_log2
        qk = tl.where(n_mask[None, :], qk, -float("inf"))

        # update streaming max + exp2 sums (FlashAttention-2 core)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])  # P~ (unnormalized)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)  # rescale factor

        l_i = l_i * alpha + l_ij
        o_i = o_i * alpha[:, None]
        A_i = A_i * alpha[:, None]
        B_i = B_i * alpha[:, None]
        r_i = r_i * alpha

        # V / tV: [BN, Dv]
        V_ptrs = V + v_base + n_idx[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        tV_ptrs = tV + v_base + n_idx[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        v = tl.load(V_ptrs, mask=n_mask[:, None], other=0.0)
        tv = tl.load(tV_ptrs, mask=n_mask[:, None], other=0.0)

        # ---- JVP pieces ----
        # tS = (tQ K^T + Q tK^T) * sm_scale
        tS = (tl.dot(tq, k, input_precision="ieee").to(tl.float32) + tl.dot(q, tk, input_precision="ieee").to(tl.float32)) * sm_scale
        tS = tl.where(n_mask[None, :], tS, 0.0)

        # H~ = P~ ⊙ tS  (use unnormalized P~)
        Htilde = p * tS
        r_i += tl.sum(Htilde, axis=1)

        # accumulate O~, A~, B~ (Algorithm 2)
        p = p.to(v.dtype)
        Htilde = Htilde.to(v.dtype)
        o_i = tl.dot(p, v, o_i, input_precision="ieee")
        A_i = tl.dot(p, tv, A_i, input_precision="ieee")
        B_i = tl.dot(Htilde, v, B_i, input_precision="ieee")

        m_i = m_ij

    # ---- epilogue ----
    # O = O~ / l,  LSE = (m + log2(l)) * ln2
    # tO = (A~ + B~ - diag(r) O) / l

    inv_l_i = 1.0 / l_i
    O_i = o_i * inv_l_i[:, None]
    A_i = A_i * inv_l_i[:, None]
    B_i = B_i * inv_l_i[:, None]
    mu_i = r_i * inv_l_i
    tO_i = A_i + B_i - mu_i[:, None] * O_i

    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453

    # store
    O_ptrs = O + o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od
    tO_ptrs = tO + o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od
    LSE_ptrs = LSE + pid_bh * SEQ_LEN_Q + offs_m

    tl.store(O_ptrs, O_i.to(O.type.element_ty), mask=m_mask[:, None])
    tl.store(tO_ptrs, tO_i.to(tO.type.element_ty), mask=m_mask[:, None])
    tl.store(LSE_ptrs, lse, mask=m_mask)
