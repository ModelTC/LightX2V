import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, LSE, OS,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh - idx_b * H

    q_offset = idx_b * LQ * H * D + idx_h * D
    kv_offset = idx_b * LK * H * D + idx_h * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * LQ

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + q_offset + offs_m[:, None] * (H * D) + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * (H * D) + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
        k_start = idx_n * BLOCK_N
        k_mask = (k_start + offs_n) < LK

        K_ptrs = K + kv_offset + (k_start + offs_n)[None, :] * (H * D) + offs_d[:, None]
        V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * (H * D) + offs_d[None, :]

        k = tl.load(K_ptrs, mask=k_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
        qk = tl.where(k_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs, mask=k_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < LQ)

    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < LQ)


@triton.jit
def _attn_bwd_preprocess(
    OS, DOS, DELTAS,
    LQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh - idx_b * H

    OS += idx_b * LQ * H * D + idx_h * D
    DOS += idx_b * LQ * H * D + idx_h * D
    DELTAS += idx_bh * LQ

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    o_s = tl.load(OS + offs_m[:, None] * (H * D) + offs_d[None, :], mask=offs_m[:, None] < LQ)
    do_s = tl.load(DOS + offs_m[:, None] * (H * D) + offs_d[None, :], mask=offs_m[:, None] < LQ)

    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    tl.store(DELTAS + offs_m, delta_s, mask=offs_m < LQ)


@triton.jit
def _attn_bwd_dq(
    Q, K, V, LSE, DELTAS,
    DOS, DQ, LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh - idx_b * H

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_offset = idx_b * LQ * H * D + idx_h * D
    kv_offset = idx_b * LK * H * D + idx_h * D
    lse_offset = idx_bh * LQ
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    Q_ptrs = Q + q_offset + offs_m[:, None] * (H * D) + offs_d[None, :]
    DQ_ptrs = DQ + q_offset + offs_m[:, None] * (H * D) + offs_d[None, :]
    DOS_ptrs = DOS + q_offset + offs_m[:, None] * (H * D) + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    LUT_ptr = LUT + lut_offset

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ)
    do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < LQ)
    delta_s = tl.load(DELTAS_ptrs, mask=offs_m < LQ)
    lse = tl.load(LSE_ptrs, mask=offs_m < LQ, other=float("inf"))

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for block_idx in tl.range(topk, num_stages=2):
        idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
        k_start = idx_n * BLOCK_N
        k_mask = (k_start + offs_n) < LK

        K_ptrs = K + kv_offset + (k_start + offs_n)[:, None] * (H * D) + offs_d[None, :]
        V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * (H * D) + offs_d[None, :]

        k = tl.load(K_ptrs, mask=k_mask[:, None])
        v = tl.load(V_ptrs, mask=k_mask[:, None])

        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(k_mask[None, :], p, 0.0)

        dp = tl.dot(do_s, v.T).to(tl.float32)
        ds = p * (dp - delta_s[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < LQ)


@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, DOS, DK, DV,
    qk_scale, KBID, LSE, DELTAS,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh - idx_b * H

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)

    q_offset = idx_b * LQ * H * D + idx_h * D
    kv_offset = idx_b * LK * H * D + idx_h * D
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
    lse_offset = idx_bh * LQ

    Q_ptrs = Q + q_offset + offs_m[:, None] * (H * D) + offs_d[None, :]
    DOS_ptrs = DOS + q_offset + offs_m[:, None] * (H * D) + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m

    K_ptrs = K + kv_offset + offs_n[:, None] * (H * D) + offs_d[None, :]
    V_ptrs = V + kv_offset + offs_n[:, None] * (H * D) + offs_d[None, :]
    DK_ptrs = DK + kv_offset + offs_n[:, None] * (H * D) + offs_d[None, :]
    DV_ptrs = DV + kv_offset + offs_n[:, None] * (H * D) + offs_d[None, :]

    KBID_ptr = KBID + kbid_offset + idx_n

    k = tl.load(K_ptrs, mask=offs_n[:, None] < LK)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < LK)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    for idx_m in tl.range(0, LQ, BLOCK_M2):
        kbid = tl.load(KBID_ptr)
        if kbid == 1:
            m_mask = offs_m < (LQ - idx_m)
            q = tl.load(Q_ptrs, mask=m_mask[:, None])
            lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))

            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(offs_n[:, None] < LK, pT, 0.0)

            do = tl.load(DOS_ptrs, mask=m_mask[:, None])
            dv += tl.dot(pT.to(do.dtype), do)

            delta = tl.load(DELTAS_ptrs, mask=m_mask)
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)

        Q_ptrs += BLOCK_M2 * (H * D)
        DOS_ptrs += BLOCK_M2 * (H * D)
        LSE_ptrs += BLOCK_M2
        DELTAS_ptrs += BLOCK_M2
        if (idx_m + BLOCK_M2) % BLOCK_M == 0:
            KBID_ptr += N_BLOCKS

    tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < LK)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < LK)


class _attention_ar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert k_block_id.is_contiguous() and lut.is_contiguous()
        assert BLOCK_M == 64 or BLOCK_M == 128
        assert BLOCK_N == 64 or BLOCK_N == 128

        B, LQ, H, D = q.shape
        _, LK, Hk, Dk = k.shape

        if qk_scale is None:
            qk_scale = D**-0.5

        M_BLOCKS = triton.cdiv(LQ, BLOCK_M)

        o_s = torch.empty_like(q)
        lse = torch.empty((B, H, LQ), device=q.device, dtype=torch.float32)

        grid = (M_BLOCKS, B * H)
        _attn_fwd[grid](
            q, k, v, qk_scale, topk,
            lut, lse, o_s,
            LQ, LK, M_BLOCKS,
            H, D, BLOCK_M, BLOCK_N,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=3
        )

        ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.LQ = LQ
        ctx.LK = LK
        ctx.H = H
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, LQ, H, D = q.shape
        LK = ctx.LK

        M_BLOCKS = triton.cdiv(LQ, BLOCK_M)
        N_BLOCKS = triton.cdiv(LK, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta_s = torch.empty((B, H, LQ), device=q.device, dtype=torch.float32)

        grid = (M_BLOCKS, B * H)
        _attn_bwd_preprocess[grid](
            o_s, do_s, delta_s,
            LQ, H, D, BLOCK_M,
        )

        grid = (M_BLOCKS, B * H)
        _attn_bwd_dq[grid](
            q, k, v, lse, delta_s,
            do_s, dq, lut,
            ctx.qk_scale, ctx.topk,
            LQ, LK, M_BLOCKS,
            ctx.H, D, BLOCK_M, BLOCK_N,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5
        )

        grid = (N_BLOCKS, B * H)
        _attn_bwd_dkdv[grid](
            q, k, v, do_s, dk, dv,
            ctx.qk_scale, k_block_id, lse, delta_s,
            LQ, LK, M_BLOCKS, N_BLOCKS,
            ctx.H, D, BLOCK_M, BLOCK_N,
            BLOCK_SLICE_FACTOR=BLOCK_M // 64,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5
        )

        return dq, dk, dv, None, None, None, None, None, None
