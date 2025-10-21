from typing import Optional

# Please reinstall flashinfer by referring to https://github.com/svg-project/Sparse-VideoGen
try:
    import flashinfer
except ImportError:
    flashinfer = None

import torch
import triton
import triton.language as tl

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .svg2_attn_utils import (
    batch_kmeans_Euclid,
    identify_dynamic_map,
)
from .template import AttnWeightTemplate


@triton.jit
def _permute_kernel(
    X_ptr,
    IDX_ptr,
    Y_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Each program permutes BLOCK_S tokens *all* hidden features (D). No inner python loop."""

    pid_bh = tl.program_id(0)
    tile_s = tl.program_id(1)

    # Offsets along sequence
    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    token_mask = s_offsets < S

    # Gather source indices for these tokens
    idx_ptrs = IDX_ptr + pid_bh * S + s_offsets
    src_row_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)

    # Broadcast to create 2-D pointer matrix (BLOCK_S, D)
    d_offsets = tl.arange(0, D)

    src_ptrs = X_ptr + (pid_bh * S + src_row_idx[:, None]) * D + d_offsets[None, :]
    dst_ptrs = Y_ptr + (pid_bh * S + s_offsets[:, None]) * D + d_offsets[None, :]

    full_mask = token_mask[:, None]

    values = tl.load(src_ptrs, mask=full_mask, other=0.0)
    tl.store(dst_ptrs, values, mask=full_mask)


def permute_tensor_by_labels_triton(
    tensor: torch.Tensor,
    labels: Optional[torch.Tensor],
    dim: int,
    *,
    sorted_indices: Optional[torch.Tensor] = None,
):
    """
    Permute `tensor` along `dim` according to ascending order of `labels`.

    This is a Triton-accelerated replacement for the original implementation.
    It currently supports 4-D tensors of shape [B, H, S, D] and `dim == 2`.
    If these conditions are not met or the tensors reside on CPU, we fall back
    to the reference PyTorch implementation.
    """

    # Assertions – we only support the optimized CUDA path.
    assert dim == 2, "permute_tensor_by_labels currently only supports dim==2 (sequence dimension)"
    assert tensor.dim() == 4, "Expected tensor shape [B,H,S,D]"
    assert tensor.is_cuda, "permute_tensor_by_labels requires CUDA tensors"

    B, H, S, D = tensor.shape
    BH = B * H

    # Determine sorted indices
    if sorted_indices is not None:
        sorted_indices = sorted_indices.to(torch.int32).contiguous()
    else:
        assert labels is not None, "Either `labels` or `sorted_indices` must be provided."
        labels = labels.to(tensor.device)
        sorted_indices = torch.argsort(labels, dim=-1).to(torch.int32).contiguous()

    # Flatten tensor and allocate output
    inp_flat = tensor.reshape(BH, S, D).contiguous()
    out_flat = torch.empty_like(inp_flat)

    # Triton kernel tile size
    BLOCK_S = 64  # number of tokens per program, tunable

    n_s_tiles = triton.cdiv(S, BLOCK_S)
    grid = (BH, n_s_tiles)

    _permute_kernel[grid](inp_flat, sorted_indices, out_flat, S, D, BLOCK_S, num_warps=4)

    permuted_tensor = out_flat.reshape(B, H, S, D)
    return permuted_tensor, sorted_indices


@triton.jit
def _inverse_permute_kernel(
    X_ptr,
    IDX_ptr,
    Y_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Inverse permutation: scatter BLOCK_S tokens back in one shot."""

    pid_bh = tl.program_id(0)
    tile_s = tl.program_id(1)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    token_mask = s_offsets < S

    idx_ptrs = IDX_ptr + pid_bh * S + s_offsets
    src_pos_idx = s_offsets.to(tl.int32)
    dst_pos_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)

    d_offsets = tl.arange(0, D)

    src_ptrs = X_ptr + (pid_bh * S + src_pos_idx[:, None]) * D + d_offsets[None, :]
    dst_ptrs = Y_ptr + (pid_bh * S + dst_pos_idx[:, None]) * D + d_offsets[None, :]

    full_mask = token_mask[:, None]

    values = tl.load(src_ptrs, mask=full_mask, other=0.0)
    tl.store(dst_ptrs, values, mask=full_mask)


def apply_inverse_permutation_triton(
    permuted_tensor: torch.Tensor,
    sorted_indices: torch.Tensor,
    dim: int,
):
    """
    Triton implementation of inverse permutation. Inverse the permutation applied by `permute_tensor_by_labels`.

    Args:
        permuted_tensor: (B, H, S, D).
        sorted_indices: (B, H, S).
        dim: Dimension along which to apply inverse permutation. Typically 2, meaning the sequence lengthdimension.

    Returns:
        Tensor of shape (B, H, S, D).
    """

    assert dim == 2, "apply_inverse_permutation currently only supports dim==2"
    assert permuted_tensor.dim() == 4, "Expected tensor shape [B,H,S,D]"
    assert permuted_tensor.is_cuda, "apply_inverse_permutation requires CUDA tensors"

    B, H, S, D = permuted_tensor.shape
    BH = B * H

    # Ensure index dtype
    sorted_indices = sorted_indices.to(torch.int32).contiguous()

    # Flatten inputs
    inp_flat = permuted_tensor.reshape(BH, S, D).contiguous()
    out_flat = torch.empty_like(inp_flat)

    BLOCK_S = 64
    n_s_tiles = triton.cdiv(S, BLOCK_S)
    grid = (BH, n_s_tiles)

    _inverse_permute_kernel[grid](inp_flat, sorted_indices, out_flat, S, D, BLOCK_S, num_warps=4)

    original_tensor = out_flat.reshape(B, H, S, D)
    return original_tensor


@ATTN_WEIGHT_REGISTER("svg2_attn")
class Svg2AttnWeight(AttnWeightTemplate):
    centroids_init = False
    num_q_centroids = 300
    num_k_centroids = 1000
    kmeans_iter_init = 50
    top_p_kmeans = 0.9
    min_kc_ratio = 0.10
    kmeans_iter_step = 2

    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
    ):
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        bs, num_heads, seq_len, dim = q.size()
        q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = self.semantic_aware_permutation(q, k, v)

        output_permuted = self.dynamic_block_sparse_fwd_flashinfer(q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False)

        attn_output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)

        return attn_output.reshape(bs, num_heads, seq_len, dim).transpose(1, 2).reshape(bs * seq_len, -1)

    def dynamic_block_sparse_fwd_flashinfer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask_map: torch.Tensor,
        block_row_sz: torch.Tensor,
        block_col_sz: torch.Tensor,
        is_cpu: bool = True,
    ):
        """
        Launcher for the Flashinfer dynamic block sparse attention kernel.

        Args:
            q (torch.Tensor): Query tensor, shape [B, H, S, D].
            k (torch.Tensor): Key tensor, shape [B, H, S, D].
            v (torch.Tensor): Value tensor, shape [B, H, S, D].
            block_mask_map (torch.Tensor): Boolean mask, shape [B, H, qc_num, kc_num]. Currently must on CPU.
            block_row_sz (torch.Tensor): Query block sizes, shape [B, H, qc_num]. Currently must on CPU.
            block_col_sz (torch.Tensor): Key block sizes, shape [B, H, kc_num]. Currently must on CPU.
            is_cpu (bool): Whether to run on CPU. Flashinfer default is to run on CPU. We switch to GPU for faster planning. Default is True.
        """
        # Input shape check
        B, H, S, D = q.shape
        qc_num = block_row_sz.shape[-1]
        kc_num = block_col_sz.shape[-1]
        assert block_mask_map.shape == (B, H, qc_num, kc_num)

        assert all(t.device == torch.device("cpu") for t in [block_mask_map, block_row_sz, block_col_sz]) if is_cpu else True

        # Check if block_col_sz and block_row_sz are the same for each head
        assert torch.all(block_col_sz.sum(dim=2) == block_col_sz.sum(dim=2)[0, 0])
        assert torch.all(block_row_sz.sum(dim=2) == block_row_sz.sum(dim=2)[0, 0])

        # Prepare flashinfer wrapper
        float_workspace_buffer = torch.empty(128 * 1024 * 1024, device=q.device)
        vector_sparse_indices_buffer = torch.empty(1024 * 1024 * 1024, device=q.device)
        wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(float_workspace_buffer, backend="auto")
        wrapper.reset_workspace_buffer(
            float_workspace_buffer=wrapper._float_workspace_buffer,
            int_workspace_buffer=wrapper._int_workspace_buffer,
            vector_sparse_indices_buffer=vector_sparse_indices_buffer,  # Only reset this buffer size
            vector_sparse_indptr_buffer=wrapper._vector_sparse_indptr_buffer,
        )

        block_mask_map = block_mask_map.reshape(B * H, qc_num, kc_num)
        block_row_sz = block_row_sz.reshape(B * H, qc_num)
        block_col_sz = block_col_sz.reshape(B * H, kc_num)

        wrapper.plan(
            block_mask_map=block_mask_map,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            num_qo_heads=B * H,
            num_kv_heads=B * H,
            head_dim=D,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )

        # print_memory_usage("After plan")

        q = q.reshape(B * H, S, D)
        k = k.reshape(B * H, S, D)
        v = v.reshape(B * H, S, D)
        o = wrapper.run(q, k, v)  # [num_qo_heads, qo_len, head_dim]
        o = o.reshape(B, H, S, D)
        return o

    def semantic_aware_permutation(self, query, key, value):
        cfg, num_heads, seq_len, dim = query.size()

        # 1. Kmeans clustering
        qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_clustering(query, key)

        # 2. Identify dynamic map
        q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
        k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)

        dynamic_map = identify_dynamic_map(
            qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
            kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
            q_cluster_sizes,
            k_cluster_sizes,
            self.top_p_kmeans,
            self.min_kc_ratio,
        )

        # 3. Permute the query, key, value
        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, v_sorted_indices = permute_tensor_by_labels_triton(value, klabels, dim=2, sorted_indices=k_sorted_indices)

        return q_permuted, k_permuted, v_permuted, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices

    def kmeans_clustering(self, query, key):
        if not self.centroids_init:
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_init(query, key)
            self.centroids_init = True
        else:
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_step(query, key)

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    def kmeans_init(self, query, key):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(query.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_q_centroids, max_iters=self.kmeans_iter_init)
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(key.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_k_centroids, max_iters=self.kmeans_iter_init)

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    def kmeans_step(self, query, key):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.k_centroids,
        )

        self.q_centroids = qcentroids
        self.k_centroids = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter


if __name__ == "__main__":
    q, k, v = torch.randn(32130, 40, 128, dtype=torch.bfloat16).cuda(), torch.randn(32130, 40, 128, dtype=torch.bfloat16).cuda(), torch.randn(32130, 40, 128, dtype=torch.bfloat16).cuda()

    svg2_attn = Svg2AttnWeight()
    print("Svg2AttnWeight initialized.")

    out = svg2_attn.apply(q, k, v)
    print(f"out: {out.shape}, {out.dtype}, {out.device}")
