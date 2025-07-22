import torch

try:
    import flashinfer
    from packaging import version

    flashinfer_version = version.parse(flashinfer.__version__)
    has_o_dtype = flashinfer_version >= version.parse("0.2.6.post1")
except ImportError:
    flashinfer = None

###
###  Code from radial-attention
###  https://github.com/mit-han-lab/ç/blob/main/radial_attn/attn_mask.py#L150
###


def radial_attn(
    query, key, value, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, mask_map=None, sparsity_type="radial", block_size=128, decay_factor=1, model_cls="wan"
):
    orig_seqlen, num_head, hidden_dim = query.shape
    query = pad_qkv(query, block_size=block_size)
    key = pad_qkv(key, block_size=block_size)
    value = pad_qkv(value, block_size=block_size)

    mask = mask_map.queryLogMask(query, sparsity_type, block_size=block_size, decay_factor=decay_factor, model_type=model_cls) if mask_map else None
    seqlen = query.shape[0]
    workspace_buffer = torch.empty(128 * 1024 * 1024, device=query.device, dtype=torch.uint8)
    bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(
        workspace_buffer,
        backend="fa2",
    )

    indptr = get_indptr_from_mask(mask, query)
    indices = get_indices_from_mask(mask, query)

    kwargs = dict(
        indptr=indptr,
        indices=indices,
        M=seqlen,
        N=seqlen,
        R=block_size,
        C=block_size,
        num_qo_heads=num_head,
        num_kv_heads=num_head,
        head_dim=hidden_dim,
        q_data_type=query.dtype,
        kv_data_type=key.dtype,
        use_fp16_qk_reduction=True,
    )
    if has_o_dtype:
        kwargs["o_data_type"] = query.dtype

    bsr_wrapper.plan(**kwargs)

    o = bsr_wrapper.run(query, key, value)

    return o[:orig_seqlen, :, :]


def get_indptr_from_mask(mask, query):
    # query shows the device of the indptr
    # indptr (torch.Tensor) - the block index pointer of the block-sparse matrix on row dimension,
    # shape `(MB + 1,)`, where `MB` is the number of blocks in the row dimension.
    # The first element is always 0, and the last element is the number of blocks in the row dimension.
    # The rest of the elements are the number of blocks in each row.
    # the mask is already a block sparse mask
    indptr = torch.zeros(mask.shape[0] + 1, device=query.device, dtype=torch.int32)
    indptr[0] = 0
    row_counts = mask.sum(dim=1).flatten()  # Ensure 1D output [num_blocks_row]
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return indptr


def get_indices_from_mask(mask, query):
    # indices (torch.Tensor) - the block indices of the block-sparse matrix on column dimension,
    # shape `(nnz,),` where `nnz` is the number of non-zero blocks.
    # The elements in `indices` array should be less than `NB`: the number of blocks in the column dimension.
    nonzero_indices = torch.nonzero(mask)
    indices = nonzero_indices[:, 1].to(dtype=torch.int32, device=query.device)
    return indices


def shrinkMaskStrict(mask, block_size=128):
    seqlen = mask.shape[0]
    block_num = seqlen // block_size
    mask = mask[: block_num * block_size, : block_num * block_size].view(block_num, block_size, block_num, block_size)
    col_densities = mask.sum(dim=1) / block_size
    # we want the minimum non-zero column density in the block
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1 / 3
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
    block_mask = frac_high_density_cols > 0.6
    block_mask[0:0] = True
    block_mask[-1:-1] = True
    return block_mask


def pad_qkv(input_tensor, block_size=128):
    """
    Pad the input tensor to be a multiple of the block size.
    input shape: (seqlen, num_heads, hidden_dim)
    """
    seqlen, num_heads, hidden_dim = input_tensor.shape
    # Calculate the necessary padding
    padding_length = (block_size - (seqlen % block_size)) % block_size
    # Create a padded tensor with zeros
    padded_tensor = torch.zeros((seqlen + padding_length, num_heads, hidden_dim), device=input_tensor.device, dtype=input_tensor.dtype)
    # Copy the original tensor into the padded tensor
    padded_tensor[:seqlen, :, :] = input_tensor

    return padded_tensor


def get_diagonal_split_mask(i, j, token_per_frame, sparse_type, query):
    assert sparse_type in ["radial"]
    dist = abs(i - j)
    group = dist.bit_length()
    threshold = 128  # hardcoded threshold for now, which is equal to block-size
    decay_length = 2 ** token_per_frame.bit_length() / 2**group
    if decay_length >= threshold:
        return torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)

    split_factor = int(threshold / decay_length)
    modular = dist % split_factor
    if modular == 0:
        return torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
    else:
        return torch.zeros((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)


def get_window_width(i, j, token_per_frame, sparse_type, num_frame, decay_factor=1, block_size=128, model_type=None):
    assert sparse_type in ["radial"]
    dist = abs(i - j)
    if model_type == "wan":
        if dist < 1:
            return token_per_frame
        if dist == 1:
            return token_per_frame // 2
    elif model_type == "hunyuan":
        if dist <= 1:
            return token_per_frame
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    group = dist.bit_length()
    decay_length = 2 ** token_per_frame.bit_length() / 2**group * decay_factor
    threshold = block_size
    if decay_length >= threshold:
        return decay_length
    else:
        return threshold


def gen_log_mask_shrinked(query, s, video_token_num, num_frame, block_size=128, sparse_type="log", decay_factor=0.5, model_type=None):
    """
    A more memory friendly version, we generate the attention mask of each frame pair at a time,
    shrinks it, and stores it into the final result
    """
    final_log_mask = torch.zeros((s // block_size, s // block_size), device=query.device, dtype=torch.bool)
    token_per_frame = video_token_num // num_frame
    video_text_border = video_token_num // block_size

    col_indices = torch.arange(0, token_per_frame, device=query.device).view(1, -1)
    row_indices = torch.arange(0, token_per_frame, device=query.device).view(-1, 1)
    final_log_mask[video_text_border:] = True
    final_log_mask[:, video_text_border:] = True
    for i in range(num_frame):
        for j in range(num_frame):
            local_mask = torch.zeros((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
            if j == 0:  # this is attention sink
                local_mask = torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
            else:
                window_width = get_window_width(i, j, token_per_frame, sparse_type, num_frame, decay_factor=decay_factor, block_size=block_size, model_type=model_type)
                local_mask = torch.abs(col_indices - row_indices) <= window_width
                split_mask = get_diagonal_split_mask(i, j, token_per_frame, sparse_type, query)
                local_mask = torch.logical_and(local_mask, split_mask)

            remainder_row = (i * token_per_frame) % block_size
            remainder_col = (j * token_per_frame) % block_size
            # get the padded size
            all_length_row = remainder_row + ((token_per_frame - 1) // block_size + 1) * block_size
            all_length_col = remainder_col + ((token_per_frame - 1) // block_size + 1) * block_size
            padded_local_mask = torch.zeros((all_length_row, all_length_col), device=query.device, dtype=torch.bool)
            padded_local_mask[remainder_row : remainder_row + token_per_frame, remainder_col : remainder_col + token_per_frame] = local_mask
            # shrink the mask
            block_mask = shrinkMaskStrict(padded_local_mask, block_size=block_size)
            # set the block mask to the final log mask
            block_row_start = (i * token_per_frame) // block_size
            block_col_start = (j * token_per_frame) // block_size
            block_row_end = block_row_start + block_mask.shape[0]
            block_col_end = block_col_start + block_mask.shape[1]
            final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end], block_mask)
    print(f"mask sparsity: {1 - final_log_mask.sum() / final_log_mask.numel()}")
    return final_log_mask


class MaskMap:
    def __init__(self, video_token_num=79200, num_frame=22):
        self.video_token_num = video_token_num
        self.num_frame = num_frame
        self.log_mask = None

    def queryLogMask(self, query, sparse_type, block_size=128, decay_factor=0.5, model_type=None):
        log_mask = torch.ones((query.shape[0] // block_size, query.shape[0] // block_size), device=query.device, dtype=torch.bool)
        if self.log_mask is None:
            self.log_mask = gen_log_mask_shrinked(
                query, query.shape[0], self.video_token_num, self.num_frame, sparse_type=sparse_type, decay_factor=decay_factor, model_type=model_type, block_size=block_size
            )
        block_bound = self.video_token_num // block_size
        log_mask[:block_bound, :block_bound] = self.log_mask[:block_bound, :block_bound]
        return log_mask
