import triton
import triton.language as tl


@triton.jit
def wan_hidden_states_placement_kernel(
    hidden_states_ptr,  # [cfg, num_heads, seq_len, head_dim] seq_len = context_length + num_frame * frame_size
    hidden_states_out_ptr,  # [cfg, num_heads, seq_len, head_dim]
    best_mask_idx_ptr,  # [cfg, num_heads]
    hidden_states_stride_b,
    hidden_states_stride_h,
    hidden_states_stride_s,
    hidden_states_stride_d,
    mask_idx_stride_b,
    mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    context_length: tl.constexpr,
    num_frame: tl.constexpr,
    frame_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Copy hidden_states to output
    # range: [b, h, block_id * block_size: block_id * block_size + block_size, :]
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    start_id = block_id * BLOCK_SIZE
    end_id = start_id + BLOCK_SIZE
    end_id = tl.where(end_id > seq_len, seq_len, end_id)

    # Load best mask idx (0 is spatial, 1 is temporal)
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)

    offset_token = tl.arange(0, BLOCK_SIZE) + start_id
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim)

    if is_temporal:
        patch_id = offset_token // num_frame
        frame_id = offset_token - patch_id * num_frame
        offset_store_token = tl.where(offset_token >= seq_len - context_length, offset_token, frame_id * frame_size + patch_id)

        offset_load = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_token[:, None] * hidden_states_stride_s) + offset_d[None, :] * hidden_states_stride_d
        offset_hidden_states = hidden_states_ptr + offset_load

        offset_store = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_store_token[:, None] * hidden_states_stride_s) + offset_d[None, :] * hidden_states_stride_d
        offset_hidden_states_out = hidden_states_out_ptr + offset_store

        # Maybe tune the pipeline here
        hidden_states = tl.load(offset_hidden_states, mask=offset_mask[:, None])
        tl.store(offset_hidden_states_out, hidden_states, mask=offset_mask[:, None])
    else:
        offset_load = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_token[:, None] * hidden_states_stride_s) + offset_d[None, :] * hidden_states_stride_d
        offset_hidden_states = hidden_states_ptr + offset_load

        offset_store = offset_load
        offset_hidden_states_out = hidden_states_out_ptr + offset_store

        # Maybe tune the pipeline here
        hidden_states = tl.load(offset_hidden_states, mask=offset_mask[:, None])
        tl.store(offset_hidden_states_out, hidden_states, mask=offset_mask[:, None])


@triton.jit
def wan_sparse_head_placement_kernel(
    query_ptr,
    key_ptr,
    value_ptr,  # [cfg, num_heads, seq_len, head_dim] seq_len = context_length + num_frame * frame_size
    query_out_ptr,
    key_out_ptr,
    value_out_ptr,  # [cfg, num_heads, seq_len, head_dim]
    best_mask_idx_ptr,  # [cfg, num_heads]
    query_stride_b,
    query_stride_h,
    query_stride_s,
    query_stride_d,
    mask_idx_stride_b,
    mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    context_length: tl.constexpr,
    num_frame: tl.constexpr,
    frame_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Copy query, key, value to output
    # range: [b, h, block_id * block_size: block_id * block_size + block_size, :]
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    start_id = block_id * BLOCK_SIZE
    end_id = start_id + BLOCK_SIZE
    end_id = tl.where(end_id > seq_len, seq_len, end_id)

    # Load best mask idx (0 is spatial, 1 is temporal)
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)

    offset_token = tl.arange(0, BLOCK_SIZE) + start_id
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim)

    if is_temporal:
        frame_id = offset_token // frame_size
        patch_id = offset_token - frame_id * frame_size
        offset_store_token = tl.where(offset_token >= seq_len - context_length, offset_token, patch_id * num_frame + frame_id)

        offset_load = (cfg * query_stride_b + head * query_stride_h + offset_token[:, None] * query_stride_s) + offset_d[None, :] * query_stride_d
        offset_query = query_ptr + offset_load
        offset_key = key_ptr + offset_load
        offset_value = value_ptr + offset_load

        offset_store = (cfg * query_stride_b + head * query_stride_h + offset_store_token[:, None] * query_stride_s) + offset_d[None, :] * query_stride_d
        offset_query_out = query_out_ptr + offset_store
        offset_key_out = key_out_ptr + offset_store
        offset_value_out = value_out_ptr + offset_store

        # Maybe tune the pipeline here
        query = tl.load(offset_query, mask=offset_mask[:, None])
        tl.store(offset_query_out, query, mask=offset_mask[:, None])
        key = tl.load(offset_key, mask=offset_mask[:, None])
        tl.store(offset_key_out, key, mask=offset_mask[:, None])
        value = tl.load(offset_value, mask=offset_mask[:, None])
        tl.store(offset_value_out, value, mask=offset_mask[:, None])

    else:
        offset_load = (cfg * query_stride_b + head * query_stride_h + offset_token[:, None] * query_stride_s) + offset_d[None, :] * query_stride_d
        offset_query = query_ptr + offset_load
        offset_key = key_ptr + offset_load
        offset_value = value_ptr + offset_load

        offset_store = offset_load
        offset_query_out = query_out_ptr + offset_store
        offset_key_out = key_out_ptr + offset_store
        offset_value_out = value_out_ptr + offset_store

        # Maybe tune the pipeline here
        query = tl.load(offset_query, mask=offset_mask[:, None])
        tl.store(offset_query_out, query, mask=offset_mask[:, None])
        key = tl.load(offset_key, mask=offset_mask[:, None])
        tl.store(offset_key_out, key, mask=offset_mask[:, None])
        value = tl.load(offset_value, mask=offset_mask[:, None])
        tl.store(offset_value_out, value, mask=offset_mask[:, None])
