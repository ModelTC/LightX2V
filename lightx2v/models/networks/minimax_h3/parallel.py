"""Weight and packed-sequence partitioning shared by H3 DiTs."""

import torch
import torch.distributed as dist

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3SequenceParallelState


def tensor_parallel_split_type(key, *, text_refiner_tensor_parallel=True):
    if key.startswith("token_refiner.") and not text_refiner_tensor_parallel:
        return None
    if any(part in key for part in (".attn.to_q.", ".attn.to_k.", ".attn.to_v.")):
        return "col"
    if ".attn.to_out.0." in key or ".ff.net.2." in key:
        return "row"
    if ".ff.net.0.proj." in key:
        return "ff_fused_col"
    if ".adaln_proj.linear." in key:
        return "col"
    return None


def split_tensor_parallel_weight(key, tensor, tp_size, tp_rank, *, text_refiner_tensor_parallel=True):
    if tp_size == 1:
        return tensor
    split_type = tensor_parallel_split_type(key, text_refiner_tensor_parallel=text_refiner_tensor_parallel)
    if split_type is None or tensor.ndim == 0:
        return tensor
    if split_type == "row":
        # Add row biases after reduction; per-output-channel scales are shared.
        if tensor.ndim < 2 or key.endswith(".weight_scale"):
            return tensor
        if tensor.shape[1] % tp_size:
            raise ValueError(f"Cannot row-shard {key} shape {tuple(tensor.shape)} across TP size {tp_size}")
        return torch.chunk(tensor, tp_size, dim=1)[tp_rank].contiguous()
    if split_type == "ff_fused_col":
        if tensor.shape[0] % 2:
            raise ValueError(f"Invalid H3 fused SwiGLU tensor {key} with shape {tuple(tensor.shape)}")
        half = tensor.shape[0] // 2
        if half % tp_size:
            raise ValueError(f"Cannot fused-column-shard {key} shape {tuple(tensor.shape)} across TP size {tp_size}")
        value, gate = tensor.split(half, dim=0)
        return torch.cat((value.chunk(tp_size, dim=0)[tp_rank], gate.chunk(tp_size, dim=0)[tp_rank]), dim=0).contiguous()
    if tensor.shape[0] % tp_size:
        raise ValueError(f"Cannot column-shard {key} shape {tuple(tensor.shape)} across TP size {tp_size}")
    return torch.chunk(tensor, tp_size, dim=0)[tp_rank].contiguous()


def shard_packed_sequence(pre, group, *, allow_media_prefix=False):
    world_size, rank = dist.get_world_size(group), dist.get_rank(group)
    total_length = pre.hidden_states.shape[0]
    text_length = pre.text_indices.numel()
    expected_text_indices = torch.arange(text_length, device=pre.text_indices.device)
    if not torch.equal(pre.text_indices, expected_text_indices):
        raise ValueError("MiniMax-H3 sequence parallel requires the conditioner rows to be a contiguous prefix")
    remainder = total_length % world_size
    if text_length < remainder and not allow_media_prefix:
        raise ValueError(f"MiniMax-H3 cannot split packed length {total_length} over {world_size} ranks with only {text_length} prefix rows")
    aux_length = text_length - ((text_length - remainder) % world_size)
    if allow_media_prefix:
        # Dense attention treats media rows in the replicated prefix exactly
        # like text rows, so short prompts need neither padding nor a mask.
        aux_length = max(aux_length, remainder)
    main_length = total_length - aux_length
    if main_length == 0 and allow_media_prefix:
        return pre
    if main_length <= 0 or main_length % world_size:
        raise ValueError(f"MiniMax-H3 sequence split is invalid: total={total_length}, aux={aux_length}, seq_p_size={world_size}")
    main_shard_length = main_length // world_size
    shard_start = aux_length + rank * main_shard_length
    shard_end = shard_start + main_shard_length

    def shard(tensor):
        return torch.cat((tensor[:aux_length], tensor[shard_start:shard_end]), dim=0)

    pre.sequence_parallel_state = MiniMaxH3SequenceParallelState(
        aux_length=aux_length,
        main_shard_length=main_shard_length,
        timestep_indices=pre.timestep_indices,
        adaln_indices=pre.adaln_indices,
        rotary_emb=pre.rotary_emb,
    )
    pre.hidden_states = shard(pre.hidden_states)
    pre.timestep_indices = shard(pre.timestep_indices)
    pre.adaln_indices = shard(pre.adaln_indices)
    pre.rotary_emb = tuple(shard(tensor) for tensor in pre.rotary_emb)
    return pre


def gather_packed_sequence(output, pre, group):
    state = pre.sequence_parallel_state
    if state is None:
        raise RuntimeError("MiniMax-H3 sequence-parallel metadata is missing")
    local_main = output[state.aux_length :]
    if local_main.shape[0] != state.main_shard_length:
        raise RuntimeError(f"MiniMax-H3 local sequence length changed from {state.main_shard_length} to {local_main.shape[0]}")
    gathered = [torch.empty_like(local_main) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local_main.contiguous(), group=group)
    output = torch.cat((output[: state.aux_length], *gathered), dim=0)
    pre.timestep_indices = state.timestep_indices
    pre.adaln_indices = state.adaln_indices
    pre.rotary_emb = state.rotary_emb
    pre.sequence_parallel_state = None
    return output
