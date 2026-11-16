"""Pre-RoPE KV for fixed conditions and committed clean AV chunks."""

import torch

from lightx2v.common.ops.rope.torch_rope import TorchRealRope

from .fp8_communication import FP8QKV
from .kv_cache_kernel import gather_rope_qkv
from .streaming import CleanCacheKey, MiniMaxH3CleanStreamingPlan


class MiniMaxH3CleanKVCache:
    """One selective KV cache for conditions and clean media history.

    Diffusion steps do not own independent
    buffers.  Noisy chunk rows are ephemeral; a completed chunk is evaluated at
    clean timestep once and retained under its chunk key. Motion prompts are
    replaced at segment boundaries, so at most one prompt is resident. Chunk
    storage rotates through free slots without overwriting the active sink.
    Returned attention views are valid until the next call for that layer.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        plan: "MiniMaxH3CleanStreamingPlan",
    ) -> None:
        self.plan = plan
        self._buffers: list[dict["CleanCacheKey", tuple[torch.Tensor, torch.Tensor]]] = [{} for _ in range(num_blocks)]
        self._prefix_keys: tuple["CleanCacheKey", ...] = ()
        self._entry_key: "CleanCacheKey" | None = None
        self._persist = False
        self._chunk_rows = max(plan.chunk_row_slice(index).stop - plan.chunk_row_slice(index).start for index in range(plan.num_chunks))
        self._num_chunk_slots = plan.policy.sink_chunk_num + plan.policy.latest_chunk_num
        self._chunk_slots: dict["CleanCacheKey", int] = {}
        self._next_chunk_slot = 0
        self._chunk_storage = [None] * num_blocks
        self._prompt_storage = [None] * num_blocks
        self._workspaces = [None] * num_blocks
        self._workspace_generation = [-1] * num_blocks
        self._generation = 0
        self._fixed_rows = plan.global_position_ids.shape[0] + plan.reference_position_ids.shape[0]
        # Only the first motion prompt is encoded here. Later prompts can grow
        # the workspace when their chunk begins.
        self._workspace_capacity = self._fixed_rows + plan.prompt_position_ids[0].shape[0] + self._num_chunk_slots * self._chunk_rows

    def prepare(
        self,
        entry_key: "CleanCacheKey",
        prefix_keys: tuple["CleanCacheKey", ...],
        *,
        persist: bool,
    ) -> None:
        # The scheduler supplies ordered, distinct prefixes from the plan.
        is_prompt = entry_key[0] == "prompt"
        if persist:
            if is_prompt:
                for block in self._buffers:
                    for key in tuple(block):
                        if key[0] == "prompt":
                            del block[key]
            if entry_key[0] == "chunk":
                occupied = set(self._chunk_slots.values())
                for offset in range(self._num_chunk_slots):
                    slot = (self._next_chunk_slot + offset) % self._num_chunk_slots
                    if slot not in occupied:
                        self._chunk_slots[entry_key] = slot
                        self._next_chunk_slot = (slot + 1) % self._num_chunk_slots
                        break
                else:
                    raise RuntimeError("CleanAR has no free history slot; finish the previous chunk before committing another")
            elif is_prompt:
                rows = self.plan.entry_position_ids(entry_key).shape[0]
                self._workspace_capacity = max(self._workspace_capacity, self._fixed_rows + rows + self._num_chunk_slots * self._chunk_rows)
        self._prefix_keys = prefix_keys
        self._entry_key = entry_key
        self._persist = persist
        self._generation += 1

    def _write_slot(self, shape, block_idx, *, device, dtypes):
        """Reserve raw KV independently of the mutable attention workspace."""
        kind = self._entry_key[0]
        if kind == "chunk":
            if shape[0] > self._chunk_rows:
                raise ValueError("CleanAR chunk exceeds its reserved history slot")
            storage = self._chunk_storage[block_idx]
            if storage is None:
                slot_shape = (self._num_chunk_slots, self._chunk_rows, *shape[1:])
                storage = self._chunk_storage[block_idx] = tuple(torch.empty(slot_shape, device=device, dtype=dtype) for dtype in dtypes)
            slot = self._chunk_slots[self._entry_key]
            target = tuple(value[slot, : shape[0]] for value in storage)
        elif kind == "prompt":
            storage = self._prompt_storage[block_idx]
            if storage is None or storage[0].shape[0] < shape[0]:
                storage = self._prompt_storage[block_idx] = tuple(torch.empty(shape, device=device, dtype=dtype) for dtype in dtypes)
            target = tuple(value[: shape[0]] for value in storage)
        else:
            # Fixed conditions are written once per request.
            target = tuple(torch.empty(shape, device=device, dtype=dtype) for dtype in dtypes)
        self._buffers[block_idx][self._entry_key] = target
        return target

    def prefix_position_ids(self, current_chunk_index: int) -> torch.Tensor:
        return self.plan.prefix_position_ids(self._prefix_keys, current_chunk_index)

    def condition_prefix_position_ids(self) -> torch.Tensor:
        positions = [self.plan.entry_position_ids(key) for key in self._prefix_keys]
        if positions:
            return torch.cat(positions)
        return self.plan.global_position_ids.new_empty((0, 3))

    @torch.compiler.disable
    def get_attention_qkv(
        self,
        qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | FP8QKV,
        block_idx: int,
        *,
        rope,
        query_freqs,
        key_freqs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Commit raw clean KV and return rotated Q/K and raw V for attention."""
        # Dictionary keys change at each chunk; keep session state outside Dynamo.
        if isinstance(qkv, FP8QKV):
            shape, device, dtype = qkv.shape, qkv.data.device, qkv.dtype
            dtypes = (dtype, dtype)
        else:
            q, k, v = qkv
            if k.ndim != 3 or q.shape != k.shape or k.shape != v.shape:
                raise ValueError("H3 QKV tensors must have matching shape [L, H, D]")
            shape, device, dtype = k.shape, k.device, q.dtype
            dtypes = (k.dtype, v.dtype)
        block = self._buffers[block_idx]
        start = sum(block[key][0].shape[0] for key in self._prefix_keys)
        end = start + shape[0]
        stored = self._write_slot(shape, block_idx, device=device, dtypes=dtypes) if self._persist else None
        capacity = max(self._workspace_capacity, end)
        workspace = self._workspaces[block_idx]
        if workspace is None or workspace[0].shape[0] < capacity:
            workspace_shape = (capacity, *shape[1:])
            workspace = self._workspaces[block_idx] = tuple(torch.empty(workspace_shape, device=device, dtype=value_dtype) for value_dtype in dtypes)
            self._workspace_generation[block_idx] = -1
        read_prefix = self._workspace_generation[block_idx] != self._generation
        if device.type == "cuda" and dtype == dtypes[0] == dtypes[1] == torch.bfloat16 and isinstance(rope, TorchRealRope) and rope.compute_dtype == torch.bfloat16 and rope.layout == "split_half":
            q = gather_rope_qkv(tuple(block[key] for key in self._prefix_keys), qkv, stored, workspace, key_freqs, read_prefix=read_prefix)
            key = workspace[0][:end]
        else:
            q, k, v = qkv.unpack() if isinstance(qkv, FP8QKV) else qkv
            if stored is not None:
                stored[0].copy_(k)
                stored[1].copy_(v)
            if read_prefix:
                torch.cat([*(block[key][0] for key in self._prefix_keys), k], out=workspace[0][:end])
                torch.cat([*(block[key][1] for key in self._prefix_keys), v], out=workspace[1][:end])
            else:
                workspace[0][start:end].copy_(k)
                workspace[1][start:end].copy_(v)
            q = rope.apply_single(q, query_freqs, rotary_dim=query_freqs[0].shape[-1])
            key = rope.apply_single(workspace[0][:end], key_freqs, rotary_dim=key_freqs[0].shape[-1])
        self._workspace_generation[block_idx] = self._generation
        return q, key, workspace[1][:end]

    def finalize_condition(self) -> None:
        """Release condition entries disabled by the attention policy."""
        keep = set(self.plan.fixed_keys())
        for block in self._buffers:
            for key in tuple(block):
                if key[0] in {"global", "reference"} and key not in keep:
                    del block[key]
        self._prefix_keys = ()
        self._entry_key = None
        self._persist = False

    def finish_chunk(self, chunk_index: int) -> None:
        next_chunk = chunk_index + 1
        keep_chunks = set(self.plan.clean_history_indices(next_chunk)) if next_chunk < self.plan.num_chunks else set()
        for block in self._buffers:
            for key in tuple(block):
                if key[0] == "chunk" and key[1] not in keep_chunks:
                    del block[key]
        self._chunk_slots = {key: slot for key, slot in self._chunk_slots.items() if key[1] in keep_chunks}

    def clear(self) -> None:
        self._buffers.clear()
        self._chunk_storage.clear()
        self._prompt_storage.clear()
        self._workspaces.clear()
        self._workspace_generation.clear()
        self._chunk_slots.clear()
        self._prefix_keys = ()
        self._entry_key = None
        self._persist = False
