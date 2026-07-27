from typing import Dict, Tuple

import torch

from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer

from ..utils import compute_freqs_causvid


class WanCausVidPreInfer(WanPreInfer):
    """Prepare the CausVid RoPE cache once for each autoregressive chunk."""

    def __init__(self, config):
        super().__init__(config)
        self.scheduler = None
        self._causvid_rope_cache: Dict[tuple, Tuple[torch.Tensor, torch.Tensor | None]] = {}
        self._reset_request_cache()

    def set_scheduler(self, scheduler):
        super().set_scheduler(scheduler)
        self._reset_request_cache()
        self._rope_cache_request_id = scheduler.rope_request_id

    def set_rope(self, rope):
        super().set_rope(rope)
        self._reset_request_cache()
        if self.scheduler is not None:
            self._rope_cache_request_id = self.scheduler.rope_request_id

    def _reset_request_cache(self):
        self._rope_cache_request_id = None
        self._causvid_rope_cache.clear()
        self._causvid_start_frame = 0
        self.cos_sin = None
        self.rope_positions = None
        self.grid_sizes = (0, 0, 0)

    def _sync_request_cache(self):
        request_id = self.scheduler.rope_request_id
        if request_id == self._rope_cache_request_id:
            return

        self._reset_request_cache()
        self._rope_cache_request_id = request_id

    def _rope_cache_key(self, grid_sizes, start_frame):
        device = self.freqs.device
        return (
            tuple(grid_sizes),
            int(start_frame),
            device.type,
            device.index,
            type(self.rope),
            id(self.rope),
        )

    def prepare_cos_sin(self, grid_sizes, freqs):
        return compute_freqs_causvid(
            self.head_size // 2,
            grid_sizes,
            freqs,
            start_frame=self._causvid_start_frame,
        )

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        if self.seq_p_group is not None:
            raise NotImplementedError("Sequence parallel inference is not implemented for CausVid.")
        if kv_start < 0 or kv_end <= kv_start:
            raise ValueError(f"Invalid CausVid KV range: [{kv_start}, {kv_end}).")
        if self.scheduler is None:
            raise RuntimeError("WanCausVidPreInfer scheduler is not initialized.")

        self._sync_request_cache()

        # The previous local grid is the most accurate source of the number of
        # spatial tokens. On the first chunk, fall back to the configured value;
        # the first chunk normally starts at frame zero.
        if self.grid_sizes[1] > 0 and self.grid_sizes[2] > 0:
            spatial_tokens = self.grid_sizes[1] * self.grid_sizes[2]
        else:
            spatial_tokens = self.config["frame_seq_length"]
        self._causvid_start_frame = kv_start // spatial_tokens

        cached = None
        if self.grid_sizes != (0, 0, 0):
            cached = self._causvid_rope_cache.get(self._rope_cache_key(self.grid_sizes, self._causvid_start_frame))
        if cached is None:
            # Force WanPreInfer to call the CausVid-specific prepare_cos_sin.
            self.cos_sin = None
            self.rope_positions = None
        else:
            self.cos_sin, self.rope_positions = cached

        output = super().infer(weights, inputs, kv_start=kv_start, kv_end=kv_end)

        grid_sizes = output.grid_sizes.tuple
        spatial_tokens = grid_sizes[1] * grid_sizes[2]
        start_frame = kv_start // spatial_tokens
        if start_frame != self._causvid_start_frame:
            # This can only happen when the configured frame_seq_length differs
            # from the actual patched resolution on the first invocation.
            self._causvid_start_frame = start_frame
            raw_freqs = self.prepare_cos_sin(grid_sizes, self.freqs.clone())
            self.cos_sin = self.prepare_rope_cache(raw_freqs)

        cache_key = self._rope_cache_key(grid_sizes, start_frame)
        cached = self._causvid_rope_cache.get(cache_key)
        if cached is None:
            cached = (self.cos_sin, self.rope_positions)
            self._causvid_rope_cache[cache_key] = cached

        output.cos_sin, output.rope_positions = cached
        return output
