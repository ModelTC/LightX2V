import math

import torch
import torch.distributed as dist

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d


class WanAnimate2PreInfer(WanPreInfer):
    def __init__(self, config):
        super().__init__(config)
        self._rope_freqs_cache = {}

    def clear_rope_cache(self):
        """Drop the small per-clip GPU frequency cache on runner cleanup."""
        self._rope_freqs_cache.clear()

    def _sequence_parallel_size(self):
        return dist.get_world_size(self.seq_p_group) if self.seq_p_group is not None else 1

    def _patchify(self, weights, latent, conditioning):
        video = torch.cat([latent, conditioning], dim=0).to(self.infer_dtype)
        x = weights.patch_embedding.apply(video.unsqueeze(0))
        grid = tuple(int(value) for value in x.shape[2:])
        x = x.flatten(2).transpose(1, 2).squeeze(0).contiguous()
        valid_len = x.shape[0]
        world_size = self._sequence_parallel_size()
        padded_len = math.ceil(valid_len / world_size) * world_size
        if padded_len > valid_len:
            x = torch.cat([x, x.new_zeros(padded_len - valid_len, x.shape[-1])], dim=0)
        return x, grid, valid_len

    def _time_embeddings(self, weights, timestep):
        embedding = sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(self.sensitive_layer_dtype)
        embedding = weights.time_embedding_0.apply(embedding)
        embedding = torch.nn.functional.silu(embedding)
        embedding = weights.time_embedding_2.apply(embedding)
        embedding0 = weights.time_projection_1.apply(torch.nn.functional.silu(embedding))
        return embedding, embedding0.unflatten(1, (6, self.dim)).squeeze(0)

    def _context(self, weights, text_context, clip_features):
        text = weights.text_embedding_0.apply(text_context.squeeze(0).to(self.sensitive_layer_dtype))
        text = torch.nn.functional.gelu(text, approximate="tanh")
        text = weights.text_embedding_2.apply(text)

        # The registered LN backend owns its accumulation precision.  Cast at
        # component boundaries only so every GEMM sees the same dtype as its
        # weight (MMWeight uses torch.mm(..., out=...) and will not autocast).
        clip = weights.proj_0.apply(clip_features.to(self.sensitive_layer_dtype))
        clip = weights.proj_1.apply(clip.to(self.infer_dtype))
        clip = torch.nn.functional.gelu(clip, approximate="none")
        clip = weights.proj_3.apply(clip)
        clip = weights.proj_4.apply(clip.to(self.sensitive_layer_dtype))
        return torch.cat([clip.to(self.infer_dtype), text.to(self.infer_dtype)], dim=0)

    def _rope_freqs(
        self,
        grid,
        padded_len,
        device,
        *,
        offset_t=0,
        offset_h=0,
        offset_w=0,
        time_stride=1,
    ):
        """Assemble Wan RoPE frequencies with Animate2 reference offsets."""
        grid = tuple(int(value) for value in grid)
        key = (
            grid,
            int(padded_len),
            str(device),
            int(offset_t),
            int(offset_h),
            int(offset_w),
            int(time_stride),
        )
        cached = self._rope_freqs_cache.get(key)
        if cached is not None:
            return cached

        frames, height, width = grid
        valid_len = frames * height * width
        if padded_len < valid_len:
            raise ValueError(f"Wan-Animate-2 RoPE padded length {padded_len} is smaller than the grid token count {valid_len}.")
        if frames <= 0 or height <= 0 or width <= 0 or time_stride <= 0:
            raise ValueError(f"Invalid Wan-Animate-2 RoPE grid/stride: grid={grid}, stride={time_stride}.")
        if offset_t < 0 or offset_h < 0 or offset_w < 0:
            raise ValueError(f"Wan-Animate-2 RoPE offsets must be resolved to non-negative values before table construction, got {(offset_t, offset_h, offset_w)}.")
        max_position = max(
            offset_t + (frames - 1) * time_stride,
            offset_h + height - 1,
            offset_w + width - 1,
        )
        if max_position >= self.freqs.shape[0]:
            raise ValueError(f"Wan-Animate-2 RoPE position {max_position} exceeds Wan's frequency table length {self.freqs.shape[0]}.")

        complex_dim = self.head_size // 2
        table_t, table_h, table_w = self.freqs.split(
            [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3],
            dim=1,
        )
        freqs = torch.cat(
            [
                table_t[offset_t : offset_t + frames * time_stride : time_stride].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                table_h[offset_h : offset_h + height].view(1, height, 1, -1).expand(frames, height, width, -1),
                table_w[offset_w : offset_w + width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(valid_len, 1, -1)
        if padded_len > valid_len:
            freqs = torch.cat(
                [
                    freqs,
                    torch.ones(
                        padded_len - valid_len,
                        1,
                        freqs.shape[-1],
                        dtype=freqs.dtype,
                        device=freqs.device,
                    ),
                ],
                dim=0,
            )
        if self.rope is None:
            raise RuntimeError("Wan RoPE must be initialized before Animate2 inference.")
        freqs = self.rope.prepare_freqs(freqs.to(device), rotary_dim=self.head_size)
        self._rope_freqs_cache[key] = freqs
        return freqs

    @staticmethod
    def _grid_output(grid, device):
        return GridOutput(
            tensor=torch.tensor([grid], dtype=torch.int32, device=device),
            tuple=grid,
        )

    def infer_reference(self, weights, inputs):
        animate = inputs["animate2"]
        x, grid, valid_len = self._patchify(
            weights,
            animate["reference_latents"],
            animate["reference_y"],
        )
        timestep = torch.ones(1, dtype=torch.int64, device=x.device)
        embedding, embedding0 = self._time_embeddings(weights, timestep)
        context = self._context(
            weights,
            inputs["text_encoder_output"]["context_ref"],
            animate["reference_clip"],
        )

        generation_grid = (
            int(animate["generation_y"].shape[1]),
            int(animate["generation_y"].shape[2]) // 2,
            int(animate["generation_y"].shape[3]) // 2,
        )
        offset_t = int(self.config.get("refer_offset_t", 0))
        offset_h = int(self.config.get("refer_offset_h", 0))
        offset_w = int(self.config.get("refer_offset_w", 0))
        if offset_t < 0:
            offset_t = generation_grid[0]
        if offset_h < 0:
            offset_h = generation_grid[1]
        if offset_w < 0:
            offset_w = generation_grid[2]

        return WanPreInferModuleOutput(
            embed=embedding,
            grid_sizes=self._grid_output(grid, x.device),
            x=x,
            embed0=embedding0,
            context=context,
            valid_token_len=valid_len,
            adapter_args={
                "mode": "reference",
                "reference_kv_cache": animate["reference_kv_cache"],
                "rope_freqs": self._rope_freqs(
                    grid,
                    x.shape[0],
                    x.device,
                    offset_t=offset_t,
                    offset_h=offset_h,
                    offset_w=offset_w,
                    time_stride=int(self.config.get("refer_stride", 1)),
                ),
            },
        )

    def infer(self, weights, inputs):
        animate = inputs["animate2"]
        context_key = "context" if self.scheduler.infer_condition else "context_null"
        context_tensor = inputs["text_encoder_output"].get(context_key)
        if context_tensor is None:
            raise RuntimeError(f"Wan-Animate-2 is missing text encoder output {context_key!r}.")

        x, grid, valid_len = self._patchify(
            weights,
            self.scheduler.latents,
            animate["generation_y"],
        )
        embedding, embedding0 = self._time_embeddings(weights, self.scheduler.timestep_input)
        context = self._context(weights, context_tensor, animate["generation_clip"])
        reference_grid = tuple(int(value) for value in animate["reference_latents"].shape[1:])
        reference_grid = (reference_grid[0], reference_grid[1] // 2, reference_grid[2] // 2)

        return WanPreInferModuleOutput(
            embed=embedding,
            grid_sizes=self._grid_output(grid, x.device),
            x=x,
            embed0=embedding0,
            context=context,
            valid_token_len=valid_len,
            adapter_args={
                "mode": "generation",
                "reference_kv_cache": animate["reference_kv_cache"],
                "reference_grid": reference_grid,
                "rope_freqs": self._rope_freqs(grid, x.shape[0], x.device),
                "origin_len": int(animate["origin_len"]),
                "origin_area": tuple(int(value) for value in animate["origin_area"]),
                "is_uncondition": not self.scheduler.infer_condition,
            },
        )
