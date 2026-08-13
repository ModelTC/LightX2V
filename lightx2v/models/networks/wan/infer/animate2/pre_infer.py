import math

import torch
import torch.distributed as dist

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


def sinusoidal_embedding_1d_source(dim, position):
    """Wan-Animate-2's float64 sinusoidal embedding, returned as float32."""
    if dim % 2:
        raise ValueError(f"Sinusoidal embedding dimension must be even, got {dim}.")
    half = dim // 2
    position = position.to(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).float()


def rope_params_source(max_seq_len, dim, theta=10000, offset=0):
    """Build the same CPU float64-complex table as the source implementation."""
    if dim % 2:
        raise ValueError(f"RoPE dimension must be even, got {dim}.")
    positions = torch.arange(max_seq_len) + int(offset)
    inv_freq = 1.0 / torch.pow(
        theta,
        torch.arange(0, dim, 2, dtype=torch.float64).div(dim),
    )
    angles = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(angles), angles)


def build_rope_table(head_dim, *, offset_t=0, offset_h=0, offset_w=0, max_seq_len=512):
    return torch.cat(
        [
            rope_params_source(max_seq_len, head_dim - 4 * (head_dim // 6), offset=offset_t),
            rope_params_source(max_seq_len, 2 * (head_dim // 6), offset=offset_h),
            rope_params_source(max_seq_len, 2 * (head_dim // 6), offset=offset_w),
        ],
        dim=1,
    )


class WanAnimate2PreInfer:
    def __init__(self, config):
        self.config = config
        self.freq_dim = int(config["freq_dim"])
        self.dim = int(config["dim"])
        self.num_heads = int(config["num_heads"])
        self.head_dim = self.dim // self.num_heads
        self.text_len = int(config["text_len"])
        self.infer_dtype = GET_DTYPE()
        self.sensitive_dtype = GET_SENSITIVE_DTYPE()
        self.seq_p_group = config.get("device_mesh").get_group(mesh_dim="seq_p") if config.get("seq_parallel", False) else None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

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

    @staticmethod
    def _mm_fp32(module, value):
        value = value.float()
        weight = module._get_actual_weight().float()
        bias = module._get_actual_bias()
        output = torch.mm(value, weight) if bias is None else torch.addmm(bias.float(), value, weight)
        if getattr(module, "has_lora_branch", False):
            hidden = torch.mm(value, module.lora_down.float().t())
            lora = torch.mm(hidden, module.lora_up.float().t())
            output = output + float(module.lora_strength * module.lora_scale) * lora
        return output

    def _time_embeddings(self, weights, timestep):
        embedding = sinusoidal_embedding_1d_source(self.freq_dim, timestep.flatten())
        embedding = self._mm_fp32(weights.time_embedding_0, embedding)
        embedding = torch.nn.functional.silu(embedding)
        embedding = self._mm_fp32(weights.time_embedding_2, embedding)
        embedding0 = self._mm_fp32(weights.time_projection_1, torch.nn.functional.silu(embedding))
        return embedding, embedding0.unflatten(1, (6, self.dim)).squeeze(0)

    @staticmethod
    def _layer_norm_fp32(module, value):
        weight = module._get_actual_weight()
        bias = module._get_actual_bias()
        return torch.nn.functional.layer_norm(
            value.float(),
            (value.shape[-1],),
            None if weight is None else weight.float(),
            None if bias is None else bias.float(),
            module.eps,
        )

    def _context(self, weights, text_context, clip_features):
        text_context = text_context.squeeze(0)
        text = weights.text_embedding_0.apply(text_context.to(self.sensitive_dtype))
        text = torch.nn.functional.gelu(text, approximate="tanh")
        text = weights.text_embedding_2.apply(text)

        # CUDA autocast runs LayerNorm in FP32 in the released pipeline.  The
        # first result is cast by the following Linear, while the final
        # LayerNorm remains FP32 and promotes the concatenated context.
        clip = self._layer_norm_fp32(weights.proj_0, clip_features)
        clip = weights.proj_1.apply(clip.to(self.infer_dtype))
        clip = torch.nn.functional.gelu(clip, approximate="none")
        clip = weights.proj_3.apply(clip)
        clip = self._layer_norm_fp32(weights.proj_4, clip)
        return torch.cat([clip, text], dim=0)

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
                "rope_table": build_rope_table(
                    self.head_dim,
                    offset_t=offset_t,
                    offset_h=offset_h,
                    offset_w=offset_w,
                ).to(x.device),
                "refer_stride": int(self.config.get("refer_stride", 1)),
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

        offset_t = int(self.config.get("refer_offset_t", 0))
        offset_h = int(self.config.get("refer_offset_h", 0))
        offset_w = int(self.config.get("refer_offset_w", 0))
        if offset_t < 0:
            offset_t = grid[0]
        if offset_h < 0:
            offset_h = grid[1]
        if offset_w < 0:
            offset_w = grid[2]

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
                "rope_table": build_rope_table(self.head_dim).to(x.device),
                "reference_rope_table": build_rope_table(
                    self.head_dim,
                    offset_t=offset_t,
                    offset_h=offset_h,
                    offset_w=offset_w,
                ).to(x.device),
                "refer_stride": int(self.config.get("refer_stride", 1)),
                "origin_len": int(animate["origin_len"]),
                "origin_area": tuple(int(value) for value in animate["origin_area"]),
                "is_uncondition": not self.scheduler.infer_condition,
            },
        )
