from typing import Any

import torch
from einops import rearrange

from lightx2v.models.networks.wan.infer.module_io import GridOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE


def _matrix_game3_forward_tensor_probe(tensor: torch.Tensor, head_values: int = 8) -> dict[str, Any]:
    tensor = tensor.detach()
    tensor_fp32 = tensor.to(dtype=torch.float32)
    flattened = tensor_fp32.reshape(-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor_fp32.min().item()),
        "max": float(tensor_fp32.max().item()),
        "mean": float(tensor_fp32.mean().item()),
        "std": float(tensor_fp32.std(unbiased=False).item()),
        "head": flattened[:head_values].cpu().tolist(),
    }


def _matrix_game3_forward_value_probe(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return _matrix_game3_forward_tensor_probe(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "len": len(value),
            "head": list(value[:8]),
        }
    if isinstance(value, dict):
        return {k: _matrix_game3_forward_value_probe(v) for k, v in value.items()}
    return str(value)


class WanMtxg3PreInferOutput:
    """Container for MG3 pre-inference outputs passed to the transformer."""

    __slots__ = [
        "x",
        "embed",
        "embed0",
        "grid_sizes",
        "cos_sin",
        "context",
        "freqs",
        "plucker_emb",
        "mouse_cond",
        "keyboard_cond",
        "mouse_cond_memory",
        "keyboard_cond_memory",
        "memory_length",
        "memory_latent_idx",
        "predict_latent_idx",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class WanMtxg3PreInfer(WanPreInfer):
    """Pre-inference for Matrix-Game-3.0.

    Builds:
    - Patch embeddings + plucker camera embeddings
    - Text embeddings (no CLIP image encoder — MG3 uses direct text conditioning)
    - Time embeddings
    - Passes through conditioning signals (keyboard, mouse, plucker, memory)
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_memory = True
        self.sigma_theta = config.get("sigma_theta", 0.0)

        # Build RoPE frequencies with optional sigma_theta head-specific theta
        d = config["dim"] // config["num_heads"]
        num_heads = config["num_heads"]
        if self.sigma_theta > 0:
            self.freqs = self._build_sigma_theta_freqs(d, num_heads, self.sigma_theta)
        else:
            self.freqs = torch.cat(
                [
                    self.rope_params(2048, d - 4 * (d // 6)),
                    self.rope_params(2048, 2 * (d // 6)),
                    self.rope_params(2048, 2 * (d // 6)),
                ],
                dim=1,
            ).to(torch.device(AI_DEVICE))

    def _build_sigma_theta_freqs(self, d, num_heads, sigma_theta):
        """Build head-specific RoPE with sigma_theta perturbation as in official MG3."""
        c = d // 2
        c_t = c - 2 * (c // 3)
        c_h = c // 3
        c_w = c // 3
        max_seq_len = 2048

        rope_epsilon = torch.linspace(-1, 1, num_heads, dtype=torch.float64)
        theta_base = 10000.0
        theta_hat = theta_base * (1 + sigma_theta * rope_epsilon)

        def build_freqs(seq_len, c_part):
            exp = torch.arange(c_part, dtype=torch.float64) / c_part
            omega = 1.0 / torch.pow(theta_hat.unsqueeze(1), exp.unsqueeze(0))
            pos = torch.arange(seq_len, dtype=torch.float64)
            angles = pos.view(1, -1, 1) * omega.unsqueeze(1)
            return torch.polar(torch.ones_like(angles), angles)

        freqs_t = build_freqs(max_seq_len, c_t)
        freqs_h = build_freqs(max_seq_len, c_h)
        freqs_w = build_freqs(max_seq_len, c_w)
        return torch.cat([freqs_t, freqs_h, freqs_w], dim=2).to(torch.device(AI_DEVICE))

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        """Build pre-inference outputs for the MG3.0 transformer."""
        x = self.scheduler.latents
        t = self.scheduler.timestep_input

        # Official MG3 feeds a per-token timestep map where the fixed conditioning
        # latent slots are forced to zero. LightX2V's generic Wan scheduler only
        # builds that map for plain `wan2.2`, so the MG3 adapter reconstructs it
        # here when the scheduler exposes a scalar timestep.
        if t.numel() == 1:
            mask = getattr(self.scheduler, "mask", None)
            if mask is not None:
                timestep_scalar = t.reshape(1).to(device=x.device, dtype=x.dtype)
                t = (mask[0][:, ::2, ::2].to(device=x.device, dtype=x.dtype) * timestep_scalar).flatten()
            else:
                t = t.reshape(-1).to(device=x.device, dtype=x.dtype)

        # Text context (MG3 uses text conditioning only, no CLIP image encoder)
        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        # Matrix-Game-3 conditions are staged in the standard LightX2V
        # `image_encoder_output["dit_cond_dict"]` container by the runner.
        image_encoder_output = inputs.get("image_encoder_output", {})
        dit_cond_dict = image_encoder_output.get("dit_cond_dict") or {}

        if self.scheduler.infer_condition:
            plucker_emb = dit_cond_dict.get("plucker_emb_with_memory", None)
            if plucker_emb is None:
                plucker_emb = dit_cond_dict.get("c2ws_plucker_emb", None)
            mouse_cond = dit_cond_dict.get("mouse_cond", None)
            keyboard_cond = dit_cond_dict.get("keyboard_cond", None)
            x_memory = dit_cond_dict.get("x_memory", None)
            timestep_memory = dit_cond_dict.get("timestep_memory", None)
            mouse_cond_memory = dit_cond_dict.get("mouse_cond_memory", None)
            keyboard_cond_memory = dit_cond_dict.get("keyboard_cond_memory", None)
            memory_latent_idx = dit_cond_dict.get("memory_latent_idx", None)
        else:
            plucker_emb = dit_cond_dict.get("c2ws_plucker_emb", None)
            mouse_source = dit_cond_dict.get("mouse_cond", None)
            keyboard_source = dit_cond_dict.get("keyboard_cond", None)
            mouse_cond = torch.ones_like(mouse_source) if mouse_source is not None else None
            keyboard_cond = -torch.ones_like(keyboard_source) if keyboard_source is not None else None
            x_memory = None
            timestep_memory = None
            mouse_cond_memory = None
            keyboard_cond_memory = None
            memory_latent_idx = None
        predict_latent_idx = dit_cond_dict.get("predict_latent_idx", None)

        if hasattr(self.scheduler, "forward_kwargs_cond"):
            debug_payload = {
                "x": _matrix_game3_forward_value_probe(x.unsqueeze(0)),
                "t": _matrix_game3_forward_value_probe(t.reshape(1, -1)),
                "context": _matrix_game3_forward_value_probe(context),
                "mouse_cond": _matrix_game3_forward_value_probe(mouse_cond),
                "keyboard_cond": _matrix_game3_forward_value_probe(keyboard_cond),
                "x_memory": _matrix_game3_forward_value_probe(x_memory),
                "timestep_memory": _matrix_game3_forward_value_probe(timestep_memory),
                "mouse_cond_memory": _matrix_game3_forward_value_probe(mouse_cond_memory),
                "keyboard_cond_memory": _matrix_game3_forward_value_probe(keyboard_cond_memory),
                "plucker_emb": _matrix_game3_forward_value_probe(plucker_emb),
                "memory_latent_idx": _matrix_game3_forward_value_probe(memory_latent_idx),
                "predict_latent_idx": _matrix_game3_forward_value_probe(predict_latent_idx),
            }
            total_latent_frames_debug = x.shape[1] + (int(x_memory.shape[2]) if x_memory is not None else 0)
            patch_h_debug, patch_w_debug = tuple(self.config.get("patch_size", (1, 2, 2)))[1:]
            debug_payload["seq_len"] = int(total_latent_frames_debug * x.shape[2] * x.shape[3] // (patch_h_debug * patch_w_debug))
            if self.scheduler.infer_condition:
                self.scheduler.forward_kwargs_cond = debug_payload
            else:
                self.scheduler.forward_kwargs_uncond = debug_payload

        memory_length = 0
        if x_memory is not None:
            memory_length = int(x_memory.shape[2])
            x = torch.cat([x_memory.squeeze(0).to(device=x.device, dtype=x.dtype), x], dim=1)
            if timestep_memory is not None:
                t = torch.cat([timestep_memory.squeeze(0).to(device=x.device, dtype=x.dtype), t.to(device=x.device, dtype=x.dtype)], dim=0)

        # Patch embedding
        x = weights.patch_embedding.apply(x.unsqueeze(0)).to(self.infer_dtype)
        grid_sizes_t, grid_sizes_h, grid_sizes_w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()

        # Time embedding
        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        if self.sensitive_layer_dtype != self.infer_dtype:
            embed = weights.time_embedding_0.apply(embed.to(self.sensitive_layer_dtype))
        else:
            embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed).float()
        # Official MG3 keeps both the time embedding and its 6-way modulation
        # projection in fp32 before each block consumes them.
        modulation_dtype = self.sensitive_layer_dtype if self.sensitive_layer_dtype != self.infer_dtype else self.infer_dtype
        embed0 = torch.nn.functional.silu(embed).to(modulation_dtype)
        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim)).float()

        # Text embedding
        if self.sensitive_layer_dtype != self.infer_dtype:
            out = weights.text_embedding_0.apply(context.squeeze(0).to(self.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(context.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out).to(self.infer_dtype)

        # Grid sizes and RoPE
        grid_sizes = GridOutput(
            tensor=torch.tensor(
                [[grid_sizes_t, grid_sizes_h, grid_sizes_w]],
                dtype=torch.int32,
                device=x.device,
            ),
            tuple=(grid_sizes_t, grid_sizes_h, grid_sizes_w),
        )

        # MG3 can use head-specific 3D RoPE frequencies when `sigma_theta > 0`.
        # The shared LightX2V `prepare_cos_sin()` only handles the standard 2D
        # RoPE table, so MG3 keeps passing raw `freqs` downstream and lets the
        # MG3 transformer apply indexed RoPE itself.
        self.grid_sizes = grid_sizes.tuple
        self.cos_sin = None

        # Process plucker embedding through the global camera layers
        if plucker_emb is not None:
            # Match the official MG3 implementation: plucker embeddings arrive as
            # [B, C, F, H, W] (or an equivalent list form), must be patchified into
            # [B, L, C'] tokens, and only then can they pass through the global
            # camera-control linear projection.
            if torch.is_tensor(plucker_emb):
                plucker_items = [u.unsqueeze(0) for u in plucker_emb]
            else:
                plucker_items = [u.unsqueeze(0) if u.dim() == 4 else u for u in plucker_emb]

            patch_t, patch_h, patch_w = self.config.get("patch_size", (1, 2, 2))
            plucker_emb = [
                rearrange(
                    item,
                    "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                    c1=patch_t,
                    c2=patch_h,
                    c3=patch_w,
                )
                for item in plucker_items
            ]
            plucker_emb = torch.cat(plucker_emb, dim=1)
            if plucker_emb.size(1) < x.size(1):
                plucker_emb = torch.cat(
                    [
                        plucker_emb,
                        plucker_emb.new_zeros(plucker_emb.size(0), x.size(1) - plucker_emb.size(1), plucker_emb.size(2)),
                    ],
                    dim=1,
                )

            plucker_emb = weights.patch_embedding_wancamctrl.apply(plucker_emb.squeeze(0))
            plucker_hidden = weights.c2ws_hidden_states_layer2.apply(torch.nn.functional.silu(weights.c2ws_hidden_states_layer1.apply(plucker_emb)))
            plucker_emb = (plucker_emb + plucker_hidden).to(self.infer_dtype)

        return WanMtxg3PreInferOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x.squeeze(0),
            embed0=embed0.squeeze(0),
            context=context,
            cos_sin=self.cos_sin,
            freqs=self.freqs,
            plucker_emb=plucker_emb,
            mouse_cond=mouse_cond,
            keyboard_cond=keyboard_cond,
            mouse_cond_memory=mouse_cond_memory,
            keyboard_cond_memory=keyboard_cond_memory,
            memory_length=memory_length,
            memory_latent_idx=memory_latent_idx,
            predict_latent_idx=predict_latent_idx,
        )
