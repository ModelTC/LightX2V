from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.module_io import GridOutput
from lightx2v.models.networks.wan.infer.self_forcing.pre_infer import WanSFPreInfer, WanSFPreInferModuleOutput
from lightx2v_platform.base.global_var import AI_DEVICE


def sinusoidal_embedding_1d(dim, positions):
    positions = positions.to(torch.float64)
    half = dim // 2
    frequencies = torch.pow(10000, -torch.arange(half, dtype=torch.float64, device=positions.device).div(half))
    angles = torch.outer(positions, frequencies)
    return torch.cat((angles.cos(), angles.sin()), dim=1)


def rope_params(max_seq_len, dim):
    frequencies = 1.0 / torch.pow(10000, torch.arange(0, dim, 2, dtype=torch.float64).div(dim))
    angles = torch.outer(torch.arange(max_seq_len), frequencies)
    return torch.polar(torch.ones_like(angles), angles)


@dataclass
class WanMatrixGame4PreInferModuleOutput(WanSFPreInferModuleOutput):
    adapter_args: dict = field(default_factory=dict)


class WanMatrixGame4PreInfer(WanSFPreInfer):
    def __init__(self, config):
        super().__init__(config)
        head_dim = config["dim"] // config["num_heads"]
        max_frames = config.get("rope_max_len", 8192)
        self.freqs = torch.cat(
            (
                rope_params(max_frames, head_dim - 4 * (head_dim // 6)),
                rope_params(max_frames, 2 * (head_dim // 6)),
                rope_params(max_frames, 2 * (head_dim // 6)),
            ),
            dim=1,
        ).to(AI_DEVICE)

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        latents = self.scheduler.latents_input
        control = self.scheduler.control_latents_input
        control = torch.zeros_like(latents) if control is None else control.to(latents)
        padding = latents.new_zeros((self.config["in_dim"] - 2 * latents.shape[0], *latents.shape[1:]))
        model_input = torch.cat((latents, control, padding), dim=0)
        patches = weights.patch_embedding.apply(model_input.unsqueeze(0))
        frames, height, width = patches.shape[2:]
        x = patches.flatten(2).transpose(1, 2).squeeze(0).contiguous()

        timesteps = self.scheduler.timestep_input
        embed = self.time_embedding(weights, sinusoidal_embedding_1d(self.freq_dim, timesteps.flatten()).to(x))
        embed0 = self.time_projection(weights, embed)

        context_name = "context" if self.scheduler.infer_condition else "context_null"
        context = inputs["text_encoder_output"][context_name].squeeze(0)
        context = F.pad(context, (0, 0, 0, self.config["text_len"] - context.shape[0]))
        context = weights.text_embedding_0.apply(context.to(self.sensitive_layer_dtype))
        context = weights.text_embedding_2.apply(F.gelu(context, approximate="tanh"))

        grid_sizes = GridOutput(
            tensor=torch.tensor([[frames, height, width]], dtype=torch.int32, device=x.device),
            tuple=(frames, height, width),
        )
        return WanMatrixGame4PreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x,
            embed0=embed0,
            seq_lens=torch.tensor([x.shape[0]], dtype=torch.int32),
            freqs=self.freqs,
            context=context,
            adapter_args={"is_ref_prefill": self.scheduler.is_reference_prefill},
        )
