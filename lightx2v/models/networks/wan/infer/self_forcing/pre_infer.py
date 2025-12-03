from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from lightx2v.models.networks.wan.infer.module_io import GridOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@dataclass
class WanSFPreInferModuleOutput:
    embed: torch.Tensor
    grid_sizes: GridOutput
    x: torch.Tensor
    embed0: torch.Tensor
    seq_lens: torch.Tensor
    freqs: torch.Tensor
    context: torch.Tensor
    conditional_dict: Dict[str, Any] = field(default_factory=dict)


class WanSFPreInfer(WanPreInfer):
    def __init__(self, config):
        super().__init__(config)
        d = config["dim"] // config["num_heads"]
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        ).to(AI_DEVICE)

    def time_embedding(self, weights, embed):
        embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)

        return embed

    def time_projection(self, weights, embed):
        embed0 = torch.nn.functional.silu(embed)
        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))
        return embed0

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        x = self.scheduler.latents_input
        t = self.scheduler.timestep_input

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        # embeddings
        x = weights.patch_embedding.apply(x.unsqueeze(0))
        grid_sizes_t, grid_sizes_h, grid_sizes_w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        seq_lens = torch.tensor(x.size(1), dtype=torch.int32, device=x.device).unsqueeze(0)

        embed_tmp = sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        embed = self.time_embedding(weights, embed_tmp)
        embed0 = self.time_projection(weights, embed)

        # text embeddings
        if self.sensitive_layer_dtype != self.infer_dtype:  # False
            out = weights.text_embedding_0.apply(context.squeeze(0).to(self.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(context.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)
        if self.clean_cuda_cache:
            del out
            torch.cuda.empty_cache()

        if self.clean_cuda_cache:
            if self.config.get("use_image_encoder", True):
                del context_clip
            torch.cuda.empty_cache()

        grid_sizes = GridOutput(tensor=torch.tensor([[grid_sizes_t, grid_sizes_h, grid_sizes_w]], dtype=torch.int32, device=x.device), tuple=(grid_sizes_t, grid_sizes_h, grid_sizes_w))

        return WanSFPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x.squeeze(0),
            embed0=embed0.squeeze(0),
            seq_lens=seq_lens,
            freqs=self.freqs,
            context=context,
        )
