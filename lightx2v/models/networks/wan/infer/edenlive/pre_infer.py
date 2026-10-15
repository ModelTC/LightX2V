import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer


class EdenLivePreInfer(WanPreInfer):
    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        scheduler = self.scheduler
        x = scheduler.latents_input.to(self.infer_dtype)
        start, frames = scheduler.chunk_start, x.shape[1]
        condition = inputs["image_encoder_output"]
        if self.config["eden_variant"] == "14b":
            x = torch.cat((x, condition["vae_encoder_out"][:, start : start + frames]), dim=0)
        control = inputs["proxy_features"]
        if not scheduler.infer_condition and self.config["proxy_unconditional_control"] == "zero":
            control = torch.zeros_like(control)
        control = F.interpolate(control, size=x.shape[1:], mode="trilinear", align_corners=False)[0]
        if self.config["proxy_condition_rmsnorm"]:
            control = F.rms_norm(control.permute(1, 2, 3, 0).float(), (1024,), weights.condition_norm.tensor, 1e-6)
            control = control.permute(3, 0, 1, 2).to(x.dtype)
        x = weights.patch_embedding.apply(torch.cat((x, control.to(x)), dim=0).unsqueeze(0))
        grid = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)[0].contiguous()
        t = scheduler.timestep_input
        if t.numel() > 1:
            t = t.repeat_interleave(grid[1] * grid[2])
        frequencies = torch.exp(-math.log(10000) * torch.arange(self.freq_dim // 2, device=t.device, dtype=torch.float32) / (self.freq_dim // 2))
        angles = t.flatten().float()[:, None] * frequencies[None]
        embed = torch.cat((angles.cos(), angles.sin()), dim=-1).to(self.sensitive_layer_dtype)
        embed = weights.time_embedding_2.apply(F.silu(weights.time_embedding_0.apply(embed)))
        embed0 = weights.time_projection_1.apply(F.silu(embed)).unflatten(1, (6, self.dim))
        text_key = "context" if scheduler.infer_condition else "context_null"
        context = inputs["text_encoder_output"][text_key].squeeze(0).to(self.sensitive_layer_dtype)
        context = weights.text_embedding_2.apply(F.gelu(weights.text_embedding_0.apply(context), approximate="tanh"))
        if self.config["use_image_encoder"]:
            image = condition["clip_encoder_out"]
            image = weights.proj_1.apply(weights.proj_0.apply(image).to(self.infer_dtype))
            image = weights.proj_4.apply(weights.proj_3.apply(F.gelu(image, approximate="none")))
            context = torch.cat((image.squeeze(0).to(self.infer_dtype), context), dim=0)
        return WanPreInferModuleOutput(
            embed=embed,
            embed0=embed0,
            x=x,
            context=context,
            grid_sizes=GridOutput(torch.tensor([grid], device=x.device, dtype=torch.int32), grid),
            adapter_args={"text_segments": inputs.get("text_segments")},
        )
