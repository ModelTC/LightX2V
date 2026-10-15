import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.utils import sinusoidal_embedding_1d


class EdenLivePreInfer(WanPreInfer):
    def _get_latents_input(self, inputs):
        return self.scheduler.latents_input.to(self.infer_dtype)

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        scheduler = self.scheduler
        x = self._get_latents_input(inputs)
        condition = inputs["image_encoder_output"]
        control = inputs["proxy_features"].to(x)
        if not scheduler.infer_condition and self.config["proxy_unconditional_control"] == "zero":
            control = torch.zeros_like(control)
        control = F.interpolate(control, size=x.shape[1:], mode="trilinear", align_corners=False)[0]
        if self.config["proxy_condition_rmsnorm"]:
            control = F.rms_norm(control.permute(1, 2, 3, 0), (1024,), weights.condition_norm.tensor, 1e-6)
            control = control.permute(3, 0, 1, 2)
        x = weights.patch_embedding.apply(torch.cat((x, control), dim=0).unsqueeze(0))
        grid = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)[0].contiguous()
        t = scheduler.timestep_input
        if t.numel() > 1:
            t = t.repeat_interleave(grid[1] * grid[2])
        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        embed = weights.time_embedding_2.apply(F.silu(weights.time_embedding_0.apply(embed), inplace=True))
        embed0 = weights.time_projection_1.apply(F.silu(embed)).unflatten(1, (6, self.dim))
        text_key = "context" if scheduler.infer_condition else "context_null"
        context = inputs["text_encoder_output"][text_key].squeeze(0).to(self.sensitive_layer_dtype)
        context = weights.text_embedding_2.apply(F.gelu(weights.text_embedding_0.apply(context), approximate="tanh"))
        if self.config["use_image_encoder"]:
            image = condition["clip_encoder_out"].squeeze(0)
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


class Wan21EdenLivePreInfer(EdenLivePreInfer):
    def _get_latents_input(self, inputs):
        x = super()._get_latents_input(inputs)
        start = self.scheduler.chunk_start
        condition = inputs["image_encoder_output"]["vae_encoder_out"][:, start : start + x.shape[1]]
        return torch.cat((x, condition), dim=0)
