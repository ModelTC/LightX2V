from typing import List, Optional

import torch
from einops import rearrange

from .causal_model import CausalWanModel
from .scheduler import FlowMatchScheduler


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_config="",
        timestep_shift=5.0,
        is_causal=True,
    ):
        super().__init__()
        print(model_config)
        self.model = CausalWanModel.from_config(model_config)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 15 * 880  # 32760  # [1, 15, 16, 60, 104]

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations

        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(lambda x: x.double().to(flow_pred.device), [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps])

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        kv_cache_mouse: Optional[List[dict]] = None,
        kv_cache_keyboard: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
    ) -> torch.Tensor:
        assert noisy_image_or_video.shape[1] == 16
        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep
        logits = None

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.to(self.model.dtype),  # .permute(0, 2, 1, 3, 4),
                t=input_timestep,
                **conditional_dict,
                # seq_len=self.seq_len,
                kv_cache=kv_cache,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
            )  # .permute(0, 2, 1, 3, 4)

        else:
            flow_pred = self.model(
                noisy_image_or_video.to(self.model.dtype),  # .permute(0, 2, 1, 3, 4),
                t=input_timestep,
                **conditional_dict,
            )
            # .permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=rearrange(flow_pred, "b c f h w -> (b f) c h w"),  # .flatten(0, 1),
            xt=rearrange(noisy_image_or_video, "b c f h w -> (b f) c h w"),  # .flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        )  # .unflatten(0, flow_pred.shape[:2])
        pred_x0 = rearrange(pred_x0, "(b f) c h w -> b c f h w", b=flow_pred.shape[0])
        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0
