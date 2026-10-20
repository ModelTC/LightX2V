"""FewStep rollout extension using verl-omni's existing Wan serving adapter.

Only the denoising algorithm changes. Token encoding, VAE decode, media
transport, weight loading and distributed serving belong to the upstream
adapter. Initial support is single-transformer Wan text-to-video.
"""

import torch
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.wan22_dance_grpo.vllm_omni_rollout_adapter import (
    Wan22DanceGRPOPipelineWithLogProb,
    _unwrap_single_request,
)
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx

from lightx2v_rl.algorithms.fewstep import FewStepConfig
from lightx2v_rl.rollout.trajectory import sample_trajectory


class _FewStepWanRollout(Wan22DanceGRPOPipelineWithLogProb):
    bridge_name: str
    supports_request_batch = False

    @torch.no_grad()
    def forward(self, req, **kwargs):
        request = _unwrap_single_request(req)
        if isinstance(request.prompt, dict) and request.prompt.get("prompt") == "dummy run":
            return super().forward(req, **kwargs)
        if getattr(self, "_fewstep_active", False):
            raise RuntimeError("The FewStep Wan adapter requires serial request execution.")
        if self.transformer is None or self.transformer_2 is not None or self.expand_timesteps:
            raise ValueError("FewStep currently supports single-transformer Wan T2V checkpoints only.")
        prompt = request.prompt if isinstance(request.prompt, dict) else {}
        if kwargs.get("image") is not None or (prompt.get("multi_modal_data") or {}).get("image") is not None:
            raise ValueError("The FewStep Wan adapter does not support image conditioning.")
        params = request.sampling_params
        original_extras = params.extra_args
        extras = dict(original_extras or {})
        if not extras.get("fewstep"):
            raise ValueError("Rollout sampling requires the external FewStepRolloutAlgoConfig.fewstep options.")
        config = FewStepConfig(**dict(extras["fewstep"]))
        if config.bridge != self.bridge_name:
            raise ValueError("The rollout registration and FewStep bridge configuration must agree.")
        # Keep the learner's CFG value authoritative. Wan's inherited default is
        # 5.0, which is incorrect for a distilled model trained at CFG=1.
        cfg_scale = extras.get("true_cfg_scale", getattr(params, "true_cfg_scale", None))
        if cfg_scale is None:
            raise ValueError("The FewStep rollout requires an explicit true_cfg_scale matching the learner.")
        if params.guidance_scale_provided and params.guidance_scale != cfg_scale:
            raise ValueError("Wan guidance_scale must equal true_cfg_scale for rollout/replay parity.")
        if params.guidance_scale_2 is not None and params.guidance_scale_2 != cfg_scale:
            raise ValueError("FewStep single-transformer Wan requires a single CFG scale.")
        cycle_index = max(int(extras.get("global_steps", 1)) - 1, 0)
        original_steps = params.num_inference_steps
        params.num_inference_steps = config.min_steps + cycle_index % (config.max_steps - config.min_steps + 1)
        extras.update(sde_window_size=None, shift=config.timestep_shift)
        params.extra_args = extras
        self._fewstep_active = True
        self._fewstep_config = config
        self._fewstep_cycle_index = cycle_index
        self._fewstep_trajectory = None
        kwargs.update(guidance_scale=cfg_scale, sde_window_size=None)
        try:
            output = super().forward(req, **kwargs)
            trajectory = self._fewstep_trajectory
            if trajectory is not None and trajectory.selected_step is not None:
                # The upstream transport collects custom training tensors from
                # metadata.rl, alongside native trajectory_* fields.
                metadata = output.output.setdefault("metadata", {})
                metadata.setdefault("rl", {})["all_next_timesteps"] = trajectory.all_next_timesteps.cpu()
            return output
        finally:
            params.extra_args = original_extras
            params.num_inference_steps = original_steps
            self._fewstep_active = False
            self._fewstep_trajectory = None

    def diffuse(
        self,
        latents,
        timesteps,
        prompt_embeds,
        negative_prompt_embeds,
        guidance_low,
        guidance_high,
        boundary_timestep,
        dtype,
        attention_kwargs,
        latent_condition,
        first_frame_mask,
        noise_level,
        sde_window,
        sde_type,
        generator,
        logprobs,
    ):
        if latent_condition is not None or first_frame_mask is not None:
            raise ValueError("FewStep currently supports text-to-video without latent conditioning.")
        step_index = 0

        def predict_velocity(state, sigma):
            nonlocal step_index
            if self.interrupt:
                raise RuntimeError("FewStep rollout interrupted before completing the video.")
            self._current_timestep = sigma * 1000.0
            set_forward_context_denoise_step_idx(step_index)
            step_index += 1
            positive = {
                "hidden_states": state.to(dtype),
                "timestep": self._current_timestep,
                "encoder_hidden_states": prompt_embeds,
                "attention_kwargs": attention_kwargs or {},
                "return_dict": False,
                "current_model": self.transformer,
            }
            do_cfg = guidance_low > 1.0
            if do_cfg and negative_prompt_embeds is None:
                raise ValueError("CFG requires negative prompt embeddings.")
            negative = {**positive, "encoder_hidden_states": negative_prompt_embeds} if do_cfg else None
            return self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_cfg,
                true_cfg_scale=guidance_low,
                positive_kwargs=positive,
                negative_kwargs=negative,
                cfg_normalize=False,
            ).float()

        trajectory = sample_trajectory(
            predict_velocity,
            latents,
            self._fewstep_config,
            cycle_index=self._fewstep_cycle_index,
            generator=generator,
            deterministic=not logprobs,
        )
        self._fewstep_trajectory = trajectory
        if trajectory.selected_step is None:
            return trajectory.final_latents, None, None, None
        return trajectory.final_latents, trajectory.all_latents, trajectory.log_probs, trajectory.all_timesteps


@VllmOmniPipelineBase.register("WanPipeline", algorithm="fewstep_geo")
class GeoWanRollout(_FewStepWanRollout):
    """Geo-FSRL rollout (ported from FewStepRL's IMMB)."""

    bridge_name = "geo"


@VllmOmniPipelineBase.register("WanPipeline", algorithm="fewstep_cps")
class CPSWanRollout(_FewStepWanRollout):
    bridge_name = "cps"


@VllmOmniPipelineBase.register("WanPipeline", algorithm="fewstep_flow_grpo")
class FlowGRPOWanRollout(_FewStepWanRollout):
    bridge_name = "flow_grpo"
