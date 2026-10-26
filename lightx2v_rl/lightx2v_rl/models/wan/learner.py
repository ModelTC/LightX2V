"""Train a sampled FewStep Wan transition through the generic verl engine."""

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from verl_omni.pipelines.model_base import DiffusionModelBase

from lightx2v_rl.algorithms.fewstep import bridge_step
from lightx2v_rl.backends.verl_omni.config import get_fewstep_config


@DiffusionModelBase.register("WanPipeline", algorithm="fewstep_geo")
@DiffusionModelBase.register("WanPipeline", algorithm="fewstep_cps")
@DiffusionModelBase.register("WanPipeline", algorithm="fewstep_flow_grpo")
class FewStepWanModel(DiffusionModelBase):
    """Diffusers Wan adapter; rollout and learner use the same bridge function.

    The batch stores the sampled transition only: ``all_latents`` is
    ``[B, 2, C, F, H, W]`` and both timestep tensors are ``[B, 1]``. Explicit
    next timesteps are necessary for random and variable-length schedules.
    """

    @classmethod
    def preserve_fp32_modules(cls):
        return False

    @classmethod
    def cast_forward_inputs(cls):
        # FSDP's blanket BF16 cast rounds e.g. timestep=937.5 to 936, so the
        # learner would evaluate a different model than the rollout policy.
        return False

    @classmethod
    def build_scheduler(cls, model_config):
        fewstep = get_fewstep_config(model_config)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=fewstep.timestep_shift,
        )
        cls.set_timesteps(scheduler, model_config, "cpu")
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config, device):
        # The runtime expects a scheduler object. Bridge recomputation reads
        # the exact transition from the batch, never from this nominal grid.
        scheduler.set_timesteps(get_fewstep_config(model_config).max_steps, device=device)

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config,
        latents,
        timesteps,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        micro_batch,
        step,
    ):
        if latents.ndim != 6 or latents.shape[1] != 2:
            raise ValueError("FewStep all_latents must have shape [B, 2, C, F, H, W].")
        if timesteps.shape != (latents.shape[0], 1) or step != 0:
            raise ValueError("FewStep batches contain exactly one sampled transition per video.")
        # The initial recipes use BF16 storage and compute on both actor and
        # reference modules. Read the actual module dtype so unsharded models
        # and CPU test models work as well; do not cast the stored trajectory.
        module_dtype = getattr(module, "dtype", None)
        if not isinstance(module_dtype, torch.dtype):
            module_dtype = next(module.parameters()).dtype
        prompt_embeds = prompt_embeds.to(dtype=module_dtype)
        if prompt_embeds_mask is not None:
            prompt_embeds = prompt_embeds * prompt_embeds_mask.unsqueeze(-1).to(prompt_embeds.dtype)
        inputs = {
            "hidden_states": latents[:, 0].to(dtype=module_dtype),
            # FewStep can sample fractional timesteps. Integer rounding would
            # change the policy between rollout and log-prob recomputation.
            "timestep": timesteps[:, 0].float(),
            "encoder_hidden_states": prompt_embeds,
            "return_dict": False,
        }
        negative_inputs = None
        if model_config.pipeline.true_cfg_scale > 1:
            if negative_prompt_embeds is None:
                raise ValueError("CFG requires negative prompt embeddings from the rollout.")
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=module_dtype)
            if negative_prompt_embeds_mask is not None:
                negative_prompt_embeds = negative_prompt_embeds * negative_prompt_embeds_mask.unsqueeze(-1).to(
                    negative_prompt_embeds.dtype,
                )
            negative_inputs = {**inputs, "encoder_hidden_states": negative_prompt_embeds}
        return inputs, negative_inputs

    @classmethod
    def forward(cls, module, model_config, model_inputs, negative_model_inputs=None):
        prediction = module(**model_inputs)[0]
        scale = model_config.pipeline.true_cfg_scale
        if scale > 1:
            if negative_model_inputs is None:
                raise ValueError("CFG requires negative model inputs.")
            negative_prediction = module(**negative_model_inputs)[0]
            prediction = negative_prediction + scale * (prediction - negative_prediction)
        return prediction

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler,
        model_config,
        model_inputs,
        negative_model_inputs,
        scheduler_inputs,
        step,
    ):
        if scheduler_inputs is None or "all_next_timesteps" not in scheduler_inputs:
            raise ValueError("FewStep replay requires explicit all_next_timesteps from the rollout.")
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]
        next_timesteps = scheduler_inputs["all_next_timesteps"]
        if step != 0 or timesteps.shape != next_timesteps.shape or timesteps.shape[1] != 1:
            raise ValueError("FewStep requires matching [B, 1] current and next timesteps.")
        prediction = cls.forward(module, model_config, model_inputs, negative_model_inputs)
        transition = bridge_step(
            prediction.float(),
            latents[:, 0].float(),
            timesteps[:, 0].float() / 1000,
            next_timesteps[:, 0].float() / 1000,
            config=get_fewstep_config(model_config),
            next_sample=latents[:, 1].float(),
        )
        return {
            "log_probs": transition.log_prob,
            "prev_sample_mean": transition.mean,
            "std_dev_t": transition.std_dev_t,
            "sqrt_dt": transition.sqrt_dt,
            "transition_std": transition.std,
            "sigma": transition.sigma,
            "next_sigma": transition.next_sigma,
        }
