"""Differentiable Qwen-Image-2.1 replay through verl-omni's generic engine."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from verl_omni.pipelines.model_base import DiffusionModelBase

from lightx2v_rl.algorithms.fewstep import bridge_step
from lightx2v_rl.backends.verl_omni.config import get_fewstep_config
from lightx2v_rl.models.qwen_image.rollout import validate_turbo_schedule


def _module_dtype(module) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else next(module.parameters()).dtype


def _shared_length(batch, key, default):
    lengths = batch.get(key)
    if lengths is None:
        return default
    if lengths.ndim != 2 or lengths.shape[1] != 1 or not torch.all(lengths == lengths[:1]):
        raise ValueError("QwenImage21 replay needs identical lengths; set micro_batch_size_per_gpu=1.")
    length = int(lengths[0, 0])
    if length < 0 or length > default:
        raise ValueError(f"Invalid recorded {key}: {length} (capacity {default}).")
    return length


def prepare_qwen_replay_inputs(module, batch, prompt_embeds=None, prompt_embeds_mask=None):
    """Restore exactly the full-prefix inputs recorded at the sampled step.

    Batch size one is the general case. Larger microbatches are valid only when
    their prompt image-slot layout and reference shapes coincide; the official
    transformer itself reads that layout from row zero.
    """
    latents = batch["all_latents"]
    if latents.ndim != 4 or latents.shape[1] != 2:
        raise ValueError("Qwen FewStep all_latents must have shape [B, 2, tokens, channels].")
    batch_size = latents.shape[0]
    dtype = _module_dtype(module)
    device = latents.device
    shapes = batch["qwen_img_shapes"].to(device=device, dtype=torch.long)
    mask = batch["qwen_img_mask"].to(device=device, dtype=torch.bool)
    if shapes.ndim != 3 or shapes.shape[0] != batch_size or shapes.shape[-1] != 3:
        raise ValueError("qwen_img_shapes must have shape [B, images, 3].")
    if mask.ndim != 2 or mask.shape[0] != batch_size:
        raise ValueError("qwen_img_mask must have shape [B, sequence].")
    shape_count = _shared_length(batch, "qwen_shape_counts", shapes.shape[1])
    if shape_count < 1:
        raise ValueError("At least the target image shape must be recorded.")
    shapes = shapes[:, :shape_count]
    mask_length = _shared_length(batch, "qwen_img_mask_lengths", mask.shape[1])
    mask = mask[:, :mask_length]
    if not torch.all(shapes == shapes[:1]) or not torch.all(mask == mask[:1]):
        raise ValueError("QwenImage21 replay needs identical layouts; set micro_batch_size_per_gpu=1.")
    # Generic transports may right-pad per-request tensors. Actual lengths are
    # recoverable from the image shapes and recorded prompt-mask dimensions.
    target_tokens = int(shapes[0, -1].prod())
    reference_tokens = int(shapes[0, :-1].prod(dim=-1).sum())
    if _shared_length(batch, "qwen_reference_lengths", reference_tokens) != reference_tokens:
        raise ValueError("Recorded reference token count does not agree with image shapes.")
    if target_tokens != latents.shape[2]:
        raise ValueError("Recorded target image shape does not match the sampled latent tokens.")
    image_latents = batch["qwen_image_latents"][:, :reference_tokens].to(device=device, dtype=dtype)
    if image_latents.shape != (batch_size, reference_tokens, latents.shape[-1]):
        raise ValueError("Reference latent shape does not agree with qwen_img_shapes.")
    if prompt_embeds is None:
        prompt_embeds = batch["prompt_embeds"]
    if prompt_embeds_mask is None:
        prompt_embeds_mask = batch.get("prompt_embeds_mask")
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    prompt_length = _shared_length(batch, "qwen_prompt_lengths", prompt_embeds.shape[1])
    prompt_embeds = prompt_embeds[:, :prompt_length]
    if mask.shape[1] != prompt_length + target_tokens // 4:
        raise ValueError("Recorded image-slot mask does not cover the joint prompt/target sequence.")
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask[:, :prompt_length].to(device=device, dtype=torch.bool)
        if bool(prompt_embeds_mask.all()):
            prompt_embeds_mask = None
    timestep = batch["qwen_model_timesteps"]
    if timestep.shape != (batch_size, 1):
        raise ValueError("qwen_model_timesteps must have shape [B, 1].")
    return {
        "hidden_states": torch.cat((image_latents, latents[:, 0].to(dtype=dtype)), dim=1),
        "timestep": timestep[:, 0].to(device=device, dtype=dtype),
        "encoder_hidden_states": prompt_embeds,
        "encoder_hidden_states_mask": prompt_embeds_mask,
        "img_shapes": [[tuple(shape) for shape in row] for row in shapes.tolist()],
        "img_mask": mask,
        "attention_kwargs": {},
        "kv_cache": None,
        "kv_cache_mode": None,
        "return_dict": False,
    }


def qwen_target_prediction(module, model_inputs):
    """The official model returns joint text/reference/target sequence output."""
    target_tokens = 1
    for size in model_inputs["img_shapes"][0][-1]:
        target_tokens *= size
    return module(**model_inputs)[0][:, -target_tokens:]


@DiffusionModelBase.register("QwenImage21Pipeline", algorithm="fewstep_geo")
@DiffusionModelBase.register("QwenImage21Pipeline", algorithm="fewstep_cps")
@DiffusionModelBase.register("QwenImage21Pipeline", algorithm="fewstep_flow_grpo")
class FewStepQwenImage21Model(DiffusionModelBase):
    @classmethod
    def preserve_fp32_modules(cls):
        return False

    @classmethod
    def cast_forward_inputs(cls):
        return False

    @classmethod
    def build_module(cls, model_config, torch_dtype):
        from diffusers import QwenImage21Transformer2DModel

        module = QwenImage21Transformer2DModel.from_pretrained(
            model_config.local_path,
            subfolder=model_config.transformer_subfolder,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        # verl's generic checkpoint manager expects the Transformers config
        # save API; Diffusers owns equivalent serialization on the model.
        module.config.save_pretrained = module.save_config
        return module

    @classmethod
    def build_scheduler(cls, model_config):
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_config.local_path, subfolder="scheduler", local_files_only=True,
        )
        cls.set_timesteps(scheduler, model_config, "cpu")
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config, device):
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

        with (Path(model_config.local_path) / "model_index.json").open() as stream:
            sigmas = validate_turbo_schedule(json.load(stream).get("sample_sigmas"))
        pipeline = model_config.pipeline
        image_seq_len = (pipeline.height // 16) * (pipeline.width // 16)
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)

    @classmethod
    def prepare_model_inputs(
        cls, module, model_config, latents, timesteps, prompt_embeds,
        prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask,
        micro_batch, step,
    ):
        if step != 0 or timesteps.shape != (latents.shape[0], 1):
            raise ValueError("Qwen FewStep batches contain one sampled transition per candidate.")
        if model_config.pipeline.true_cfg_scale != 1:
            raise ValueError("Qwen-Image-2.1-Turbo replay requires official CFG=1.")
        return prepare_qwen_replay_inputs(module, micro_batch, prompt_embeds, prompt_embeds_mask), None

    @classmethod
    def forward(cls, module, model_config, model_inputs, negative_model_inputs=None):
        if negative_model_inputs is not None:
            raise ValueError("Turbo replay does not use negative-prompt guidance.")
        return qwen_target_prediction(module, model_inputs)

    @classmethod
    def forward_and_sample_previous_step(
        cls, module, scheduler, model_config, model_inputs, negative_model_inputs,
        scheduler_inputs, step,
    ):
        if step != 0 or scheduler_inputs is None:
            raise ValueError("Qwen FewStep replay requires the sampled transition batch.")
        latents = scheduler_inputs["all_latents"]
        prediction = cls.forward(module, model_config, model_inputs, negative_model_inputs)
        transition = bridge_step(
            prediction.float(), latents[:, 0].float(),
            # Retain scheduler sigmas verbatim: a t=1000*sigma round-trip can
            # change float32 values, and the BF16 model timestep is different.
            scheduler_inputs["qwen_sigmas"][:, 0].float(),
            scheduler_inputs["qwen_next_sigmas"][:, 0].float(),
            config=get_fewstep_config(model_config), next_sample=latents[:, 1].float(),
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
