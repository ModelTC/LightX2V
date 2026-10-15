"""Host the official Diffusers Qwen pipeline in verl's vLLM-Omni workers.

The serving runtime owns requests, process placement, sleep/wake and weight
transport. This adapter owns only the model call and its RL trajectory payload.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from verl_omni.pipelines.diffusion_rollout_output import rollout_output
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.rollout_request import condition_images_from_payload

from lightx2v_rl.algorithms.fewstep import FewStepConfig
from lightx2v_rl.models.qwen_image.rollout import QwenImage21FewStepRollout, pad_qwen_replay_tensors


def copy_transformer_weights(module: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """Accept exact Diffusers names; never silently omit an actor tensor."""
    destinations = dict(module.named_parameters()) | dict(module.named_buffers())
    loaded = set()
    with torch.no_grad():
        for name, value in weights:
            if not name.startswith("transformer."):
                raise ValueError(f"Only transformer weights may be synchronized, got {name!r}.")
            target_name = name.removeprefix("transformer.")
            if target_name not in destinations:
                raise KeyError(f"Unknown QwenImage21 weight: {name}")
            target = destinations[target_name]
            if target.shape != value.shape:
                raise ValueError(f"Shape mismatch for {name}: {tuple(value.shape)} != {tuple(target.shape)}")
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded.add(name)
    return loaded


class _QwenImage21Serving(nn.Module):
    # The generic engine warmup has no task or reference-image metadata.
    dummy_run_num_frames = 0
    support_image_input = True
    color_format = "RGBA"
    supports_request_batch = False
    supports_step_execution = False
    diffusion_io_spec = DiffusionIOSpec(primary=MediaSpec("image"))
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules = ["vae"]
    bridge_name: str

    def __init__(self, *, od_config, prefix=""):
        super().__init__()
        from diffusers import QwenImage21Pipeline

        parallel = od_config.parallel_config
        for name in ("tensor_parallel_size", "sequence_parallel_size", "cfg_parallel_size", "text_encoder_tp_size"):
            if int(getattr(parallel, name, 1) or 1) != 1:
                raise ValueError("Official QwenImage21 rollout currently requires one GPU per replica.")
        self.od_config = od_config
        # Load on CPU first. The runtime's subsequent Module.to(cuda) uses its
        # tracked allocator, allowing sleep() to release every component.
        self._pipeline = QwenImage21Pipeline.from_pretrained(
            od_config.model, torch_dtype=od_config.dtype, local_files_only=True,
        )
        for name, component in self._pipeline.components.items():
            if isinstance(component, nn.Module):
                self.add_module(name, component)
        self._pipeline.set_progress_bar_config(disable=True)
        # from_pretrained has already loaded every component, with the official
        # checkpoint layout. Subsequent load_weights calls are actor updates.
        self.weights_sources = []
        self._loaded_parameter_names = set(dict(self.named_parameters()))

    def load_weights(self, weights):
        loaded = copy_transformer_weights(self.transformer, weights)
        # Cover the already-loaded frozen encoder/VAE during the loader's
        # initial empty-source check. Their bytes are not rewritten by RL.
        return self._loaded_parameter_names | loaded

    @torch.no_grad()
    def forward(self, req, **kwargs):
        requests = getattr(req, "requests", None)
        if requests is not None:
            if len(requests) != 1:
                raise ValueError("QwenImage21 uses one prompt/reference set per serving request.")
            request = requests[0]
        else:
            request = req
        payload = request.prompt
        if not isinstance(payload, dict):
            raise TypeError("QwenImage21 requires the typed diffusion request payload.")
        metadata = payload.get("mm_processor_kwargs") or {}
        prompt = metadata.get("qwen_image21_prompt")
        if not isinstance(prompt, str):
            raise ValueError("Use QwenImage21AgentLoop to preserve the raw official prompt.")
        references = condition_images_from_payload(payload)
        task_type = metadata.get("task_type")
        params = request.sampling_params
        extras = dict(params.extra_args or {})
        config = FewStepConfig(**dict(extras.get("fewstep") or {}))
        if config.bridge != self.bridge_name:
            raise ValueError("Serving registration and FewStep configuration disagree.")
        if int(getattr(params, "num_outputs_per_prompt", 1) or 1) != 1:
            raise ValueError("verl supplies each GRPO candidate as a separate request.")
        seed = getattr(params, "seed", None)
        generator = torch.Generator("cpu")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(seed))
        is_training = bool(extras.get("logprobs", False))
        result = QwenImage21FewStepRollout(self._pipeline, config).generate(
            prompt,
            task_type=task_type,
            reference_images=references,
            cycle_index=max(int(extras.get("global_steps", 1)) - 1, 0),
            generator=generator,
            height=int(params.height),
            width=int(params.width),
            output_resolution=int(extras.get("output_resolution", 1024)),
            deterministic=not is_training,
            use_kv_cache=bool(extras.get("use_kv_cache", not is_training)),
            output_type="pil",
        )
        # Preserve official RGBA pixels. Reward clients decide how to present
        # alpha to RGB-only scorers; rollout never drops the alpha channel.
        pixels = torch.from_numpy(np.array(result.images[0], copy=True)).permute(2, 0, 1).contiguous()
        if not is_training:
            return rollout_output(media=pixels, to_cpu=True)
        tensors = pad_qwen_replay_tensors(
            result.training_tensors(),
            max_prompt_length=int(extras.get("replay_prompt_length", 4096)),
            max_reference_tokens=int(extras.get("max_reference_tokens", 8192)),
            max_reference_images=int(extras.get("max_reference_images", 1)),
        )
        return rollout_output(
            media=pixels,
            trajectory_latents=tensors.pop("all_latents"),
            trajectory_timesteps=tensors.pop("all_timesteps"),
            trajectory_log_probs=tensors.pop("rollout_log_probs"),
            rl=tensors,
            to_cpu=True,
        )


@VllmOmniPipelineBase.register("QwenImage21Pipeline", algorithm="fewstep_geo")
class GeoQwenImage21Serving(_QwenImage21Serving):
    bridge_name = "geo"


@VllmOmniPipelineBase.register("QwenImage21Pipeline", algorithm="fewstep_cps")
class CPSQwenImage21Serving(_QwenImage21Serving):
    bridge_name = "cps"


@VllmOmniPipelineBase.register("QwenImage21Pipeline", algorithm="fewstep_flow_grpo")
class FlowGRPOQwenImage21Serving(_QwenImage21Serving):
    bridge_name = "flow_grpo"


def register_serving_architecture():
    from vllm_omni.diffusion.registry import register_diffusion_model

    register_diffusion_model(
        "QwenImage21Pipeline", __name__, "GeoQwenImage21Serving",
    )


register_serving_architecture()
