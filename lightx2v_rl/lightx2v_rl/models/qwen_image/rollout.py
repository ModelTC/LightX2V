"""FewStep sampling around the official QwenImage21Pipeline.

The upstream pipeline owns prompt templates, RGBA handling, reference resizing,
Qwen3-VL encoding, VAE normalization, latent layout and the checkpoint's sigma
grid. Only one scheduler transition is replaced by the selected RL bridge.
No copy of the upstream pipeline is maintained here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lightx2v_rl.algorithms.fewstep import FewStepConfig, bridge_step

CONDITION_KEYS = (
    "prompt_embeds",
    "prompt_embeds_mask",
    "qwen_image_latents",
    "qwen_img_shapes",
    "qwen_img_mask",
    "qwen_model_timesteps",
)


@dataclass(frozen=True)
class QwenImage21RolloutOutput:
    images: Any
    tensors: dict[str, torch.Tensor]
    selected_step: int | None

    def training_tensors(self, *, cpu: bool = True) -> dict[str, torch.Tensor]:
        if self.selected_step is None:
            raise ValueError("Deterministic evaluation has no stochastic transition to train on.")
        return {key: value.detach().cpu() if cpu else value for key, value in self.tensors.items()}


def validate_turbo_schedule(sigmas: Any) -> list[float]:
    """Require the checkpoint's actual eight-step grid, never synthesize one."""
    if sigmas is None:
        raise ValueError("Qwen-Image-2.1-Turbo requires sample_sigmas in model_index.json.")
    values = torch.as_tensor(list(sigmas), dtype=torch.float64)
    if values.shape != (8,) or not torch.isfinite(values).all():
        raise ValueError("Expected the Turbo checkpoint's eight finite sampling sigmas.")
    if values[0] != 1 or values[-1] <= 0 or not torch.all(values[:-1] > values[1:]):
        raise ValueError("Turbo sampling sigmas must descend from 1 to a positive final sampling sigma.")
    return values.tolist()


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def pad_qwen_replay_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    max_prompt_length: int,
    max_reference_tokens: int,
    max_reference_images: int = 1,
) -> dict[str, torch.Tensor]:
    """Make variable reference/prompt fields safe for generic tensor concatenation.

    Counts travel with the padded rows so the learner restores the original
    layout before its forward. Padding never moves image slots or changes the
    conditioning. Overflow is an error, never truncation.
    """
    if min(max_prompt_length, max_reference_tokens, max_reference_images) < 1:
        raise ValueError("Qwen transport padding capacities must all be positive.")
    batch_size, _, target_tokens, _ = tensors["all_latents"].shape
    lengths = {
        "qwen_prompt_lengths": tensors["prompt_embeds"].shape[1],
        "qwen_reference_lengths": tensors["qwen_image_latents"].shape[1],
        "qwen_shape_counts": tensors["qwen_img_shapes"].shape[1],
        "qwen_img_mask_lengths": tensors["qwen_img_mask"].shape[1],
    }
    capacities = {
        "prompt_embeds": max_prompt_length,
        "prompt_embeds_mask": max_prompt_length,
        "qwen_image_latents": max_reference_tokens,
        "qwen_img_shapes": max_reference_images + 1,
        "qwen_img_mask": max_prompt_length + target_tokens // 4,
    }
    padded = dict(tensors)
    for key, capacity in capacities.items():
        value = tensors[key]
        if value.shape[1] > capacity:
            raise ValueError(f"{key} length {value.shape[1]} exceeds configured transport capacity {capacity}.")
        padded_value = value.new_zeros((batch_size, capacity, *value.shape[2:]))
        padded_value[:, :value.shape[1]] = value
        padded[key] = padded_value
    for key, length in lengths.items():
        padded[key] = torch.full(
            (batch_size, 1), length, dtype=torch.long, device=tensors["all_latents"].device,
        )
    return padded


class QwenImage21FewStepRollout:
    """Serial, one-input-group sampler for both text generation and editing.

    A call can generate several candidates, all with the same prompt and ordered
    reference list. Different references require separate calls, as in the
    official pipeline. Training disables prefix caching: all prefix projections
    must participate in differentiable replay, and reduced-precision cached and
    uncached forwards are not numerically identical. Evaluation uses the
    official cached path by default.
    """

    def __init__(self, pipeline: Any, config: FewStepConfig):
        self.pipeline = pipeline
        self.config = config
        validate_turbo_schedule(pipeline.config.sample_sigmas)
        if config.min_steps != 8 or config.max_steps != 8:
            raise ValueError("Qwen-Image-2.1-Turbo keeps its checkpoint's eight-step schedule.")
        if config.timestep_sampling_mode != "fixed_uniform" or config.timestep_shift != 1:
            raise ValueError("Turbo uses its saved grid; random timestep sampling or an extra shift is unsupported.")
        if getattr(pipeline.scheduler.config, "stochastic_sampling", False):
            raise ValueError("The official deterministic Euler scheduler is required outside the RL transition.")
        self._active = False

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        task_type: str,
        reference_images: list[Any] | None = None,
        num_images_per_prompt: int = 1,
        cycle_index: int = 0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        height: int = 2048,
        width: int = 2048,
        output_resolution: int = 1024,
        deterministic: bool = False,
        output_type: str = "pil",
        latents: torch.Tensor | None = None,
        use_kv_cache: bool | None = None,
    ) -> QwenImage21RolloutOutput:
        if self._active:
            raise RuntimeError("QwenImage21FewStepRollout requires serial pipeline execution.")
        if not isinstance(prompt, str):
            raise TypeError("Call the sampler once per prompt/reference group.")
        if task_type not in {"t2i", "i2i"}:
            raise ValueError("task_type must be 't2i' or 'i2i'.")
        references = list(reference_images or [])
        if (task_type == "t2i" and references) or (task_type == "i2i" and not references):
            raise ValueError("T2I requires no reference images; I2I requires at least one.")
        if len(references) > 10:
            raise ValueError("The released Qwen-Image-2.1 model supports at most ten references.")
        if num_images_per_prompt < 1 or height < 32 or width < 32:
            raise ValueError("Use a positive candidate count and image dimensions of at least 32 pixels.")
        if height % 32 or width % 32:
            raise ValueError("Use dimensions divisible by 32 to avoid silent official-pipeline resizing.")
        if not deterministic and use_kv_cache:
            raise ValueError("Training must disable KV caching to match differentiable prefix replay.")
        call_kwargs = dict(
            prompt=prompt,
            image=references or None,
            true_cfg_scale=1.0,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            height=height,
            width=width,
            output_resolution=output_resolution,
            output_type=output_type,
            latents=latents,
            use_kv_cache=(True if use_kv_cache is None else use_kv_cache) if deterministic else False,
        )
        # Deliberately pass neither sigmas nor num_inference_steps: the saved
        # author-provided schedule remains authoritative in the official call.
        self._active = True
        try:
            if deterministic:
                return QwenImage21RolloutOutput(self.pipeline(**call_kwargs).images, {}, None)
            return self._generate_training(call_kwargs, cycle_index, generator)
        finally:
            self._active = False

    def _generate_training(self, call_kwargs, cycle_index, generator):
        pipeline = self.pipeline
        scheduler = pipeline.scheduler
        original_step = scheduler.step
        had_step_override = "step" in vars(scheduler)
        previous_step_override = vars(scheduler).get("step")
        step_index = 0
        selected_step = None
        tensors = {}

        def capture_inputs(module, args, kwargs):
            nonlocal selected_step
            if args:
                raise RuntimeError("The official QwenImage21 transformer call must use named arguments.")
            num_steps = len(scheduler.timesteps)
            if num_steps != 8:
                raise ValueError("The loaded Turbo scheduler did not produce eight denoising steps.")
            if selected_step is None:
                if self.config.cycle_train_idx:
                    selected_step = cycle_index % (num_steps - 1)
                else:
                    rng = generator[0] if isinstance(generator, list) else generator
                    rng_device = "cpu" if rng is None else rng.device
                    selected_step = int(torch.randint(num_steps - 1, (), device=rng_device, generator=rng))
            if step_index != selected_step:
                return
            if kwargs.get("kv_cache") is not None or kwargs.get("kv_cache_mode") is not None:
                raise ValueError("Training rollout unexpectedly enabled prefix KV caching.")
            if kwargs.get("attention_kwargs"):
                raise ValueError("Custom attention kwargs require an explicit matching replay implementation.")
            hidden = kwargs["hidden_states"]
            batch_size = hidden.shape[0]
            shapes = torch.as_tensor(kwargs["img_shapes"], device=hidden.device, dtype=torch.long)
            if shapes.shape[0] == 1:
                shapes = shapes.expand(batch_size, -1, -1)
            if shapes.shape[0] != batch_size or not torch.all(shapes == shapes[:1]):
                raise ValueError("One Qwen rollout group must share the same reference/target layout.")
            target_tokens = int(shapes[0, -1].prod())
            embedding = kwargs["encoder_hidden_states"]
            embedding_mask = kwargs.get("encoder_hidden_states_mask")
            if embedding_mask is None:
                embedding_mask = torch.ones(embedding.shape[:2], dtype=torch.bool, device=embedding.device)
            image_mask = kwargs["img_mask"]
            if image_mask.shape[0] == 1:
                image_mask = image_mask.expand(batch_size, -1)
            tensors.update(
                prompt_embeds=_clone_tensor(embedding),
                prompt_embeds_mask=_clone_tensor(embedding_mask),
                qwen_image_latents=_clone_tensor(hidden[:, :-target_tokens]),
                qwen_img_shapes=_clone_tensor(shapes),
                qwen_img_mask=_clone_tensor(image_mask),
                # Official code casts t to latent dtype BEFORE dividing by
                # 1000. Reconstructing from float32 sigma would change policy.
                qwen_model_timesteps=_clone_tensor(kwargs["timestep"]).reshape(batch_size, 1),
            )

        def bridge_scheduler_step(model_output, timestep, sample, *args, **kwargs):
            nonlocal step_index
            index = step_index
            step_index += 1
            sigma = scheduler.sigmas[index].float().to(sample.device)
            next_sigma = scheduler.sigmas[index + 1].float().to(sample.device)
            # Preserve scheduler state progression and every unselected Euler
            # update (including dtype handling) exactly as upstream implements it.
            official_output = original_step(model_output, timestep, sample, *args, **kwargs)
            if index != selected_step:
                return official_output
            if not isinstance(official_output, tuple):
                raise RuntimeError("The official QwenImage21 pipeline must request scheduler return_dict=False.")
            transition = bridge_step(
                model_output.float(), sample.float(), sigma, next_sigma,
                config=self.config, generator=generator,
            )
            # Keep the same latent dtype as the actual scheduler. Score the
            # rounded action that the model will observe at the following step.
            next_sample = transition.next_sample.to(official_output[0].dtype)
            scored = bridge_step(
                model_output.float(), sample.float(), sigma, next_sigma,
                config=self.config, next_sample=next_sample.float(),
            )
            batch_size = sample.shape[0]
            tensors.update(
                all_latents=torch.stack((sample.float(), next_sample.float()), dim=1),
                all_timesteps=(sigma * 1000).expand(batch_size, 1).clone(),
                all_next_timesteps=(next_sigma * 1000).expand(batch_size, 1).clone(),
                qwen_sigmas=sigma.expand(batch_size, 1).clone(),
                qwen_next_sigmas=next_sigma.expand(batch_size, 1).clone(),
                rollout_log_probs=scored.log_prob[:, None],
            )
            return (next_sample, *official_output[1:])

        handle = pipeline.transformer.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        scheduler.step = bridge_scheduler_step
        try:
            output = pipeline(**call_kwargs)
            if step_index != 8 or selected_step is None or "all_latents" not in tensors:
                raise RuntimeError("The rollout ended before completing its trainable Turbo trajectory.")
            return QwenImage21RolloutOutput(output.images, tensors, selected_step)
        finally:
            handle.remove()
            if had_step_override:
                scheduler.step = previous_step_override
            else:
                del scheduler.step
