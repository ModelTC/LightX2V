"""LightX2V-native wrapper around the official PyTorch pi0.5 architecture."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import Pi0Config
from .infer import OpenPIPostInfer, OpenPIPreInfer, OpenPITransformerInfer
from .weights import load_pi05_libero_weights


def _read_config(config: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), path.parent
    if isinstance(config, Mapping):
        return dict(config), None
    try:
        return dict(config), None
    except (TypeError, ValueError) as exc:
        raise TypeError("OpenPI config must be a mapping or a JSON path") from exc


def _resolve_path(value: str | Path | None, base_dir: Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


class OpenPIModel(nn.Module):
    """Native model/pipeline split into pre-, transformer-, post-infer stages.

    Runtime imports no code from ``/data/liuhongda/openpi`` and requires no
    JAX stack.  The one-time checkpoint conversion remains an offline step.
    """

    def __init__(
        self,
        core_model: nn.Module,
        pre_infer: OpenPIPreInfer,
        transformer_infer: OpenPITransformerInfer,
        post_infer: OpenPIPostInfer,
        model_config: Pi0Config,
        device: torch.device | str,
        seed: int | None = 0,
    ):
        super().__init__()
        self.core_model = core_model
        self.pre_infer = pre_infer
        self.transformer_infer = transformer_infer
        self.post_infer = post_infer
        self.model_config = model_config
        self.device = torch.device(device)
        self.seed = None if seed is None or int(seed) < 0 else int(seed)
        self._generator: torch.Generator | None = None
        self.reset()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | str | Path) -> "OpenPIModel":
        values, base_dir = _read_config(config)
        model_values = dict(values.get("model", {}))
        # Flat LightX2V configs remain supported; nested model values win.
        model_values = {**values, **model_values}
        model_config = Pi0Config.from_mapping(model_values)
        model_config.validate_pi05_libero()

        checkpoint_value = values.get("checkpoint_dir", values.get("model_path"))
        checkpoint_path = _resolve_path(checkpoint_value, base_dir)
        # A runner/ROS --model_path override must win over the static JSON
        # weight_path.  With no override, the explicit JSON weight_path is used.
        if checkpoint_path is not None:
            weight_path = checkpoint_path if checkpoint_path.suffix == ".safetensors" else checkpoint_path / "model.safetensors"
        else:
            weight_path = _resolve_path(values.get("weight_path"), base_dir)
            if weight_path is None:
                raise ValueError("OpenPI config requires checkpoint_dir/model_path or weight_path")
        checkpoint_dir = weight_path.parent

        norm_stats_path = _resolve_path(values.get("norm_stats_path"), base_dir)
        if norm_stats_path is None:
            norm_stats_path = checkpoint_dir / "assets/physical-intelligence/libero/norm_stats.json"
        tokenizer_path = _resolve_path(values.get("tokenizer_path"), base_dir)
        if tokenizer_path is None:
            tokenizer_candidates = [
                checkpoint_dir / "assets/paligemma_tokenizer.model",
                checkpoint_dir / "paligemma_tokenizer.model",
            ]
            tokenizer_path = next((path for path in tokenizer_candidates if path.is_file()), tokenizer_candidates[0])

        device = torch.device(values.get("device", "cuda"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("OpenPI config requests CUDA, but torch.cuda.is_available() is false")
        num_steps = int(values.get("num_inference_steps", values.get("num_flow_steps", values.get("num_steps", 10))))
        output_action_dim = int(values.get("output_action_dim", 7))
        if output_action_dim != 7:
            raise ValueError("The released pi05_libero policy must output 7-D LIBERO actions")

        core_model = load_pi05_libero_weights(weight_path, model_config, device)
        return cls(
            core_model=core_model,
            pre_infer=OpenPIPreInfer(
                norm_stats_path=norm_stats_path,
                tokenizer_path=tokenizer_path,
                device=device,
                action_dim=model_config.action_dim,
                max_token_len=model_config.max_token_len,
                discrete_state_input=model_config.discrete_state_input,
            ),
            transformer_infer=OpenPITransformerInfer(num_steps=num_steps),
            post_infer=OpenPIPostInfer(norm_stats_path, output_action_dim=output_action_dim),
            model_config=model_config,
            device=device,
            seed=values.get("seed", 0),
        )

    def _make_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    def reset(self) -> None:
        self._generator = None if self.seed is None else self._make_generator(self.seed)

    def _sample_noise(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            generator = self._make_generator(int(seed))
        else:
            generator = self._generator
        return torch.randn(
            (1, self.model_config.action_horizon, self.model_config.action_dim),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

    @torch.no_grad()
    def predict_normalized_action_chunk(
        self,
        images: dict,
        state,
        task_description: str,
        *,
        seed: int | None = None,
        noise: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        observation = self.pre_infer.infer(images, state, task_description)
        if noise is None:
            noise_tensor = self._sample_noise(seed)
        else:
            noise_tensor = torch.as_tensor(noise, dtype=torch.float32, device=self.device)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor.unsqueeze(0)
            expected = (1, self.model_config.action_horizon, self.model_config.action_dim)
            if tuple(noise_tensor.shape) != expected:
                raise ValueError(f"Noise must have shape {expected}, got {tuple(noise_tensor.shape)}")
        return self.transformer_infer.infer(self.core_model, observation, self.device, noise=noise_tensor)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        images: dict,
        state,
        task_description: str,
        seed: int | None = None,
    ) -> np.ndarray:
        normalized = self.predict_normalized_action_chunk(images, state, task_description, seed=seed)
        actions = self.post_infer.infer(normalized)
        expected = (self.model_config.action_horizon, 7)
        if actions.shape != expected:
            raise RuntimeError(f"OpenPI returned {actions.shape}; expected {expected}")
        return actions

    def next_action(self, images: dict, state, task_description: str) -> np.ndarray:
        return self.predict_action_chunk(images, state, task_description)[0]

    def forward(self, observation, actions, noise=None, time=None):
        """Expose the official training loss for future LightX2V fine-tuning."""
        return self.core_model(observation, actions, noise=noise, time=time)

    def close(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
