"""LightX2V-native wrapper around the official PyTorch pi0.5 architecture."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import Pi0Config
from .infer import OpenPIPostInfer, OpenPIPreInfer, OpenPITransformerInfer
from .weights import load_pi05_libero_weights


class OpenPIModel(nn.Module):
    """PyTorch pi0.5 model with LightX2V inference stages."""

    def __init__(
        self,
        core_model: nn.Module,
        pre_infer: OpenPIPreInfer,
        transformer_infer: OpenPITransformerInfer,
        post_infer: OpenPIPostInfer,
        model_config: Pi0Config,
        device: torch.device | str,
        seed: int = 0,
    ):
        super().__init__()
        self.core_model = core_model
        self.pre_infer = pre_infer
        self.transformer_infer = transformer_infer
        self.post_infer = post_infer
        self.model_config = model_config
        self.device = torch.device(device)
        self.seed = seed
        self._generator: torch.Generator
        self.reset()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "OpenPIModel":
        values = dict(config)
        model_config = Pi0Config.from_mapping(values)

        checkpoint_dir = Path(values["model_path"]).expanduser().resolve()
        weight_path = checkpoint_dir / "model.safetensors"
        norm_stats_path = checkpoint_dir / "assets/physical-intelligence/libero/norm_stats.json"
        tokenizer_path = checkpoint_dir / "assets/paligemma_tokenizer.model"

        device = torch.device(values["device"])
        core_model = load_pi05_libero_weights(weight_path, model_config, device)
        return cls(
            core_model=core_model,
            pre_infer=OpenPIPreInfer(
                norm_stats_path=norm_stats_path,
                tokenizer_path=tokenizer_path,
                device=device,
                action_dim=model_config.action_dim,
                max_token_len=model_config.max_token_len,
            ),
            transformer_infer=OpenPITransformerInfer(num_steps=values["num_inference_steps"]),
            post_infer=OpenPIPostInfer(norm_stats_path),
            model_config=model_config,
            device=device,
            seed=values.get("seed", 0),
        )

    def _make_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return generator

    def reset(self) -> None:
        self._generator = self._make_generator(self.seed)

    def get_rng_state(self) -> torch.Tensor:
        return self._generator.get_state()

    def set_rng_state(self, state: torch.Tensor) -> None:
        self._generator.set_state(state)

    def _sample_noise(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            generator = self._make_generator(seed)
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
        images: dict[str, np.ndarray],
        state: np.ndarray,
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
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task_description: str,
        seed: int | None = None,
    ) -> np.ndarray:
        normalized = self.predict_normalized_action_chunk(images, state, task_description, seed=seed)
        return self.post_infer.infer(normalized)
