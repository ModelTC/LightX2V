"""Training wrapper around LightX2V's localized OpenPI pi0.5 network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from lightx2v_train.runtime.distributed import get_device
from lightx2v_train.utils.registry import MODEL_REGISTER


@MODEL_REGISTER("openpi_pi05_libero")
class OpenPIPi05LiberoModel:
    """Load a converted SafeTensors checkpoint as model-only initialization."""

    def __init__(self, config: dict[str, Any]):
        self.model_config = dict(config["model"])
        self.initialization_path = Path(self.model_config["checkpoint_dir"]).expanduser().resolve()
        self.device = get_device()
        self.core_model: torch.nn.Module | None = None

    def load_components(self, *, load_transformer: bool, load_vae: bool, load_condition_encoder: bool) -> None:
        del load_vae, load_condition_encoder
        if not load_transformer:
            raise ValueError("OpenPI training requires load_transformer=true")
        if self.core_model is not None:
            return

        weight_path = self.initialization_path / "model.safetensors"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Converted OpenPI weights not found: {weight_path}")

        from lightx2v.models.networks.openpi.config import Pi0Config
        from lightx2v.models.networks.openpi.weights import load_pi05_libero_weights

        if self.model_config["parameter_dtype"] != "float32":
            raise ValueError("OpenPI training requires model.parameter_dtype='float32'")
        if not self.model_config["require_fp32_checkpoint"]:
            raise ValueError("OpenPI training requires model.require_fp32_checkpoint=true")
        pi0_config = Pi0Config.from_mapping(self.model_config)
        pi0_config.validate_pi05_libero()
        self.core_model = load_pi05_libero_weights(weight_path, pi0_config, self.device)
        self.core_model.train()
        logger.info("[openpi:model] model-only initialization checkpoint={}", self.initialization_path)

    def require_core_model(self) -> torch.nn.Module:
        if self.core_model is None:
            raise RuntimeError("OpenPI model components have not been loaded")
        return self.core_model

    def enable_gradient_checkpointing(self) -> None:
        model = self.require_core_model()
        model.gradient_checkpointing_enable()

    def architecture_metadata(self) -> dict[str, Any]:
        keys = (
            "pi05",
            "discrete_state_input",
            "paligemma_variant",
            "action_expert_variant",
            "action_dim",
            "action_horizon",
            "max_token_len",
        )
        metadata = {key: self.model_config[key] for key in keys}
        metadata["compute_dtype"] = self.model_config["dtype"]
        metadata["parameter_dtype"] = self.model_config["parameter_dtype"]
        metadata["require_fp32_checkpoint"] = self.model_config["require_fp32_checkpoint"]
        return metadata
