"""External config types, keeping algorithm options out of verl-omni core."""

from dataclasses import dataclass, field, fields
from typing import Any

from verl_omni.trainer.config import DiffusionAlgoConfig
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionRolloutAlgoConfig

from lightx2v_rl.algorithms.fewstep import FewStepConfig
from lightx2v_rl.models.registry import get_model_spec

FEWSTEP_LOSS_MODES = frozenset({"fewstep_geo", "fewstep_cps", "fewstep_flow_grpo"})


@dataclass
class FewStepAlgoConfig(DiffusionAlgoConfig):
    """External advantage options; verl-omni only dispatches the generic hook."""

    custom_advantage_function: str = "lightx2v_rl.backends.verl_omni.advantage.compute_advantages"
    weight_advantages: bool = True
    reward_advantage: dict[str, Any] = field(default_factory=lambda: {
        "normalization": "max_group_std", "epsilon": 1e-4, "weights": {},
    })


def get_fewstep_config(model_config: Any) -> FewStepConfig:
    """Read the same bridge configuration used by the rollout producer."""
    options = getattr(model_config, "fewstep", None)
    if options is None:
        raise ValueError("FewStep training requires FewStepModelConfig.fewstep.")
    config = options if isinstance(options, FewStepConfig) else FewStepConfig(**dict(options))
    expected = f"fewstep_{config.bridge}"
    if model_config.algorithm != expected:
        raise ValueError(f"Model algorithm {model_config.algorithm!r} must match {expected!r}.")
    return config


@dataclass
class FewStepModelConfig(DiffusionModelConfig):
    """Model config with a single, explicit FewStep algorithm namespace."""

    fewstep: dict[str, Any] = field(default_factory=dict)
    rl_adapter: str | None = None

    def __post_init__(self):
        get_fewstep_config(self)
        super().__post_init__()
        get_model_spec(self.architecture, self.rl_adapter)


@dataclass
class FewStepRolloutAlgoConfig(DiffusionRolloutAlgoConfig):
    """Serialize FewStep options through the existing rollout request contract."""

    fewstep: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        FewStepConfig(**dict(self.fewstep))


@dataclass
class FewStepLossConfig(DiffusionLossConfig):
    """Allow namespaced external losses without modifying upstream validation."""

    loss_mode: str = "fewstep_geo"
    clip_ratio: float = 0.001

    def __post_init__(self):
        if self.loss_mode not in FEWSTEP_LOSS_MODES:
            raise ValueError(f"Expected a FewStep loss mode, got {self.loss_mode!r}.")
        # Reuse validation of the base options; the base class's fixed list of
        # builtin loss names deliberately does not govern external algorithms.
        options = {item.name: getattr(self, item.name) for item in fields(DiffusionLossConfig)}
        options["loss_mode"] = "flow_grpo"
        DiffusionLossConfig(**options)
