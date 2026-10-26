"""FewStepRL's Wan algorithms with IMMB exposed as Geo-FSRL (``geo``).

Only PyTorch is required. Reward and advantage computation remain the
responsibility of the host trainer.
"""

from .bridge import BridgeStepOutput, bridge_step, euler_step, geo_coefficients, geo_reweight_scale
from .config import FewStepConfig
from .objective import PolicyLossOutput, compute_policy_loss, compute_reweight_scales
from .timesteps import TimestepPathBatch, build_timestep_paths, timestep_shift

__all__ = [
    "BridgeStepOutput",
    "FewStepConfig",
    "PolicyLossOutput",
    "TimestepPathBatch",
    "bridge_step",
    "build_timestep_paths",
    "compute_policy_loss",
    "compute_reweight_scales",
    "euler_step",
    "geo_coefficients",
    "geo_reweight_scale",
    "timestep_shift",
]
