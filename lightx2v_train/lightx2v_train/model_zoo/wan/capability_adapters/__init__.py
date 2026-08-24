"""Capabilities implemented specifically by Wan-family models."""

from .wan_consistency_model_capability import LingBotConsistencyModelCapability, WanConsistencyModelCapability
from .wan_flow_matching_capability import WanFlowMatchingCapability

__all__ = [
    "LingBotConsistencyModelCapability",
    "WanConsistencyModelCapability",
    "WanFlowMatchingCapability",
]
