"""Capability adapters for MiniMax-H3."""

from .common import MiniMaxH3JointLatents, MiniMaxH3LatentShape
from .minimax_h3_consistency_model_capability import MiniMaxH3ConsistencyModelCapability
from .minimax_h3_distribution_matching_capability import MiniMaxH3DistributionMatchingCapability

__all__ = [
    "MiniMaxH3ConsistencyModelCapability",
    "MiniMaxH3DistributionMatchingCapability",
    "MiniMaxH3JointLatents",
    "MiniMaxH3LatentShape",
]
