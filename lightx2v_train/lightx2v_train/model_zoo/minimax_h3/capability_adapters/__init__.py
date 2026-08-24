"""Capability adapters for MiniMax-H3."""

from .common import MiniMaxH3JointLatents, MiniMaxH3LatentShape
from .minimax_h3_distribution_matching_capability import MiniMaxH3DistributionMatchingCapability
from .minimax_h3_flow_matching_capability import MiniMaxH3FlowMatchingCapability

__all__ = [
    "MiniMaxH3DistributionMatchingCapability",
    "MiniMaxH3FlowMatchingCapability",
    "MiniMaxH3JointLatents",
    "MiniMaxH3LatentShape",
]
