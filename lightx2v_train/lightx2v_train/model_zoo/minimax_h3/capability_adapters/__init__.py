"""Capability adapters for MiniMax-H3."""

from .common import MiniMaxH3JointLatents, MiniMaxH3LatentShape
from .minimax_h3_distillation_capability import MiniMaxH3DistillationCapability

__all__ = [
    "MiniMaxH3DistillationCapability",
    "MiniMaxH3JointLatents",
    "MiniMaxH3LatentShape",
]
