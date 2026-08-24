"""Capabilities implemented specifically by Flux2 models."""

from .flux2_consistency_model_capability import Flux2ConsistencyModelCapability
from .flux2_edit_distribution_matching_capability import Flux2EditDistributionMatchingCapability

__all__ = [
    "Flux2ConsistencyModelCapability",
    "Flux2EditDistributionMatchingCapability",
]
