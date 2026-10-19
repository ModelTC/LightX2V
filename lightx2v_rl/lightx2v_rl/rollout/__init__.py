"""Trajectory collection independent of the serving or training runtime."""

from .trajectory import FewStepTrajectory, sample_trajectory

__all__ = ["FewStepTrajectory", "sample_trajectory"]
