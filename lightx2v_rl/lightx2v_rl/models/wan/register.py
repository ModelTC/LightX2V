"""Register this model family and shared FewStep losses in learner workers."""

from lightx2v_rl.backends.verl_omni.loss import FewStepLoss

from .learner import FewStepWanModel

__all__ = ["FewStepLoss", "FewStepWanModel"]
