"""Register this model family and shared FewStep losses in learner workers."""

from lightx2v_rl.backends.verl_omni.loss import FewStepLoss

from .learner import FewStepQwenImage21Model

__all__ = ["FewStepLoss", "FewStepQwenImage21Model"]
