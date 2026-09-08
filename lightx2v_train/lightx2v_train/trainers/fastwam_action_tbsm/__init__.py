__all__ = ["FastWAMActionTBSMTrainer"]


def __getattr__(name):
    if name == "FastWAMActionTBSMTrainer":
        from .trainer import FastWAMActionTBSMTrainer

        return FastWAMActionTBSMTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
