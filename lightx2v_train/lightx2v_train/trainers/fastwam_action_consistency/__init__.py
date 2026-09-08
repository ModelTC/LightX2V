__all__ = ["FastWAMActionConsistencyTrainer"]


def __getattr__(name):
    if name == "FastWAMActionConsistencyTrainer":
        from .trainer import FastWAMActionConsistencyTrainer

        return FastWAMActionConsistencyTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
