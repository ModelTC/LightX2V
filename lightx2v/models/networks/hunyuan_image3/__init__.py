__all__ = ["HunyuanImage3Model"]


def __getattr__(name):
    if name == "HunyuanImage3Model":
        from .model import HunyuanImage3Model

        globals()[name] = HunyuanImage3Model
        return HunyuanImage3Model
    raise AttributeError(name)
