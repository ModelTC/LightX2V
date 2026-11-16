__all__ = ["HunyuanImage3Runner"]


def __getattr__(name):
    if name == "HunyuanImage3Runner":
        from .hunyuan_image3_runner import HunyuanImage3Runner

        globals()[name] = HunyuanImage3Runner
        return HunyuanImage3Runner
    raise AttributeError(name)
