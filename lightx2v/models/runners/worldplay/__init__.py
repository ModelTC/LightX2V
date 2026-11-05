from importlib import import_module

__all__ = ["WorldPlayARRunner", "WorldPlayBIRunner", "WorldPlayDistillRunner"]


def __getattr__(name):
    modules = {"WorldPlayARRunner": "worldplay_ar_runner", "WorldPlayBIRunner": "worldplay_bi_runner", "WorldPlayDistillRunner": "worldplay_distill_runner"}
    if name in modules:
        value = getattr(import_module(f"{__name__}.{modules[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
