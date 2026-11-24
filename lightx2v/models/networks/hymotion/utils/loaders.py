"""Resolve only modules included in this adapter, never the upstream checkout."""

from importlib import import_module


def load_object(name, args):
    name = name.replace("/", ".")
    prefixes = {"hymotion.network.": "lightx2v.models.networks.hymotion.weights.", "hymotion.pipeline.": "lightx2v.models.networks.hymotion.weights."}
    for prefix, replacement in prefixes.items():
        if name.startswith(prefix):
            name = replacement + name[len(prefix) :]
            break
    if not name.startswith("lightx2v.models.networks.hymotion."):
        raise ValueError(f"Unsupported HY-Motion component: {name}")
    module, cls = name.rsplit(".", 1)
    return getattr(import_module(module), cls)(**args)
