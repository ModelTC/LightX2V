"""Load pure Torch unit-test modules without LightX2V's eager device bootstrap."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType


@contextmanager
def cpu_imports():
    root = Path(__file__).resolve().parents[1]
    saved = {name: module for name, module in sys.modules.items() if name.startswith("lightx2v")}
    for name in saved:
        del sys.modules[name]
    packages = (
        "lightx2v",
        "lightx2v.common",
        "lightx2v.common.ops",
        "lightx2v.common.ops.attn",
        "lightx2v.common.ops.conv",
        "lightx2v.utils",
        "lightx2v.models",
        "lightx2v_platform",
        "lightx2v_platform.base",
    )
    try:
        for name in packages:
            module = ModuleType(name)
            module.__path__ = [str(root.joinpath(*name.split(".")))]
            sys.modules[name] = module
        yield
    finally:
        for name in list(sys.modules):
            if name.startswith("lightx2v"):
                del sys.modules[name]
        sys.modules.update(saved)
