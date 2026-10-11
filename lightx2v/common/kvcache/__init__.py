from .fifo import FIFOKVCachePool
from .manager import KVCacheManager
from .rolling import HybridStepRollingKVCachePool, RollingKVCachePool, SpatialRollingKVCachePool
from .static import StaticKVCachePool

__all__ = [
    "FIFOKVCachePool",
    "HybridStepRollingKVCachePool",
    "KVCacheManager",
    "RollingKVCachePool",
    "SpatialRollingKVCachePool",
    "KIVIQuantRollingKVCachePool",
    "StepKiviQuantRollingKVCachePool",
    "StaticKVCachePool",
]


def __getattr__(name):
    if name in {"KIVIQuantRollingKVCachePool", "StepKiviQuantRollingKVCachePool"}:
        from . import quant

        return getattr(quant, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
