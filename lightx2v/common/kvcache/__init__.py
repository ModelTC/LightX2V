from .fifo import FIFOKVCachePool
from .manager import KVCacheManager
from .rolling import HybridStepRollingKVCachePool, RollingKVCachePool, SpatialRollingKVCachePool
from .static import StaticKVCachePool

try:
    from . import quant
except ImportError as error:
    quant = None
    _quant_import_error = error
else:
    _quant_import_error = None

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
        if _quant_import_error is not None:
            raise _quant_import_error

        return getattr(quant, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
