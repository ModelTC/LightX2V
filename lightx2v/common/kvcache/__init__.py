"""
KV cache for autoregressive transformer inference.

- ``base``: cross-attention pool
- ``rolling``: ``RollingKVCachePool`` (bf16 rolling-window cache)
- ``quant``:   ``CalibRollingKVCachePool`` / ``QuantRollingKVCachePool``
- ``offload``: ``OffloadRollingKVCachePool`` / ``OffloadQuantRollingKVCachePool``
- ``manager``: ``KVCacheManager``
"""

from .manager import KVCacheManager
from .offload import OffloadQuantRollingKVCachePool, OffloadRollingKVCachePool
from .quant import CalibRollingKVCachePool, QuantRollingKVCachePool
from .rolling import RollingKVCachePool

__all__ = [
    "KVCacheManager",
    "RollingKVCachePool",
    "CalibRollingKVCachePool",
    "QuantRollingKVCachePool",
    "OffloadRollingKVCachePool",
    "OffloadQuantRollingKVCachePool",
]
