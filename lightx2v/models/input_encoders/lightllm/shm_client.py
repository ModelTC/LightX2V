"""
Shared Memory Client for LightLLM Hidden States

支持从 LightLLM 服务的共享内存中直接读取 hidden states，
实现零拷贝数据传输，显著降低通信延迟。
"""

from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np
from loguru import logger


class ShmClient:
    """共享内存客户端，用于读取 LightLLM 服务的 hidden states"""

    def __init__(self):
        self._cache = {}  # 缓存已打开的共享内存对象

    def read_hidden_states(
        self,
        shm_name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.uint8,
    ) -> np.ndarray:
        """
        从共享内存读取 hidden states 数据

        Args:
            shm_name: 共享内存名称
            shape: 数据形状
            dtype: 数据类型（默认 uint8，需要后续 view 为 bfloat16）

        Returns:
            numpy 数组（数据的副本，可安全使用）
        """
        try:
            # 打开共享内存
            shm = shared_memory.SharedMemory(name=shm_name)

            # 创建 numpy 数组视图
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

            # 复制数据（确保数据独立，不依赖共享内存生命周期）
            result = arr.copy()

            # 关闭共享内存（不 unlink，因为服务端负责管理生命周期）
            shm.close()

            logger.debug(f"Read hidden states from shm '{shm_name}': shape={shape}")
            return result

        except FileNotFoundError:
            logger.error(f"Shared memory '{shm_name}' not found")
            raise
        except Exception as e:
            logger.error(f"Failed to read from shared memory '{shm_name}': {e}")
            raise

    def read_hidden_states_zero_copy(
        self,
        shm_name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.uint8,
    ) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
        """
        从共享内存读取 hidden states 数据（零拷贝模式）

        注意：此模式返回的数组直接引用共享内存，调用者需要负责：
        1. 在使用完数据后调用 shm.close()
        2. 不要在共享内存关闭后继续使用数组

        Args:
            shm_name: 共享内存名称
            shape: 数据形状
            dtype: 数据类型

        Returns:
            (numpy 数组, SharedMemory 对象) - 调用者需要管理 shm 对象的生命周期
        """
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            logger.debug(f"Zero-copy read from shm '{shm_name}': shape={shape}")
            return arr, shm
        except Exception as e:
            logger.error(f"Failed to zero-copy read from shared memory '{shm_name}': {e}")
            raise

    def is_shm_available(self, shm_name: str) -> bool:
        """检查共享内存是否可用"""
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False


# 全局单例
_shm_client: Optional[ShmClient] = None


def get_shm_client() -> ShmClient:
    """获取共享内存客户端单例"""
    global _shm_client
    if _shm_client is None:
        _shm_client = ShmClient()
    return _shm_client
