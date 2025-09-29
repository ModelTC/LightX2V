import asyncio
import time
from functools import wraps

import torch
from loguru import logger

from lightx2v.utils.envs import *


class MetricsProfilingContext:
    def __init__(self, metrics_func, labels=None):
        self.metrics_func = metrics_func

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        if self.labels:
            metrics_func.labels(self.labels).observe(elapsed)
        else:
            metrics_func.observe(elapsed)
        return False

    async def __aenter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        if self.labels:
            metrics_func.labels(self.labels).observe(elapsed)
        else:
            metrics_func.observe(elapsed)
        return False

    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper

