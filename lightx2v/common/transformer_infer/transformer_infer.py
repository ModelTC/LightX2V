import math
from abc import ABC, abstractmethod

import torch
from loguru import logger

from lightx2v.utils.registry_factory import COMPILE_BACKEND_REGISTER


class BaseTransformerInfer(ABC):
    def init_compile(self, config):
        self.use_compile = config.get("use_compile", False)
        self.compile_backend = config.get("compile_backend", "default")
        self.compiled_blocks = {}
        self._compile_backend_obj = self._create_compile_backend() if self.use_compile else None
        if self._compile_backend_obj is not None:
            logger.info(f"[Compile] Using torch.compile (backend={self.compile_backend}) for {type(self).__name__}")

    def _create_compile_backend(self):
        """Create one explicitly selected platform backend for this instance."""
        if self.compile_backend == "default":
            return None
        backend_factory = COMPILE_BACKEND_REGISTER.get(self.compile_backend)
        if backend_factory is None:
            raise ValueError(f"Unknown compile_backend={self.compile_backend!r}; expected 'default' or a registered platform backend.")
        return backend_factory()

    def get_compiled_block(self, block_idx, block):
        key = self.get_compile_block_key(block_idx, block)
        cached = self.compiled_blocks.get(key)
        if cached is not None and cached[0] is block:
            return cached[1]

        def block_runner(*args):
            return self.infer_block(block, *args)

        if self.compile_backend == "default":
            compiled = torch.compile(block_runner, dynamic=None)
        else:
            compiled = torch.compile(block_runner, dynamic=None, backend=self._compile_backend_obj)
        self.compiled_blocks[key] = (block, compiled)
        return compiled

    def get_compile_block_key(self, block_idx, block):
        return block_idx

    def run_block(self, block_idx, block, *args):
        if self.use_compile:
            return self.get_compiled_block(block_idx, block)(*args)
        return self.infer_block(block, *args)

    @abstractmethod
    def infer(self):
        pass

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.scheduler.transformer_infer = self


class BaseTaylorCachingTransformerInfer(BaseTransformerInfer):
    @abstractmethod
    def infer_calculating(self):
        pass

    @abstractmethod
    def infer_using_cache(self):
        pass

    @abstractmethod
    def get_taylor_step_diff(self):
        pass

    # 1. when fully calcualted, stored in cache
    def derivative_approximation(self, block_cache, module_name, out):
        if module_name not in block_cache:
            block_cache[module_name] = {0: out}
        else:
            step_diff = self.get_taylor_step_diff()

            previous_out = block_cache[module_name][0]
            block_cache[module_name][0] = out
            block_cache[module_name][1] = (out - previous_out) / step_diff

    def taylor_formula(self, tensor_dict):
        x = self.get_taylor_step_diff()

        output = 0
        for i in range(len(tensor_dict)):
            output += (1 / math.factorial(i)) * tensor_dict[i] * (x**i)

        return output
