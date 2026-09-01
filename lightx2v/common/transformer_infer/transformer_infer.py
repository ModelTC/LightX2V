import math
from abc import ABC, abstractmethod

import torch
from loguru import logger


class BaseTransformerInfer(ABC):
    def init_compile(self, config):
        self.use_compile = config.get("use_compile", False)
        # compile_backend: "default" -> plain torch.compile; "mindie" -> MindieSDBackend
        self.compile_backend = config.get("compile_backend", "default")
        # compile_dynamic: "None" (default auto), True, or False. H3 per-step
        # sequence is fixed, so False skips per-call shape guard checks.
        self.compile_dynamic = config.get("compile_dynamic", None)
        self.compiled_blocks = {}
        self._compile_backend_obj = self._create_compile_backend() if self.use_compile else None
        if self.use_compile:
            logger.info(f"[Compile] Using torch.compile (backend={self.compile_backend}) for {type(self).__name__}")

    def _create_compile_backend(self):
        """Instantiate the configured compile backend, or None for the default.

        Reuse ONE backend instance: a fresh MindieSDBackend() per call makes
        Dynamo see a different backend callable each time (BACKEND_MATCH
        recompilation until recompile_limit, then silent eager fallback).
        Unknown names and unavailable optional backends degrade to the default
        torch.compile with a warning.
        """
        if self.compile_backend not in ("default", "mindie"):
            logger.warning(f"[Compile] Unknown compile_backend={self.compile_backend!r}; expected 'default' or 'mindie'. Falling back to 'default'.")
            self.compile_backend = "default"
            return None
        if self.compile_backend == "default":
            return None
        try:
            from mindiesd.compilation import MindieSDBackend

            return MindieSDBackend()
        except Exception as e:  # pragma: no cover - optional dependency path
            logger.warning(f"[Compile] MindieSDBackend unavailable ({e}); falling back to default torch.compile")
            self.compile_backend = "default"
            return None

    def get_compiled_block(self, block_idx, block):
        key = self.get_compile_block_key(block_idx, block)
        cached = self.compiled_blocks.get(key)
        if cached is not None and cached[0] is block:
            return cached[1]

        def block_runner(*args):
            return self.infer_block(block, *args)

        compile_kwargs = {}
        if self.compile_backend == "mindie" and self._compile_backend_obj is not None:
            compile_kwargs["backend"] = self._compile_backend_obj
        # dynamic=False: H3 per-step sequence is fixed (9467 local); fixed-shape
        # graphs skip Dynamo's per-call shape guard checks. Fall back to
        # dynamic=None if the graph contains truly dynamic inputs (Sym symbols).
        dynamic_mode = getattr(self, "compile_dynamic", None)
        compiled = torch.compile(block_runner, dynamic=dynamic_mode, **compile_kwargs)
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
