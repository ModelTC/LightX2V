from functools import partial

import torch

from lightx2v.common.kvcache.condition import ConditionKVCache


class AutoregressiveTransformerInferMixin:
    """Share block compilation and cross-attention cache lifetime."""

    def __init__(self, config):
        super().__init__(config)
        self.cross_attn_kv_cache = ConditionKVCache(enabled=config.get("ar_config", {}).get("use_cross_attn_kv_cache", True))
        self._cross_attn_branch = True
        self.compiled_kv_blocks = {}

    def get_compiled_block(self, block_idx, block):
        key = self.get_compile_block_key(block_idx, block)
        cached = self.compiled_blocks.get(key)
        if cached is not None and cached[0] is block:
            return cached[1]
        if self.compile_backend == "default":
            options = {"options": self.compile_options} if self.compile_options else {}
            compiled = torch.compile(self.infer_block, dynamic=None, **options)
        else:
            compiled = torch.compile(self.infer_block, dynamic=None, backend=self._compile_backend_obj)
        runner = partial(compiled, block)
        self.compiled_blocks[key] = block, runner
        return runner

    def get_compiled_kv_block(self, block_idx, block):
        cached = self.compiled_kv_blocks.get(block_idx)
        if cached is not None and cached[0] is block:
            return cached[1]
        options = {"options": self.compile_options} if self.compile_options else {}
        compiled = torch.compile(self.infer_kv_block, dynamic=None, **options)
        runner = partial(compiled, block)
        self.compiled_kv_blocks[block_idx] = block, runner
        return runner

    def run_kv_block(self, block_idx, block, *args):
        if self.use_compile:
            return self.get_compiled_kv_block(block_idx, block)(*args)
        return self.infer_kv_block(block, *args)

    def set_cross_attn_context(self, *sources, branch=True, kind="text", cache_key=None):
        self._cross_attn_branch = branch
        return self.cross_attn_kv_cache.set_context((branch, kind), *sources, cache_key=cache_key)

    def reset_cross_attn_cache(self):
        self.cross_attn_kv_cache.reset()
