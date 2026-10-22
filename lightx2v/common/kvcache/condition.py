import torch


class ConditionKVCache:
    """Projected condition KV, isolated by branch, modality, layer and variant."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.generation = 0
        self._contexts = {}
        self._values = {}

    @staticmethod
    def _source_signature(source: torch.Tensor):
        try:
            version = source._version
        except RuntimeError:
            # Inference tensors cannot report in-place condition changes.
            return None
        return source.data_ptr(), tuple(source.shape), tuple(source.stride()), source.dtype, source.device, version

    def set_context(self, name, *sources: torch.Tensor, cache_key=None) -> bool:
        if not self.enabled:
            return False
        signatures = tuple(self._source_signature(source) for source in sources)
        signature = (signatures, cache_key)
        cached = self._contexts.get(name)
        reusable = all(item is not None for item in signatures)
        if reusable and cached is not None and cached[0] == signature:
            return False
        self.reset(name)
        if reusable:
            # Retain source storage so an allocator cannot recycle its identity.
            self._contexts[name] = signature, sources
            self._values[name] = {}
        return True

    def get(self, name, layer_id, variant=None):
        if not self.enabled:
            return None
        return self._values.get(name, {}).get((layer_id, variant))

    def store(self, name, layer_id, kv, variant=None):
        if self.enabled and name in self._values:
            self._values[name][layer_id, variant] = kv
        return kv

    def reset(self, name=None):
        if name is None:
            self._contexts.clear()
            self._values.clear()
        else:
            self._contexts.pop(name, None)
            self._values.pop(name, None)
        self.generation += 1
