from abc import ABCMeta, abstractmethod

from lightx2v.common.ops.compute_stats import attention_flops
from lightx2v.utils.core_op_stats import CoreOpStatistics

from .compute_stats import from_mask, from_ranges, from_sparsity


class AttnWeightTemplate(CoreOpStatistics, metaclass=ABCMeta):
    compute_family = "attention"
    compute_methods = ("apply", "apply_with_lse")

    def compute_stats(self, arguments, output):
        if any(name not in arguments for name in ("q", "k", "v")):
            return {"unsupported_reason": "attention contract requires explicit Q/K/V"}
        q, k, v = (arguments[name] for name in ("q", "k", "v"))
        if any(not hasattr(value, "ndim") for value in (q, k, v)):
            return {"unsupported_reason": "packed/quantized KV needs a backend-specific logical shape contract"}
        if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
            return {"unsupported_reason": "attention apply requires SHD or BSHD Q/K/V"}
        batch = q.shape[0] if q.ndim == 4 else 1
        sq, sk, heads = q.shape[-3], k.shape[-3], q.shape[-2]
        facts = {
            "shape": {"q": list(q.shape), "k": list(k.shape), "v": list(v.shape), "layout": "SHD" if q.ndim == 3 else "BSHD"},
            "work": {
                "dense_equivalent_flops": attention_flops(batch * heads * sq * sk, q.shape[-1], v.shape[-1]),
                "definition": "Dense-equivalent QK+PV only; actual mask, causal, varlen and routing work require backend evidence",
            },
        }
        if getattr(self, "sparse_compute_gap", None):
            facts["sparse_schedule_gap"] = self.sparse_compute_gap
            facts["note"] = "只有稠密等价量，实际稀疏计算量未知：" + self.sparse_compute_gap
        return facts

    # Pure contracts for shape/top-k, final masks, and final ranges; sessions and switches live elsewhere.
    compute_sparse_stats_from_sparsity = staticmethod(from_sparsity)
    compute_sparse_stats = staticmethod(from_mask)
    compute_sparse_range_stats = staticmethod(from_ranges)

    def __init__(self, weight_name):
        self.weight_name = weight_name
        self.config = {}

    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cpu(self, non_blocking=False):
        pass

    def to_cuda(self, non_blocking=False):
        pass

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        return {}

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        pass
