"""Core-op statistics entry; MM, attention, conv, and MoE bases provide the formulas.

Subclasses still implement apply. Install a default-off diagnostic branch without changing arguments or returns.
Collect only when an explicit WorkloadStats session selects the family; do not double-count super().apply.
"""

from functools import wraps
from inspect import signature

from lightx2v.utils import workload_stats


def _observed(method):
    parameters = signature(method)

    @wraps(method)
    def call(self, *args, **kwargs):
        if not workload_stats.enabled(self.compute_family):
            return method(self, *args, **kwargs)
        return workload_stats.capture_core_op(self, method, parameters, args, kwargs)

    return call


class CoreOpStatistics:
    compute_family = None
    compute_methods = ("apply",)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Wrap only entrypoints implemented by this class at creation; leave inherited wrappers intact.
        # Attention also declares apply_with_lse to cover the compute entry actually used by Ring.
        for name in cls.compute_methods:
            method = cls.__dict__.get(name)
            if method is not None and not getattr(method, "__isabstractmethod__", False):
                setattr(cls, name, _observed(method))

    def compute_stats(self, arguments, output):
        """Return a reason for unsupported contracts; sessions must not infer FLOPs from class names."""
        return {"unsupported_reason": "this core op has no compute contract"}
