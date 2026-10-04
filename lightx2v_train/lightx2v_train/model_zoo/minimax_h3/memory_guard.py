"""Optional topology and peak-memory checks for H3 DMD."""

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.distributed as dist

from lightx2v_train.runtime.distributed import (
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_world_size,
)


@dataclass(frozen=True)
class MiniMaxH3MemoryGuardConfig:
    """Runtime guard for the documented sub-80-GiB H3 topology.

    ``device_limit_gib`` is the hard reserved-memory limit. Peak allocated
    memory must additionally stay below ``device_limit_gib -
    safety_margin_gib``, leaving communication/allocator headroom instead of
    treating 80 GiB of live tensors as a target.
    """

    enabled: bool = False
    device_limit_gib: float = 80.0
    safety_margin_gib: float = 2.0
    min_fsdp_size: int = 4
    continuous: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping | None):
        value = value or {}
        config = cls(
            enabled=bool(value.get("enabled", False)),
            device_limit_gib=float(value.get("device_limit_gib", 80.0)),
            safety_margin_gib=float(value.get("safety_margin_gib", 2.0)),
            min_fsdp_size=int(value.get("min_fsdp_size", 4)),
            continuous=bool(value.get("continuous", False)),
        )
        if config.device_limit_gib <= 0:
            raise ValueError("training.minimax_h3.memory_guard.device_limit_gib must be positive.")
        if config.safety_margin_gib < 0:
            raise ValueError("training.minimax_h3.memory_guard.safety_margin_gib must be non-negative.")
        if config.safety_margin_gib >= config.device_limit_gib:
            raise ValueError("training.minimax_h3.memory_guard.safety_margin_gib must be smaller than device_limit_gib.")
        if config.min_fsdp_size < 1:
            raise ValueError("training.minimax_h3.memory_guard.min_fsdp_size must be positive.")
        return config

    @property
    def peak_threshold_gib(self):
        return self.device_limit_gib - self.safety_margin_gib


def _configured_h3_parallel_sizes(config):
    distributed = config.get("distributed", {})
    sequence_parallel = distributed.get("sequence_parallel", {})
    if isinstance(sequence_parallel, Mapping):
        sp_enabled = bool(sequence_parallel.get("enabled", False))
        sp_size = int(sequence_parallel.get("size", 1)) if sp_enabled else 1
    else:
        sp_size = 1
    fsdp = distributed.get("fsdp2", {})
    fsdp_enabled = bool(isinstance(fsdp, Mapping) and fsdp.get("enabled", False))
    fsdp_size = int(fsdp.get("size", 1)) if fsdp_enabled else 1
    stream_load_pretrained = bool(isinstance(fsdp, Mapping) and fsdp.get("stream_load_pretrained", False))
    ddp = distributed.get("dp", {})
    ddp_enabled = bool(isinstance(ddp, Mapping) and ddp.get("enabled", False))
    return (
        sp_size,
        fsdp_size,
        fsdp_enabled,
        ddp_enabled,
        stream_load_pretrained,
    )


def _validate_h3_memory_topology(
    guard,
    *,
    configured_sp_size,
    configured_fsdp_size,
    fsdp_enabled,
    ddp_enabled,
    stream_load_pretrained,
    runtime_sp_size,
    runtime_fsdp_size,
    num_attention_heads,
    fsdp_wrapped_roles,
    gradient_checkpointing,
    adaptive_video_regularization,
    student_sla,
):
    """Fail early when the sub-80-GiB guarantee topology is not in use."""

    if not guard.enabled:
        return
    if ddp_enabled or not fsdp_enabled:
        raise ValueError("MiniMax-H3 memory_guard requires FSDP2 only (distributed.fsdp2.enabled=true and distributed.dp disabled).")
    if not stream_load_pretrained:
        raise ValueError("MiniMax-H3 memory_guard requires distributed.fsdp2.stream_load_pretrained=true so setup never stacks a full teacher checkpoint above existing shards.")
    if configured_sp_size <= 1:
        raise ValueError("MiniMax-H3 memory_guard requires sequence parallelism with size > 1.")
    if configured_fsdp_size < guard.min_fsdp_size:
        raise ValueError(
            "MiniMax-H3 memory_guard requires distributed.fsdp2.size >= "
            f"{guard.min_fsdp_size}; got {configured_fsdp_size}. Three H3 denoisers cannot "
            "fit below the configured device limit with a smaller weight shard group."
        )
    if runtime_sp_size != configured_sp_size or runtime_fsdp_size != configured_fsdp_size:
        raise RuntimeError(f"MiniMax-H3 runtime/config parallel topology mismatch: runtime SP{runtime_sp_size}/FSDP{runtime_fsdp_size}, configured SP{configured_sp_size}/FSDP{configured_fsdp_size}.")
    if num_attention_heads % runtime_sp_size:
        raise ValueError(f"MiniMax-H3 num_attention_heads={num_attention_heads} must be divisible by sequence_parallel.size={runtime_sp_size}.")
    unwrapped = sorted(role for role, wrapped in fsdp_wrapped_roles.items() if not wrapped)
    if unwrapped:
        raise RuntimeError(f"MiniMax-H3 memory_guard requires every DMD denoiser to be FSDP2-wrapped; unwrapped roles={unwrapped}.")
    if not gradient_checkpointing:
        raise ValueError("MiniMax-H3 memory_guard requires training.gradient_checkpointing=true.")
    if adaptive_video_regularization:
        raise ValueError("MiniMax-H3 memory_guard requires Adaptive Video Distillation loss to be disabled.")
    if student_sla:
        raise ValueError("MiniMax-H3 memory_guard requires student_sparse_attention.enabled=false when SP is enabled.")


def _validate_h3_peak_memory(reports, guard, context="first full outer iteration"):
    """Validate per-rank ``(allocated GiB, reserved GiB)`` peak reports."""

    if not reports:
        raise ValueError("MiniMax-H3 memory guard received no CUDA memory reports.")
    peak_allocated = max(float(report[0]) for report in reports)
    peak_reserved = max(float(report[1]) for report in reports)
    allocated_threshold = guard.peak_threshold_gib
    reserved_threshold = guard.device_limit_gib
    if peak_allocated > allocated_threshold or peak_reserved >= reserved_threshold:
        raise RuntimeError(
            f"MiniMax-H3 CUDA memory guard failed during {context}: "
            f"peak_allocated={peak_allocated:.2f} GiB, peak_reserved={peak_reserved:.2f} GiB; "
            f"required allocated <= {allocated_threshold:.2f} GiB and reserved < "
            f"{reserved_threshold:.2f} GiB ({guard.safety_margin_gib:.2f} GiB "
            "allocated-memory safety margin)."
        )
    return peak_allocated, peak_reserved


def _validate_h3_checkpoint_topology(saved, current, state_path):
    """Validate exact SP/FSDP layout; return False for legacy dense state."""

    if saved is None:
        if current["sequence_parallel_size"] > 1:
            raise RuntimeError(f"Cannot resume a checkpoint without MiniMax-H3 parallel-topology metadata using sequence parallelism: {state_path}")
        return False
    if saved != current:
        raise RuntimeError(f"MiniMax-H3 checkpoint parallel-topology mismatch: checkpoint={saved!r}, current={current!r}, path={state_path}")
    return True


class MiniMaxH3MemoryGuard:
    def __init__(self, config, model):
        self.config = MiniMaxH3MemoryGuardConfig.from_mapping(config)
        self.model = model
        self.measuring = False
        self.checked = False

    def validate(self, models, adaptive_video_regularization, student_sla):
        if not self.config.enabled:
            return
        sp, fsdp, fsdp_enabled, ddp_enabled, streamed = _configured_h3_parallel_sizes(self.model.config)
        _validate_h3_memory_topology(
            self.config,
            configured_sp_size=sp,
            configured_fsdp_size=fsdp,
            fsdp_enabled=fsdp_enabled,
            ddp_enabled=ddp_enabled,
            stream_load_pretrained=streamed,
            runtime_sp_size=get_sequence_parallel_world_size(),
            runtime_fsdp_size=get_data_parallel_world_size(),
            num_attention_heads=int(self.model.denoiser_module().config.num_attention_heads),
            fsdp_wrapped_roles={role: model.is_fsdp2_wrapped() for role, model in models.items()},
            gradient_checkpointing=bool(self.model.config.get("training", {}).get("gradient_checkpointing", False)),
            adaptive_video_regularization=adaptive_video_regularization,
            student_sla=student_sla,
        )

    def start(self):
        if not self.config.enabled or (self.checked and not self.config.continuous):
            return
        if not torch.cuda.is_available():
            raise RuntimeError("H3 memory_guard requires CUDA devices.")
        torch.cuda.synchronize(self.model.device)
        torch.cuda.reset_peak_memory_stats(self.model.device)
        self.measuring = True

    def finish(self):
        if not self.measuring:
            return
        torch.cuda.synchronize(self.model.device)
        report = (
            torch.cuda.max_memory_allocated(self.model.device) / 1024**3,
            torch.cuda.max_memory_reserved(self.model.device) / 1024**3,
        )
        reports = [report]
        if dist.is_initialized():
            reports = [None] * get_world_size()
            dist.all_gather_object(reports, report)
        _validate_h3_peak_memory(reports, self.config)
        self.measuring = False
        self.checked = True
