from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator, Mapping

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.tensor.device_mesh import DeviceMesh, init_device_mesh

from lightx2v_platform.base.global_var import AI_DEVICE

_PHASES = {"ar", "denoise"}


@dataclass(frozen=True)
class _PhaseParallelState:
    tp_group: dist.ProcessGroup
    tp_rank: int
    tp_size: int
    seq_group: dist.ProcessGroup | None
    seq_rank: int
    seq_size: int
    logical_tp_rank: int
    logical_gather_order: tuple[int, ...]


class HunyuanImage3ParallelContext:
    """Static process groups and phase-dependent active topology."""

    def __init__(
        self,
        *,
        device_mesh: DeviceMesh,
        storage_tp_group: dist.ProcessGroup,
        storage_tp_rank: int,
        storage_tp_size: int,
        local_micro_shard_id: int,
        micro_shard_count: int,
        ar_tp_size: int,
        denoise_tp_size: int,
        denoise_seq_size: int,
        phase_states: Mapping[str, _PhaseParallelState],
        initial_phase: str = "denoise",
    ):
        self.device_mesh = device_mesh
        self.denoise_device_mesh = device_mesh
        self.storage_tp_group = storage_tp_group
        self.storage_tp_rank = int(storage_tp_rank)
        self.storage_tp_size = int(storage_tp_size)
        self.local_micro_shard_id = int(local_micro_shard_id)
        self.micro_shard_count = int(micro_shard_count)
        self.ar_tp_size = int(ar_tp_size)
        self.denoise_tp_size = int(denoise_tp_size)
        self.denoise_seq_size = int(denoise_seq_size)
        self._phase_states = dict(phase_states)
        self._phase = self._validate_phase(initial_phase)
        self.ar_custom_all_reduce = None

    @staticmethod
    def _validate_phase(name: str) -> str:
        if name not in _PHASES:
            raise ValueError(f"Unsupported HunyuanImage3 parallel phase {name!r}; expected 'ar' or 'denoise'.")
        return name

    @property
    def phase(self) -> str:
        return self._phase

    def _assert_distributed_phase_consensus(self, target_phase: str | None) -> None:
        """Fail collectively when phase state or transition targets diverge."""
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return

        phase_codes = {"ar": 0, "denoise": 1}
        status_device = torch.device("cpu")
        if dist.get_backend() == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("HunyuanImage3 NCCL phase consensus requires CUDA.")
            status_device = torch.device("cuda", torch.cuda.current_device())
        local_status = torch.tensor(
            [phase_codes.get(self._phase, -1), phase_codes.get(target_phase, -1)],
            device=status_device,
            dtype=torch.int32,
        )
        world_size = dist.get_world_size()
        gathered_status = torch.empty(world_size * 2, device=status_device, dtype=torch.int32)
        dist.all_gather_into_tensor(gathered_status, local_status, group=dist.group.WORLD)
        gathered_status = gathered_status.view(world_size, 2).cpu().tolist()
        current_codes = {int(status[0]) for status in gathered_status}
        target_codes = {int(status[1]) for status in gathered_status}
        code_names = {value: key for key, value in phase_codes.items()} | {-1: "<invalid>"}
        invalid_current = len(current_codes) != 1 or -1 in current_codes
        divergent_target = len(target_codes) != 1
        if invalid_current or divergent_target:
            current_names = sorted(code_names.get(code, f"<invalid:{code}>") for code in current_codes)
            target_names = sorted(code_names.get(code, f"<invalid:{code}>") for code in target_codes)
            raise RuntimeError(f"HunyuanImage3 distributed phase state diverged across ranks: current={current_names}, target={target_names}.")

    def activate_phase(self, name: str) -> HunyuanImage3ParallelContext:
        """Synchronize all ranks and select a pre-built topology."""
        phase = name if name in _PHASES else None
        self._assert_distributed_phase_consensus(phase)
        if phase is None:
            self._validate_phase(name)
        if phase == self._phase:
            return self

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist.is_available() and dist.is_initialized():
            if dist.get_backend() == "nccl" and torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()

        self._phase = phase
        allocated_gib = reserved_gib = 0.0
        if torch.cuda.is_available():
            allocated_gib = torch.cuda.memory_allocated() / 2**30
            reserved_gib = torch.cuda.memory_reserved() / 2**30
        logger.info(
            "HunyuanImage3 activated phase={} global_rank={} active_tp_rank={}/{} logical_tp_rank={} active_sp_rank={}/{} cuda_allocated_gib={:.3f} cuda_reserved_gib={:.3f}",
            phase,
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
            self.active_tp_rank,
            self.active_tp_size,
            self.logical_tp_rank,
            self.active_seq_rank,
            self.active_seq_size,
            allocated_gib,
            reserved_gib,
        )
        return self

    @contextmanager
    def stage(self, name: str) -> Iterator[HunyuanImage3ParallelContext]:
        """Temporarily activate a phase and restore the previous phase."""

        previous = self._phase
        self.activate_phase(name)
        try:
            yield self
        finally:
            self.activate_phase(previous)

    def tensor_parallel_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """SUM across active TP, using custom all-reduce only during AR."""

        if self.active_tp_size <= 1:
            return tensor

        reducer = self.ar_custom_all_reduce
        if self.phase == "ar" and reducer is not None:
            return reducer.all_reduce(tensor, is_decode=tensor.numel() == tensor.shape[-1])

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.active_tp_group)
        return tensor

    def custom_all_reduce_capture(self):
        reducer = self.ar_custom_all_reduce
        if reducer is None:
            return nullcontext()
        return reducer.capture()

    def close(self) -> None:
        reducer = self.ar_custom_all_reduce
        if reducer is not None:
            reducer.close()
            self.ar_custom_all_reduce = None

    @property
    def _active(self) -> _PhaseParallelState:
        return self._phase_states[self._phase]

    @property
    def active_tp_group(self) -> dist.ProcessGroup:
        return self._active.tp_group

    @property
    def active_tp_rank(self) -> int:
        return self._active.tp_rank

    @property
    def active_tp_size(self) -> int:
        return self._active.tp_size

    @property
    def active_seq_group(self) -> dist.ProcessGroup | None:
        return self._active.seq_group

    @property
    def active_seq_rank(self) -> int:
        return self._active.seq_rank

    @property
    def active_seq_size(self) -> int:
        return self._active.seq_size

    @property
    def active_seq_parallel(self) -> bool:
        return self.active_seq_size > 1

    @property
    def logical_tp_rank(self) -> int:
        return self._active.logical_tp_rank

    @property
    def logical_gather_order(self) -> tuple[int, ...]:
        return self._active.logical_gather_order


def _positive_int(value, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"HunyuanImage3 {name} must be a positive integer, got {value!r}.") from error
    if result < 1:
        raise ValueError(f"HunyuanImage3 {name} must be a positive integer, got {result}.")
    return result


def build_hunyuan_image3_parallel_context(config) -> HunyuanImage3ParallelContext:
    """Create process groups for phase-aware HunyuanImage3 inference."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("HunyuanImage3 phase-aware parallelism requires torch.distributed initialization.")

    parallel = config.get("parallel") or {}
    if not parallel.get("phase_aware", False):
        raise ValueError("HunyuanImage3 phase-aware parallel context requires parallel.phase_aware=true.")

    storage_tp_size = _positive_int(parallel.get("storage_tensor_p_size"), "parallel.storage_tensor_p_size")
    ar = parallel.get("ar") or {}
    denoise = parallel.get("denoise") or {}
    ar_tp_size = _positive_int(ar.get("tensor_p_size"), "parallel.ar.tensor_p_size")
    ar_seq_size = _positive_int(ar.get("seq_p_size", 1), "parallel.ar.seq_p_size")
    denoise_tp_size = _positive_int(denoise.get("tensor_p_size"), "parallel.denoise.tensor_p_size")
    denoise_seq_size = _positive_int(denoise.get("seq_p_size"), "parallel.denoise.seq_p_size")

    if ar_seq_size != 1:
        raise ValueError(f"HunyuanImage3 phase-aware AR supports tensor parallel only; parallel.ar.seq_p_size must be 1, got {ar_seq_size}.")
    if storage_tp_size != denoise_tp_size:
        raise ValueError(f"HunyuanImage3 phase-aware storage TP must equal denoise TP so denoise can reuse the loaded shards: storage={storage_tp_size}, denoise_tp={denoise_tp_size}.")
    if ar_tp_size % storage_tp_size:
        raise ValueError(f"HunyuanImage3 AR TP size ({ar_tp_size}) must be divisible by storage TP size ({storage_tp_size}).")

    micro_shard_count = ar_tp_size // storage_tp_size
    if micro_shard_count != denoise_seq_size:
        raise ValueError(f"HunyuanImage3 local micro-shard count must equal denoise SP size for the ABAB layout: ar_tp/storage_tp={micro_shard_count}, denoise_sp={denoise_seq_size}.")

    world_size = dist.get_world_size()
    if ar_tp_size != world_size or denoise_tp_size * denoise_seq_size != world_size:
        raise ValueError(f"HunyuanImage3 phase sizes must each cover the distributed world: ar_tp={ar_tp_size}, denoise_tp={denoise_tp_size}, denoise_sp={denoise_seq_size}, world_size={world_size}.")

    device_mesh = init_device_mesh(
        AI_DEVICE,
        (denoise_seq_size, denoise_tp_size),
        mesh_dim_names=("seq_p", "tensor_p"),
    )
    storage_tp_group = device_mesh.get_group(mesh_dim="tensor_p")
    storage_tp_rank = dist.get_rank(storage_tp_group)
    if denoise_seq_size > 1:
        denoise_seq_group = device_mesh.get_group(mesh_dim="seq_p")
        denoise_seq_rank = dist.get_rank(denoise_seq_group)
    else:
        denoise_seq_group = None
        denoise_seq_rank = 0

    local_micro_shard_id = denoise_seq_rank
    logical_tp_rank = storage_tp_rank * micro_shard_count + local_micro_shard_id

    # WORLD gathers in physical global-rank order. Map those positions to the
    # logical contiguous shard order required by lm_head and similar outputs.
    physical_to_logical = []
    for physical_rank in range(world_size):
        storage_rank = physical_rank % denoise_tp_size
        micro_shard_id = physical_rank // denoise_tp_size
        physical_to_logical.append(storage_rank * micro_shard_count + micro_shard_id)
    if sorted(physical_to_logical) != list(range(ar_tp_size)):
        raise RuntimeError(f"Invalid HunyuanImage3 AR logical TP mapping: {physical_to_logical}.")
    logical_gather_order = tuple(sorted(range(world_size), key=physical_to_logical.__getitem__))

    ar_metadata_group = None
    if config.get("enable_ar_custom_all_reduce", False):
        # vLLM exchanges CUDA IPC metadata over Gloo.
        ar_metadata_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

    phase_states = {
        "ar": _PhaseParallelState(
            tp_group=dist.group.WORLD,
            tp_rank=dist.get_rank(),
            tp_size=world_size,
            seq_group=None,
            seq_rank=0,
            seq_size=1,
            logical_tp_rank=logical_tp_rank,
            logical_gather_order=logical_gather_order,
        ),
        "denoise": _PhaseParallelState(
            tp_group=storage_tp_group,
            tp_rank=storage_tp_rank,
            tp_size=denoise_tp_size,
            seq_group=denoise_seq_group,
            seq_rank=denoise_seq_rank,
            seq_size=denoise_seq_size,
            logical_tp_rank=storage_tp_rank,
            logical_gather_order=tuple(range(denoise_tp_size)),
        ),
    }

    context = HunyuanImage3ParallelContext(
        device_mesh=device_mesh,
        storage_tp_group=storage_tp_group,
        storage_tp_rank=storage_tp_rank,
        storage_tp_size=storage_tp_size,
        local_micro_shard_id=local_micro_shard_id,
        micro_shard_count=micro_shard_count,
        ar_tp_size=ar_tp_size,
        denoise_tp_size=denoise_tp_size,
        denoise_seq_size=denoise_seq_size,
        phase_states=phase_states,
    )
    if ar_metadata_group is not None:
        from lightx2v.models.networks.hunyuan_image3.custom_all_reduce import HunyuanImage3CustomAllReduce

        reducer = HunyuanImage3CustomAllReduce(
            metadata_group=ar_metadata_group,
            fallback_group=dist.group.WORLD,
            config=config,
            device=torch.device(AI_DEVICE, torch.cuda.current_device()),
            phase_getter=lambda: context.phase,
        )
        context.ar_custom_all_reduce = reducer
        reducer.initialize()
    return context


def initialize_hunyuan_image3_parallel_runtime(config) -> HunyuanImage3ParallelContext:
    context = build_hunyuan_image3_parallel_context(config)
    config["parallel_context"] = context
    config["device_mesh"] = context.denoise_device_mesh
    config["tensor_parallel"] = context.storage_tp_size > 1
    config["seq_parallel"] = context.denoise_seq_size > 1
    config["cfg_parallel"] = False
    return context
