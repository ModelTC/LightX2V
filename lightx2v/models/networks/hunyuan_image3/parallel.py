from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.tensor.device_mesh import DeviceMesh, init_device_mesh

from lightx2v_platform.base.global_var import AI_DEVICE

_PHASE_ALIASES = {
    "ar": "ar",
    "autoregressive": "ar",
    "denoise": "denoise",
    "diffusion": "denoise",
}


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
    """Static process groups with a lightweight AR/denoise phase selector.

    The storage and denoise tensor-parallel layouts are identical. Sequence
    ranks therefore hold replicas of the same storage shard (``ABAB`` for the
    four-rank TP2+SP2 layout). AR further slices each storage shard locally and
    uses every process as one logical TP rank without redistributing weights.

    ``active_tp_rank`` is the rank in the physical process group. AR's logical
    weight order differs from that physical order, so weight selection must use
    ``logical_tp_rank`` and ordered gathers must apply
    ``logical_gather_order``.
    """

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
        self._phase = self._normalize_phase(initial_phase)

    @staticmethod
    def _normalize_phase(name: str) -> str:
        normalized = _PHASE_ALIASES.get(str(name).strip().lower())
        if normalized is None:
            supported = ", ".join(sorted(_PHASE_ALIASES))
            raise ValueError(f"Unsupported HunyuanImage3 parallel phase {name!r}; expected one of: {supported}.")
        return normalized

    @property
    def phase(self) -> str:
        return self._phase

    def _assert_distributed_phase_consensus(self, target_phase: str | None) -> None:
        """Fail all ranks together when local phase state or target diverges.

        A rank-local early return is unsafe if another rank believes it still
        needs to enter the transition barrier.  This small WORLD collective is
        therefore also executed for no-op activations.
        """

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
        if len(current_codes) != 1 or len(target_codes) != 1 or -1 in current_codes or -1 in target_codes:
            current_names = sorted(code_names.get(code, f"<invalid:{code}>") for code in current_codes)
            target_names = sorted(code_names.get(code, f"<invalid:{code}>") for code in target_codes)
            raise RuntimeError(
                "HunyuanImage3 distributed phase state diverged across ranks: "
                f"current={current_names}, target={target_names}."
            )

    def activate_phase(self, name: str) -> HunyuanImage3ParallelContext:
        """Synchronize all ranks, then select a pre-built compute topology.

        No process group or weight tensor is created here.  The synchronization
        prevents one rank from entering a collective for the next phase while
        another rank still has work queued for the previous topology.
        """

        phase = _PHASE_ALIASES.get(str(name).strip().lower())
        self._assert_distributed_phase_consensus(phase)
        if phase is None:
            # Single-process callers do not enter the consensus collective;
            # preserve the detailed public validation error for that case.
            self._normalize_phase(name)
            raise AssertionError("Unreachable HunyuanImage3 phase validation path.")
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
            "HunyuanImage3 activated phase={} global_rank={} active_tp_rank={}/{} logical_tp_rank={} "
            "active_sp_rank={}/{} cuda_allocated_gib={:.3f} cuda_reserved_gib={:.3f}",
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

    # Compatibility aliases used by existing HunyuanImage3 model/infer code.
    @property
    def tp_group(self) -> dist.ProcessGroup:
        return self.active_tp_group

    @property
    def tp_rank(self) -> int:
        return self.active_tp_rank

    @property
    def tp_size(self) -> int:
        return self.active_tp_size

    @property
    def seq_p_group(self) -> dist.ProcessGroup | None:
        return self.active_seq_group

    @property
    def seq_p_rank(self) -> int:
        return self.active_seq_rank

    @property
    def seq_p_size(self) -> int:
        return self.active_seq_size


# The context owns both the static topology and the active phase selector. Keep
# this semantic alias for callers that prefer the topology-oriented name.
HunyuanImage3ParallelTopology = HunyuanImage3ParallelContext


def _positive_int(value, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"HunyuanImage3 {name} must be a positive integer, got {value!r}.") from error
    if result < 1:
        raise ValueError(f"HunyuanImage3 {name} must be a positive integer, got {result}.")
    return result


def build_hunyuan_image3_parallel_context(config) -> HunyuanImage3ParallelContext:
    """Create all groups for the phase-aware four-rank topology once.

    The canonical rank matrix is ``[denoise_seq, denoise_tensor]``. With the
    requested sizes this is ``[[0, 1], [2, 3]]``: storage/denoise TP groups are
    ``[0, 1]`` and ``[2, 3]`` while SP groups are ``[0, 2]`` and ``[1, 3]``.
    AR uses WORLD in physical rank order and exposes the logical permutation
    ``(0, 2, 1, 3)`` for ordered outputs such as vocabulary logits.
    """

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
        raise ValueError(
            "HunyuanImage3 phase-aware storage TP must equal denoise TP so denoise can reuse the loaded shards: "
            f"storage={storage_tp_size}, denoise_tp={denoise_tp_size}."
        )
    if ar_tp_size % storage_tp_size:
        raise ValueError(f"HunyuanImage3 AR TP size ({ar_tp_size}) must be divisible by storage TP size ({storage_tp_size}).")

    micro_shard_count = ar_tp_size // storage_tp_size
    if micro_shard_count != denoise_seq_size:
        raise ValueError(
            "HunyuanImage3 local micro-shard count must equal denoise SP size for the ABAB layout: "
            f"ar_tp/storage_tp={micro_shard_count}, denoise_sp={denoise_seq_size}."
        )

    world_size = dist.get_world_size()
    if ar_tp_size != world_size or denoise_tp_size * denoise_seq_size != world_size:
        raise ValueError(
            "HunyuanImage3 phase sizes must each cover the distributed world: "
            f"ar_tp={ar_tp_size}, denoise_tp={denoise_tp_size}, denoise_sp={denoise_seq_size}, world_size={world_size}."
        )

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

    return HunyuanImage3ParallelContext(
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


def build_hunyuan_image3_parallel_topology(config) -> HunyuanImage3ParallelContext:
    return build_hunyuan_image3_parallel_context(config)


def get_hunyuan_image3_parallel_context(config, *, required: bool = False) -> HunyuanImage3ParallelContext | None:
    context = config.get("parallel_context") or config.get("hunyuan_image3_parallel_context")
    if context is None and required:
        raise RuntimeError("HunyuanImage3 phase-aware parallel context is not initialized.")
    return context
