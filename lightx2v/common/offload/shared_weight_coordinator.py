"""Distributed coordination for process-shared CPU weight arenas.

This module is deliberately model agnostic.  A model adapter supplies a
manifest and a callback that writes tensors into the leader's views; the
coordinator discovers GPU/NUMA topology, elects one leader per memory domain,
and makes every rank attach and CUDA-register the same physical pages.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.common.offload.shared_pinned_arena import (
    DEFAULT_REGISTER_CHUNK_BYTES,
    ReplicaGroup,
    ReplicaPlanner,
    SharedPinnedArena,
    SharedWeightManifest,
    TopologyRecord,
)


class SharedWeightCoordinationError(RuntimeError):
    """A coordinated arena operation failed on at least one rank."""


def validate_shared_weight_config(config):
    """Check shared-memory policy before adapters read checkpoint payloads."""
    if config.get("shared_cpu_weight_backend", "sysv") != "sysv":
        raise ValueError("shared_cpu_weight_backend must be 'sysv'")
    if config.get("shared_cpu_weight_scope", "auto") not in {"auto", "host", "numa"}:
        raise ValueError("shared_cpu_weight_scope must be 'auto', 'host', or 'numa'")
    if type(config.get("shared_cpu_weight_strict_numa", True)) is not bool:
        raise ValueError("shared_cpu_weight_strict_numa must be a bool")
    chunk_mb = config.get("shared_cpu_weight_register_chunk_mb", 128)
    if type(chunk_mb) is not int or chunk_mb <= 0:
        raise ValueError("shared_cpu_weight_register_chunk_mb must be a positive integer")


@dataclass(frozen=True)
class ArenaDescriptor:
    """Process-local handle published by a replica leader."""

    leader_rank: int
    shmid: int


@dataclass
class SharedArenaAllocation:
    """The local arena attachment and its placement metadata."""

    arena: SharedPinnedArena
    topology: TopologyRecord
    group: ReplicaGroup

    def tensor_views(self) -> Mapping[str, torch.Tensor]:
        return self.arena.tensor_views()

    def close(self, *, synchronize_cuda: bool = True) -> None:
        self.arena.close(remove=False, synchronize_cuda=synchronize_cuda)


def _distributed_rank_and_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _all_gather_object(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _failure_message(stage: str, statuses: list[Mapping[str, Any]]) -> str | None:
    failures = [status for status in statuses if not status.get("ok", False)]
    if not failures:
        return None
    detail = "; ".join(f"rank {status.get('rank')}: {status.get('error', 'unknown error')}" for status in failures)
    return f"shared CPU weight {stage} failed: {detail}"


def coordinate_rank_local_error(stage: str, error: BaseException | None) -> None:
    """Make every rank fail when one rank reports a local shared-weight error."""

    rank, world_size = _distributed_rank_and_world()
    if world_size == 1:
        if error is not None:
            raise error
        return

    status = {
        "rank": rank,
        "ok": error is None,
        "error": None if error is None else f"{type(error).__name__}: {error}",
    }
    statuses = _all_gather_object(status, world_size)
    failure = _failure_message(stage, statuses)
    if failure is None:
        return

    coordinated_error = SharedWeightCoordinationError(failure)
    if error is not None:
        raise coordinated_error from error
    raise coordinated_error


def _close_after_failure(arena: SharedPinnedArena | None, *, creator: bool) -> None:
    if arena is None:
        return
    try:
        arena.close(remove=creator, synchronize_cuda=False)
    except Exception as error:  # cleanup must not hide the coordinated failure
        logger.error(f"Failed to clean up shared CPU weight arena: {error}")


def materialize_shared_weight_arena(
    manifest: SharedWeightManifest,
    populate: Callable[[Mapping[str, torch.Tensor]], None],
    *,
    scope: str = "auto",
    strict_numa: bool = True,
    register_chunk_bytes: int = DEFAULT_REGISTER_CHUNK_BYTES,
) -> SharedArenaAllocation:
    """Create one immutable arena per selected topology domain.

    All ranks execute this function.  Only the elected leader of each replica
    group invokes ``populate``; followers never read shared checkpoint tensor
    payloads.  Failures are exchanged at every stage so one bad rank cannot
    strand its peers in a later collective.
    """

    rank, world_size = _distributed_rank_and_world()

    try:
        if not callable(populate):
            raise TypeError("populate must be callable")
        if not isinstance(scope, str):
            raise TypeError("scope must be a string")
        if type(strict_numa) is not bool:
            raise TypeError("strict_numa must be a bool")
        if type(register_chunk_bytes) is not int or register_chunk_bytes <= 0:
            raise ValueError("register_chunk_bytes must be a positive integer")
        if not torch.cuda.is_available():
            raise SharedWeightCoordinationError("shared pinned weight arenas currently require CUDA")
        cuda_device = torch.cuda.current_device()
        local_rank = int(os.environ.get("LOCAL_RANK", cuda_device))
        topology = TopologyRecord.discover(
            rank,
            manifest.weight_signature,
            local_rank=local_rank,
            cuda_device=cuda_device,
        )
        discovery_status = {
            "rank": rank,
            "ok": True,
            "topology": topology,
            "manifest_digest": manifest.digest,
            "manifest_nbytes": manifest.nbytes,
            "policy": (scope, strict_numa, register_chunk_bytes),
        }
    except Exception as error:
        topology = None
        cuda_device = None
        local_rank = None
        discovery_status = {"rank": rank, "ok": False, "error": repr(error)}

    discovery_statuses = _all_gather_object(discovery_status, world_size)
    failure = _failure_message("preflight/topology discovery", discovery_statuses)
    if failure is not None:
        raise SharedWeightCoordinationError(failure)

    manifest_layouts = {(status["manifest_digest"], status["manifest_nbytes"]) for status in discovery_statuses}
    if len(manifest_layouts) != 1:
        raise SharedWeightCoordinationError(f"shared CPU weight manifests differ across ranks: {sorted(manifest_layouts)}")
    policies = {tuple(status["policy"]) for status in discovery_statuses}
    if len(policies) != 1:
        raise SharedWeightCoordinationError(f"shared CPU weight policies differ across ranks: {sorted(policies)}")

    records = [status["topology"] for status in discovery_statuses]
    try:
        plan = ReplicaPlanner.plan(records, scope=scope)
        group = plan.group_for_rank(rank)
    except Exception as error:
        raise SharedWeightCoordinationError(f"shared CPU weight replica planning failed: {error}") from error

    is_leader = rank == group.leader_rank
    arena: SharedPinnedArena | None = None
    descriptor: ArenaDescriptor | None = None
    completed = False
    try:
        try:
            if is_leader:
                arena = SharedPinnedArena.create(
                    manifest=manifest,
                    numa_node=group.key.numa_node,
                    strict_numa=strict_numa,
                    register_cuda=True,
                    cuda_device=cuda_device,
                    register_chunk_bytes=register_chunk_bytes,
                    auto_remove=True,
                )
                populate(arena.tensor_views())
                descriptor = ArenaDescriptor(leader_rank=rank, shmid=arena.shmid)
            leader_status = {"rank": rank, "ok": True, "descriptor": descriptor}
        except Exception as error:
            leader_status = {"rank": rank, "ok": False, "error": repr(error), "descriptor": None}

        leader_statuses = _all_gather_object(leader_status, world_size)
        failure = _failure_message("leader creation/population", leader_statuses)
        if failure is not None:
            raise SharedWeightCoordinationError(failure)

        try:
            descriptors = {status["descriptor"].leader_rank: status["descriptor"] for status in leader_statuses if status.get("descriptor") is not None}
            group_descriptor = descriptors.get(group.leader_rank)
            if group_descriptor is None:
                raise SharedWeightCoordinationError(f"no arena descriptor was published for replica leader rank {group.leader_rank}")
            if not is_leader:
                arena = SharedPinnedArena.attach(
                    group_descriptor.shmid,
                    manifest=manifest,
                    register_cuda=True,
                    cuda_device=cuda_device,
                    register_chunk_bytes=register_chunk_bytes,
                )
            attach_status = {"rank": rank, "ok": True}
        except Exception as error:
            attach_status = {"rank": rank, "ok": False, "error": repr(error)}

        attach_statuses = _all_gather_object(attach_status, world_size)
        failure = _failure_message("attach/CUDA registration", attach_statuses)
        if failure is not None:
            raise SharedWeightCoordinationError(failure)

        event = {
            "event": "shared_cpu_weight_arena",
            "rank": rank,
            "pid": os.getpid(),
            "local_rank": local_rank,
            "cuda_ordinal": cuda_device,
            "pci_bdf": topology.pci_bus_id,
            "host_id": topology.host_id,
            "ipc_namespace": topology.ipc_namespace,
            "numa_node": topology.numa_node,
            "group_ranks": list(group.ranks),
            "leader_rank": group.leader_rank,
            "shmid": arena.shmid,
            "host_address": arena.address,
            "arena_bytes": arena.nbytes,
            "registered_bytes": arena.registered_nbytes,
            "registered_chunks": arena.registration_chunk_count,
            "manifest_hash": manifest.digest,
            "scope": scope,
        }
        logger.info(f"[SharedCPUWeights] {json.dumps(event, sort_keys=True)}")
        allocation = SharedArenaAllocation(arena=arena, topology=topology, group=group)
        completed = True
        return allocation
    finally:
        if not completed:
            _close_after_failure(arena, creator=is_leader)
