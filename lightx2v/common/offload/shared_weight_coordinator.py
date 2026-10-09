"""Distributed coordination for process-shared CPU weight arenas.

This module is deliberately model agnostic.  A model adapter supplies a
manifest and a callback that writes tensors into the leader's views; the
coordinator discovers GPU/NUMA topology, elects one leader per memory domain,
and makes every rank attach and CUDA-register the same physical pages.

CPU status waits use the job's Store, never the default GPU process group.
LIGHTX2V_SHARED_WEIGHT_TIMEOUT_SECONDS bounds each wait (default: 3600 seconds).
Failures abort subsequent shared-weight initialization in the same job. A rank
busy in CPU loading observes the failure when it next reaches coordination.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.distributed_c10d import _get_default_store

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


class _CPUStatusExchange:
    """Exchange small, trusted job metadata without enqueuing GPU collectives.

    The Store belongs to the distributed job; our prefix and deadline are
    independent of its process groups. Keep a terminal failure so late ranks
    and enclosing initialization error handlers observe the same cause.
    """

    def __init__(self, store):
        self.store = dist.PrefixStore("lightx2v/shared_cpu_weights/", store)
        self.sequence = 0

    def _abort(self, message: str) -> None:
        first_failure = self.store.compare_set("failure", "", message).decode()
        raise SharedWeightCoordinationError(first_failure)

    def exchange(self, stage: str, status: Mapping[str, Any], world_size: int) -> list[Any]:
        sequence = self.sequence
        self.sequence += 1
        rank = status["rank"]
        started = time.monotonic()
        try:
            timeout = float(os.environ.get("LIGHTX2V_SHARED_WEIGHT_TIMEOUT_SECONDS", "3600"))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("LIGHTX2V_SHARED_WEIGHT_TIMEOUT_SECONDS must be finite and positive")
            self.store.set(f"{sequence}/{rank}", pickle.dumps((stage, status)))
            failure = _failure_message(stage, [status])
            if failure is not None:
                self._abort(failure)

            pending = set(range(world_size))
            statuses = [None] * world_size
            while True:
                if self.store.check(["failure"]):
                    raise SharedWeightCoordinationError(self.store.get("failure").decode())
                for peer in list(pending):
                    key = f"{sequence}/{peer}"
                    if not self.store.check([key]):
                        continue
                    peer_stage, peer_status = pickle.loads(self.store.get(key))
                    if peer_stage != stage:
                        self._abort(f"shared CPU weight stage mismatch at exchange {sequence}: rank {rank} entered {stage!r}, rank {peer} entered {peer_stage!r}")
                    failure = _failure_message(stage, [peer_status])
                    if failure is not None:
                        self._abort(failure)
                    statuses[peer] = peer_status
                    pending.remove(peer)
                elapsed = time.monotonic() - started
                if not pending:
                    if elapsed >= 1:
                        logger.info(f"[SharedCPUWeights] CPU coordination {stage!r}: rank {rank} waited {elapsed:.1f}s")
                    return statuses
                if elapsed >= timeout:
                    self._abort(f"shared CPU weight {stage} timed out after {timeout:g}s on rank {rank}; missing ranks: {sorted(pending)}")
                time.sleep(min(0.1, timeout - elapsed))
        except SharedWeightCoordinationError:
            raise
        except Exception as error:
            message = f"shared CPU weight {stage} CPU coordination failed on rank {rank}: {type(error).__name__}: {error}"
            # A disconnected Store cannot propagate an error, but local arena
            # cleanup must still run and retain the transport failure as cause.
            try:
                self._abort(message)
            except SharedWeightCoordinationError:
                raise
            except Exception:
                raise SharedWeightCoordinationError(message) from error


_cpu_status_exchanges = WeakKeyDictionary()


def _exchange_status(stage: str, status: Mapping[str, Any], world_size: int) -> list[Any]:
    if world_size == 1:
        return [status]
    group = dist.group.WORLD
    if group not in _cpu_status_exchanges:
        _cpu_status_exchanges[group] = _CPUStatusExchange(_get_default_store())
    return _cpu_status_exchanges[group].exchange(stage, status, world_size)


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
    try:
        statuses = _exchange_status(stage, status, world_size)
    except SharedWeightCoordinationError as coordinated_error:
        if error is not None:
            raise coordinated_error from error
        raise
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

    discovery_statuses = _exchange_status("preflight/topology discovery", discovery_status, world_size)
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

        leader_statuses = _exchange_status("leader creation/population", leader_status, world_size)
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

        attach_statuses = _exchange_status("attach/CUDA registration", attach_status, world_size)
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
