"""Process-shared, NUMA-local pinned host memory for weight offload.

The primitives in this module deliberately know nothing about a model or a
distributed backend.  A caller is expected to exchange ``shmid`` and a
``SharedWeightManifest`` with the ranks in a ``ReplicaGroup`` using its
existing control plane (for example, torch.distributed).

The lifetime rule is important: each process attaches the SysV segment at a
potentially different virtual address, so each process must also register its
own address with CUDA.  Physical pages are shared; virtual mappings and CUDA
registrations are process-local.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import platform
import socket
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

IPC_PRIVATE = 0
IPC_CREAT = 0o1000
IPC_EXCL = 0o2000
IPC_RMID = 0
IPC_STAT = 2
SHM_RDONLY = 0o10000

MPOL_BIND = 2

CUDA_HOST_REGISTER_PORTABLE = 1
DEFAULT_REGISTER_CHUNK_BYTES = 128 * 1024 * 1024
DEFAULT_MANIFEST_ALIGNMENT = 64

# If CUDA refuses to unregister an external address, retaining its owner is
# safer than unmapping memory that the driver still considers registered.
_QUARANTINED_LIFETIMES: list[Any] = []


class SharedPinnedArenaError(RuntimeError):
    """Base error for shared pinned arena operations."""


class SysVSharedMemoryError(SharedPinnedArenaError):
    """A System V shared-memory operation failed."""


class NumaBindingError(SharedPinnedArenaError):
    """Installing a NUMA policy failed."""


class CudaHostRegistrationError(SharedPinnedArenaError):
    """Registering or unregistering externally allocated host memory failed."""


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    return name.removeprefix("torch.")


@cache
def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype {name!r}")
    return dtype


def _contiguous_stride(shape: Sequence[int]) -> tuple[int, ...]:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= max(size, 1)
    return tuple(reversed(stride))


@dataclass(frozen=True)
class TensorSpec:
    """Description of one tensor view inside a byte arena.

    ``storage_shape`` describes the contiguous bytes in the arena. ``shape``
    and ``stride`` describe the logical view exposed to the model.  Keeping
    both representations supports layouts such as a transposed FP8 matrix
    without creating a private copy.
    """

    name: str
    offset: int
    nbytes: int
    dtype: str
    storage_shape: tuple[int, ...]
    shape: tuple[int, ...] | None = None
    stride: tuple[int, ...] | None = None
    storage_offset: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", self.dtype.removeprefix("torch."))
        object.__setattr__(self, "storage_shape", tuple(self.storage_shape))
        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(self.shape))
        if self.stride is not None:
            object.__setattr__(self, "stride", tuple(self.stride))
        self.validate()

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return self.storage_shape if self.shape is None else self.shape

    @property
    def logical_stride(self) -> tuple[int, ...]:
        if self.stride is not None:
            return self.stride
        return _contiguous_stride(self.logical_shape)

    @cached_property
    def itemsize(self) -> int:
        return _dtype_from_name(self.dtype).itemsize

    def validate(self) -> None:
        if not self.name:
            raise ValueError("tensor name must be non-empty")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError(f"{self.name!r}: offset must be non-negative")
        if type(self.nbytes) is not int or self.nbytes < 0:
            raise ValueError(f"{self.name!r}: nbytes must be non-negative")
        if type(self.storage_offset) is not int or self.storage_offset < 0:
            raise ValueError(f"{self.name!r}: storage_offset must be non-negative")
        if any(type(size) is not int or size < 0 for size in self.storage_shape):
            raise ValueError(f"{self.name!r}: storage_shape must contain non-negative integers")
        if any(type(size) is not int or size < 0 for size in self.logical_shape):
            raise ValueError(f"{self.name!r}: shape must contain non-negative integers")
        if len(self.logical_stride) != len(self.logical_shape):
            raise ValueError(f"{self.name!r}: shape and stride must have the same rank")
        if any(type(value) is not int or value < 0 for value in self.logical_stride):
            raise ValueError(f"{self.name!r}: negative strides are not supported")

        itemsize = self.itemsize
        storage_numel = math.prod(self.storage_shape)
        expected_nbytes = storage_numel * itemsize
        if self.nbytes != expected_nbytes:
            raise ValueError(f"{self.name!r}: nbytes={self.nbytes}, expected {expected_nbytes} for {self.storage_shape} {self.dtype}")
        if self.offset % itemsize:
            raise ValueError(f"{self.name!r}: offset {self.offset} is not aligned to dtype itemsize {itemsize}")

        logical_numel = math.prod(self.logical_shape)
        if logical_numel == 0:
            if self.storage_offset > storage_numel:
                raise ValueError(f"{self.name!r}: empty view starts outside its storage")
            return
        max_element = self.storage_offset + sum((size - 1) * stride for size, stride in zip(self.logical_shape, self.logical_stride))
        if max_element >= storage_numel:
            raise ValueError(f"{self.name!r}: logical view exceeds its storage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "dtype": self.dtype,
            "storage_shape": list(self.storage_shape),
            "shape": None if self.shape is None else list(self.shape),
            "stride": None if self.stride is None else list(self.stride),
            "storage_offset": self.storage_offset,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorSpec":
        return cls(
            name=value["name"],
            offset=value["offset"],
            nbytes=value["nbytes"],
            dtype=value["dtype"],
            storage_shape=tuple(value["storage_shape"]),
            shape=None if value.get("shape") is None else tuple(value["shape"]),
            stride=None if value.get("stride") is None else tuple(value["stride"]),
            storage_offset=value.get("storage_offset", 0),
        )


@dataclass(frozen=True)
class SharedWeightManifest:
    """Stable, serializable layout shared by arena creators and attachers."""

    nbytes: int
    tensors: tuple[TensorSpec, ...]
    weight_signature: str
    alignment: int = DEFAULT_MANIFEST_ALIGNMENT
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", tuple(self.tensors))
        self.validate()

    def validate(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError(f"unsupported manifest version {self.version}")
        if not self.tensors:
            raise ValueError("manifest must contain at least one tensor")
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("manifest nbytes must be positive")
        if type(self.alignment) is not int or self.alignment <= 0:
            raise ValueError("manifest alignment must be positive")
        if not self.weight_signature:
            raise ValueError("weight_signature must be non-empty")
        names: set[str] = set()
        previous_end = 0
        for spec in sorted(self.tensors, key=lambda item: (item.offset, item.name)):
            if spec.name in names:
                raise ValueError(f"duplicate tensor name {spec.name!r}")
            names.add(spec.name)
            if spec.offset < previous_end:
                raise ValueError(f"tensor {spec.name!r} overlaps a previous tensor")
            if spec.offset + spec.nbytes > self.nbytes:
                raise ValueError(f"tensor {spec.name!r} exceeds arena size {self.nbytes}")
            previous_end = spec.offset + spec.nbytes

    @cached_property
    def by_name(self) -> Mapping[str, TensorSpec]:
        return MappingProxyType({spec.name: spec for spec in self.tensors})

    @classmethod
    def from_tensors(
        cls,
        tensors: Mapping[str, torch.Tensor],
        weight_signature: str,
        alignment: int = DEFAULT_MANIFEST_ALIGNMENT,
    ) -> "SharedWeightManifest":
        if alignment <= 0:
            raise ValueError("alignment must be positive")

        offset = 0
        specs = []
        for name in sorted(tensors):
            tensor = tensors[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name!r} is not a tensor")
            if tensor.layout != torch.strided or tensor.is_quantized:
                raise ValueError(f"{name!r} cannot be represented in a shared byte arena")
            item_alignment = math.lcm(alignment, tensor.element_size())
            offset = _align_up(offset, item_alignment)
            nbytes = tensor.numel() * tensor.element_size()
            specs.append(
                TensorSpec(
                    name=name,
                    offset=offset,
                    nbytes=nbytes,
                    dtype=_dtype_name(tensor.dtype),
                    storage_shape=tuple(tensor.shape),
                )
            )
            offset += nbytes

        return cls(
            nbytes=_align_up(offset, alignment),
            tensors=tuple(specs),
            weight_signature=weight_signature,
            alignment=alignment,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nbytes": self.nbytes,
            "alignment": self.alignment,
            "weight_signature": self.weight_signature,
            "tensors": [spec.to_dict() for spec in self.tensors],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedWeightManifest":
        return cls(
            version=value["version"],
            nbytes=value["nbytes"],
            alignment=value["alignment"],
            weight_signature=value["weight_signature"],
            tensors=tuple(TensorSpec.from_dict(item) for item in value["tensors"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "SharedWeightManifest":
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError("manifest JSON must contain an object")
        return cls.from_dict(decoded)

    @cached_property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def build_tensor_aware_registration_regions(
    nbytes: int,
    tensors: Sequence[TensorSpec],
    *,
    target_chunk_bytes: int = DEFAULT_REGISTER_CHUNK_BYTES,
    page_size: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Partition an arena without splitting a tensor across registrations.

    CUDA permits adjacent registered ranges, but an asynchronous copy whose
    source spans two independently registered ranges can fail. Internal range
    boundaries are therefore page-aligned and chosen only outside the open
    storage interval of every tensor. A large tensor may make a range larger
    than ``target_chunk_bytes``.

    Returned tuples are ``(offset, size)`` relative to the arena base and form
    one continuous cover of ``[0, nbytes)``.
    """

    if nbytes <= 0:
        raise ValueError("registration size must be positive")
    if target_chunk_bytes <= 0:
        raise ValueError("target_chunk_bytes must be positive")
    page_size = os.sysconf("SC_PAGE_SIZE") if page_size is None else page_size
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if target_chunk_bytes % page_size:
        raise ValueError(f"target_chunk_bytes must be a multiple of page size {page_size}")

    ranges = []
    for spec in sorted(tensors, key=lambda item: (item.offset, item.name)):
        end = spec.offset + spec.nbytes
        if end > nbytes:
            raise ValueError(f"tensor {spec.name!r} exceeds registration size {nbytes}")
        if spec.nbytes:
            ranges.append((spec.offset, end))

    def crossing(boundary: int) -> tuple[int, int] | None:
        for start, end in ranges:
            if start >= boundary:
                return None
            if start < boundary < end:
                return start, end
        return None

    def valid_boundary_before(desired: int, lower_bound: int) -> int | None:
        boundary = desired // page_size * page_size
        while boundary > lower_bound:
            interval = crossing(boundary)
            if interval is None:
                return boundary
            boundary = interval[0] // page_size * page_size
        return None

    def valid_boundary_after(desired: int) -> int | None:
        boundary = _align_up(desired, page_size)
        while boundary < nbytes:
            interval = crossing(boundary)
            if interval is None:
                return boundary
            boundary = _align_up(interval[1], page_size)
        return None

    boundaries = [0]
    cursor = 0
    while nbytes - cursor > target_chunk_bytes:
        desired = cursor + target_chunk_bytes
        before = valid_boundary_before(desired, cursor)
        after = valid_boundary_after(desired)
        candidates = [boundary for boundary in (before, after) if boundary is not None and cursor < boundary < nbytes]
        if not candidates:
            break
        boundary = min(candidates, key=lambda value: (abs(value - desired), value))
        boundaries.append(boundary)
        cursor = boundary
    boundaries.append(nbytes)

    regions = tuple((start, end - start) for start, end in zip(boundaries, boundaries[1:]))
    for boundary in boundaries[1:-1]:
        if boundary % page_size:
            raise AssertionError("internal registration boundary is not page-aligned")
        if crossing(boundary) is not None:
            raise AssertionError("registration boundary splits a tensor")
    return regions


@dataclass(frozen=True)
class TopologyRecord:
    """Topology facts for one rank; records may also be injected by tests."""

    rank: int
    host_id: str
    ipc_namespace: str
    numa_node: int | None
    weight_signature: str
    local_rank: int | None = None
    cuda_device: int | None = None
    pci_bus_id: str | None = None

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if not self.host_id:
            raise ValueError("host_id must be non-empty")
        if not self.ipc_namespace:
            raise ValueError("ipc_namespace must be non-empty")
        if self.numa_node is not None and self.numa_node < 0:
            raise ValueError("numa_node must be non-negative")
        if not self.weight_signature:
            raise ValueError("weight_signature must be non-empty")

    @classmethod
    def discover(
        cls,
        rank: int,
        weight_signature: str,
        *,
        local_rank: int | None = None,
        cuda_device: int | None = None,
        host_id: str | None = None,
        ipc_namespace: str | None = None,
        pci_bus_id: str | None = None,
        numa_node: int | None = None,
        sysfs_root: str | os.PathLike[str] = "/sys/bus/pci/devices",
    ) -> "TopologyRecord":
        if host_id is None:
            host_id = discover_host_id()
        if ipc_namespace is None:
            ipc_namespace = discover_ipc_namespace()
        if cuda_device is None and torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
        if pci_bus_id is None and cuda_device is not None:
            pci_bus_id = CudaRuntime.load().device_pci_bus_id(cuda_device)
        if numa_node is None and pci_bus_id is not None:
            numa_node = discover_gpu_numa_node(pci_bus_id, sysfs_root=sysfs_root)

        return cls(
            rank=rank,
            host_id=host_id,
            ipc_namespace=ipc_namespace,
            numa_node=numa_node,
            weight_signature=weight_signature,
            local_rank=local_rank,
            cuda_device=cuda_device,
            pci_bus_id=pci_bus_id,
        )


@dataclass(frozen=True)
class ReplicaKey:
    host_id: str
    ipc_namespace: str
    weight_signature: str
    numa_node: int | None


@dataclass(frozen=True)
class ReplicaGroup:
    key: ReplicaKey
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError("replica group must contain at least one rank")
        if tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("replica group ranks must be sorted and unique")

    @property
    def leader_rank(self) -> int:
        return self.ranks[0]


@dataclass(frozen=True)
class ReplicaPlan:
    groups: tuple[ReplicaGroup, ...]

    def group_for_rank(self, rank: int) -> ReplicaGroup:
        matches = [group for group in self.groups if rank in group.ranks]
        if len(matches) != 1:
            raise KeyError(f"rank {rank} belongs to {len(matches)} replica groups")
        return matches[0]


class ReplicaPlanner:
    """Group ranks that may share one physical weight replica."""

    _VALID_SCOPES = {"auto", "host", "numa"}

    @classmethod
    def plan(
        cls,
        records: Sequence[TopologyRecord],
        *,
        scope: str = "auto",
    ) -> ReplicaPlan:
        if scope not in cls._VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(cls._VALID_SCOPES)}, got {scope!r}")
        if not records:
            raise ValueError("at least one topology record is required")

        ordered = sorted(records, key=lambda item: item.rank)
        ranks = [item.rank for item in ordered]
        if len(set(ranks)) != len(ranks):
            raise ValueError("topology records contain duplicate ranks")

        cohorts: dict[tuple[str, str, str], list[TopologyRecord]] = {}
        for record in ordered:
            cohort_key = (record.host_id, record.ipc_namespace, record.weight_signature)
            cohorts.setdefault(cohort_key, []).append(record)

        grouped: dict[ReplicaKey, list[int]] = {}
        for (host_id, ipc_namespace, weight_signature), cohort in cohorts.items():
            cohort_scope = scope
            if scope == "auto":
                cohort_scope = "numa" if all(record.numa_node is not None for record in cohort) else "host"
            if cohort_scope == "numa" and any(record.numa_node is None for record in cohort):
                missing = [record.rank for record in cohort if record.numa_node is None]
                raise ValueError(f"NUMA topology is unknown for ranks {missing}")

            for record in cohort:
                memory_domain = record.numa_node if cohort_scope == "numa" else None
                key = ReplicaKey(host_id, ipc_namespace, weight_signature, memory_domain)
                grouped.setdefault(key, []).append(record.rank)

        groups = []
        for key, group_ranks in grouped.items():
            members = tuple(sorted(group_ranks))
            groups.append(ReplicaGroup(key=key, ranks=members))
        groups.sort(key=lambda group: group.leader_rank)
        return ReplicaPlan(groups=tuple(groups))


def discover_host_id() -> str:
    """Return an identity that remains distinct for same-named cluster nodes."""

    hostname = socket.gethostname()
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError as error:
        raise SharedPinnedArenaError(f"cannot identify the current host: {error}") from error
    if not boot_id:
        raise SharedPinnedArenaError("host boot ID is empty")
    return f"{hostname}:{boot_id}"


def discover_ipc_namespace() -> str:
    try:
        return os.readlink("/proc/self/ns/ipc")
    except OSError as error:
        raise SharedPinnedArenaError(f"cannot identify the current IPC namespace: {error}") from error


def _normalize_pci_bus_id(value: str) -> str:
    parts = value.strip().lower().split(":")
    if len(parts) != 3 or "." not in parts[2]:
        raise ValueError(f"invalid PCI bus id {value!r}")
    domain = int(parts[0], 16)
    bus = int(parts[1], 16)
    device_text, function_text = parts[2].split(".", 1)
    device = int(device_text, 16)
    function = int(function_text, 16)
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def discover_gpu_numa_node(pci_bus_id: str, *, sysfs_root: str | os.PathLike[str] = "/sys/bus/pci/devices") -> int | None:
    normalized = _normalize_pci_bus_id(pci_bus_id)
    path = Path(sysfs_root) / normalized / "numa_node"
    try:
        value = int(path.read_text().strip())
    except (OSError, ValueError) as error:
        raise SharedPinnedArenaError(f"cannot read NUMA node for GPU {normalized} from {path}: {error}") from error
    if value == -1:
        return None
    if value < -1:
        raise SharedPinnedArenaError(f"invalid NUMA node {value} for GPU {normalized} in {path}")
    return value


class _IpcPerm(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_int),
        ("uid", ctypes.c_uint),
        ("gid", ctypes.c_uint),
        ("cuid", ctypes.c_uint),
        ("cgid", ctypes.c_uint),
        ("mode", ctypes.c_uint),
        ("seq", ctypes.c_ushort),
        ("pad2", ctypes.c_ushort),
        ("reserved1", ctypes.c_ulong),
        ("reserved2", ctypes.c_ulong),
    ]


class _ShmidDs(ctypes.Structure):
    _fields_ = [
        ("perm", _IpcPerm),
        ("segment_size", ctypes.c_size_t),
        ("attach_time", ctypes.c_long),
        ("detach_time", ctypes.c_long),
        ("change_time", ctypes.c_long),
        ("creator_pid", ctypes.c_int),
        ("last_pid", ctypes.c_int),
        ("attach_count", ctypes.c_ulong),
        ("reserved5", ctypes.c_ulong),
        ("reserved6", ctypes.c_ulong),
    ]


@dataclass(frozen=True)
class SharedMemoryStat:
    size: int
    uid: int
    gid: int
    mode: int
    creator_pid: int
    last_pid: int
    attach_count: int


def _load_libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.shmget.argtypes = (ctypes.c_int, ctypes.c_size_t, ctypes.c_int)
    libc.shmget.restype = ctypes.c_int
    libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
    libc.shmat.restype = ctypes.c_void_p
    libc.shmdt.argtypes = (ctypes.c_void_p,)
    libc.shmdt.restype = ctypes.c_int
    libc.shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    libc.shmctl.restype = ctypes.c_int
    libc.syscall.restype = ctypes.c_long
    return libc


def _errno_message(operation: str) -> str:
    error = ctypes.get_errno()
    return f"{operation} failed: [errno {error}] {os.strerror(error)}"


class SysVSegment:
    """One attached System V shared-memory segment."""

    def __init__(self, shmid: int, size: int, address: int, *, creator: bool, readonly: bool = False) -> None:
        self.shmid = shmid
        self.size = size
        self.address = address
        self.creator = creator
        self.readonly = readonly
        self._marked_for_deletion = False
        self._closed = False

    @classmethod
    def create(cls, size: int, *, permissions: int = 0o600, auto_remove: bool = False) -> "SysVSegment":
        if size <= 0:
            raise ValueError("shared memory size must be positive")
        if permissions & ~0o777:
            raise ValueError("permissions must contain only Unix permission bits")

        libc = _load_libc()
        shmid = libc.shmget(IPC_PRIVATE, ctypes.c_size_t(size), permissions | IPC_CREAT | IPC_EXCL)
        if shmid < 0:
            raise SysVSharedMemoryError(_errno_message(f"shmget({size})"))
        address = None
        segment = None
        try:
            address = libc.shmat(shmid, None, 0)
            if address == ctypes.c_void_p(-1).value:
                raise SysVSharedMemoryError(_errno_message(f"shmat({shmid})"))
            segment = cls(shmid, size, int(address), creator=True)
            if auto_remove:
                # A marked segment remains attachable by shmid while the
                # creator is attached, and disappears after the last detach.
                segment.mark_for_deletion()
            actual = segment.stat().size
            if actual != size:
                raise SysVSharedMemoryError(f"created segment {shmid} has size {actual}, expected {size}")
            return segment
        except BaseException as error:
            try:
                if segment is not None:
                    segment.close(remove=True)
                elif libc.shmctl(shmid, IPC_RMID, None) != 0:
                    raise SysVSharedMemoryError(_errno_message(f"shmctl({shmid}, IPC_RMID)"))
            except BaseException as cleanup_error:
                error.add_note(f"shared-memory cleanup also failed: {cleanup_error!r}")
            raise

    @classmethod
    def attach(cls, shmid: int, *, expected_size: int | None = None, readonly: bool = False) -> "SysVSegment":
        if shmid < 0:
            raise ValueError("shmid must be non-negative")
        stat = cls.stat_by_id(shmid)
        if expected_size is not None and stat.size != expected_size:
            raise SysVSharedMemoryError(f"segment {shmid} has size {stat.size}, expected {expected_size}")

        flags = SHM_RDONLY if readonly else 0
        address = _load_libc().shmat(shmid, None, flags)
        if address == ctypes.c_void_p(-1).value:
            raise SysVSharedMemoryError(_errno_message(f"shmat({shmid})"))
        return cls(shmid, stat.size, int(address), creator=False, readonly=readonly)

    @staticmethod
    def stat_by_id(shmid: int) -> SharedMemoryStat:
        raw = _ShmidDs()
        result = _load_libc().shmctl(shmid, IPC_STAT, ctypes.byref(raw))
        if result != 0:
            raise SysVSharedMemoryError(_errno_message(f"shmctl({shmid}, IPC_STAT)"))
        return SharedMemoryStat(
            size=int(raw.segment_size),
            uid=int(raw.perm.uid),
            gid=int(raw.perm.gid),
            mode=int(raw.perm.mode),
            creator_pid=int(raw.creator_pid),
            last_pid=int(raw.last_pid),
            attach_count=int(raw.attach_count),
        )

    def stat(self) -> SharedMemoryStat:
        return self.stat_by_id(self.shmid)

    @property
    def is_attached(self) -> bool:
        return not self._closed and self.address != 0

    @property
    def marked_for_deletion(self) -> bool:
        return self._marked_for_deletion

    def bind_to_numa(self, numa_node: int, *, strict: bool = True) -> bool:
        if not self.is_attached:
            raise SysVSharedMemoryError("cannot bind a detached shared-memory segment")
        return bind_address_to_numa(self.address, self.size, numa_node, strict=strict)

    def mark_for_deletion(self) -> None:
        if self._marked_for_deletion:
            return
        if _load_libc().shmctl(self.shmid, IPC_RMID, None) != 0:
            raise SysVSharedMemoryError(_errno_message(f"shmctl({self.shmid}, IPC_RMID)"))
        self._marked_for_deletion = True

    def detach(self) -> None:
        if not self.is_attached:
            return
        address = self.address
        if _load_libc().shmdt(ctypes.c_void_p(address)) != 0:
            raise SysVSharedMemoryError(_errno_message(f"shmdt({address:#x})"))
        self.address = 0
        self._closed = True

    def close(self, *, remove: bool | None = None) -> None:
        if remove is None:
            remove = self.creator
        if remove and not self._marked_for_deletion:
            # Do not detach a creator when IPC_RMID fails. Keeping the mapping
            # and shmid makes cleanup retryable instead of orphaning a segment.
            self.mark_for_deletion()
        self.detach()

    def __enter__(self) -> "SysVSegment":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def bind_address_to_numa(address: int, size: int, numa_node: int, *, strict: bool = True) -> bool:
    """Install ``MPOL_BIND`` before an arena's pages are first touched."""

    if address <= 0:
        raise ValueError("address must be positive")
    if size <= 0:
        raise ValueError("size must be positive")
    if numa_node < 0:
        raise ValueError("numa_node must be non-negative")

    syscall_number = {"x86_64": 237, "amd64": 237, "aarch64": 235}.get(platform.machine().lower())
    if syscall_number is None:
        message = f"mbind is unsupported on architecture {platform.machine()}"
        if strict:
            raise NumaBindingError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return False

    word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
    word_count = numa_node // word_bits + 1
    mask_type = ctypes.c_ulong * word_count
    mask = mask_type()
    mask[numa_node // word_bits] = 1 << (numa_node % word_bits)
    # The raw Linux syscall ABI decrements maxnode before determining how many
    # words to copy. Passing the complete mask width plus one preserves the
    # highest word (including the common node-0-only mask).
    maxnode = word_count * word_bits + 1

    libc = _load_libc()
    ctypes.set_errno(0)
    result = libc.syscall(
        ctypes.c_long(syscall_number),
        ctypes.c_void_p(address),
        ctypes.c_ulong(size),
        ctypes.c_int(MPOL_BIND),
        ctypes.cast(mask, ctypes.c_void_p),
        ctypes.c_ulong(maxnode),
        ctypes.c_uint(0),
    )
    if result == 0:
        return True

    message = _errno_message(f"mbind({address:#x}, {size}, node={numa_node})")
    if strict:
        raise NumaBindingError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return False


class CudaRuntime:
    """Small ctypes wrapper around the CUDA runtime calls used by the arena."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        library.cudaSetDevice.argtypes = (ctypes.c_int,)
        library.cudaSetDevice.restype = ctypes.c_int
        library.cudaDeviceSynchronize.argtypes = ()
        library.cudaDeviceSynchronize.restype = ctypes.c_int
        library.cudaHostRegister.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
        library.cudaHostRegister.restype = ctypes.c_int
        library.cudaHostUnregister.argtypes = (ctypes.c_void_p,)
        library.cudaHostUnregister.restype = ctypes.c_int
        library.cudaDeviceGetPCIBusId.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
        library.cudaDeviceGetPCIBusId.restype = ctypes.c_int
        library.cudaGetErrorString.argtypes = (ctypes.c_int,)
        library.cudaGetErrorString.restype = ctypes.c_char_p

    @classmethod
    def load(cls, path: str | None = None) -> "CudaRuntime":
        candidates = []
        if path:
            candidates.append(path)
        configured = os.environ.get("LIGHTX2V_CUDART_PATH")
        if configured:
            candidates.append(configured)
        discovered = ctypes.util.find_library("cudart")
        if discovered:
            candidates.append(discovered)
        candidates.extend(
            [
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so",
                "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so",
            ]
        )

        errors = []
        for candidate in dict.fromkeys(candidates):
            try:
                return cls(ctypes.CDLL(candidate, use_errno=True))
            except OSError as error:
                errors.append(f"{candidate}: {error}")
        detail = "; ".join(errors) if errors else "no candidate paths found"
        raise CudaHostRegistrationError(f"cannot load CUDA runtime: {detail}")

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        raw = self._library.cudaGetErrorString(result)
        detail = raw.decode("utf-8", errors="replace") if raw else "unknown CUDA error"
        raise CudaHostRegistrationError(f"{operation} failed with CUDA error {result}: {detail}")

    def set_device(self, device: int) -> None:
        self._check(self._library.cudaSetDevice(device), f"cudaSetDevice({device})")

    def synchronize(self) -> None:
        self._check(self._library.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def host_register(self, address: int, size: int, flags: int) -> None:
        self._check(
            self._library.cudaHostRegister(ctypes.c_void_p(address), ctypes.c_size_t(size), ctypes.c_uint(flags)),
            f"cudaHostRegister({address:#x}, {size})",
        )

    def host_unregister(self, address: int) -> None:
        self._check(self._library.cudaHostUnregister(ctypes.c_void_p(address)), f"cudaHostUnregister({address:#x})")

    def device_pci_bus_id(self, device: int) -> str:
        result = ctypes.create_string_buffer(32)
        self._check(self._library.cudaDeviceGetPCIBusId(result, len(result), device), f"cudaDeviceGetPCIBusId({device})")
        return result.value.decode("ascii")


class CudaHostRegistration:
    """Own chunked CUDA host registrations for one mapped address range."""

    def __init__(
        self,
        address: int,
        size: int,
        *,
        device: int,
        chunk_bytes: int = DEFAULT_REGISTER_CHUNK_BYTES,
        flags: int = CUDA_HOST_REGISTER_PORTABLE,
        runtime: Any | None = None,
        regions: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        if address <= 0:
            raise ValueError("address must be positive")
        if size <= 0:
            raise ValueError("registration size must be positive")
        if device < 0:
            raise ValueError("CUDA device must be non-negative")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if chunk_bytes % page_size:
            raise ValueError(f"chunk_bytes must be a multiple of the system page size {page_size}")

        self.address = address
        self.size = size
        self.device = device
        self.chunk_bytes = chunk_bytes
        self.flags = flags
        self._runtime = CudaRuntime.load() if runtime is None else runtime
        self._regions = self._validate_regions(regions, page_size)
        self._registered: list[tuple[int, int]] = []
        self._started = False
        self._closed = True

    def _validate_regions(self, regions: Sequence[tuple[int, int]] | None, page_size: int) -> tuple[tuple[int, int], ...] | None:
        if regions is None:
            return None
        normalized = tuple((int(offset), int(length)) for offset, length in regions)
        if not normalized:
            raise ValueError("registration regions must be non-empty")
        cursor = 0
        for index, (offset, length) in enumerate(normalized):
            if offset != cursor:
                raise ValueError(f"registration regions must continuously cover the arena; expected offset {cursor}, got {offset}")
            if length <= 0:
                raise ValueError("registration region sizes must be positive")
            if offset % page_size:
                raise ValueError(f"registration region offset {offset} is not page-aligned")
            cursor = offset + length
            if index + 1 < len(normalized) and cursor % page_size:
                raise ValueError(f"internal registration boundary {cursor} is not page-aligned")
        if cursor != self.size:
            raise ValueError(f"registration regions cover {cursor} bytes, expected {self.size}")
        return normalized

    @property
    def chunks(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._registered)

    @property
    def registered_nbytes(self) -> int:
        return sum(size for _, size in self._registered)

    @property
    def chunk_count(self) -> int:
        return len(self._registered)

    def register(self) -> None:
        if self._started:
            raise RuntimeError("CUDA host registration has already been attempted")
        self._started = True
        self._closed = False
        try:
            self._runtime.set_device(self.device)
            regions = self._regions
            if regions is None:
                regions = []
                offset = 0
                while offset < self.size:
                    chunk_size = min(self.chunk_bytes, self.size - offset)
                    regions.append((offset, chunk_size))
                    offset += chunk_size
            for offset, chunk_size in regions:
                chunk_address = self.address + offset
                self._runtime.host_register(chunk_address, chunk_size, self.flags)
                self._registered.append((chunk_address, chunk_size))
        except BaseException as error:
            try:
                self._rollback()
            except BaseException as rollback_error:
                error.add_note(f"CUDA registration rollback also failed: {rollback_error!r}")
            raise

    def _rollback(self) -> None:
        failed_addresses = set()
        errors = []
        for address, _ in reversed(tuple(self._registered)):
            try:
                self._runtime.host_unregister(address)
            except BaseException as error:
                failed_addresses.add(address)
                errors.append(error)
        self._registered = [chunk for chunk in self._registered if chunk[0] in failed_addresses]
        self._closed = not self._registered
        if errors:
            raise CudaHostRegistrationError(f"CUDA registration rollback failed: {errors}") from errors[0]

    def close(self, *, synchronize: bool = True) -> None:
        if self._closed:
            return
        # If synchronization fails, none of the ranges may be unregistered:
        # an in-flight DMA could still be reading them. Leave all state intact
        # so a caller can retry close after recovering the CUDA context.
        self._runtime.set_device(self.device)
        if synchronize:
            self._runtime.synchronize()

        failed_addresses = set()
        errors = []
        for address, _ in reversed(tuple(self._registered)):
            try:
                self._runtime.host_unregister(address)
            except BaseException as error:
                failed_addresses.add(address)
                errors.append(error)
        self._registered = [chunk for chunk in self._registered if chunk[0] in failed_addresses]
        self._closed = not self._registered
        if errors:
            raise CudaHostRegistrationError(f"failed to unregister {len(errors)} CUDA host region(s): {errors}") from errors[0]

    def __enter__(self) -> "CudaHostRegistration":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _ArenaLifetime:
    """Resource owner retained by the tensor buffer protocol object."""

    def __init__(self, segment: SysVSegment, registration: CudaHostRegistration | None) -> None:
        self.segment = segment
        self.registration = registration
        self.closed = False

    def close(self, *, remove: bool | None = None, synchronize_cuda: bool = True) -> None:
        if self.closed:
            return
        if self.registration is not None:
            # A failed synchronization/unregister leaves the mapping alive.
            # Detaching it would invalidate an address still known to CUDA.
            self.registration.close(synchronize=synchronize_cuda)
            self.registration = None
        self.segment.close(remove=remove)
        self.closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            _QUARANTINED_LIFETIMES.append(self)


def _external_cpu_uint8_tensor(address: int, size: int, owner: Any) -> tuple[torch.Tensor, Any]:
    if address <= 0:
        raise ValueError("address must be positive")
    if size <= 0:
        raise ValueError("size must be positive")
    buffer_type = ctypes.c_ubyte * size
    backing = buffer_type.from_address(address)
    # torch.frombuffer retains ``backing``. Keeping the lifetime on that object
    # prevents implicit unmap/unregister while derived tensor views still live.
    backing._lightx2v_owner = owner
    return torch.frombuffer(backing, dtype=torch.uint8, count=size), backing


class SharedPinnedArena:
    """A SysV shared segment optionally registered as CUDA pinned memory."""

    def __init__(
        self,
        lifetime: _ArenaLifetime,
        raw: torch.Tensor,
        backing: Any,
        manifest: SharedWeightManifest | None,
    ) -> None:
        self._lifetime = lifetime
        self._raw = raw
        self._backing = backing
        self.manifest = manifest
        self._views: Mapping[str, torch.Tensor] | None = None
        self._closed = False

    @staticmethod
    def _resolve_size(nbytes: int | None, manifest: SharedWeightManifest | None) -> int:
        if nbytes is None and manifest is None:
            raise ValueError("nbytes or manifest is required")
        resolved = manifest.nbytes if nbytes is None else nbytes
        if resolved <= 0:
            raise ValueError("arena nbytes must be positive")
        if manifest is not None and manifest.nbytes != resolved:
            raise ValueError(f"arena nbytes {resolved} does not match manifest nbytes {manifest.nbytes}")
        return resolved

    @classmethod
    def create(
        cls,
        nbytes: int | None = None,
        *,
        manifest: SharedWeightManifest | None = None,
        numa_node: int | None = None,
        strict_numa: bool = True,
        register_cuda: bool = True,
        cuda_device: int | None = None,
        register_chunk_bytes: int = DEFAULT_REGISTER_CHUNK_BYTES,
        cuda_runtime: Any | None = None,
        permissions: int = 0o600,
        auto_remove: bool = False,
    ) -> "SharedPinnedArena":
        size = cls._resolve_size(nbytes, manifest)
        segment = SysVSegment.create(size, permissions=permissions, auto_remove=auto_remove)
        lifetime = _ArenaLifetime(segment, None)
        try:
            if numa_node is not None:
                segment.bind_to_numa(numa_node, strict=strict_numa)
            registration = cls._make_registration(
                segment,
                manifest=manifest,
                register_cuda=register_cuda,
                cuda_device=cuda_device,
                register_chunk_bytes=register_chunk_bytes,
                cuda_runtime=cuda_runtime,
            )
            lifetime.registration = registration
            if registration is not None:
                registration.register()
            raw, backing = _external_cpu_uint8_tensor(segment.address, size, lifetime)
            return cls(lifetime, raw, backing, manifest)
        except BaseException as error:
            try:
                lifetime.close(remove=True, synchronize_cuda=False)
            except BaseException as cleanup_error:
                _QUARANTINED_LIFETIMES.append(lifetime)
                error.add_note(f"arena cleanup also failed: {cleanup_error!r}")
            raise

    @classmethod
    def attach(
        cls,
        shmid: int,
        nbytes: int | None = None,
        *,
        manifest: SharedWeightManifest | None = None,
        register_cuda: bool = True,
        cuda_device: int | None = None,
        register_chunk_bytes: int = DEFAULT_REGISTER_CHUNK_BYTES,
        cuda_runtime: Any | None = None,
    ) -> "SharedPinnedArena":
        size = cls._resolve_size(nbytes, manifest)
        segment = SysVSegment.attach(shmid, expected_size=size)
        lifetime = _ArenaLifetime(segment, None)
        try:
            registration = cls._make_registration(
                segment,
                manifest=manifest,
                register_cuda=register_cuda,
                cuda_device=cuda_device,
                register_chunk_bytes=register_chunk_bytes,
                cuda_runtime=cuda_runtime,
            )
            lifetime.registration = registration
            if registration is not None:
                registration.register()
            raw, backing = _external_cpu_uint8_tensor(segment.address, size, lifetime)
            return cls(lifetime, raw, backing, manifest)
        except BaseException as error:
            try:
                lifetime.close(remove=False, synchronize_cuda=False)
            except BaseException as cleanup_error:
                _QUARANTINED_LIFETIMES.append(lifetime)
                error.add_note(f"arena cleanup also failed: {cleanup_error!r}")
            raise

    @staticmethod
    def _make_registration(
        segment: SysVSegment,
        *,
        manifest: SharedWeightManifest | None,
        register_cuda: bool,
        cuda_device: int | None,
        register_chunk_bytes: int,
        cuda_runtime: Any | None,
    ) -> CudaHostRegistration | None:
        if not register_cuda:
            return None
        if cuda_device is None:
            if not torch.cuda.is_available():
                raise CudaHostRegistrationError("CUDA registration requested but CUDA is unavailable and cuda_device was not provided")
            cuda_device = torch.cuda.current_device()
        regions = None
        if manifest is not None:
            regions = build_tensor_aware_registration_regions(
                segment.size,
                manifest.tensors,
                target_chunk_bytes=register_chunk_bytes,
            )
        return CudaHostRegistration(
            segment.address,
            segment.size,
            device=cuda_device,
            chunk_bytes=register_chunk_bytes,
            runtime=cuda_runtime,
            regions=regions,
        )

    @property
    def shmid(self) -> int:
        return self._lifetime.segment.shmid

    @property
    def address(self) -> int:
        return self._lifetime.segment.address

    @property
    def nbytes(self) -> int:
        return self._lifetime.segment.size

    @property
    def raw(self) -> torch.Tensor:
        if self._closed:
            raise SharedPinnedArenaError("arena is closed")
        return self._raw

    @property
    def is_cuda_registered(self) -> bool:
        return not self._closed and self._lifetime.registration is not None

    @property
    def registered_nbytes(self) -> int:
        registration = self._lifetime.registration
        return 0 if self._closed or registration is None else registration.registered_nbytes

    @property
    def registration_chunk_count(self) -> int:
        registration = self._lifetime.registration
        return 0 if self._closed or registration is None else registration.chunk_count

    def mark_for_deletion(self) -> None:
        if self._closed:
            raise SharedPinnedArenaError("arena is closed")
        self._lifetime.segment.mark_for_deletion()

    def tensor_views(self) -> Mapping[str, torch.Tensor]:
        if self._closed:
            raise SharedPinnedArenaError("arena is closed")
        if self.manifest is None:
            raise ValueError("a manifest is required to create tensor views")
        if self._views is not None:
            return self._views

        views = {}
        for spec in self.manifest.tensors:
            byte_view = self.raw.narrow(0, spec.offset, spec.nbytes)
            storage = byte_view.view(_dtype_from_name(spec.dtype)).view(spec.storage_shape)
            logical_shape = spec.logical_shape
            logical_stride = spec.logical_stride
            if logical_shape == spec.storage_shape and logical_stride == _contiguous_stride(spec.storage_shape) and spec.storage_offset == 0:
                views[spec.name] = storage
            else:
                views[spec.name] = torch.as_strided(
                    storage,
                    size=logical_shape,
                    stride=logical_stride,
                    storage_offset=storage.storage_offset() + spec.storage_offset,
                )
        self._views = MappingProxyType(views)
        return self._views

    def copy_from(self, tensors: Mapping[str, torch.Tensor]) -> None:
        views = self.tensor_views()
        expected = set(views)
        provided = set(tensors)
        if expected != provided:
            raise ValueError(f"tensor names do not match manifest (missing={sorted(expected - provided)}, unexpected={sorted(provided - expected)})")
        for name, view in views.items():
            source = tensors[name]
            if not isinstance(source, torch.Tensor):
                raise TypeError(f"{name!r} is not a tensor")
            if source.device.type != "cpu":
                raise ValueError(f"{name!r} must be on CPU, got {source.device}")
            if source.dtype != view.dtype or tuple(source.shape) != tuple(view.shape):
                raise ValueError(f"{name!r} has {tuple(source.shape)} {source.dtype}, expected {tuple(view.shape)} {view.dtype}")
            view.copy_(source)

    def stat(self) -> SharedMemoryStat:
        return self._lifetime.segment.stat()

    def close(self, *, remove: bool | None = None, synchronize_cuda: bool = True) -> None:
        """Release this process mapping.

        Explicit close invalidates every tensor view exported by this arena.
        The caller must stop prefetch and retire all DMA before calling it.
        """

        if self._closed:
            return
        self._lifetime.close(remove=remove, synchronize_cuda=synchronize_cuda)
        self._views = None
        self._raw = torch.empty(0, dtype=torch.uint8)
        self._backing = None
        self._closed = True

    def __enter__(self) -> "SharedPinnedArena":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
