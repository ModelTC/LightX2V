"""Lightweight, process-local routing across replicated reward services."""

import hashlib
import os
import socket
import threading
from collections.abc import Sequence
from urllib.parse import urlsplit

_lock = threading.Lock()
_counters = {}


def normalize_endpoints(value, *, name="reward endpoint"):
    """Validate one HTTP(S) base URL or a nonempty sequence of base URLs.

    Return a tuple without rewriting URLs. OmegaConf ListConfig implements
    Sequence, so a YAML list works without requiring a configuration dependency.
    """
    if isinstance(value, str):
        endpoints = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        endpoints = tuple(value)
    else:
        raise ValueError(f"{name} must be an HTTP(S) base URL or a nonempty list of URLs")
    if not endpoints:
        raise ValueError(f"{name} must contain at least one HTTP(S) base URL")
    for index, endpoint in enumerate(endpoints):
        valid = isinstance(endpoint, str) and bool(endpoint) and not any(c.isspace() for c in endpoint)
        if valid:
            try:
                parts = urlsplit(endpoint)
                valid = parts.scheme in {"http", "https"} and bool(parts.hostname) and not parts.query and not parts.fragment
                # Accessing port also rejects malformed or out-of-range ports.
                parts.port
            except ValueError:
                valid = False
        if not valid:
            raise ValueError(f"{name}[{index}] must be a valid HTTP(S) base URL without query or fragment")
    return endpoints


def _initial_offset(size):
    # A global process rank makes initial distribution exact when one is provided.
    # Ray reward actors commonly lack RANK: disperse those processes using a stable
    # hostname/PID digest, without importing Ray or making a coordinator request.
    for variable in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        try:
            rank = int(os.environ.get(variable, ""))
        except ValueError:
            continue
        if rank >= 0:
            return rank % size
    identity = f"{socket.gethostname()}:{os.getpid()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % size


def select_endpoint(value, *, name="reward endpoint"):
    """Select one replica using thread-safe round robin and a worker offset.

    Counters are independent for each endpoint pool in each worker process. This
    does not probe health, retry another replica, or turn failed scores into zero.
    """
    endpoints = normalize_endpoints(value, name=name)
    if len(endpoints) == 1:
        return endpoints[0]
    with _lock:
        index = _counters.get(endpoints)
        if index is None:
            index = _initial_offset(len(endpoints))
        _counters[endpoints] = (index + 1) % len(endpoints)
    return endpoints[index]


def _after_fork():
    # A fork must not inherit a lock held by another thread or its parent's cursor.
    global _lock, _counters
    _lock = threading.Lock()
    _counters = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
