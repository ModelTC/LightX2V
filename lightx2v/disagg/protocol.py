import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class TensorMetadata:
    id: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass
class AllocationRequest:
    """
    Request sent from Sender (Encoder) to Receiver (Transformer).
    - bootstrap_room: Unique ID for the transfer slot/session.
    - buffer_sizes: Precomputed upper-bound buffer sizes.
    """

    bootstrap_room: int
    buffer_sizes: List[int]


@dataclass
class RemoteBuffer:
    addr: int
    nbytes: int


@dataclass
class MemoryHandle:
    """
    Handle sent from Receiver (Transformer) to Sender (Encoder).
    - buffers: List of remote buffer details corresponding to the tensor_specs in the request.
    """

    buffers: List[RemoteBuffer]
