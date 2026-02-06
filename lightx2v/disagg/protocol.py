import zmq
import torch
import pickle
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from lightx2v.disagg.mooncake import MooncakeTransferEngine

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
    - config: Inference config used to estimate upper-bound buffer sizes.
    """
    bootstrap_room: str
    config: Dict[str, Any]

@dataclass
class RemoteBuffer:
    addr: int
    session_id: str
    nbytes: int

@dataclass
class MemoryHandle:
    """
    Handle sent from Receiver (Transformer) to Sender (Encoder).
    - buffers: List of remote buffer details corresponding to the tensor_specs in the request.
    """
    buffers: List[RemoteBuffer]
