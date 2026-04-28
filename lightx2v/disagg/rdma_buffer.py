from __future__ import annotations

import ctypes
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from lightx2v.disagg.rdma_client import RDMAClient
    from lightx2v.disagg.rdma_server import RDMAServer
except ImportError:
    RDMAClient = None
    RDMAServer = None

logger = logging.getLogger(__name__)

_U64_MASK = (1 << 64) - 1


@dataclass
class RDMABufferDescriptor:
    slot_addr: int
    slot_bytes: int
    slot_size: int
    buffer_size: int
    head_addr: int
    tail_addr: int
    rkey: int = 0
    head_bytes: int = 8
    tail_bytes: int = 8


class RDMABuffer:
    """Ring buffer backed by RDMA-accessible memory.

    Role model:
    - server: producer side, owns and registers memory regions.
    - client: consumer side, reads slots remotely and updates head by rdma_faa.

    The ring stores serialized JSON configs in fixed-size slots.
    """

    def __init__(
        self,
        role: str,
        buffer_size: int = 128,
        slot_size: int = 4096,
        *,
        rdma_server: Optional[RDMAServer] = None,
        rdma_client: Optional[RDMAClient] = None,
        remote: Optional[RDMABufferDescriptor] = None,
    ):
        if role not in {"server", "client"}:
            raise ValueError("role must be 'server' or 'client'")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if slot_size <= 8:
            raise ValueError("slot_size must be greater than 8")

        self.role = role
        self.buffer_size = int(buffer_size)
        self.slot_size = int(slot_size)

        self.rdma_server: Optional[RDMAServer] = rdma_server
        self.rdma_client: Optional[RDMAClient] = rdma_client

        self._lock = threading.Lock()

        # Local backing store (server side). Client can also allocate local scratch.
        self._slot_mem = bytearray(self.buffer_size * self.slot_size)
        self._head_mem = bytearray(8)
        self._tail_mem = bytearray(8)

        self._slot_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._slot_mem))
        self._head_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._head_mem))
        self._tail_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._tail_mem))

        # Initialize head/tail to 0.
        self._write_local_u64(self._head_mem, 0)
        self._write_local_u64(self._tail_mem, 0)

        self._descriptor: Optional[RDMABufferDescriptor] = None
        if self.role == "server":
            if self.rdma_server is not None:
                info = self.rdma_server.get_local_info()
                base_addr = int(info["addr"])
                need_bytes = 16 + self.buffer_size * self.slot_size
                self.rdma_server.register_memory(base_addr, need_bytes)
                self.rdma_server.write_memory(base_addr, (0).to_bytes(8, byteorder="little", signed=False))
                self.rdma_server.write_memory(base_addr + 8, (0).to_bytes(8, byteorder="little", signed=False))
                self._descriptor = RDMABufferDescriptor(
                    slot_addr=base_addr + 16,
                    slot_bytes=self.buffer_size * self.slot_size,
                    slot_size=self.slot_size,
                    buffer_size=self.buffer_size,
                    head_addr=base_addr,
                    tail_addr=base_addr + 8,
                    rkey=int(info.get("rkey", 0)),
                )
            else:
                self._descriptor = RDMABufferDescriptor(
                    slot_addr=self._slot_addr,
                    slot_bytes=len(self._slot_mem),
                    slot_size=self.slot_size,
                    buffer_size=self.buffer_size,
                    head_addr=self._head_addr,
                    tail_addr=self._tail_addr,
                )
        else:
            if remote is None:
                raise ValueError("client role requires remote descriptor")
            self._descriptor = remote
            self.buffer_size = int(remote.buffer_size)
            self.slot_size = int(remote.slot_size)

    @property
    def descriptor(self) -> RDMABufferDescriptor:
        if self._descriptor is None:
            raise RuntimeError("descriptor is not initialized")
        return self._descriptor

    def _write_local_u64(self, buf: bytearray, value: int):
        buf[:8] = (int(value) & _U64_MASK).to_bytes(8, byteorder="little", signed=False)

    def _read_local_u64(self, buf: bytearray) -> int:
        return int.from_bytes(bytes(buf[:8]), byteorder="little", signed=False)

    def _u64_distance(self, newer: int, older: int) -> int:
        """Return unsigned circular distance on a 64-bit counter space."""
        return (int(newer) - int(older)) & _U64_MASK

    def _rdma_faa(self, ptr_addr: int, add_value: int) -> int:
        if self.rdma_client is not None:
            return self.rdma_client.rdma_faa(ptr_addr, int(add_value), rkey=self.descriptor.rkey)

        if self.rdma_server is not None:
            with self._lock:
                old = self._read_remote_u64(ptr_addr)
                new = (old + int(add_value)) & ((1 << 64) - 1)
                self._rdma_write_bytes(ptr_addr, new.to_bytes(8, byteorder="little", signed=False))
                return old

        # Fallback: local atomic emulation (useful for single-process validation).
        with self._lock:
            if ptr_addr == self.descriptor.head_addr:
                old = self._read_local_u64(self._head_mem)
                self._write_local_u64(self._head_mem, (old + int(add_value)) & _U64_MASK)
                return old
            if ptr_addr == self.descriptor.tail_addr:
                old = self._read_local_u64(self._tail_mem)
                self._write_local_u64(self._tail_mem, (old + int(add_value)) & _U64_MASK)
                return old
        raise RuntimeError("rdma_faa failed and no local fallback for ptr")

    def _rdma_cas(self, ptr_addr: int, compare_value: int, swap_value: int) -> int:
        if self.rdma_client is not None:
            return self.rdma_client.rdma_cas(
                ptr_addr,
                int(compare_value),
                int(swap_value),
                rkey=self.descriptor.rkey,
            )

        if self.rdma_server is not None:
            with self._lock:
                old = self._read_remote_u64(ptr_addr)
                if old == (int(compare_value) & _U64_MASK):
                    new = int(swap_value) & _U64_MASK
                    self._rdma_write_bytes(ptr_addr, new.to_bytes(8, byteorder="little", signed=False))
                return old

        # Local fallback for single-process testing.
        with self._lock:
            if ptr_addr == self.descriptor.head_addr:
                old = self._read_local_u64(self._head_mem)
                if old == (int(compare_value) & _U64_MASK):
                    self._write_local_u64(self._head_mem, int(swap_value) & _U64_MASK)
                return old
            if ptr_addr == self.descriptor.tail_addr:
                old = self._read_local_u64(self._tail_mem)
                if old == (int(compare_value) & _U64_MASK):
                    self._write_local_u64(self._tail_mem, int(swap_value) & _U64_MASK)
                return old
        raise RuntimeError("rdma_cas failed and no local fallback for ptr")

    def _rdma_read_bytes(self, remote_addr: int, length: int) -> bytes:
        if self.rdma_server is not None and self._descriptor is not None:
            base = self._descriptor.head_addr
            end = base + 16 + self.buffer_size * self.slot_size
            if base <= remote_addr < end:
                return self.rdma_server.read_memory(int(remote_addr), int(length))

        if self.rdma_client is not None:
            data = self.rdma_client.rdma_read_from(int(remote_addr), int(length), rkey=self.descriptor.rkey)
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            raise RuntimeError("rdma_read_from returned non-bytes payload")

        # Local fallback for single-process testing.
        if remote_addr == self.descriptor.head_addr:
            return bytes(self._head_mem[:length])
        if remote_addr == self.descriptor.tail_addr:
            return bytes(self._tail_mem[:length])
        base = self.descriptor.slot_addr
        end = base + self.descriptor.slot_bytes
        if base <= remote_addr < end:
            off = remote_addr - base
            return bytes(self._slot_mem[off : off + length])
        raise RuntimeError("rdma_read failed and no local fallback for addr")

    def _rdma_write_bytes(self, remote_addr: int, payload: bytes):
        if self.rdma_server is not None and self._descriptor is not None:
            base = self._descriptor.head_addr
            end = base + 16 + self.buffer_size * self.slot_size
            if base <= remote_addr < end:
                self.rdma_server.write_memory(int(remote_addr), payload)
                return

        if self.rdma_client is not None:
            self.rdma_client.rdma_write_to(int(remote_addr), payload, rkey=self.descriptor.rkey)
            return

        # Local fallback for single-process testing.
        if remote_addr == self.descriptor.head_addr:
            self._head_mem[: len(payload)] = payload
            return
        if remote_addr == self.descriptor.tail_addr:
            self._tail_mem[: len(payload)] = payload
            return
        base = self.descriptor.slot_addr
        end = base + self.descriptor.slot_bytes
        if base <= remote_addr < end:
            off = remote_addr - base
            self._slot_mem[off : off + len(payload)] = payload
            return
        raise RuntimeError("rdma_write failed and no local fallback for addr")

    def _read_remote_u64(self, remote_addr: int) -> int:
        raw = self._rdma_read_bytes(remote_addr, 8)
        return int.from_bytes(raw, byteorder="little", signed=False)

    def _slot_offset(self, index: int) -> int:
        return (index % self.buffer_size) * self.slot_size

    def _serialize_config(self, config: Dict[str, Any]) -> bytes:
        payload = json.dumps(config, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if len(payload) > self.slot_size - 4:
            raise ValueError(f"config payload too large: {len(payload)} > {self.slot_size - 4}")
        return len(payload).to_bytes(4, byteorder="little", signed=False) + payload

    def _deserialize_config(self, raw_slot: bytes) -> Dict[str, Any]:
        if len(raw_slot) < 4:
            raise ValueError("invalid slot payload")
        plen = int.from_bytes(raw_slot[:4], byteorder="little", signed=False)
        if plen == 0:
            raise ValueError("slot payload is not committed yet")
        if plen > self.slot_size - 4:
            raise ValueError(f"invalid slot payload length: {plen}")
        data = raw_slot[4 : 4 + plen]
        try:
            return json.loads(data.decode("utf-8"))
        except Exception as exc:
            raise ValueError("slot payload is incomplete or corrupted") from exc

    def produce(self, config: Dict[str, Any]) -> int:
        """Produce one config into ring buffer and advance tail by rdma_faa."""
        if self.rdma_server is None and self.rdma_client is None:
            raise RuntimeError("produce requires rdma_server or rdma_client")

        # Read current indices first, write the slot fully, then publish by advancing tail.
        old_tail = self._read_remote_u64(self.descriptor.tail_addr)
        cur_head = self._read_remote_u64(self.descriptor.head_addr)
        if self._u64_distance(old_tail, cur_head) >= self.buffer_size:
            raise BufferError("ring buffer is full")

        slot_idx = old_tail % self.buffer_size
        offset = self._slot_offset(slot_idx)
        payload = self._serialize_config(config)
        payload_len_header = payload[:4]
        payload_body = payload[4:]

        # Write payload to the selected slot (works for both server-local and client-remote paths).
        slot_addr = self.descriptor.slot_addr + offset
        self._rdma_write_bytes(slot_addr, b"\x00" * self.slot_size)
        if payload_body:
            self._rdma_write_bytes(slot_addr + 4, payload_body)
        # Write length header last so consumers never parse a half-written payload.
        self._rdma_write_bytes(slot_addr, payload_len_header)
        self._rdma_faa(self.descriptor.tail_addr, 1)
        logger.info("Produced config to RDMA buffer slot %d", slot_idx)
        return slot_idx

    def consume(self) -> Optional[Dict[str, Any]]:
        """Consume one config from ring buffer and advance head by rdma_faa."""
        if self.role != "client":
            raise RuntimeError("consume is only allowed in client role")

        max_claim_retries = max(8, self.buffer_size * 2)
        claim_retry_sleep_seconds = 0.001

        for _ in range(max_claim_retries):
            try:
                cur_head = self._read_remote_u64(self.descriptor.head_addr)
                cur_tail = self._read_remote_u64(self.descriptor.tail_addr)
            except Exception:
                return None

            # Fast path: empty queue, do not touch head.
            if self._u64_distance(cur_tail, cur_head) == 0:
                return None

            slot_idx = cur_head % self.buffer_size
            slot_addr = self.descriptor.slot_addr + self._slot_offset(slot_idx)
            max_read_retries = 5
            retry_sleep_seconds = 0.002
            last_error: Optional[Exception] = None
            config: Optional[Dict[str, Any]] = None

            for _ in range(max_read_retries):
                try:
                    raw = self._rdma_read_bytes(slot_addr, self.slot_size)
                    config = self._deserialize_config(raw)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(retry_sleep_seconds)

            if config is None:
                # Keep head unchanged so this slot can be retried later.
                logger.warning(
                    "RDMA buffer slot %d read incomplete after retries, keeping head unchanged: %s",
                    slot_idx,
                    last_error,
                )
                return None

            try:
                old_head = self._rdma_cas(
                    self.descriptor.head_addr,
                    cur_head,
                    (cur_head + 1) & _U64_MASK,
                )
            except Exception as exc:
                logger.warning("RDMA buffer head CAS failed for slot %d: %s", slot_idx, exc)
                return None

            if old_head != cur_head:
                # Another consumer advanced head first; retry from latest head.
                time.sleep(claim_retry_sleep_seconds)
                continue

            logger.info("Consumed config from RDMA buffer slot %d", slot_idx)
            return config

        logger.warning("RDMA buffer consume contention is too high, skip this round")
        return None

    def pending_count(self) -> int:
        """Return current queue length inferred from ring tail/head counters."""
        cur_head = self._read_remote_u64(self.descriptor.head_addr)
        cur_tail = self._read_remote_u64(self.descriptor.tail_addr)
        pending = int(self._u64_distance(cur_tail, cur_head))
        if pending <= 0:
            return 0
        if pending > self.buffer_size:
            return self.buffer_size
        return pending
