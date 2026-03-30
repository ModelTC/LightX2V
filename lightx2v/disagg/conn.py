from __future__ import annotations

import logging
import struct
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import zmq

from lightx2v.disagg.mooncake import MooncakeTransferEngine

logger = logging.getLogger(__name__)


class DisaggregationPhase(Enum):
    NULL = "null"
    PHASE1 = "phase1"
    PHASE2 = "phase2"


class DisaggregationMode(Enum):
    NULL = "null"
    ENCODE = "encode"
    TRANSFORMER = "transformer"
    DECODE = "decode"


def group_concurrent_contiguous(src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
    src_groups = []
    dst_groups = []
    current_src = [src_indices[0]]
    current_dst = [dst_indices[0]]

    for i in range(1, len(src_indices)):
        src_contiguous = src_indices[i] == src_indices[i - 1] + 1
        dst_contiguous = dst_indices[i] == dst_indices[i - 1] + 1
        if src_contiguous and dst_contiguous:
            current_src.append(src_indices[i])
            current_dst.append(dst_indices[i])
        else:
            src_groups.append(current_src)
            dst_groups.append(current_dst)
            current_src = [src_indices[i]]
            current_dst = [dst_indices[i]]

    src_groups.append(current_src)
    dst_groups.append(current_dst)

    return src_groups, dst_groups


@dataclass
class DataArgs:
    sender_engine_rank: int
    receiver_engine_rank: int
    data_ptrs: list[int]
    data_lens: list[int]
    data_item_lens: list[int]
    ib_device: Optional[str] = None


class DataPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


RequestPoolType = Dict[int, List[int]]
WaitingPoolType = Dict[int, Tuple[str, list[int]]]
MONITOR_POLLING_PORT = 7788
REQUEST_POLLING_PORT = 12788
DATASENDER_POLLING_PORT = 17788
DATARECEIVER_POLLING_PORT = 27788


class DataManager:
    # TODO: make it general and support multiple transfer backend before merging
    def __init__(self, disaggregation_phase: DisaggregationPhase, disaggregation_mode: DisaggregationMode):
        self.engine = MooncakeTransferEngine()
        self.context = zmq.Context.instance()
        self.pool_lock = threading.Lock()
        self.data_args: Dict[int, DataArgs] = {}
        self.room_threads: Dict[int, List[threading.Thread]] = {}
        self.room_sockets: Dict[int, zmq.Socket] = {}
        self.room_stop_events: Dict[int, threading.Event] = {}
        self.disaggregation_phase = disaggregation_phase
        self.disaggregation_mode = disaggregation_mode
        self.request_pool: RequestPoolType = {}
        self.request_status: Dict[int, DataPoll] = {}
        self.transfer_event: Optional[threading.Event] = None
        self.transfer_stop_event: Optional[threading.Event] = None
        self.transfer_thread: Optional[threading.Thread] = None
        if self.disaggregation_phase == DisaggregationPhase.PHASE1:
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                self.waiting_pool: WaitingPoolType = {}
                self.start_transfer_thread()
            elif self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                pass
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        elif self.disaggregation_phase == DisaggregationPhase.PHASE2:
            if self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.waiting_pool: WaitingPoolType = {}
                self.start_transfer_thread()
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                pass
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        else:
            raise ValueError(f"Unsupported DisaggregationPhase: {self.disaggregation_phase}")

    def start_transfer_thread(self):
        self.transfer_event = threading.Event()
        self.transfer_stop_event = threading.Event()

        def transfer_loop():
            while self.transfer_stop_event is not None and not self.transfer_stop_event.is_set():
                self.transfer_event.wait()
                if self.transfer_stop_event.is_set():
                    break
                self.transfer_event.clear()

                while True:
                    with self.pool_lock:
                        if not hasattr(self, "waiting_pool"):
                            break
                        pending_room = None
                        for room in list(self.waiting_pool.keys()):
                            if room in self.request_pool:
                                pending_room = room
                                break
                        if pending_room is None:
                            break

                        status = DataPoll.Transferring
                        self.request_status[pending_room] = status
                        endpoint, mooncake_session_id, receiver_ptrs = self.waiting_pool.pop(pending_room)
                        sender_data_ptrs = self.request_pool.pop(pending_room)

                    self.sync_status_to_transformer_endpoint(endpoint, pending_room)
                    ret = self.send_data(
                        pending_room,
                        mooncake_session_id,
                        sender_data_ptrs,
                        receiver_ptrs,
                    )
                    with self.pool_lock:
                        if ret != 0:
                            self.request_status[pending_room] = DataPoll.Failed
                        else:
                            self.request_status[pending_room] = DataPoll.Success
                    self.sync_status_to_transformer_endpoint(endpoint, pending_room)

        self.transfer_thread = threading.Thread(target=transfer_loop, name="data-transfer-thread")
        self.transfer_thread.start()

    def init(self, args: DataArgs, room: int):
        if room in self.data_args:
            self.remove(room)
        self.data_args[room] = args
        self.register_buffer_to_engine(room)
        if self.disaggregation_phase == DisaggregationPhase.PHASE1:
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                self.start_phase1_encode_thread(room)
            elif self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.start_phase1_transformer_thread(room)
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        elif self.disaggregation_phase == DisaggregationPhase.PHASE2:
            if self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.start_phase2_transformer_thread(room)
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.start_phase2_decode_thread(room)
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        else:
            raise ValueError(f"Unsupported DisaggregationPhase: {self.disaggregation_phase}")

    def remove(self, room: int):
        if self.disaggregation_phase == DisaggregationPhase.PHASE1:
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                self.end_phase1_encode_thread(room)
            elif self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.end_phase1_transformer_thread(room)
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        elif self.disaggregation_phase == DisaggregationPhase.PHASE2:
            if self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.end_phase2_transformer_thread(room)
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.end_phase2_decode_thread(room)
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        else:
            raise ValueError(f"Unsupported DisaggregationPhase: {self.disaggregation_phase}")

        # Recycle room-scoped mappings.
        args = self.data_args.pop(room, None)
        if args is not None:
            for data_ptr in args.data_ptrs:
                self.engine.deregister(data_ptr)

        with self.pool_lock:
            self.request_pool.pop(room, None)
            self.request_status.pop(room, None)
            if hasattr(self, "waiting_pool"):
                self.waiting_pool.pop(room, None)

    def release(self):
        if self.transfer_stop_event is not None:
            self.transfer_stop_event.set()
        if self.transfer_event is not None:
            self.transfer_event.set()
        if self.transfer_thread is not None and self.transfer_thread.is_alive():
            self.transfer_thread.join(timeout=1.0)
        self.transfer_thread = None

        for room in list(self.room_threads.keys()):
            self.end_room_threads(room)

        for room in list(self.data_args.keys()):
            args = self.data_args.pop(room, None)
            if args is None:
                continue
            for data_ptr in args.data_ptrs:
                self.engine.deregister(data_ptr)

        with self.pool_lock:
            self.request_pool.clear()
            self.request_status.clear()
            if hasattr(self, "waiting_pool"):
                self.waiting_pool.clear()

    def register_buffer_to_engine(self, room: int):
        args = self.data_args[room]
        for data_ptr, data_len in zip(args.data_ptrs, args.data_lens):
            self.engine.register(data_ptr, data_len)

    def prepare_room_threads(self, room: int):
        self.room_stop_events[room] = threading.Event()
        self.room_threads[room] = []

    def register_room_thread(self, room: int, thread: threading.Thread):
        self.room_threads.setdefault(room, []).append(thread)

    def get_or_create_room_socket(self, room: int, port: int):
        socket = self.room_sockets.get(room)
        if socket is not None:
            return socket
        socket = self.context.socket(zmq.PULL)
        socket.setsockopt(zmq.RCVTIMEO, 200)
        socket.bind(f"tcp://*:{port}")
        self.room_sockets[room] = socket
        return socket

    def close_room_socket(self, room: int):
        socket = self.room_sockets.pop(room, None)
        if socket is not None:
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()

    def end_room_threads(self, room: int):
        stop_event = self.room_stop_events.get(room)
        if stop_event is not None:
            stop_event.set()
        threads = self.room_threads.get(room, [])
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self.close_room_socket(room)
        self.room_threads.pop(room, None)
        self.room_stop_events.pop(room, None)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def send_data(
        self,
        room: int,
        mooncake_session_id: str,
        sender_data_ptrs: List[int],
        receiver_ptrs: list[int],
    ):
        # TODO: transfer data in batch if there are many tensors or large tensors, instead of sending one by one.
        args = self.data_args[room]
        tensor_num = int(len(args.data_ptrs))
        for tensor_id in range(tensor_num):
            sender_addr = sender_data_ptrs[tensor_id]
            item_len = args.data_item_lens[tensor_id]
            receiver_addr = receiver_ptrs[tensor_id]

            # TODO: mooncake transfer engine can do async transfer. Do async later
            status = self.engine.transfer_sync(
                mooncake_session_id,
                sender_addr,
                receiver_addr,
                item_len,
            )
            if status != 0:
                return status
        return 0

    def sync_status_to_transformer_endpoint(self, remote: str, room: int):
        if ":" in remote:
            remote = remote.split(":")[0]
        receiver_rank = self.data_args[room].receiver_engine_rank
        receiver_rank_port = DATARECEIVER_POLLING_PORT + receiver_rank + room * 10
        self._connect("tcp://" + remote + ":" + str(receiver_rank_port)).send_multipart(
            [
                str(room).encode("ascii"),
                str(self.request_status[room]).encode("ascii"),
            ]
        )

    def start_phase1_encode_thread(self, room: int):
        self.prepare_room_threads(room)
        stop_event = self.room_stop_events[room]
        sender_rank_port = DATASENDER_POLLING_PORT + self.data_args[room].sender_engine_rank + room * 10
        logger.info("Encoder sender_rank_port=%s", sender_rank_port)
        room_socket = self.get_or_create_room_socket(room, sender_rank_port)

        def encode_thread():
            while not stop_event.is_set():
                try:
                    (
                        endpoint,
                        mooncake_session_id,
                        bootstrap_room,
                        transformer_ptrs,
                    ) = room_socket.recv_multipart()
                except zmq.Again:
                    continue
                if bootstrap_room.decode("ascii") == "None":
                    continue
                endpoint = endpoint.decode("ascii")
                mooncake_session_id = mooncake_session_id.decode("ascii")
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                transformer_ptrs = list(struct.unpack(f"{len(transformer_ptrs) // 8}Q", transformer_ptrs))
                logger.info(
                    "Encoder received ZMQ: endpoint=%s session_id=%s room=%s transformer_ptrs=%s",
                    endpoint,
                    mooncake_session_id,
                    bootstrap_room,
                    transformer_ptrs,
                )
                with self.pool_lock:
                    self.waiting_pool[bootstrap_room] = (
                        endpoint,
                        mooncake_session_id,
                        transformer_ptrs,
                    )
                if self.transfer_event is not None:
                    self.transfer_event.set()

        encode_worker = threading.Thread(target=encode_thread)
        encode_worker.start()
        self.register_room_thread(room, encode_worker)

    def end_phase1_encode_thread(self, room: int):
        self.end_room_threads(room)

    def start_phase1_transformer_thread(self, room: int):
        self.prepare_room_threads(room)
        stop_event = self.room_stop_events[room]
        receiver_rank_port = DATARECEIVER_POLLING_PORT + self.data_args[room].receiver_engine_rank + room * 10
        room_socket = self.get_or_create_room_socket(room, receiver_rank_port)

        def transformer_thread():
            while not stop_event.is_set():
                try:
                    (bootstrap_room, status) = room_socket.recv_multipart()
                except zmq.Again:
                    continue
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.request_status[bootstrap_room] = status

        transformer_worker = threading.Thread(target=transformer_thread)
        transformer_worker.start()
        self.register_room_thread(room, transformer_worker)

    def end_phase1_transformer_thread(self, room: int):
        self.end_room_threads(room)

    def start_phase2_transformer_thread(self, room: int):
        self.prepare_room_threads(room)
        stop_event = self.room_stop_events[room]
        sender_rank_port = DATASENDER_POLLING_PORT + self.data_args[room].sender_engine_rank + room * 10
        logger.info("Transformer sender_rank_port=%s", sender_rank_port)
        room_socket = self.get_or_create_room_socket(room, sender_rank_port)

        def transformer_thread():
            while not stop_event.is_set():
                try:
                    (
                        endpoint,
                        mooncake_session_id,
                        bootstrap_room,
                        decode_ptrs,
                    ) = room_socket.recv_multipart()
                except zmq.Again:
                    continue
                if bootstrap_room.decode("ascii") == "None":
                    continue
                endpoint = endpoint.decode("ascii")
                mooncake_session_id = mooncake_session_id.decode("ascii")
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                decode_ptrs = list(struct.unpack(f"{len(decode_ptrs) // 8}Q", decode_ptrs))
                logger.info(
                    "Transformer received ZMQ: endpoint=%s session_id=%s room=%s decode_ptrs=%s",
                    endpoint,
                    mooncake_session_id,
                    bootstrap_room,
                    decode_ptrs,
                )
                with self.pool_lock:
                    self.waiting_pool[bootstrap_room] = (
                        endpoint,
                        mooncake_session_id,
                        decode_ptrs,
                    )
                if self.transfer_event is not None:
                    self.transfer_event.set()

        transformer_worker = threading.Thread(target=transformer_thread)
        transformer_worker.start()
        self.register_room_thread(room, transformer_worker)

    def end_phase2_transformer_thread(self, room: int):
        self.end_room_threads(room)

    def start_phase2_decode_thread(self, room: int):
        self.prepare_room_threads(room)
        stop_event = self.room_stop_events[room]
        receiver_rank_port = DATARECEIVER_POLLING_PORT + self.data_args[room].receiver_engine_rank + room * 10
        room_socket = self.get_or_create_room_socket(room, receiver_rank_port)

        def decode_thread():
            while not stop_event.is_set():
                try:
                    (bootstrap_room, status) = room_socket.recv_multipart()
                except zmq.Again:
                    continue
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.request_status[bootstrap_room] = status

        decode_worker = threading.Thread(target=decode_thread)
        decode_worker.start()
        self.register_room_thread(room, decode_worker)

    def end_phase2_decode_thread(self, room: int):
        self.end_room_threads(room)

    def enqueue_request(
        self,
        bootstrap_room: int,
        data_ptrs: List[int],
    ):
        with self.pool_lock:
            self.request_pool[bootstrap_room] = data_ptrs
            self.request_status[bootstrap_room] = DataPoll.WaitingForInput
        if (
            self.disaggregation_phase == DisaggregationPhase.PHASE1
            and self.disaggregation_mode == DisaggregationMode.ENCODE
            or self.disaggregation_phase == DisaggregationPhase.PHASE2
            and self.disaggregation_mode == DisaggregationMode.TRANSFORMER
        ):
            if self.transfer_event is not None:
                self.transfer_event.set()

    def check_status(self, bootstrap_room: int):
        with self.pool_lock:
            if (
                self.disaggregation_phase == DisaggregationPhase.PHASE1
                and self.disaggregation_mode == DisaggregationMode.TRANSFORMER
                or self.disaggregation_phase == DisaggregationPhase.PHASE2
                and self.disaggregation_mode == DisaggregationMode.DECODE
            ) and self.request_status[bootstrap_room] == DataPoll.Success:
                if bootstrap_room in self.request_pool:
                    self.request_pool.pop(bootstrap_room)

            return self.request_status[bootstrap_room]

    def set_status(self, bootstrap_room: int, status: DataPoll):
        with self.pool_lock:
            self.request_status[bootstrap_room] = status

    def get_localhost(self):
        return self.engine.get_localhost()

    def get_session_id(self):
        return self.engine.get_session_id()


class DataSender:
    def __init__(self, mgr: DataManager, bootstrap_addr: str, bootstrap_room: int):
        self.data_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.data_mgr.set_status(bootstrap_room, DataPoll.WaitingForInput)

    def init(self):
        pass

    def send(self, data_ptrs: List[int]):
        self.data_mgr.enqueue_request(self.bootstrap_room, data_ptrs)

    def poll(self) -> DataPoll:
        return self.data_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake DataSender Exception")


class DataReceiver:
    def __init__(self, mgr: DataManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.data_mgr = mgr
        if self.bootstrap_room is None:
            raise ValueError("bootstrap_room is required for DataReceiver")
        args = self.data_mgr.data_args[self.bootstrap_room]
        sender_rank_port = DATASENDER_POLLING_PORT + args.sender_engine_rank + self.bootstrap_room * 10
        self.sender_server_url = bootstrap_addr.split(":")[0] + ":" + str(sender_rank_port)
        logger.info("DataReceiver sender_server_url=%s", self.sender_server_url)
        self.receiver_ip = self.data_mgr.get_localhost()
        self.session_id = self.data_mgr.get_session_id()
        self.data_mgr.set_status(bootstrap_room, DataPoll.WaitingForInput)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def init(self):
        args = self.data_mgr.data_args[self.bootstrap_room]
        packed_data_ptrs = b"".join(struct.pack("Q", ptr) for ptr in args.data_ptrs)
        self.data_mgr.enqueue_request(self.bootstrap_room, packed_data_ptrs)
        self._connect("tcp://" + self.sender_server_url).send_multipart(
            [
                self.receiver_ip.encode("ascii"),
                self.session_id.encode("ascii"),
                str(self.bootstrap_room).encode("ascii"),
                packed_data_ptrs,
            ]
        )

    def poll(self) -> DataPoll:
        return self.data_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake DataReceiver Exception")


class ReqManager:
    def __init__(self):
        self.context = zmq.Context.instance()
        self.push_sockets: Dict[str, zmq.Socket] = {}
        self.pull_sockets: Dict[int, zmq.Socket] = {}

    def send(self, ip: str, port: int, config: Any):
        def _to_builtin(value: Any):
            if isinstance(value, Mapping):
                return {k: _to_builtin(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_builtin(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_to_builtin(v) for v in value)
            return value

        endpoint = f"tcp://{ip}:{port}"
        socket = self.push_sockets.get(endpoint)
        if socket is None:
            socket = self.context.socket(zmq.PUSH)
            socket.connect(endpoint)
            self.push_sockets[endpoint] = socket
        socket.send_pyobj(_to_builtin(config))

    def receive(self, port: int):
        socket = self.pull_sockets.get(port)
        if socket is None:
            socket = self.context.socket(zmq.PULL)
            socket.bind(f"tcp://*:{port}")
            self.pull_sockets[port] = socket
        return socket.recv_pyobj()

    def receive_non_block(self, port: int):
        socket = self.pull_sockets.get(port)
        if socket is None:
            socket = self.context.socket(zmq.PULL)
            socket.bind(f"tcp://*:{port}")
            self.pull_sockets[port] = socket
        try:
            return socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
