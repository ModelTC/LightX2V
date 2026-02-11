from __future__ import annotations

import asyncio
import logging
import struct
import threading
import torch
from functools import cache
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import zmq
from aiohttp import web

from lightx2v.disagg.mooncake import MooncakeTransferEngine

logger = logging.getLogger(__name__)

class DisaggregationMode(Enum):
    NULL = "null"
    ENCODE = "encode"
    TRANSFORMER = "transformer"

def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]
) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
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
WaitingPoolType = Dict[
    int, Tuple[str, list[int]]
]
DATASENDER_POLLING_PORT = 17788
DATARECEIVER_POLLING_PORT = 27788


class DataManager:
    # TODO: make it general and support multiple transfer backend before merging
    def __init__(self, args: DataArgs, disaggregation_mode: DisaggregationMode):
        self.engine = MooncakeTransferEngine()
        self.data_args = args
        self.disaggregation_mode = disaggregation_mode
        self.request_pool: RequestPoolType = {}
        self.request_status: Dict[int, DataPoll] = {}
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
        if self.disaggregation_mode == DisaggregationMode.ENCODE:
            self.waiting_pool: WaitingPoolType = {}
            self.transfer_event = threading.Event()
            self.start_encode_thread()
        elif self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
            self.start_transformer_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def register_buffer_to_engine(self):
        for data_ptr, data_len in zip(
            self.data_args.data_ptrs, self.data_args.data_lens
        ):
            self.engine.register(data_ptr, data_len)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def send_data(
        self,
        mooncake_session_id: str,
        encode_data_ptrs: List[int],
        transformer_ptrs: list[int],
    ):
        tensor_num = int(len(self.data_args.data_ptrs))
        for tensor_id in range(tensor_num):
            encode_addr = encode_data_ptrs[tensor_id]
            item_len = self.data_args.data_item_lens[tensor_id]
            transformer_addr = transformer_ptrs[tensor_id]

            # TODO: mooncake transfer engine can do async transfer. Do async later
            status = self.engine.transfer_sync(
                mooncake_session_id,
                encode_addr,
                transformer_addr,
                item_len,
            )
            if status != 0:
                return status
        return 0

    def sync_status_to_transformer_endpoint(self, remote: str, room: int):
        if ":" in remote:
            remote = remote.split(":")[0]
        self._connect(
            "tcp://"
            + remote
            + ":"
            + str(DATARECEIVER_POLLING_PORT + self.data_args.receiver_engine_rank)
        ).send_multipart(
            [
                str(room).encode("ascii"),
                str(self.request_status[room]).encode("ascii"),
            ]
        )

    def start_encode_thread(self):
        sender_rank_port = DATASENDER_POLLING_PORT + self.data_args.sender_engine_rank
        logger.info("Encoder sender_rank_port=%s", sender_rank_port)
        self.server_socket.bind("tcp://*:" + str(sender_rank_port))

        def encode_thread():
            while True:
                (
                    endpoint,
                    mooncake_session_id,
                    bootstrap_room,
                    transformer_ptrs,
                ) = self.server_socket.recv_multipart()
                if bootstrap_room.decode("ascii") == "None":
                    continue
                endpoint = endpoint.decode("ascii")
                mooncake_session_id = mooncake_session_id.decode("ascii")
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                transformer_ptrs = list(
                    struct.unpack(f"{len(transformer_ptrs)//8}Q", transformer_ptrs)
                )
                logger.info(
                    "Encoder received ZMQ: endpoint=%s session_id=%s room=%s transformer_ptrs=%s",
                    endpoint,
                    mooncake_session_id,
                    bootstrap_room,
                    transformer_ptrs,
                )
                self.waiting_pool[bootstrap_room] = (
                    endpoint,
                    mooncake_session_id,
                    transformer_ptrs,
                )
                self.transfer_event.set()

        threading.Thread(target=encode_thread).start()

        def transfer_thread():
            while True:
                self.transfer_event.wait()
                self.transfer_event.clear()
                bootstrap_room_ready = self.request_pool.keys()
                bootstrap_room_request = self.waiting_pool.keys()
                for room in list(bootstrap_room_request):
                    if room not in list(bootstrap_room_ready):
                        continue
                    status = DataPoll.Transferring
                    self.request_status[room] = status
                    (
                        endpoint,
                        mooncake_session_id,
                        transformer_ptrs,
                    ) = self.waiting_pool.pop(room)
                    self.sync_status_to_transformer_endpoint(endpoint, room)
                    encode_data_ptrs = self.request_pool.pop(room)
                    ret = self.send_data(
                        mooncake_session_id,
                        encode_data_ptrs,
                        transformer_ptrs,
                    )
                    if ret != 0:
                        status = DataPoll.Failed
                        self.sync_status_to_transformer_endpoint(endpoint, room)
                        continue
                    status = DataPoll.Success
                    self.request_status[room] = status
                    self.sync_status_to_transformer_endpoint(endpoint, room)

        threading.Thread(target=transfer_thread).start()

    def start_transformer_thread(self):
        receiver_rank_port = DATARECEIVER_POLLING_PORT + self.data_args.receiver_engine_rank
        self.server_socket.bind("tcp://*:" + str(receiver_rank_port))

        def transformer_thread():
            while True:
                (bootstrap_room, status) = self.server_socket.recv_multipart()
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.request_status[bootstrap_room] = status

        threading.Thread(target=transformer_thread).start()

    def enqueue_request(
        self,
        bootstrap_room: int,
        data_ptrs: List[int],
    ):
        self.request_pool[bootstrap_room] = data_ptrs
        self.request_status[bootstrap_room] = DataPoll.WaitingForInput
        if self.disaggregation_mode == DisaggregationMode.ENCODE:
            self.transfer_event.set()

    def check_status(self, bootstrap_room: int):
        if (
            self.disaggregation_mode == DisaggregationMode.TRANSFORMER
            and self.request_status[bootstrap_room] == DataPoll.Success
        ):
            if bootstrap_room in self.request_pool:
                self.request_pool.pop(bootstrap_room)

        return self.request_status[bootstrap_room]

    def set_status(self, bootstrap_room: int, status: DataPoll):
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

    def init(self, num_data_indices: int):
        self.num_data_indices = num_data_indices

    def send(self, data_ptrs: List[int]):
        self.data_mgr.enqueue_request(self.bootstrap_room, data_ptrs)

    def poll(self) -> DataPoll:
        return self.data_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake DataSender Exception")


class DataReceiver:

    def __init__(
        self, mgr: DataManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.data_mgr = mgr
        self.encode_server_url = (
            bootstrap_addr.split(":")[0]
            + ":"
            + str(DATASENDER_POLLING_PORT + self.data_mgr.data_args.sender_engine_rank)
        )
        logger.info("DataReceiver encode_server_url=%s", self.encode_server_url)
        self.transformer_ip = self.data_mgr.get_localhost()
        self.session_id = self.data_mgr.get_session_id()
        self.data_mgr.set_status(bootstrap_room, DataPoll.WaitingForInput)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def init(self):
        packed_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.data_mgr.data_args.data_ptrs
        )
        self.data_mgr.enqueue_request(self.bootstrap_room, packed_data_ptrs)
        self._connect("tcp://" + self.encode_server_url).send_multipart(
            [
                self.transformer_ip.encode("ascii"),
                self.session_id.encode("ascii"),
                str(self.bootstrap_room).encode("ascii"),
                packed_data_ptrs,
            ]
        )

    def poll(self) -> DataPoll:
        return self.data_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake DataReceiver Exception")

