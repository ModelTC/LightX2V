import hashlib
import json
import os
import threading
import time
from collections import deque
from multiprocessing import resource_tracker, shared_memory
from typing import Any, List, Optional

import numpy as np
import torch
import zmq
from urllib.error import URLError
from urllib.request import Request, urlopen

from lightx2v.disagg.conn import MONITOR_POLLING_PORT, REQUEST_POLLING_PORT, DataArgs, DataManager, DataPoll, DataReceiver, DataSender, DisaggregationMode, DisaggregationPhase, ReqManager
from lightx2v.disagg.monitor import Reporter
from lightx2v.disagg.protocol import AllocationRequest, MemoryHandle, RemoteBuffer
from lightx2v.disagg.rdma_buffer import RDMABuffer, RDMABufferDescriptor
from lightx2v.disagg.rdma_client import RDMAClient
from lightx2v.disagg.services.base import BaseService
from lightx2v.disagg.services.data_mgr_sidecar import DataMgrSidecar
from lightx2v.disagg.utils import (
    estimate_encoder_buffer_sizes,
    estimate_transformer_buffer_sizes,
    load_wan_transformer,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import seed_all
from lightx2v_platform.base.global_var import AI_DEVICE

_SHM_TRACKING_PATCHED = False


def _disable_shared_memory_tracking_for_process():
    global _SHM_TRACKING_PATCHED
    if _SHM_TRACKING_PATCHED:
        return

    original_register = resource_tracker.register
    original_unregister = resource_tracker.unregister

    def _register(name, rtype):
        if rtype == "shared_memory":
            return
        return original_register(name, rtype)

    def _unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return original_unregister(name, rtype)

    resource_tracker.register = _register
    resource_tracker.unregister = _unregister
    _SHM_TRACKING_PATCHED = True


class TransformerService(BaseService):
    def __init__(self, config: dict):
        super().__init__()
        _disable_shared_memory_tracking_for_process()
        self.config = config
        self.encoder_engine_rank = int(self.config.get("encoder_engine_rank", 0))
        self.transformer_engine_rank = int(self.config.get("transformer_engine_rank", 1))
        self.decoder_engine_rank = int(self.config.get("decoder_engine_rank", 2))
        self._phase1_rdma_client: Optional[RDMAClient] = None
        self._phase1_rdma_buffer: Optional[RDMABuffer] = None
        self._phase2_rdma_client: Optional[RDMAClient] = None
        self._phase2_rdma_buffer: Optional[RDMABuffer] = None
        self._centralized_request_mgr = ReqManager()
        self._centralized_request_port = REQUEST_POLLING_PORT + self.transformer_engine_rank
        data_bootstrap_addr = str(self.config.get("data_bootstrap_addr", "127.0.0.1"))
        monitor_bind_host = str(self.config.get("local_hostname", data_bootstrap_addr))
        shared_slots = int(self.config.get("rdma_buffer_slots", "128"))
        shared_slot_size = int(self.config.get("rdma_buffer_slot_size", "4096"))
        self._centralized_request_mode = str(os.getenv("IS_CENTRALIZED", "0")).strip().lower() in {"1", "true", "yes", "on"}
        self._phase1_server_ip = str(self.config.get("rdma_phase1_host", data_bootstrap_addr))
        self._phase1_handshake_port = int(self.config.get("rdma_phase1_handshake_port", "5567"))
        self._phase1_slots = shared_slots
        self._phase1_slot_size = shared_slot_size
        self._phase2_server_ip = str(self.config.get("rdma_phase2_host", data_bootstrap_addr))
        self._phase2_handshake_port = int(self.config.get("rdma_phase2_handshake_port", "5568"))
        self._phase2_slots = shared_slots
        self._phase2_slot_size = shared_slot_size
        self._last_phase1_connect_retry_ts = 0.0
        self._last_phase2_connect_retry_ts = 0.0
        self.transformer = None
        self.scheduler = None
        self.rdma_buffer1: dict[int, List[torch.Tensor]] = {}
        self.rdma_buffer2: dict[int, List[torch.Tensor]] = {}
        self.data_mgr1 = DataManager(DisaggregationPhase.PHASE1, DisaggregationMode.TRANSFORMER)
        self.data_mgr2 = DataManager(DisaggregationPhase.PHASE2, DisaggregationMode.TRANSFORMER)
        self.data_receiver: dict[int, DataReceiver] = {}
        self.data_sender: dict[int, Optional[DataSender]] = {}
        self._phase2_remote_rooms: set[int] = set()
        self._phase2_remote_shared_memory: dict[int, list[shared_memory.SharedMemory]] = {}
        self.reporter = Reporter(
            service_type="transformer",
            gpu_id=self.transformer_engine_rank,
            bind_address=f"tcp://{monitor_bind_host}:{MONITOR_POLLING_PORT + self.transformer_engine_rank}",
        )
        self._queue_metrics_lock = threading.Lock()
        self._queue_metrics: dict[str, Any] = {
            "queue_sizes": {},
            "queue_total_pending": 0,
            "all_queues_empty": True,
        }
        self.reporter.set_extra_metrics_provider(self._get_queue_metrics)
        self._reporter_thread: Optional[threading.Thread] = threading.Thread(
            target=self.reporter.serve_forever,
            name="transformer-reporter",
            daemon=True,
        )
        self._reporter_thread.start()
        self._data_mgr_sidecar = DataMgrSidecar()
        self.sync_comm = str(os.getenv("SYNC_COMM", "")).strip().lower() not in ("", "0", "false", "no", "off")
        self.load_models()

    def _wait_sender_success(self, room: int, sender: DataSender):
        while True:
            status = sender.poll()
            if status == DataPoll.Success:
                return
            if status == DataPoll.Failed:
                raise RuntimeError(f"DataSender transfer failed for room={room}")
            time.sleep(0.001)

    def _report_stage_metrics_to_controller(self, stage_name: str, config: dict[str, Any]):
        if not self._centralized_request_mode:
            return

        controller_host = str(config.get("controller_result_host", "127.0.0.1"))
        controller_port_raw = config.get("controller_result_port")
        if controller_port_raw is None:
            return

        try:
            controller_port = int(controller_port_raw)
        except (TypeError, ValueError):
            return

        request_metrics = config.get("request_metrics")
        if not isinstance(request_metrics, dict):
            return

        stage_metrics = request_metrics.get("stages", {}).get(stage_name)
        if not isinstance(stage_metrics, dict):
            return

        payload_request_metrics: dict[str, Any] = {
            "request_id": request_metrics.get("request_id", config.get("data_bootstrap_room")),
            "stages": {stage_name: stage_metrics},
        }
        if request_metrics.get("controller_send_ts") is not None:
            payload_request_metrics["controller_send_ts"] = request_metrics.get("controller_send_ts")

        self._centralized_request_mgr.send(
            controller_host,
            controller_port,
            {
                "message_type": "stage_metrics",
                "stage_name": stage_name,
                "data_bootstrap_room": int(config.get("data_bootstrap_room", 0)),
                "request_metrics": payload_request_metrics,
            },
        )
        self.logger.info(
            "Reported %s stage metrics to controller: room=%s target=%s:%s",
            stage_name,
            config.get("data_bootstrap_room"),
            controller_host,
            controller_port,
        )

    def _wait_for_controller_ok(self, stage_name: str, config: dict[str, Any]):
        if not self._centralized_request_mode:
            return

        controller_host = str(config.get("controller_control_host", config.get("controller_result_host", "127.0.0.1")))
        controller_port_raw = config.get("controller_control_port")
        if controller_port_raw is None:
            return

        try:
            controller_port = int(controller_port_raw)
        except (TypeError, ValueError):
            return

        request_body = json.dumps(
            {
                "control": "OK",
                "stage_name": stage_name,
                "data_bootstrap_room": int(config.get("data_bootstrap_room", 0)),
            }
        ).encode("utf-8")
        request = Request(
            f"http://{controller_host}:{controller_port}/ok",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                reply = json.loads(response.read().decode("utf-8"))
            if not isinstance(reply, dict) or not reply.get("ok", False):
                raise RuntimeError(f"unexpected controller OK reply: {reply}")
        except URLError:
            self.logger.exception("Failed to wait for controller OK reply for %s room=%s", stage_name, config.get("data_bootstrap_room"))
            return
        except Exception:
            self.logger.exception("Failed to wait for controller OK reply for %s room=%s", stage_name, config.get("data_bootstrap_room"))
            return

    def _attach_remote_shared_memory(self, shm_name: str) -> shared_memory.SharedMemory:
        # Python 3.12 supports `track=False`, which avoids duplicate cleanup from non-owner processes.
        try:
            return shared_memory.SharedMemory(name=shm_name, create=False, track=False)
        except TypeError:
            return shared_memory.SharedMemory(name=shm_name, create=False)

    def _get_queue_metrics(self) -> dict[str, Any]:
        with self._queue_metrics_lock:
            queue_sizes = dict(self._queue_metrics.get("queue_sizes", {}))
            return {
                "queue_sizes": queue_sizes,
                "queue_total_pending": int(self._queue_metrics.get("queue_total_pending", 0)),
                "all_queues_empty": bool(self._queue_metrics.get("all_queues_empty", True)),
            }

    def _update_queue_metrics(
        self,
        queue_sizes: dict[str, int],
        phase1_transfer_sizes: Optional[dict[str, int]] = None,
        phase2_transfer_sizes: Optional[dict[str, int]] = None,
    ):
        merged_sizes = {k: int(v) for k, v in queue_sizes.items()}
        if phase1_transfer_sizes is not None:
            for key, value in phase1_transfer_sizes.items():
                merged_sizes[f"phase1_{key}"] = int(value)
        if phase2_transfer_sizes is not None:
            for key, value in phase2_transfer_sizes.items():
                merged_sizes[f"phase2_{key}"] = int(value)
        total_pending = sum(max(v, 0) for v in merged_sizes.values())
        with self._queue_metrics_lock:
            self._queue_metrics = {
                "queue_sizes": merged_sizes,
                "queue_total_pending": total_pending,
                "all_queues_empty": total_pending == 0,
            }

    def _ensure_phase1_request_buffer(self) -> bool:
        if self._phase1_rdma_buffer is not None:
            return True
        now = time.time()
        if now - self._last_phase1_connect_retry_ts < 1.0:
            return False
        self._last_phase1_connect_retry_ts = now

        if self._phase1_rdma_client is None:
            self._phase1_rdma_client = RDMAClient(local_buffer_size=self._phase1_slot_size)
        self._phase1_rdma_client.connect_to_server(self._phase1_server_ip, self._phase1_handshake_port)
        remote_info = self._phase1_rdma_client.remote_info
        base_addr = int(remote_info["addr"])
        self._phase1_rdma_buffer = RDMABuffer(
            role="client",
            rdma_client=self._phase1_rdma_client,
            remote=RDMABufferDescriptor(
                slot_addr=base_addr + 16,
                slot_bytes=self._phase1_slots * self._phase1_slot_size,
                slot_size=self._phase1_slot_size,
                buffer_size=self._phase1_slots,
                head_addr=base_addr,
                tail_addr=base_addr + 8,
                rkey=int(remote_info.get("rkey", 0)),
            ),
        )
        return True

    def _reconnect_phase1_request_buffer(self):
        self._phase1_rdma_buffer = None
        self._last_phase1_connect_retry_ts = 0.0

        if self._phase1_rdma_client is not None:
            sock = getattr(self._phase1_rdma_client, "sock", None)
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
                self._phase1_rdma_client.sock = None

        self._ensure_phase1_request_buffer()

    def _ensure_phase2_meta_buffer(self) -> bool:
        if self._phase2_rdma_buffer is not None:
            return True
        now = time.time()
        if now - self._last_phase2_connect_retry_ts < 1.0:
            return False
        self._last_phase2_connect_retry_ts = now

        if self._phase2_rdma_client is None:
            self._phase2_rdma_client = RDMAClient(local_buffer_size=self._phase2_slot_size)
        self._phase2_rdma_client.connect_to_server(self._phase2_server_ip, self._phase2_handshake_port)
        remote_info = self._phase2_rdma_client.remote_info
        base_addr = int(remote_info["addr"])
        self._phase2_rdma_buffer = RDMABuffer(
            role="client",
            rdma_client=self._phase2_rdma_client,
            remote=RDMABufferDescriptor(
                slot_addr=base_addr + 16,
                slot_bytes=self._phase2_slots * self._phase2_slot_size,
                slot_size=self._phase2_slot_size,
                buffer_size=self._phase2_slots,
                head_addr=base_addr,
                tail_addr=base_addr + 8,
                rkey=int(remote_info.get("rkey", 0)),
            ),
        )
        return True

    def _reconnect_phase2_meta_buffer(self):
        self._phase2_rdma_buffer = None
        self._last_phase2_connect_retry_ts = 0.0

        if self._phase2_rdma_client is not None:
            sock = getattr(self._phase2_rdma_client, "sock", None)
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
                self._phase2_rdma_client.sock = None

        self._ensure_phase2_meta_buffer()

    def _produce_phase2_request_with_retry(self, room: int, payload: dict[str, Any]):
        retries = max(1, int(os.getenv("RDMA_PHASE2_PRODUCE_RETRIES", "3")))
        retry_delay_s = max(0.01, float(os.getenv("RDMA_PHASE2_PRODUCE_RETRY_DELAY_S", "0.2")))
        last_exc: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                if self._phase2_rdma_buffer is None:
                    self._ensure_phase2_meta_buffer()
                if self._phase2_rdma_buffer is None:
                    raise RuntimeError("phase2 RDMA buffer is not ready")
                self._phase2_rdma_buffer.produce(payload)
                return
            except Exception as exc:
                last_exc = exc
                self.logger.warning(
                    "Phase2 RDMA produce failed for room=%s attempt=%s/%s: %s",
                    room,
                    attempt,
                    retries,
                    exc,
                )
                if attempt >= retries:
                    break
                try:
                    self._reconnect_phase2_meta_buffer()
                except Exception as reconnect_exc:
                    self.logger.warning(
                        "Phase2 RDMA reconnect failed for room=%s attempt=%s/%s: %s",
                        room,
                        attempt,
                        retries,
                        reconnect_exc,
                    )
                time.sleep(retry_delay_s)

        raise RuntimeError(f"Failed to produce phase2 RDMA request for room={room} after {retries} attempts") from last_exc

    def init(self, config):
        self._sync_runtime_config(config)
        self.encoder_engine_rank = int(self.config.get("encoder_engine_rank", self.encoder_engine_rank))
        self.transformer_engine_rank = int(self.config.get("transformer_engine_rank", self.transformer_engine_rank))
        self.decoder_engine_rank = int(self.config.get("decoder_engine_rank", self.decoder_engine_rank))
        shared_slots = int(self.config.get("rdma_buffer_slots", self._phase1_slots))
        shared_slot_size = int(self.config.get("rdma_buffer_slot_size", 4096))
        self._phase1_server_ip = str(self.config.get("rdma_phase1_host", self._phase1_server_ip))
        self._phase1_handshake_port = int(self.config.get("rdma_phase1_handshake_port", self._phase1_handshake_port))
        self._phase1_slots = shared_slots
        self._phase1_slot_size = shared_slot_size
        self._phase2_server_ip = str(self.config.get("rdma_phase2_host", self._phase2_server_ip))
        self._phase2_handshake_port = int(self.config.get("rdma_phase2_handshake_port", self._phase2_handshake_port))
        self._phase2_slots = shared_slots
        self._phase2_slot_size = shared_slot_size

        if self.scheduler is not None:
            self.scheduler.refresh_from_config(self.config)

        # Set global seed if present in config, though specific process calls might reuse it
        if "seed" in self.config:
            seed_all(self.config["seed"])

        data_bootstrap_addr = self.config.get("data_bootstrap_addr", "127.0.0.1")
        data_bootstrap_room = self.config.get("data_bootstrap_room", 0)

        if data_bootstrap_addr is None or data_bootstrap_room is None:
            return

        if not self._centralized_request_mode:
            phase_deadline = time.time() + 30.0
            while time.time() < phase_deadline:
                try:
                    self._ensure_phase1_request_buffer()
                    self._ensure_phase2_meta_buffer()
                except Exception:
                    self.logger.exception("Failed to connect phase RDMA buffers, will retry")
                if self._phase1_rdma_buffer is not None and self._phase2_rdma_buffer is not None:
                    break
                time.sleep(0.1)

            if self._phase1_rdma_buffer is None or self._phase2_rdma_buffer is None:
                raise RuntimeError("phase RDMA buffers are not ready")

        buffer_sizes = estimate_encoder_buffer_sizes(self.config)
        request = AllocationRequest(
            bootstrap_room=data_bootstrap_room,
            buffer_sizes=buffer_sizes,
        )
        handle = self.alloc_memory(DisaggregationPhase.PHASE1, request)
        data_ptrs = [buf.addr for buf in handle.buffers]
        data_lens = [buf.nbytes for buf in handle.buffers]
        data_args = DataArgs(
            sender_engine_rank=self.encoder_engine_rank,
            receiver_engine_rank=self.transformer_engine_rank,
            data_ptrs=data_ptrs,
            data_lens=data_lens,
            data_item_lens=data_lens,
            ib_device=None,
        )
        self.data_mgr1.init(data_args, data_bootstrap_room)
        phase1_bootstrap_addr = str(self.config.get("encoder_node_address", data_bootstrap_addr))
        self.data_receiver[data_bootstrap_room] = DataReceiver(self.data_mgr1, phase1_bootstrap_addr, data_bootstrap_room)
        self.data_receiver[data_bootstrap_room].init()

        buffer_sizes = [int(v) for v in estimate_transformer_buffer_sizes(self.config)]
        remote_room: dict[str, Any] | None = None
        room_init_retries = max(1, int(os.getenv("DISAGG_TRANSFORMER_REMOTE_OUTPUT_INIT_RETRIES", "3")))
        room_init_retry_sleep_s = max(0.01, float(os.getenv("DISAGG_TRANSFORMER_REMOTE_OUTPUT_INIT_RETRY_SLEEP_S", "0.2")))

        for attempt in range(1, room_init_retries + 1):
            try:
                remote_room = self._data_mgr_sidecar.init_transformer_output_room(
                    room=data_bootstrap_room,
                    sender_engine_rank=self.transformer_engine_rank,
                    receiver_engine_rank=self.decoder_engine_rank,
                    data_lens=buffer_sizes,
                    bootstrap_addr=data_bootstrap_addr,
                )
            except Exception:
                self.logger.exception(
                    "Failed to initialize remote transformer output room=%s attempt=%s/%s",
                    data_bootstrap_room,
                    attempt,
                    room_init_retries,
                )
                remote_room = None

            if isinstance(remote_room, dict):
                break

            if attempt < room_init_retries:
                time.sleep(room_init_retry_sleep_s)

        if not isinstance(remote_room, dict):
            raise RuntimeError(f"remote transformer output room init failed for room={data_bootstrap_room}; sidecar ownership is required to keep transfers alive during service reclaim")

        shm_names_raw = remote_room.get("shm_names")
        data_lens_raw = remote_room.get("data_lens", buffer_sizes)
        if not isinstance(shm_names_raw, list) or not isinstance(data_lens_raw, list) or len(shm_names_raw) != len(data_lens_raw):
            raise RuntimeError(f"invalid remote output room metadata for room={data_bootstrap_room}: {remote_room}")

        shm_handles: list[shared_memory.SharedMemory] = []
        phase2_buffers: list[torch.Tensor] = []
        try:
            for shm_name, nbytes in zip(shm_names_raw, data_lens_raw):
                shm = self._attach_remote_shared_memory(str(shm_name))
                np_view = np.ndarray((int(nbytes),), dtype=np.uint8, buffer=shm.buf)
                tensor = torch.from_numpy(np_view)
                tensor.zero_()
                shm_handles.append(shm)
                phase2_buffers.append(tensor)
        except Exception:
            for shm in shm_handles:
                try:
                    shm.close()
                except Exception:
                    pass
            self._data_mgr_sidecar.remove_transformer_output_room(data_bootstrap_room)
            raise

        self._phase2_remote_rooms.add(int(data_bootstrap_room))
        self._phase2_remote_shared_memory[int(data_bootstrap_room)] = shm_handles
        self.rdma_buffer2[int(data_bootstrap_room)] = phase2_buffers
        self.data_sender[int(data_bootstrap_room)] = None

    def load_models(self):
        self.logger.info("Loading Transformer Models...")

        self.transformer = load_wan_transformer(self.config)

        # Initialize scheduler
        self.scheduler = WanScheduler(self.config)
        self.transformer.set_scheduler(self.scheduler)

        self.logger.info("Transformer Models loaded successfully.")

    def alloc_memory(self, phase: DisaggregationPhase, request: AllocationRequest) -> MemoryHandle:
        """
        Args:
            request: AllocationRequest containing precomputed buffer sizes.

        Returns:
            MemoryHandle with RDMA-registered buffer addresses.
        """
        buffer_sizes = request.buffer_sizes
        room = int(request.bootstrap_room)

        # torch.cuda.set_device(self.receiver_engine_rank)

        if phase == DisaggregationPhase.PHASE1:
            self.rdma_buffer1[room] = []
            target_buffers = self.rdma_buffer1[room]
        elif phase == DisaggregationPhase.PHASE2:
            self.rdma_buffer2[room] = []
            target_buffers = self.rdma_buffer2[room]
        else:
            raise ValueError(f"unsupported disaggregation phase: {phase}")

        buffers: List[RemoteBuffer] = []
        for nbytes in buffer_sizes:
            if nbytes <= 0:
                continue
            buf = torch.empty(
                (nbytes,),
                dtype=torch.uint8,
                # device=torch.device(f"cuda:{self.receiver_engine_rank}"),
            )
            ptr = buf.data_ptr()
            target_buffers.append(buf)
            buffers.append(RemoteBuffer(addr=ptr, nbytes=nbytes))

        return MemoryHandle(buffers=buffers)

    def process(self, config):
        """
        Executes the diffusion process and video decoding.
        """
        self.logger.info("Starting processing in TransformerService...")
        # Re-sync scheduler with the current request to avoid cross-request config bleed.
        if self.scheduler is not None:
            self.scheduler.refresh_from_config(config)
        room = config.get("data_bootstrap_room", 0)
        transformer_metrics = config.setdefault("request_metrics", {}).setdefault("stages", {}).setdefault("transformer", {})
        transformer_metrics["compute_start_ts"] = time.time()

        phase1_buffers = self.rdma_buffer1.get(room)
        phase2_buffers = self.rdma_buffer2.get(room)
        receiver = self.data_receiver.get(room)
        sender = self.data_sender.get(room)
        use_remote_phase2 = room in self._phase2_remote_rooms

        if phase1_buffers is None:
            raise RuntimeError(f"phase1 RDMA buffers are not initialized for room={room}.")
        if phase2_buffers is None:
            raise RuntimeError(f"phase2 RDMA buffers are not initialized for room={room}.")
        if receiver is None:
            raise RuntimeError(f"DataReceiver is not initialized for room={room}.")
        if sender is None and not use_remote_phase2:
            raise RuntimeError(f"DataSender is not initialized for room={room}.")

        def _buffer_view(buf: torch.Tensor, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
            view = torch.empty(0, dtype=dtype, device=buf.device)
            view.set_(buf.untyped_storage(), 0, shape)
            return view

        def _sha256_tensor(tensor: Optional[torch.Tensor]) -> Optional[str]:
            if tensor is None:
                return None
            data_tensor = tensor.detach()
            if data_tensor.dtype == torch.bfloat16:
                data_tensor = data_tensor.to(torch.float32)
            data = data_tensor.contiguous().cpu().numpy().tobytes()
            return hashlib.sha256(data).hexdigest()

        # Reconstruct inputs from rdma_buffer1
        enable_cfg = bool(config.get("enable_cfg", False))
        task = config.get("task", "i2v")
        use_image_encoder = bool(config.get("use_image_encoder", True))

        buffer_index = 0

        context_buf = phase1_buffers[buffer_index]
        buffer_index += 1

        context_null_buf = None
        if enable_cfg:
            context_null_buf = phase1_buffers[buffer_index]
            buffer_index += 1

        clip_buf = None
        vae_buf = None
        if task == "i2v":
            if use_image_encoder:
                clip_buf = phase1_buffers[buffer_index]
                buffer_index += 1

            vae_buf = phase1_buffers[buffer_index]
            buffer_index += 1

        latent_buf = phase1_buffers[buffer_index]
        buffer_index += 1

        meta_buf = phase1_buffers[buffer_index]
        strict_meta_hash_check = str(os.getenv("LIGHTX2V_STRICT_META_HASH", "0")).strip().lower() in {"1", "true", "yes", "on"}

        def _load_phase1_meta(max_retries: int = 20, retry_sleep_s: float = 0.05) -> dict:
            last_error: Optional[Exception] = None
            last_preview = ""

            required_shape_keys = ["context_shape", "latent_shape"]
            if enable_cfg:
                required_shape_keys.append("context_null_shape")
            if task == "i2v":
                required_shape_keys.append("vae_shape")
                if use_image_encoder:
                    required_shape_keys.append("clip_shape")

            for attempt in range(1, max_retries + 1):
                meta_bytes = _buffer_view(meta_buf, torch.uint8, (meta_buf.numel(),)).detach().contiguous().cpu().numpy().tobytes()
                raw_payload = meta_bytes.split(b"\x00", 1)[0] if meta_bytes else b""
                if not raw_payload:
                    last_error = ValueError("missing metadata from encoder")
                    if attempt < max_retries:
                        time.sleep(retry_sleep_s)
                        continue
                    break
                try:
                    meta_str = raw_payload.decode("utf-8")
                except UnicodeDecodeError as err:
                    last_error = err
                    last_preview = raw_payload[:32].hex()
                    if attempt < max_retries:
                        self.logger.warning(
                            "Invalid phase1 metadata UTF-8 for room=%s (attempt %s/%s), retrying...",
                            room,
                            attempt,
                            max_retries,
                        )
                        time.sleep(retry_sleep_s)
                        continue
                    break

                if not meta_str.strip():
                    last_error = ValueError("empty metadata payload from encoder")
                    if attempt < max_retries:
                        time.sleep(retry_sleep_s)
                        continue
                    break

                try:
                    parsed = json.loads(meta_str)
                except json.JSONDecodeError as err:
                    last_error = err
                    last_preview = meta_str[:120]
                    if attempt < max_retries:
                        self.logger.warning(
                            "Invalid phase1 metadata JSON for room=%s (attempt %s/%s), retrying...",
                            room,
                            attempt,
                            max_retries,
                        )
                        time.sleep(retry_sleep_s)
                        continue
                    break

                if not isinstance(parsed, dict):
                    last_error = TypeError(f"phase1 metadata must be a dict, got {type(parsed).__name__}")
                    last_preview = str(parsed)[:120]
                    if attempt < max_retries:
                        time.sleep(retry_sleep_s)
                        continue
                    break

                missing_shape_keys = [key for key in required_shape_keys if not isinstance(parsed.get(key), (list, tuple)) or len(parsed.get(key)) == 0]
                if missing_shape_keys:
                    last_error = ValueError(f"incomplete metadata, missing keys: {missing_shape_keys}")
                    last_preview = str({k: parsed.get(k) for k in required_shape_keys})[:180]
                    if attempt < max_retries:
                        self.logger.warning(
                            "Incomplete phase1 metadata for room=%s (attempt %s/%s), missing=%s, retrying...",
                            room,
                            attempt,
                            max_retries,
                            missing_shape_keys,
                        )
                        time.sleep(retry_sleep_s)
                        continue
                    break

                return parsed

            preview_suffix = f", preview={last_preview}" if last_preview else ""
            raise ValueError(f"failed to load phase1 metadata for room={room}: {last_error}{preview_suffix}")

        meta = _load_phase1_meta()
        meta_shapes = {k: v for k, v in meta.items() if k.endswith("_shape")}
        meta_dtypes = {k: v for k, v in meta.items() if k.endswith("_dtype")}
        self.logger.info("Transformer meta shapes: %s", meta_shapes)
        self.logger.info("Transformer meta dtypes: %s", meta_dtypes)

        def _get_shape(key: str) -> tuple[int, ...]:
            shape = meta.get(key)
            if not shape:
                raise ValueError(f"missing {key} in metadata")
            return tuple(shape)

        context_shape = _get_shape("context_shape")
        context = _buffer_view(context_buf, GET_DTYPE(), context_shape).to(torch.device(AI_DEVICE))

        context_null = None
        if enable_cfg and context_null_buf is not None:
            context_null_shape = _get_shape("context_null_shape")
            context_null = _buffer_view(context_null_buf, GET_DTYPE(), context_null_shape).to(torch.device(AI_DEVICE))

        text_encoder_output = {
            "context": context,
            "context_null": context_null,
        }

        image_encoder_output = {}
        clip_encoder_out = None
        vae_encoder_out = None

        if task == "i2v":
            if use_image_encoder and clip_buf is not None:
                clip_shape = _get_shape("clip_shape")
                clip_encoder_out = _buffer_view(clip_buf, GET_DTYPE(), clip_shape).to(torch.device(AI_DEVICE))

            if vae_buf is not None:
                vae_shape = _get_shape("vae_shape")
                vae_encoder_out = _buffer_view(vae_buf, GET_DTYPE(), vae_shape).to(torch.device(AI_DEVICE))

        latent_shape = _buffer_view(latent_buf, torch.int64, (4,)).tolist()

        if task == "i2v":
            image_encoder_output["clip_encoder_out"] = clip_encoder_out
            image_encoder_output["vae_encoder_out"] = vae_encoder_out
        else:
            image_encoder_output = None

        if meta:
            if list(context.shape) != meta.get("context_shape"):
                raise ValueError("context shape mismatch between encoder and transformer")
            if meta.get("context_hash") is not None and _sha256_tensor(context) != meta.get("context_hash"):
                msg = "context hash mismatch between encoder and transformer"
                if strict_meta_hash_check:
                    raise ValueError(msg)
                self.logger.warning("%s for room=%s, continue with non-strict mode", msg, room)
            if enable_cfg:
                if context_null is not None:
                    if list(context_null.shape) != meta.get("context_null_shape"):
                        raise ValueError("context_null shape mismatch between encoder and transformer")
                if meta.get("context_null_hash") is not None:
                    if _sha256_tensor(context_null) != meta.get("context_null_hash"):
                        msg = "context_null hash mismatch between encoder and transformer"
                        if strict_meta_hash_check:
                            raise ValueError(msg)
                        self.logger.warning("%s for room=%s, continue with non-strict mode", msg, room)
            if task == "i2v":
                if clip_encoder_out is not None:
                    if list(clip_encoder_out.shape) != meta.get("clip_shape"):
                        raise ValueError("clip shape mismatch between encoder and transformer")
                if meta.get("clip_hash") is not None:
                    if _sha256_tensor(clip_encoder_out) != meta.get("clip_hash"):
                        msg = "clip hash mismatch between encoder and transformer"
                        if strict_meta_hash_check:
                            raise ValueError(msg)
                        self.logger.warning("%s for room=%s, continue with non-strict mode", msg, room)
                if vae_encoder_out is not None:
                    if list(vae_encoder_out.shape) != meta.get("vae_shape"):
                        raise ValueError("vae shape mismatch between encoder and transformer")
                if meta.get("vae_hash") is not None:
                    if _sha256_tensor(vae_encoder_out) != meta.get("vae_hash"):
                        msg = "vae hash mismatch between encoder and transformer"
                        if strict_meta_hash_check:
                            raise ValueError(msg)
                        self.logger.warning("%s for room=%s, continue with non-strict mode", msg, room)
            if meta.get("latent_shape") is None or list(latent_shape) != meta.get("latent_shape"):
                raise ValueError("latent_shape mismatch between encoder and transformer")
            if meta.get("latent_hash") is not None:
                latent_tensor = torch.tensor(latent_shape, device=AI_DEVICE, dtype=torch.int64)
                if _sha256_tensor(latent_tensor) != meta.get("latent_hash"):
                    msg = "latent_shape hash mismatch between encoder and transformer"
                    if strict_meta_hash_check:
                        raise ValueError(msg)
                    self.logger.warning("%s for room=%s, continue with non-strict mode", msg, room)

        inputs = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
            "latent_shape": latent_shape,
        }

        seed = config.get("seed")
        if seed is None:
            raise ValueError("seed is required in config.")

        if latent_shape is None:
            raise ValueError("latent_shape is required in inputs.")

        # Scheduler Preparation
        self.logger.info(f"Preparing scheduler with seed {seed}...")
        self.scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)

        # Denoising Loop
        self.logger.info("Starting denoising loop...")
        infer_steps = self.scheduler.infer_steps

        for step_index in range(infer_steps):
            if step_index % 10 == 0:
                self.logger.info(f"Step {step_index + 1}/{infer_steps}")
            self.scheduler.step_pre(step_index=step_index)
            self.transformer.infer(inputs)
            self.scheduler.step_post()

        latents = self.scheduler.latents
        transformer_metrics["compute_end_ts"] = time.time()

        # Send latents to DecoderService
        if len(phase2_buffers) < 2:
            raise RuntimeError("phase2 RDMA buffers require [latents, meta] entries.")

        latents_to_send = latents.detach().to(GET_DTYPE()).contiguous()
        latents_nbytes = latents_to_send.numel() * latents_to_send.element_size()
        latents_buf = phase2_buffers[0]
        if latents_nbytes > latents_buf.numel():
            raise ValueError(f"latents buffer too small: need={latents_nbytes}, capacity={latents_buf.numel()}")

        latents_buf.zero_()
        latents_view = _buffer_view(latents_buf, latents_to_send.dtype, tuple(latents_to_send.shape))
        latents_view.copy_(latents_to_send)

        latents_meta = {
            "version": 1,
            "latents_shape": list(latents_to_send.shape),
            "latents_dtype": str(latents_to_send.dtype),
            "latents_hash": _sha256_tensor(latents_to_send),
        }
        meta_bytes = json.dumps(latents_meta, ensure_ascii=True).encode("utf-8")
        meta_buf = phase2_buffers[1]
        meta_view = _buffer_view(meta_buf, torch.uint8, (meta_buf.numel(),))
        if len(meta_bytes) > meta_view.numel():
            raise ValueError("phase2 metadata buffer too small for latents meta payload")
        meta_view.zero_()
        if meta_bytes:
            meta_view[: len(meta_bytes)].copy_(torch.from_numpy(np.frombuffer(meta_bytes, dtype=np.uint8)))

        buffer_ptrs = [buf.data_ptr() for buf in phase2_buffers]
        # Publish phase2 request metadata after compute so downstream can see latest metrics.
        transformer_metrics["output_enqueued_ts"] = time.time()
        phase2_request_config = dict(config)
        phase2_request_config["transformer_engine_rank"] = self.transformer_engine_rank
        transformer_node_address = ""
        transformer_session_id = ""
        if room in self._phase2_remote_rooms:
            identity = self._data_mgr_sidecar.get_transformer_output_identity(room)
            if not isinstance(identity, dict):
                raise RuntimeError(f"remote transformer output identity unavailable for room={room}")
            transformer_node_address = str(identity.get("host", "")).strip()
            transformer_session_id = str(identity.get("session_id", "")).strip()
            if not transformer_node_address or not transformer_session_id:
                raise RuntimeError(f"remote transformer output identity invalid for room={room}: {identity}")
        else:
            transformer_node_address = self.data_mgr2.get_localhost()
            transformer_session_id = self.data_mgr2.get_session_id()

        self._produce_phase2_request_with_retry(
            room,
            {
                "request_config": phase2_request_config,
                "transformer_node_address": transformer_node_address,
                "transformer_session_id": transformer_session_id,
            },
        )
        if use_remote_phase2:
            if not self._data_mgr_sidecar.send_transformer_output_room(room):
                raise RuntimeError(f"Failed to enqueue remote transformer output transfer for room={room}")
            if self.sync_comm:
                while True:
                    status = int(self._data_mgr_sidecar.get_transformer_output_status(room))
                    if status == DataPoll.Success:
                        break
                    if status == DataPoll.Failed:
                        raise RuntimeError(f"DataSender transfer failed for room={room}")
                    time.sleep(0.001)
        else:
            if sender is None:
                raise RuntimeError(f"DataSender is not initialized for room={room}")
            sender.send(buffer_ptrs)
            if self.sync_comm:
                self._wait_sender_success(room, sender)

    def release_memory(self, room: int):
        """
        Releases the RDMA buffers and clears GPU cache.
        """
        if room in self.rdma_buffer1:
            self.rdma_buffer1.pop(room, None)

        if room in self.rdma_buffer2:
            self.rdma_buffer2.pop(room, None)

        shm_handles = self._phase2_remote_shared_memory.pop(room, None)
        if isinstance(shm_handles, list):
            for shm in shm_handles:
                try:
                    shm.close()
                except Exception:
                    pass
        self._phase2_remote_rooms.discard(room)

        torch.cuda.empty_cache()

    def remove(self, room: int):
        use_remote_phase2 = room in self._phase2_remote_rooms
        self.release_memory(room)

        self.data_receiver.pop(room, None)
        self.data_sender.pop(room, None)
        self._data_mgr_sidecar.unwatch_input(room)
        self._data_mgr_sidecar.unwatch_output(room)

        if self.data_mgr1 is not None:
            self.data_mgr1.remove(room)
        if use_remote_phase2:
            self._data_mgr_sidecar.remove_transformer_output_room(room)
        elif self.data_mgr2 is not None:
            self.data_mgr2.remove(room)

    def release(self):
        room_ids = set(self.rdma_buffer1.keys()) | set(self.rdma_buffer2.keys())
        for room in list(room_ids):
            self.remove(room)
        self.reporter.stop()
        if self._reporter_thread is not None and self._reporter_thread.is_alive():
            self._reporter_thread.join(timeout=1.0)
        self._reporter_thread = None
        if self.data_mgr1 is not None:
            self.data_mgr1.release()
        if self.data_mgr2 is not None:
            self.data_mgr2.release()
        self.data_receiver.clear()
        self.data_sender.clear()
        self.transformer = None
        self.scheduler = None

    def run(self, stop_event=None):
        req_queue = deque()
        waiting_queue: dict[int, dict] = {}
        exec_queue = deque()
        complete_queue: set[int] = set()

        while True:
            phase1_transfer_sizes = self.data_mgr1.get_backlog_counts() if self.data_mgr1 is not None else {"request_pool": 0, "waiting_pool": 0}
            phase2_transfer_sizes = self.data_mgr2.get_backlog_counts() if self.data_mgr2 is not None else {"request_pool": 0, "waiting_pool": 0}
            remote_phase2_transfer_sizes = self._data_mgr_sidecar.get_transformer_output_backlog()
            for key in ("request_pool", "waiting_pool", "request_status"):
                phase2_transfer_sizes[key] = int(phase2_transfer_sizes.get(key, 0)) + int(remote_phase2_transfer_sizes.get(key, 0))
            sidecar_sizes = self._data_mgr_sidecar.get_pending_counts()
            self._update_queue_metrics(
                {
                    "req_queue": len(req_queue),
                    "waiting_queue": len(waiting_queue),
                    "exec_queue": len(exec_queue),
                },
                {
                    "request_pool": int(phase1_transfer_sizes.get("request_pool", 0)),
                    "waiting_pool": int(phase1_transfer_sizes.get("waiting_pool", 0)),
                    "sidecar_input_watch": int(sidecar_sizes.get("input_watch", 0)),
                },
                {
                    "complete_queue": len(complete_queue),
                    "request_pool": int(phase2_transfer_sizes.get("request_pool", 0)),
                    "waiting_pool": int(phase2_transfer_sizes.get("waiting_pool", 0)),
                    "sidecar_output_watch": int(sidecar_sizes.get("output_watch", 0)),
                },
            )

            if self._centralized_request_mode:
                config = self._centralized_request_mgr.receive_non_block(self._centralized_request_port)
                if config is not None:
                    if not isinstance(config, dict) or "data_bootstrap_room" not in config:
                        self.logger.warning("Ignored incomplete request packet from ZMQ: %s", config)
                        continue
                    transformer_metrics = config.setdefault("request_metrics", {}).setdefault("stages", {}).setdefault("transformer", {})
                    transformer_metrics["request_received_ts"] = time.time()
                    self.logger.info("Received request config from ZMQ: %s", {k: v for k, v in config.items()})
                    req_queue.append(config)
            else:
                if self._phase1_rdma_buffer is None:
                    try:
                        self._ensure_phase1_request_buffer()
                    except Exception:
                        self.logger.exception("Failed to connect phase1 request RDMA buffer, will retry")

                if self._phase1_rdma_client is not None and self._phase1_rdma_client.has_qp_error():
                    self.logger.warning(
                        "Phase1 request RDMA client entered error state, reconnecting: %s",
                        self._phase1_rdma_client.last_wc_error_message(),
                    )
                    try:
                        self._reconnect_phase1_request_buffer()
                    except Exception:
                        self.logger.exception("Failed to reconnect phase1 request RDMA buffer after QP error")

                if self._phase1_rdma_buffer is not None and len(req_queue) + len(waiting_queue) < 2:
                    packet = self._phase1_rdma_buffer.consume()
                    if packet is not None:
                        if isinstance(packet, dict) and "request_config" in packet:
                            config = dict(packet.get("request_config") or {})
                            config["encoder_node_address"] = packet.get("encoder_node_address", "127.0.0.1")
                        else:
                            config = packet
                        if not isinstance(config, dict) or "data_bootstrap_room" not in config:
                            self.logger.warning("Ignored incomplete phase1 packet from RDMA buffer: %s", packet)
                            continue
                        transformer_metrics = config.setdefault("request_metrics", {}).setdefault("stages", {}).setdefault("transformer", {})
                        transformer_metrics["request_received_ts"] = time.time()
                        self.logger.info("%s Received request config from RDMA buffer: %s", self.transformer_engine_rank, {k: v for k, v in config.items()})
                        req_queue.append(config)

            if req_queue:
                config = req_queue.popleft()
                room = int(config.get("data_bootstrap_room", 0))
                try:
                    self.init(config)
                    waiting_queue[room] = config
                    receiver = self.data_receiver.get(room)
                    if receiver is None:
                        raise RuntimeError(f"DataReceiver is not initialized for room={room}")
                    self._data_mgr_sidecar.watch_input(room, receiver)
                except Exception:
                    self.logger.exception("Failed to initialize request for room=%s", room)
                    self.remove(room)

            ready_rooms = self._data_mgr_sidecar.pop_ready_inputs()
            failed_rooms = self._data_mgr_sidecar.pop_failed_inputs()

            for room in ready_rooms:
                config = waiting_queue.pop(room, None)
                if config is not None:
                    exec_queue.append((room, config))

            for room in failed_rooms:
                waiting_queue.pop(room, None)
                self.logger.error("DataReceiver transfer failed for room=%s", room)
                self.remove(room)

            if exec_queue:
                room, config = exec_queue[0]
                try:
                    self.process(config)
                    if self._centralized_request_mode:
                        self._report_stage_metrics_to_controller("transformer", config)
                        self._wait_for_controller_ok("transformer", config)
                    if self.sync_comm:
                        self.remove(room)
                    else:
                        if room in self._phase2_remote_rooms:
                            complete_queue.add(room)
                        else:
                            sender = self.data_sender.get(room)
                            if sender is None:
                                self.logger.error("DataSender is not initialized for room=%s", room)
                                self.remove(room)
                            else:
                                self._data_mgr_sidecar.watch_output(room, sender)
                                complete_queue.add(room)
                except Exception:
                    self.logger.exception("Failed to process request for room=%s", room)
                    self.remove(room)
                finally:
                    exec_queue.popleft()

            completed_outputs = self._data_mgr_sidecar.pop_completed_outputs()
            for room, status in completed_outputs:
                if status == DataPoll.Failed:
                    self.logger.error("DataSender transfer failed for room=%s", room)
                complete_queue.discard(room)
                self.remove(room)

            if stop_event is not None and stop_event.is_set() and not req_queue and not waiting_queue and not exec_queue and not complete_queue:
                self.logger.info("TransformerService received stop event, exiting request loop.")
                break

            if not req_queue and not exec_queue:
                time.sleep(0.01)

        self.release()
