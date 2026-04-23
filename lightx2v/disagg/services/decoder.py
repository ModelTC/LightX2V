import hashlib
import json
import math
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import torch

from lightx2v.disagg.conn import MONITOR_POLLING_PORT, DataArgs, DataManager, DataReceiver, DisaggregationMode, DisaggregationPhase, ReqManager
from lightx2v.disagg.monitor import Reporter
from lightx2v.disagg.protocol import AllocationRequest, MemoryHandle, RemoteBuffer
from lightx2v.disagg.rdma_buffer import RDMABuffer, RDMABufferDescriptor
from lightx2v.disagg.rdma_client import RDMAClient
from lightx2v.disagg.services.base import BaseService
from lightx2v.disagg.services.data_mgr_sidecar import DataMgrSidecar
from lightx2v.disagg.utils import estimate_transformer_buffer_sizes, load_wan_vae_decoder
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import save_to_video, seed_all, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE


class DecoderService(BaseService):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.encoder_engine_rank = int(self.config.get("encoder_engine_rank", 0))
        self.transformer_engine_rank = int(self.config.get("transformer_engine_rank", 1))
        self.decoder_engine_rank = int(self.config.get("decoder_engine_rank", 2))
        self._phase2_rdma_client: Optional[RDMAClient] = None
        self._phase2_rdma_buffer: Optional[RDMABuffer] = None
        data_bootstrap_addr = str(self.config.get("data_bootstrap_addr", "127.0.0.1"))
        monitor_bind_host = str(self.config.get("local_hostname", data_bootstrap_addr))
        shared_slots = int(self.config.get("rdma_buffer_slots", "128"))
        shared_slot_size = int(self.config.get("rdma_buffer_slot_size", "4096"))
        self._phase2_server_ip = str(self.config.get("rdma_phase2_host", data_bootstrap_addr))
        self._phase2_handshake_port = int(self.config.get("rdma_phase2_handshake_port", "5568"))
        self._phase2_slots = shared_slots
        self._phase2_slot_size = shared_slot_size
        self._last_phase2_connect_retry_ts = 0.0
        self.vae_decoder = None
        self._rdma_buffers: Dict[int, List[torch.Tensor]] = {}
        self.data_mgr = DataManager(
            DisaggregationPhase.PHASE2,
            DisaggregationMode.DECODE,
        )
        self.data_receiver: Dict[int, DataReceiver] = {}
        self.req_mgr = ReqManager()
        self.reporter = Reporter(
            service_type="decoder",
            gpu_id=self.decoder_engine_rank,
            bind_address=f"tcp://{monitor_bind_host}:{MONITOR_POLLING_PORT + self.decoder_engine_rank}",
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
            name="decoder-reporter",
            daemon=True,
        )
        self._reporter_thread.start()
        self._data_mgr_sidecar = DataMgrSidecar()
        self.sync_comm = str(os.getenv("SYNC_COMM", "")).strip().lower() not in ("", "0", "false", "no", "off")
        self.load_models()

    def _get_queue_metrics(self) -> dict[str, Any]:
        with self._queue_metrics_lock:
            queue_sizes = dict(self._queue_metrics.get("queue_sizes", {}))
            return {
                "queue_sizes": queue_sizes,
                "queue_total_pending": int(self._queue_metrics.get("queue_total_pending", 0)),
                "all_queues_empty": bool(self._queue_metrics.get("all_queues_empty", True)),
            }

    def _update_queue_metrics(self, queue_sizes: dict[str, int], transfer_sizes: Optional[dict[str, int]] = None):
        merged_sizes = {k: int(v) for k, v in queue_sizes.items()}
        if transfer_sizes is not None:
            for key, value in transfer_sizes.items():
                merged_sizes[f"transfer_{key}"] = int(value)
        total_pending = sum(max(v, 0) for v in merged_sizes.values())
        with self._queue_metrics_lock:
            self._queue_metrics = {
                "queue_sizes": merged_sizes,
                "queue_total_pending": total_pending,
                "all_queues_empty": total_pending == 0,
            }

    def _ensure_phase2_request_buffer(self) -> bool:
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

    def init(self, config):
        self._sync_runtime_config(config)
        self.encoder_engine_rank = int(self.config.get("encoder_engine_rank", self.encoder_engine_rank))
        self.transformer_engine_rank = int(self.config.get("transformer_engine_rank", self.transformer_engine_rank))
        self.decoder_engine_rank = int(self.config.get("decoder_engine_rank", self.decoder_engine_rank))
        shared_slots = int(self.config.get("rdma_buffer_slots", self._phase2_slots))
        shared_slot_size = int(self.config.get("rdma_buffer_slot_size", 4096))
        self._phase2_server_ip = str(self.config.get("rdma_phase2_host", self._phase2_server_ip))
        self._phase2_handshake_port = int(self.config.get("rdma_phase2_handshake_port", self._phase2_handshake_port))
        self._phase2_slots = shared_slots
        self._phase2_slot_size = shared_slot_size

        if "seed" in self.config:
            seed_all(self.config["seed"])

        data_bootstrap_addr = self.config.get("data_bootstrap_addr", "127.0.0.1")
        data_bootstrap_room = self.config.get("data_bootstrap_room", 0)

        if data_bootstrap_addr is None or data_bootstrap_room is None:
            return

        try:
            self._ensure_phase2_request_buffer()
        except Exception:
            self.logger.exception("Failed to connect phase2 RDMA buffer, will retry")

        buffer_sizes = estimate_transformer_buffer_sizes(self.config)
        request = AllocationRequest(
            bootstrap_room=data_bootstrap_room,
            buffer_sizes=buffer_sizes,
        )
        handle = self.alloc_memory(request)
        data_ptrs = [buf.addr for buf in handle.buffers]
        data_lens = [buf.nbytes for buf in handle.buffers]
        data_args = DataArgs(
            sender_engine_rank=self.transformer_engine_rank,
            receiver_engine_rank=self.decoder_engine_rank,
            data_ptrs=data_ptrs,
            data_lens=data_lens,
            data_item_lens=data_lens,
            ib_device=None,
        )
        self.data_mgr.init(data_args, data_bootstrap_room)
        phase2_bootstrap_addr = str(self.config.get("transformer_node_address", data_bootstrap_addr))
        self.data_receiver[data_bootstrap_room] = DataReceiver(self.data_mgr, phase2_bootstrap_addr, data_bootstrap_room)
        self.data_receiver[data_bootstrap_room].init()

    def load_models(self):
        self.logger.info("Loading Decoder Models...")
        self.vae_decoder = load_wan_vae_decoder(self.config)
        self.logger.info("Decoder Models loaded successfully.")

    def alloc_memory(self, request: AllocationRequest) -> MemoryHandle:
        buffer_sizes = request.buffer_sizes
        room = request.bootstrap_room

        self._rdma_buffers[room] = []
        buffers: List[RemoteBuffer] = []
        for nbytes in buffer_sizes:
            if nbytes <= 0:
                continue
            buf = torch.empty((nbytes,), dtype=torch.uint8)
            ptr = buf.data_ptr()
            self._rdma_buffers[room].append(buf)
            buffers.append(RemoteBuffer(addr=ptr, nbytes=nbytes))

        return MemoryHandle(buffers=buffers)

    def process(self, config):
        self.logger.info("Starting processing in DecoderService...")
        room = config.get("data_bootstrap_room", 0)
        decoder_metrics = config.setdefault("request_metrics", {}).setdefault("stages", {}).setdefault("decoder", {})
        decoder_metrics["compute_start_ts"] = time.time()
        strict_meta_hash_check = str(os.getenv("LIGHTX2V_STRICT_META_HASH", "0")).strip().lower() in {"1", "true", "yes", "on"}
        room_buffers = self._rdma_buffers.get(room)
        receiver = self.data_receiver.get(room)

        if receiver is None:
            raise RuntimeError(f"DataReceiver is not initialized in DecoderService for room={room}.")
        if room_buffers is None:
            raise RuntimeError(f"No RDMA buffer available in DecoderService for room={room}.")

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

        if len(room_buffers) < 2:
            raise RuntimeError("Phase2 RDMA buffers require [latents, meta] entries.")

        meta_buf = room_buffers[1]

        def _read_phase2_meta() -> tuple[dict, str]:
            meta_bytes = _buffer_view(meta_buf, torch.uint8, (meta_buf.numel(),)).detach().contiguous().cpu().numpy().tobytes()
            meta_str = meta_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="ignore") if meta_bytes else ""
            if not meta_str:
                raise ValueError("missing latents metadata from transformer")
            parsed = json.loads(meta_str)
            if not isinstance(parsed, dict):
                raise ValueError(f"phase2 metadata type mismatch: {type(parsed)}")
            return parsed, meta_str

        def _infer_latents_shape_from_config() -> tuple[int, int, int, int]:
            z_dim = int(config.get("vae_z_dim", 16))
            vae_stride = config.get("vae_stride", (4, 8, 8))
            stride_t = int(vae_stride[0])
            stride_h = int(vae_stride[1])
            stride_w = int(vae_stride[2])
            target_video_length = int(config.get("target_video_length", 81))
            target_height = int(config.get("target_height", 480))
            target_width = int(config.get("target_width", 832))

            t_prime = 1 + (target_video_length - 1) // stride_t
            h_prime = int(math.ceil(target_height / stride_h))
            w_prime = int(math.ceil(target_width / stride_w))
            return (z_dim, t_prime, h_prime, w_prime)

        meta = None
        meta_str = ""
        for attempt in range(3):
            try:
                meta, meta_str = _read_phase2_meta()
                break
            except Exception as exc:
                if attempt < 2:
                    # Guard against rare stale/partial metadata visibility.
                    time.sleep(0.02)
                    continue
                self.logger.warning(
                    "Invalid phase2 metadata for room=%s, fallback to config-derived shape. err=%s raw_prefix=%r",
                    room,
                    exc,
                    meta_str[:128],
                )
                meta = {
                    "latents_shape": list(_infer_latents_shape_from_config()),
                    "latents_dtype": str(GET_DTYPE()),
                    "latents_hash": None,
                }

        latents_shape_val = meta.get("latents_shape")
        if not isinstance(latents_shape_val, list) or len(latents_shape_val) != 4:
            latents_shape_val = list(_infer_latents_shape_from_config())
            self.logger.warning("phase2 metadata missing/invalid latents_shape for room=%s, using fallback shape=%s", room, latents_shape_val)
        latent_shape = tuple(int(value) for value in latents_shape_val)

        dtype_map = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
        }
        latents_dtype = dtype_map.get(meta.get("latents_dtype"), GET_DTYPE())

        latents = _buffer_view(room_buffers[0], latents_dtype, latent_shape)
        if list(latents.shape) != meta.get("latents_shape"):
            raise ValueError("latents shape mismatch between transformer and decoder")
        if meta.get("latents_hash") is not None and _sha256_tensor(latents) != meta.get("latents_hash"):
            msg = "latents hash mismatch between transformer and decoder"
            if strict_meta_hash_check:
                raise ValueError(msg)
            self.logger.warning("%s for room=%s, continue with non-strict mode", msg, room)
        latents = latents.to(torch.device(AI_DEVICE)).contiguous()

        if self.vae_decoder is None:
            raise RuntimeError("VAE decoder is not loaded.")

        self.logger.info("Decoding latents in DecoderService...")
        gen_video = self.vae_decoder.decode(latents.to(GET_DTYPE()))
        gen_video_final = wan_vae_to_comfy(gen_video)
        decoder_metrics["compute_end_ts"] = time.time()

        save_path = config.get("save_path")
        if save_path is None:
            raise ValueError("save_path is required in config.")

        self.logger.info(f"Saving video to {save_path}...")
        save_to_video(gen_video_final, save_path, fps=config.get("fps", 16), method="ffmpeg")
        decoder_metrics["output_enqueued_ts"] = time.time()
        self.logger.info("Done!")

        return save_path

    def release_memory(self, room: int):
        if room in self._rdma_buffers:
            self._rdma_buffers.pop(room, None)
        torch.cuda.empty_cache()

    def remove(self, room: int):
        self.release_memory(room)

        self.data_receiver.pop(room, None)
        self._data_mgr_sidecar.unwatch_input(room)

        if self.data_mgr is None:
            return

        self.data_mgr.remove(room)

    def release(self):
        for room in list(self._rdma_buffers.keys()):
            self.remove(room)
        self.reporter.stop()
        if self._reporter_thread is not None and self._reporter_thread.is_alive():
            self._reporter_thread.join(timeout=1.0)
        self._reporter_thread = None
        if self.data_mgr is not None:
            self.data_mgr.release()
        self.data_receiver.clear()
        self.vae_decoder = None

    def run(self, stop_event=None):
        req_queue = deque()
        waiting_queue: Dict[int, dict] = {}
        exec_queue = deque()

        while True:
            transfer_sizes = self.data_mgr.get_backlog_counts() if self.data_mgr is not None else {"request_pool": 0, "waiting_pool": 0}
            sidecar_sizes = self._data_mgr_sidecar.get_pending_counts()
            self._update_queue_metrics(
                {
                    "req_queue": len(req_queue),
                    "waiting_queue": len(waiting_queue),
                    "exec_queue": len(exec_queue),
                },
                {
                    "request_pool": int(transfer_sizes.get("request_pool", 0)),
                    "waiting_pool": int(transfer_sizes.get("waiting_pool", 0)),
                    "sidecar_input_watch": int(sidecar_sizes.get("input_watch", 0)),
                },
            )

            if self._phase2_rdma_buffer is None:
                try:
                    self._ensure_phase2_request_buffer()
                except Exception:
                    self.logger.exception("Failed to connect phase2 request RDMA buffer, will retry")

            if self._phase2_rdma_buffer is not None:
                packet = self._phase2_rdma_buffer.consume()
                if packet is not None:
                    if isinstance(packet, dict) and "request_config" in packet:
                        config = dict(packet.get("request_config") or {})
                        config["transformer_node_address"] = packet.get("transformer_node_address", "127.0.0.1")
                    else:
                        config = packet
                    if not isinstance(config, dict) or "data_bootstrap_room" not in config:
                        self.logger.warning("Ignored incomplete phase2 packet from RDMA buffer: %s", packet)
                        continue
                    decoder_metrics = config.setdefault("request_metrics", {}).setdefault("stages", {}).setdefault("decoder", {})
                    decoder_metrics["request_received_ts"] = time.time()
                    self.logger.info("Received request config from RDMA buffer: %s", {k: v for k, v in config.items()})
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
                if config is None:
                    continue
                self.logger.info("Latents received successfully in DecoderService for room=%s.", room)
                exec_queue.append((room, config))

            for room in failed_rooms:
                waiting_queue.pop(room, None)
                self.logger.error("DataReceiver transfer failed for room=%s", room)
                self.remove(room)

            if exec_queue:
                room, config = exec_queue.popleft()
                try:
                    save_path = self.process(config)
                    callback_host = str(config.get("controller_result_host", "127.0.0.1"))
                    callback_port = int(config.get("controller_result_port")) if config.get("controller_result_port") is not None else None
                    if callback_port is not None:
                        self.req_mgr.send(
                            callback_host,
                            callback_port,
                            {
                                "ok": True,
                                "data_bootstrap_room": int(room),
                                "save_path": save_path,
                                "request_metrics": config.get("request_metrics"),
                            },
                        )
                except Exception:
                    self.logger.exception("Failed to process request for room=%s", room)
                    callback_host = str(config.get("controller_result_host", "127.0.0.1"))
                    callback_port = int(config.get("controller_result_port")) if config.get("controller_result_port") is not None else None
                    if callback_port is not None:
                        self.req_mgr.send(
                            callback_host,
                            callback_port,
                            {
                                "ok": False,
                                "data_bootstrap_room": int(room),
                                "save_path": None,
                                "error": "decoder process failed",
                                "request_metrics": config.get("request_metrics"),
                            },
                        )
                finally:
                    self.remove(room)

            if stop_event is not None and stop_event.is_set() and not req_queue and not waiting_queue and not exec_queue:
                self.logger.info("DecoderService received stop event, exiting request loop.")
                break

            if not req_queue and not exec_queue:
                time.sleep(0.01)

        self.release()
