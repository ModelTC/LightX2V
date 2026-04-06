import hashlib
import json
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from lightx2v.disagg.conn import MONITOR_POLLING_PORT, DataArgs, DataManager, DataPoll, DataSender, DisaggregationMode, DisaggregationPhase
from lightx2v.disagg.monitor import Reporter
from lightx2v.disagg.protocol import AllocationRequest, MemoryHandle, RemoteBuffer
from lightx2v.disagg.rdma_buffer import RDMABuffer, RDMABufferDescriptor
from lightx2v.disagg.rdma_client import RDMAClient
from lightx2v.disagg.services.base import BaseService
from lightx2v.disagg.utils import (
    estimate_encoder_buffer_sizes,
    load_wan_image_encoder,
    load_wan_text_encoder,
    load_wan_vae_encoder,
    read_image_input,
)
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import seed_all
from lightx2v_platform.base.global_var import AI_DEVICE


class EncoderService(BaseService):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.encoder_engine_rank = int(self.config.get("encoder_engine_rank", "0"))
        self.transformer_engine_rank = int(self.config.get("transformer_engine_rank", "1"))
        self.decoder_engine_rank = int(self.config.get("decoder_engine_rank", "2"))
        self._request_rdma_client: Optional[RDMAClient] = None
        self._request_rdma_buffer: Optional[RDMABuffer] = None
        self._phase1_rdma_client: Optional[RDMAClient] = None
        self._phase1_rdma_buffer: Optional[RDMABuffer] = None
        shared_slots = int(self.config.get("rdma_buffer_slots", "128"))
        shared_slot_size = int(self.config.get("rdma_buffer_slot_size", "4096"))
        self._request_server_ip = str(self.config.get("rdma_request_host", "127.0.0.1"))
        self._request_handshake_port = int(self.config.get("rdma_request_handshake_port", "5566"))
        self._request_slots = shared_slots
        self._request_slot_size = shared_slot_size
        self._phase1_server_ip = str(self.config.get("rdma_phase1_host", "127.0.0.1"))
        self._phase1_handshake_port = int(self.config.get("rdma_phase1_handshake_port", "5567"))
        self._phase1_slots = shared_slots
        self._phase1_slot_size = shared_slot_size
        self._last_request_connect_retry_ts = 0.0
        self._last_phase1_connect_retry_ts = 0.0
        self.text_encoder = None
        self.image_encoder = None
        self.vae_encoder = None
        self.data_mgr = DataManager(
            DisaggregationPhase.PHASE1,
            DisaggregationMode.ENCODE,
        )
        self.data_sender: Dict[int, DataSender] = {}
        self._rdma_buffers: Dict[int, List[torch.Tensor]] = {}
        self.reporter = Reporter(
            service_type="encoder",
            gpu_id=self.encoder_engine_rank,
            bind_address=f"tcp://{self.config.get('data_bootstrap_addr', '127.0.0.1')}:{MONITOR_POLLING_PORT + self.encoder_engine_rank}",
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
            name="encoder-reporter",
            daemon=True,
        )
        self._reporter_thread.start()
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

    def _ensure_request_buffer(self) -> bool:
        if self._request_rdma_buffer is not None:
            return True

        now = time.time()
        if now - self._last_request_connect_retry_ts < 1.0:
            return False
        self._last_request_connect_retry_ts = now

        if self._request_rdma_client is None:
            self._request_rdma_client = RDMAClient(local_buffer_size=self._request_slot_size)

        self._request_rdma_client.connect_to_server(
            server_ip=self._request_server_ip,
            port=self._request_handshake_port,
        )

        remote_info = self._request_rdma_client.remote_info
        base_addr = int(remote_info["addr"])
        descriptor = RDMABufferDescriptor(
            slot_addr=base_addr + 16,
            slot_bytes=self._request_slots * self._request_slot_size,
            slot_size=self._request_slot_size,
            buffer_size=self._request_slots,
            head_addr=base_addr,
            tail_addr=base_addr + 8,
            rkey=int(remote_info.get("rkey", 0)),
        )
        self._request_rdma_buffer = RDMABuffer(
            role="client",
            rdma_client=self._request_rdma_client,
            remote=descriptor,
        )
        self.logger.info(
            "Connected request RDMA buffer: host=%s port=%s slots=%s slot_size=%s",
            self._request_server_ip,
            self._request_handshake_port,
            self._request_slots,
            self._request_slot_size,
        )
        return True

    def _ensure_phase1_meta_buffer(self) -> bool:
        if self._phase1_rdma_buffer is not None:
            return True

        now = time.time()
        if now - self._last_phase1_connect_retry_ts < 1.0:
            return False
        self._last_phase1_connect_retry_ts = now

        if self._phase1_rdma_client is None:
            self._phase1_rdma_client = RDMAClient(local_buffer_size=self._phase1_slot_size)

        self._phase1_rdma_client.connect_to_server(
            server_ip=self._phase1_server_ip,
            port=self._phase1_handshake_port,
        )

        remote_info = self._phase1_rdma_client.remote_info
        base_addr = int(remote_info["addr"])
        descriptor = RDMABufferDescriptor(
            slot_addr=base_addr + 16,
            slot_bytes=self._phase1_slots * self._phase1_slot_size,
            slot_size=self._phase1_slot_size,
            buffer_size=self._phase1_slots,
            head_addr=base_addr,
            tail_addr=base_addr + 8,
            rkey=int(remote_info.get("rkey", 0)),
        )
        self._phase1_rdma_buffer = RDMABuffer(
            role="client",
            rdma_client=self._phase1_rdma_client,
            remote=descriptor,
        )
        self.logger.info(
            "Connected phase1 RDMA buffer: host=%s port=%s slots=%s slot_size=%s",
            self._phase1_server_ip,
            self._phase1_handshake_port,
            self._phase1_slots,
            self._phase1_slot_size,
        )
        return True

    def init(self, config):
        self.config = config
        shared_slots = int(self.config.get("rdma_buffer_slots", self._request_slots))
        shared_slot_size = int(self.config.get("rdma_buffer_slot_size", 4096))
        self._request_server_ip = str(self.config.get("rdma_request_host", self._request_server_ip))
        self._request_handshake_port = int(self.config.get("rdma_request_handshake_port", self._request_handshake_port))
        self._request_slots = shared_slots
        self._request_slot_size = shared_slot_size
        self._phase1_server_ip = str(self.config.get("rdma_phase1_host", self._phase1_server_ip))
        self._phase1_handshake_port = int(self.config.get("rdma_phase1_handshake_port", self._phase1_handshake_port))
        self._phase1_slots = shared_slots
        self._phase1_slot_size = shared_slot_size

        # Seed everything if seed is in config
        if "seed" in self.config:
            seed_all(self.config["seed"])

        data_bootstrap_addr = self.config.get("data_bootstrap_addr", "127.0.0.1")
        data_bootstrap_room = self.config.get("data_bootstrap_room", 0)

        phase1_deadline = time.time() + 30.0
        while self._phase1_rdma_buffer is None and time.time() < phase1_deadline:
            try:
                self._ensure_phase1_meta_buffer()
            except Exception:
                self.logger.exception("Failed to connect phase1 RDMA buffer, will retry")
            if self._phase1_rdma_buffer is None:
                time.sleep(0.1)

        if self._phase1_rdma_buffer is None:
            raise RuntimeError("phase1 RDMA buffer is not ready")

        if data_bootstrap_addr is None or data_bootstrap_room is None:
            return

        buffer_sizes = estimate_encoder_buffer_sizes(self.config)
        request = AllocationRequest(
            bootstrap_room=data_bootstrap_room,
            buffer_sizes=buffer_sizes,
        )
        handle = self.alloc_memory(request)
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
        self.data_mgr.init(data_args, data_bootstrap_room)
        self.data_sender[data_bootstrap_room] = DataSender(self.data_mgr, data_bootstrap_addr, data_bootstrap_room)

        phase1_meta = {
            "request_config": dict(self.config),
            "encoder_node_address": self.data_mgr.get_localhost(),
            "encoder_session_id": self.data_mgr.get_session_id(),
        }
        self._phase1_rdma_buffer.produce(phase1_meta)

    def load_models(self):
        self.logger.info("Loading Encoder Models...")

        # T5 Text Encoder
        text_encoders = load_wan_text_encoder(self.config)
        self.text_encoder = text_encoders[0] if text_encoders else None

        # CLIP Image Encoder (Optional per usage in wan_i2v.py)
        if self.config.get("use_image_encoder", False):
            self.image_encoder = load_wan_image_encoder(self.config)

        # VAE Encoder (Required for I2V)
        # Note: wan_i2v.py logic: if vae_encoder is None: raise RuntimeError
        # But we only load if needed or always? Let's check the config flags.
        # It seems always loaded for I2V task, but might be offloaded.
        # For simplicity of this service, we load it if the task implies it or just try to load.
        # But `load_wan_vae_encoder` will look at the config.
        self.vae_encoder = load_wan_vae_encoder(self.config)

        self.logger.info("Encoder Models loaded successfully.")

    def _get_latent_shape_with_lat_hw(self, latent_h, latent_w):
        return [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            latent_h,
            latent_w,
        ]

    def _compute_latent_shape_from_image(self, image_tensor: torch.Tensor):
        h, w = image_tensor.shape[2:]
        aspect_ratio = h / w
        max_area = self.config["target_height"] * self.config["target_width"]

        latent_h = round(np.sqrt(max_area * aspect_ratio) // self.config["vae_stride"][1] // self.config["patch_size"][1] * self.config["patch_size"][1])
        latent_w = round(np.sqrt(max_area / aspect_ratio) // self.config["vae_stride"][2] // self.config["patch_size"][2] * self.config["patch_size"][2])
        latent_shape = self._get_latent_shape_with_lat_hw(latent_h, latent_w)
        return latent_shape, latent_h, latent_w

    def _get_vae_encoder_output(self, first_frame: torch.Tensor, latent_h: int, latent_w: int):
        h = latent_h * self.config["vae_stride"][1]
        w = latent_w * self.config["vae_stride"][2]

        msk = torch.ones(
            1,
            self.config["target_video_length"],
            latent_h,
            latent_w,
            device=torch.device(AI_DEVICE),
        )
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, latent_h, latent_w)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [
                torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                torch.zeros(3, self.config["target_video_length"] - 1, h, w),
            ],
            dim=1,
        ).to(AI_DEVICE)

        vae_encoder_out = self.vae_encoder.encode(vae_input.unsqueeze(0).to(GET_DTYPE()))
        vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(GET_DTYPE())
        return vae_encoder_out

    def alloc_memory(self, request: AllocationRequest) -> MemoryHandle:
        """
        Args:
            request: AllocationRequest containing precomputed buffer sizes.

        Returns:
            MemoryHandle with RDMA-registered buffer addresses.
        """
        buffer_sizes = request.buffer_sizes
        room = request.bootstrap_room
        self._rdma_buffers[room] = []
        buffers: List[RemoteBuffer] = []

        for nbytes in buffer_sizes:
            if nbytes <= 0:
                continue
            buf = torch.empty(
                (nbytes,),
                dtype=torch.uint8,
                # device=torch.device(f"cuda:{self.sender_engine_rank}"),
            )
            ptr = buf.data_ptr()
            self._rdma_buffers[room].append(buf)
            buffers.append(RemoteBuffer(addr=ptr, nbytes=nbytes))

        return MemoryHandle(buffers=buffers)

    def process(self, config):
        """
        Generates encoder outputs from prompt and image input.
        """
        self.logger.info("Starting processing in EncoderService...")
        room = int(config.get("data_bootstrap_room", 0))

        room_buffers = self._rdma_buffers.get(room)
        sender = self.data_sender.get(room)

        prompt = config.get("prompt")
        negative_prompt = config.get("negative_prompt")
        if prompt is None:
            raise ValueError("prompt is required in config.")

        # 1. Text Encoding
        text_len = config.get("text_len", 512)

        context = self.text_encoder.infer([prompt])
        context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])

        if config.get("enable_cfg", False):
            if negative_prompt is None:
                raise ValueError("negative_prompt is required in config when enable_cfg is True.")
            context_null = self.text_encoder.infer([negative_prompt])
            context_null = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context_null])
        else:
            context_null = None

        text_encoder_output = {
            "context": context,
            "context_null": context_null,
        }

        task = config.get("task")
        clip_encoder_out = None

        if task == "t2v":
            latent_h = config["target_height"] // config["vae_stride"][1]
            latent_w = config["target_width"] // config["vae_stride"][2]
            latent_shape = [
                config.get("num_channels_latents", 16),
                (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
                latent_h,
                latent_w,
            ]
            image_encoder_output = None
        elif task == "i2v":
            image_path = config.get("image_path")
            if image_path is None:
                raise ValueError("image_path is required for i2v task.")

            # 2. Image Encoding + VAE Encoding
            img, _ = read_image_input(image_path)

            if self.image_encoder is not None:
                # Assuming image_encoder.visual handles list of images
                clip_encoder_out = self.image_encoder.visual([img]).squeeze(0).to(GET_DTYPE())

            if self.vae_encoder is None:
                raise RuntimeError("VAE encoder is required but was not loaded.")

            latent_shape, latent_h, latent_w = self._compute_latent_shape_from_image(img)
            vae_encoder_out = self._get_vae_encoder_output(img, latent_h, latent_w)

            image_encoder_output = {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encoder_out,
            }
        else:
            raise ValueError(f"Unsupported task: {task}")

        self.logger.info("Encode processing completed. Preparing to send data...")

        if self.data_mgr is not None and sender is not None:
            if room_buffers is None:
                raise RuntimeError(f"RDMA buffers are not initialized for room={room}")

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

            buffer_index = 0
            context_buf = room_buffers[buffer_index]
            context_buf.zero_()
            context_view = _buffer_view(context_buf, GET_DTYPE(), tuple(context.shape))
            context_view.copy_(context)
            buffer_index += 1
            if config.get("enable_cfg", False):
                context_null_buf = room_buffers[buffer_index]
                context_null_buf.zero_()
                context_null_view = _buffer_view(context_null_buf, GET_DTYPE(), tuple(context_null.shape))
                context_null_view.copy_(context_null)
                buffer_index += 1

            if task == "i2v":
                if config.get("use_image_encoder", True):
                    clip_buf = room_buffers[buffer_index]
                    clip_buf.zero_()
                    if image_encoder_output.get("clip_encoder_out") is not None:
                        clip_view = _buffer_view(clip_buf, GET_DTYPE(), tuple(image_encoder_output["clip_encoder_out"].shape))
                        clip_view.copy_(image_encoder_output["clip_encoder_out"])
                    buffer_index += 1

                vae_buf = room_buffers[buffer_index]
                vae_buf.zero_()
                vae_view = _buffer_view(
                    vae_buf,
                    GET_DTYPE(),
                    tuple(image_encoder_output["vae_encoder_out"].shape),
                )
                vae_view.copy_(image_encoder_output["vae_encoder_out"])
                buffer_index += 1

            latent_tensor = torch.tensor(latent_shape, device=AI_DEVICE, dtype=torch.int64)
            latent_buf = _buffer_view(room_buffers[buffer_index], torch.int64, (4,))
            latent_buf.copy_(latent_tensor)
            buffer_index += 1

            meta = {
                "version": 1,
                "context_shape": list(context.shape),
                "context_dtype": str(context.dtype),
                "context_hash": _sha256_tensor(context),
                "context_null_shape": list(context_null.shape) if context_null is not None else None,
                "context_null_dtype": str(context_null.dtype) if context_null is not None else None,
                "context_null_hash": _sha256_tensor(context_null),
                "clip_shape": list(clip_encoder_out.shape) if clip_encoder_out is not None else None,
                "clip_dtype": str(clip_encoder_out.dtype) if clip_encoder_out is not None else None,
                "clip_hash": _sha256_tensor(clip_encoder_out),
                "vae_shape": list(image_encoder_output["vae_encoder_out"].shape) if image_encoder_output is not None else None,
                "vae_dtype": str(image_encoder_output["vae_encoder_out"].dtype) if image_encoder_output is not None else None,
                "vae_hash": _sha256_tensor(image_encoder_output["vae_encoder_out"]) if image_encoder_output is not None else None,
                "latent_shape": list(latent_shape),
                "latent_dtype": str(latent_tensor.dtype),
                "latent_hash": _sha256_tensor(latent_tensor),
            }
            meta_shapes = {k: v for k, v in meta.items() if k.endswith("_shape")}
            meta_dtypes = {k: v for k, v in meta.items() if k.endswith("_dtype")}
            self.logger.info("Encoder meta shapes: %s", meta_shapes)
            self.logger.info("Encoder meta dtypes: %s", meta_dtypes)
            meta_bytes = json.dumps(meta, ensure_ascii=True).encode("utf-8")
            meta_buf = _buffer_view(room_buffers[buffer_index], torch.uint8, (room_buffers[buffer_index].numel(),))
            if meta_bytes and len(meta_bytes) > meta_buf.numel():
                raise ValueError("metadata buffer too small for hash/shape payload")
            meta_buf.zero_()
            if meta_bytes:
                meta_buf[: len(meta_bytes)].copy_(torch.from_numpy(np.frombuffer(meta_bytes, dtype=np.uint8)))

            buffer_ptrs = [buf.data_ptr() for buf in room_buffers]
            sender.send(buffer_ptrs)

    def release_memory(self, room: int):
        """
        Releases the RDMA buffers and clears GPU cache.
        """
        if room in self._rdma_buffers:
            self._rdma_buffers.pop(room, None)
        torch.cuda.empty_cache()

    def remove(self, room: int):
        self.release_memory(room)

        self.data_sender.pop(room, None)

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
        self.data_sender.clear()
        self.text_encoder = None
        self.image_encoder = None
        self.vae_encoder = None

    def run(self, stop_event=None):
        req_queue = deque()
        exec_queue = deque()
        complete_queue: Dict[int, dict] = {}

        while True:
            transfer_sizes = self.data_mgr.get_backlog_counts() if self.data_mgr is not None else {"request_pool": 0, "waiting_pool": 0}
            self._update_queue_metrics(
                {
                    "req_queue": len(req_queue),
                    "exec_queue": len(exec_queue),
                    "complete_queue": len(complete_queue),
                },
                {
                    "request_pool": int(transfer_sizes.get("request_pool", 0)),
                    "waiting_pool": int(transfer_sizes.get("waiting_pool", 0)),
                },
            )

            if self._request_rdma_buffer is None:
                try:
                    self._ensure_request_buffer()
                except Exception:
                    self.logger.exception("Failed to connect request RDMA buffer, will retry")

            if self._request_rdma_buffer is not None:
                config = self._request_rdma_buffer.consume()
                if config is not None:
                    self.logger.info("Received request config from RDMA buffer: %s", {k: v for k, v in config.items()})
                    req_queue.append(config)

            if req_queue:
                config = req_queue.popleft()
                room = int(config.get("data_bootstrap_room", 0))
                try:
                    self.init(config)
                    exec_queue.append((room, config))
                except Exception:
                    self.logger.exception("Failed to initialize request for room=%s", room)
                    self.remove(room)

            if exec_queue:
                room, config = exec_queue.popleft()
                try:
                    self.process(config)
                    complete_queue[room] = config
                except Exception:
                    self.logger.exception("Failed to process request for room=%s", room)
                    complete_queue.pop(room, None)
                    self.remove(room)

            completed_rooms: List[int] = []
            for room in list(complete_queue.keys()):
                sender = self.data_sender.get(room)
                if sender is None:
                    completed_rooms.append(room)
                    continue

                status = sender.poll()
                if status == DataPoll.Success:
                    completed_rooms.append(room)
                elif status == DataPoll.Failed:
                    self.logger.error("DataSender transfer failed for room=%s", room)
                    completed_rooms.append(room)

            for room in completed_rooms:
                complete_queue.pop(room, None)
                self.remove(room)

            if stop_event is not None and stop_event.is_set() and not req_queue and not exec_queue and not complete_queue:
                self.logger.info("EncoderService received stop event, exiting request loop.")
                break

            if not req_queue and not exec_queue:
                time.sleep(0.01)

        self.release()
