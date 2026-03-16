import hashlib
import json
import time
from collections import deque
from typing import List, Optional

import numpy as np
import torch

from lightx2v.disagg.conn import REQUEST_POLLING_PORT, DataArgs, DataManager, DataPoll, DataReceiver, DataSender, DisaggregationMode, DisaggregationPhase, ReqManager
from lightx2v.disagg.protocol import AllocationRequest, MemoryHandle, RemoteBuffer
from lightx2v.disagg.services.base import BaseService
from lightx2v.disagg.utils import (
    estimate_encoder_buffer_sizes,
    estimate_transformer_buffer_sizes,
    load_wan_transformer,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import seed_all
from lightx2v_platform.base.global_var import AI_DEVICE


class TransformerService(BaseService):
    def __init__(self):
        super().__init__()
        self.request_port = REQUEST_POLLING_PORT + 1
        self.req_mgr = ReqManager()
        self.transformer = None
        self.scheduler = None
        self.rdma_buffer1: dict[int, List[torch.Tensor]] = {}
        self.rdma_buffer2: dict[int, List[torch.Tensor]] = {}
        self.data_mgr1 = DataManager(DisaggregationPhase.PHASE1, DisaggregationMode.TRANSFORMER)
        self.data_mgr2 = DataManager(DisaggregationPhase.PHASE2, DisaggregationMode.TRANSFORMER)
        self.data_receiver: dict[int, DataReceiver] = {}
        self.data_sender: dict[int, DataSender] = {}

    def init(self, config):
        self.config = config
        self.transformer = None
        self.scheduler = None
        self.encoder_engine_rank = int(self.config.get("encoder_engine_rank", 0))
        self.transformer_engine_rank = int(self.config.get("transformer_engine_rank", 1))
        self.decoder_engine_rank = int(self.config.get("decoder_engine_rank", 2))

        self.load_models()

        # Set global seed if present in config, though specific process calls might reuse it
        if "seed" in self.config:
            seed_all(self.config["seed"])

        data_bootstrap_addr = self.config.get("data_bootstrap_addr", "127.0.0.1")
        data_bootstrap_room = self.config.get("data_bootstrap_room", 0)

        if data_bootstrap_addr is None or data_bootstrap_room is None:
            return

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
        self.data_receiver[data_bootstrap_room] = DataReceiver(self.data_mgr1, data_bootstrap_addr, data_bootstrap_room)
        self.data_receiver[data_bootstrap_room].init()

        buffer_sizes = estimate_transformer_buffer_sizes(self.config)
        request = AllocationRequest(
            bootstrap_room=data_bootstrap_room,
            buffer_sizes=buffer_sizes,
        )
        handle = self.alloc_memory(DisaggregationPhase.PHASE2, request)
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
        self.data_mgr2.init(data_args, data_bootstrap_room)
        self.data_sender[data_bootstrap_room] = DataSender(self.data_mgr2, data_bootstrap_addr, data_bootstrap_room)

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
        room = config.get("data_bootstrap_room", 0)

        phase1_buffers = self.rdma_buffer1.get(room)
        phase2_buffers = self.rdma_buffer2.get(room)
        receiver = self.data_receiver.get(room)
        sender = self.data_sender.get(room)

        if phase1_buffers is None:
            raise RuntimeError(f"phase1 RDMA buffers are not initialized for room={room}.")
        if phase2_buffers is None:
            raise RuntimeError(f"phase2 RDMA buffers are not initialized for room={room}.")
        if receiver is None:
            raise RuntimeError(f"DataReceiver is not initialized for room={room}.")
        if sender is None:
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
        meta_bytes = _buffer_view(meta_buf, torch.uint8, (meta_buf.numel(),)).detach().contiguous().cpu().numpy().tobytes()
        meta_str = meta_bytes.split(b"\x00", 1)[0].decode("utf-8") if meta_bytes else ""
        if not meta_str:
            raise ValueError("missing metadata from encoder")
        meta = json.loads(meta_str)
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
                raise ValueError("context hash mismatch between encoder and transformer")
            if enable_cfg:
                if context_null is not None:
                    if list(context_null.shape) != meta.get("context_null_shape"):
                        raise ValueError("context_null shape mismatch between encoder and transformer")
                if meta.get("context_null_hash") is not None:
                    if _sha256_tensor(context_null) != meta.get("context_null_hash"):
                        raise ValueError("context_null hash mismatch between encoder and transformer")
            if task == "i2v":
                if clip_encoder_out is not None:
                    if list(clip_encoder_out.shape) != meta.get("clip_shape"):
                        raise ValueError("clip shape mismatch between encoder and transformer")
                if meta.get("clip_hash") is not None:
                    if _sha256_tensor(clip_encoder_out) != meta.get("clip_hash"):
                        raise ValueError("clip hash mismatch between encoder and transformer")
                if vae_encoder_out is not None:
                    if list(vae_encoder_out.shape) != meta.get("vae_shape"):
                        raise ValueError("vae shape mismatch between encoder and transformer")
                if meta.get("vae_hash") is not None:
                    if _sha256_tensor(vae_encoder_out) != meta.get("vae_hash"):
                        raise ValueError("vae hash mismatch between encoder and transformer")
            if meta.get("latent_shape") is None or list(latent_shape) != meta.get("latent_shape"):
                raise ValueError("latent_shape mismatch between encoder and transformer")
            if meta.get("latent_hash") is not None:
                latent_tensor = torch.tensor(latent_shape, device=AI_DEVICE, dtype=torch.int64)
                if _sha256_tensor(latent_tensor) != meta.get("latent_hash"):
                    raise ValueError("latent_shape hash mismatch between encoder and transformer")

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
        sender.send(buffer_ptrs)

    def release_memory(self, room: int):
        """
        Releases the RDMA buffers and clears GPU cache.
        """
        if room in self.rdma_buffer1:
            self.rdma_buffer1.pop(room, None)

        if room in self.rdma_buffer2:
            self.rdma_buffer2.pop(room, None)

        torch.cuda.empty_cache()

    def release(self, room: int):
        self.release_memory(room)

        self.data_receiver.pop(room, None)
        self.data_sender.pop(room, None)

        if self.data_mgr1 is not None:
            self.data_mgr1.release(room)
        if self.data_mgr2 is not None:
            self.data_mgr2.release(room)

    def exec_request(self, stop_event=None):
        req_queue = deque()
        waiting_queue: dict[int, dict] = {}
        exec_queue = deque()
        complete_queue: dict[int, dict] = {}

        while True:
            while True:
                config = self.req_mgr.receive_non_block(self.request_port)
                if config is None:
                    break
                req_queue.append(config)

            if req_queue:
                config = req_queue.popleft()
                room = int(config.get("data_bootstrap_room", 0))
                try:
                    self.init(config)
                    waiting_queue[room] = config
                except Exception:
                    self.logger.exception("Failed to initialize request for room=%s", room)
                    self.release(room)

            ready_rooms: List[int] = []
            failed_rooms: List[int] = []
            for room, config in list(waiting_queue.items()):
                receiver = self.data_receiver.get(room)
                if receiver is None:
                    failed_rooms.append(room)
                    continue

                status = receiver.poll()
                if status == DataPoll.Success:
                    ready_rooms.append(room)
                elif status == DataPoll.Failed:
                    failed_rooms.append(room)

            for room in ready_rooms:
                exec_queue.append((room, waiting_queue.pop(room)))

            for room in failed_rooms:
                waiting_queue.pop(room, None)
                self.logger.error("DataReceiver transfer failed for room=%s", room)
                self.release(room)

            if exec_queue:
                room, config = exec_queue.popleft()
                try:
                    self.process(config)
                    complete_queue[room] = config
                except Exception:
                    self.logger.exception("Failed to process request for room=%s", room)
                    complete_queue.pop(room, None)
                    self.release(room)

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
                self.release(room)

            if stop_event is not None and stop_event.is_set() and not req_queue and not waiting_queue and not exec_queue and not complete_queue:
                self.logger.info("TransformerService received stop event, exiting request loop.")
                break

            if not req_queue and not exec_queue:
                time.sleep(0.01)
