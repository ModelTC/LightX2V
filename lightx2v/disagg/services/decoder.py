import hashlib
import json
import time
from collections import deque
from typing import Dict, List, Optional

import torch

from lightx2v.disagg.conn import REQUEST_POLLING_PORT, DataArgs, DataManager, DataPoll, DataReceiver, DisaggregationMode, DisaggregationPhase, ReqManager
from lightx2v.disagg.protocol import AllocationRequest, MemoryHandle, RemoteBuffer
from lightx2v.disagg.services.base import BaseService
from lightx2v.disagg.utils import estimate_transformer_buffer_sizes, load_wan_vae_decoder
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import save_to_video, seed_all, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE


class DecoderService(BaseService):
    def __init__(self):
        super().__init__()
        self.request_port = REQUEST_POLLING_PORT + 2
        self.req_mgr = ReqManager()
        self.vae_decoder = None
        self._rdma_buffers: Dict[int, List[torch.Tensor]] = {}
        self.data_mgr = DataManager(
            DisaggregationPhase.PHASE2,
            DisaggregationMode.DECODE,
        )
        self.data_receiver: Dict[int, DataReceiver] = {}

    def init(self, config):
        self.config = config
        self.vae_decoder = None

        self.encoder_engine_rank = int(self.config.get("encoder_engine_rank", 0))
        self.transformer_engine_rank = int(self.config.get("transformer_engine_rank", 1))
        self.decoder_engine_rank = int(self.config.get("decoder_engine_rank", 2))

        self.load_models()

        if "seed" in self.config:
            seed_all(self.config["seed"])

        data_bootstrap_addr = self.config.get("data_bootstrap_addr", "127.0.0.1")
        data_bootstrap_room = self.config.get("data_bootstrap_room", 0)
        if data_bootstrap_addr is None or data_bootstrap_room is None:
            return

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
        self.data_receiver[data_bootstrap_room] = DataReceiver(self.data_mgr, data_bootstrap_addr, data_bootstrap_room)
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
        meta_bytes = _buffer_view(meta_buf, torch.uint8, (meta_buf.numel(),)).detach().contiguous().cpu().numpy().tobytes()
        meta_str = meta_bytes.split(b"\x00", 1)[0].decode("utf-8") if meta_bytes else ""
        if not meta_str:
            raise ValueError("missing latents metadata from transformer")
        meta = json.loads(meta_str)

        latents_shape_val = meta.get("latents_shape")
        if not isinstance(latents_shape_val, list) or len(latents_shape_val) != 4:
            raise ValueError("invalid latents_shape in phase2 metadata")
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
            raise ValueError("latents hash mismatch between transformer and decoder")
        latents = latents.to(torch.device(AI_DEVICE)).contiguous()

        if self.vae_decoder is None:
            raise RuntimeError("VAE decoder is not loaded.")

        self.logger.info("Decoding latents in DecoderService...")
        gen_video = self.vae_decoder.decode(latents.to(GET_DTYPE()))
        gen_video_final = wan_vae_to_comfy(gen_video)

        save_path = config.get("save_path")
        if save_path is None:
            raise ValueError("save_path is required in config.")

        self.logger.info(f"Saving video to {save_path}...")
        save_to_video(gen_video_final, save_path, fps=config.get("fps", 16), method="ffmpeg")
        self.logger.info("Done!")

        return save_path

    def release_memory(self, room: int):
        if room in self._rdma_buffers:
            self._rdma_buffers.pop(room, None)
        torch.cuda.empty_cache()

    def remove(self, room: int):
        self.release_memory(room)

        self.data_receiver.pop(room, None)

        if self.data_mgr is None:
            return

        self.data_mgr.remove(room)

    def release(self):
        for room in list(self._rdma_buffers.keys()):
            self.remove(room)
        if self.data_mgr is not None:
            self.data_mgr.release()
        self.data_receiver.clear()
        self.vae_decoder = None

    def run(self, stop_event=None):
        req_queue = deque()
        waiting_queue: Dict[int, dict] = {}
        exec_queue = deque()

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
                    self.remove(room)

            ready_rooms: List[int] = []
            failed_rooms: List[int] = []
            for room in list(waiting_queue.keys()):
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
                self.logger.info("Latents received successfully in DecoderService for room=%s.", room)
                exec_queue.append((room, waiting_queue.pop(room)))

            for room in failed_rooms:
                waiting_queue.pop(room, None)
                self.logger.error("DataReceiver transfer failed for room=%s", room)
                self.remove(room)

            if exec_queue:
                room, config = exec_queue.popleft()
                try:
                    self.process(config)
                except Exception:
                    self.logger.exception("Failed to process request for room=%s", room)
                finally:
                    self.remove(room)

            if stop_event is not None and stop_event.is_set() and not req_queue and not waiting_queue and not exec_queue:
                self.logger.info("DecoderService received stop event, exiting request loop.")
                break

            if not req_queue and not exec_queue:
                time.sleep(0.01)

        self.release()
