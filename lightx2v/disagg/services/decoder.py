import hashlib
import json
import time
from typing import List, Optional

import torch

from lightx2v.disagg.conn import DataArgs, DataManager, DataPoll, DataReceiver, DisaggregationMode, DisaggregationPhase
from lightx2v.disagg.protocol import AllocationRequest, MemoryHandle, RemoteBuffer
from lightx2v.disagg.services.base import BaseService
from lightx2v.disagg.utils import estimate_transformer_buffer_sizes, load_wan_vae_decoder
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import save_to_video, seed_all, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE


class DecoderService(BaseService):
    def __init__(self, config):
        super().__init__(config)
        self.vae_decoder = None
        self._rdma_buffers: List[torch.Tensor] = []

        self.encoder_engine_rank = int(self.config.get("encoder_engine_rank", 0))
        self.transformer_engine_rank = int(self.config.get("transformer_engine_rank", 1))
        self.decoder_engine_rank = int(self.config.get("decoder_engine_rank", 2))

        self.data_mgr = None
        self.data_receiver = None

        self.load_models()

        if "seed" in self.config:
            seed_all(self.config["seed"])

        data_bootstrap_addr = self.config.get("data_bootstrap_addr", "127.0.0.1")
        data_bootstrap_room = self.config.get("data_bootstrap_room", 0)
        if data_bootstrap_addr is None or data_bootstrap_room is None:
            return

        buffer_sizes = estimate_transformer_buffer_sizes(self.config)
        request = AllocationRequest(
            bootstrap_room=str(data_bootstrap_room),
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
        self.data_mgr = DataManager(
            data_args,
            DisaggregationPhase.PHASE2,
            DisaggregationMode.DECODE,
        )
        self.data_receiver = DataReceiver(self.data_mgr, data_bootstrap_addr, int(data_bootstrap_room))
        self.data_receiver.init()

    def load_models(self):
        self.logger.info("Loading Decoder Models...")
        self.vae_decoder = load_wan_vae_decoder(self.config)
        self.logger.info("Decoder Models loaded successfully.")

    def alloc_memory(self, request: AllocationRequest) -> MemoryHandle:
        buffer_sizes = request.buffer_sizes

        self._rdma_buffers = []
        buffers: List[RemoteBuffer] = []
        for nbytes in buffer_sizes:
            if nbytes <= 0:
                continue
            buf = torch.empty((nbytes,), dtype=torch.uint8)
            ptr = buf.data_ptr()
            self._rdma_buffers.append(buf)
            buffers.append(RemoteBuffer(addr=ptr, nbytes=nbytes))

        return MemoryHandle(buffers=buffers)

    def process(self):
        self.logger.info("Starting processing in DecoderService...")

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

        if self.data_receiver is not None:
            while True:
                status = self.data_receiver.poll()
                if status == DataPoll.Success:
                    self.logger.info("Latents received successfully in DecoderService.")
                    break
                time.sleep(0.01)
        else:
            raise RuntimeError("DataReceiver is not initialized in DecoderService.")

        if not self._rdma_buffers:
            raise RuntimeError("No RDMA buffer available in DecoderService.")
        if len(self._rdma_buffers) < 2:
            raise RuntimeError("Phase2 RDMA buffers require [latents, meta] entries.")

        meta_buf = self._rdma_buffers[1]
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

        latents = _buffer_view(self._rdma_buffers[0], latents_dtype, latent_shape)
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

        save_path = self.config.get("save_path")
        if save_path is None:
            raise ValueError("save_path is required in config.")

        self.logger.info(f"Saving video to {save_path}...")
        save_to_video(gen_video_final, save_path, fps=self.config.get("fps", 16), method="ffmpeg")
        self.logger.info("Done!")

        return save_path

    def release_memory(self):
        if self._rdma_buffers:
            for buf in self._rdma_buffers:
                if self.data_mgr is not None:
                    self.data_mgr.engine.deregister(buf.data_ptr())
            self._rdma_buffers = []
        torch.cuda.empty_cache()
