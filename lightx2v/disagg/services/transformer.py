import torch
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from lightx2v.disagg.services.base import BaseService
from lightx2v.disagg.conn import DataArgs, DataManager, DataReceiver, DisaggregationMode, DataPoll
from lightx2v.disagg.utils import (
    estimate_encoder_buffer_sizes,
    load_wan_transformer,
    load_wan_vae_decoder,
)
from lightx2v.disagg.protocol import AllocationRequest, MemoryHandle, RemoteBuffer
from lightx2v.disagg.mooncake import MooncakeTransferEngine
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import seed_all, save_to_video, wan_vae_to_comfy

class TransformerService(BaseService):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = None
        self.vae_decoder = None
        self.scheduler = None
        self.transfer_engine = None
        self._rdma_buffers: List[torch.Tensor] = []
        self.engine_rank = 1
        self.data_mgr = None
        self.data_receiver = None

        self.load_models()
        
        # Set global seed if present in config, though specific process calls might reuse it
        if "seed" in self.config:
            seed_all(self.config["seed"])

        data_bootstrap_addr = self.config.get("data_bootstrap_addr", "127.0.0.1")
        data_bootstrap_room = self.config.get("data_bootstrap_room", 0)
        
        if data_bootstrap_addr is None or data_bootstrap_room is None:
            return

        request = AllocationRequest(
            bootstrap_room=str(data_bootstrap_room),
            config=self.config,
        )
        handle = self.alloc_memory(request)
        data_ptrs = [buf.addr for buf in handle.buffers]
        data_lens = [buf.nbytes for buf in handle.buffers]

        data_args = DataArgs(
            engine_rank=self.engine_rank,
            data_ptrs=data_ptrs,
            data_lens=data_lens,
            data_item_lens=data_lens,
            ib_device=None,
        )
        self.data_mgr = DataManager(data_args, DisaggregationMode.TRANSFORMER)
        self.data_receiver = DataReceiver(
            self.data_mgr, data_bootstrap_addr, int(data_bootstrap_room)
        )
        self.data_receiver.init()

    def load_models(self):
        self.logger.info("Loading Transformer Models...")
        
        self.transformer = load_wan_transformer(self.config)
        self.vae_decoder = load_wan_vae_decoder(self.config)
        
        # Initialize scheduler
        self.scheduler = WanScheduler(self.config)
        self.transformer.set_scheduler(self.scheduler)
        
        self.logger.info("Transformer Models loaded successfully.")

    def alloc_memory(self, request: AllocationRequest) -> MemoryHandle:
        """
        Estimate upper-bound memory for encoder results and allocate GPU buffers.

        Args:
            request: AllocationRequest containing config and tensor specs.

        Returns:
            MemoryHandle with RDMA-registered buffer addresses.
        """
        config = request.config
        estimated_sizes = estimate_encoder_buffer_sizes(config)
        buffer_sizes = estimated_sizes

        if self.transfer_engine is None:
            self.transfer_engine = MooncakeTransferEngine()

        self._rdma_buffers = []
        buffers: List[RemoteBuffer] = []
        for nbytes in buffer_sizes:
            if nbytes <= 0:
                continue
            buf = torch.empty((nbytes,), dtype=torch.uint8, device=torch.device(f"cuda:{self.engine_rank}"))
            ptr = buf.data_ptr()
            self.transfer_engine.register(ptr, nbytes)
            self._rdma_buffers.append(buf)
            buffers.append(
                RemoteBuffer(addr=ptr, session_id=self.transfer_engine.get_session_id(), nbytes=nbytes)
            )

        return MemoryHandle(buffers=buffers)

    def process(self):
        """
        Executes the diffusion process and video decoding.
        """
        self.logger.info("Starting processing in TransformerService...")
        
        # Poll for data from EncoderService
        import time
        if self.data_receiver is not None:
            while True:
                status = self.data_receiver.poll()
                if status == DataPoll.Success:
                    break
                time.sleep(0.01)
        else:
            self.logger.warning("DataReceiver is not initialized. Using dummy or existing data if any.")
            pass

        # Reconstruct inputs from _rdma_buffers
        text_len = int(self.config.get("text_len", 512))
        text_dim = int(self.config.get("text_encoder_dim", 4096))
        clip_dim = int(self.config.get("clip_embed_dim", 1024))
        z_dim = int(self.config.get("vae_z_dim", 16))
        
        vae_stride = self.config.get("vae_stride", (4, 8, 8))
        target_video_length = int(self.config.get("target_video_length", 81))
        target_height = int(self.config.get("target_height", 480))
        target_width = int(self.config.get("target_width", 832))

        t_prime = 1 + (target_video_length - 1) // int(vae_stride[0])
        h_prime = int(np.ceil(target_height / int(vae_stride[1])))
        w_prime = int(np.ceil(target_width / int(vae_stride[2])))
        
        enable_cfg = bool(self.config.get("enable_cfg", False))
        task = self.config.get("task", "i2v")
        use_image_encoder = bool(self.config.get("use_image_encoder", True))

        buffer_index = 0
        
        # 1. Text Context
        context_buf = self._rdma_buffers[buffer_index]
        context = context_buf.view(GET_DTYPE()).reshape(1, text_len, text_dim)
        buffer_index += 1
        
        context_null = None
        if enable_cfg:
            context_null_buf = self._rdma_buffers[buffer_index]
            context_null = context_null_buf.view(GET_DTYPE()).reshape(1, text_len, text_dim)
            buffer_index += 1
            
        text_encoder_output = {
            "context": context,
            "context_null": context_null,
        }

        image_encoder_output = {}
        clip_encoder_out = None
        vae_encoder_out_padded = None
        
        if task == "i2v":
            if use_image_encoder:
                clip_buf = self._rdma_buffers[buffer_index]
                clip_encoder_out = clip_buf.view(GET_DTYPE()).reshape(clip_dim)
                buffer_index += 1

            vae_buf = self._rdma_buffers[buffer_index]
            vae_encoder_out_padded = vae_buf.view(GET_DTYPE()).reshape(z_dim + 4, t_prime, h_prime, w_prime)
            buffer_index += 1
        
        latent_shape_buf = self._rdma_buffers[buffer_index]
        latent_shape = latent_shape_buf.view(torch.int64).tolist()
        
        vae_encoder_out = None
        if vae_encoder_out_padded is not None:
            valid_t = latent_shape[1]
            valid_h = latent_shape[2]
            valid_w = latent_shape[3]
            vae_encoder_out = vae_encoder_out_padded[:, :valid_t, :valid_h, :valid_w]

        if task == "i2v":
            image_encoder_output["clip_encoder_out"] = clip_encoder_out
            image_encoder_output["vae_encoder_out"] = vae_encoder_out
        else:
            image_encoder_output = None
        
        inputs = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
            "latent_shape": latent_shape,
        }

        seed = self.config.get("seed")
        save_path = self.config.get("save_path")
        if seed is None:
            raise ValueError("seed is required in config.")
        if save_path is None:
            raise ValueError("save_path is required in config.")
        
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
        
        # VAE Decoding
        self.logger.info("Decoding latents...")
        if self.vae_decoder is None:
             raise RuntimeError("VAE decoder is not loaded.")
             
        gen_video = self.vae_decoder.decode(latents.to(GET_DTYPE()))
        
        # Post-processing
        self.logger.info("Post-processing video...")
        gen_video_final = wan_vae_to_comfy(gen_video)
        
        # Saving
        self.logger.info(f"Saving video to {save_path}...")
        save_to_video(gen_video_final, save_path, fps=self.config.get("fps", 16), method="ffmpeg")
        self.logger.info("Done!")
        
        return save_path

    def release_memory(self):
        """
        Releases the RDMA buffers, deregisters them from transfer engine, and clears GPU cache.
        """
        if self._rdma_buffers:
            for buf in self._rdma_buffers:
                if self.transfer_engine:
                    self.transfer_engine.deregister(buf.data_ptr())
            self._rdma_buffers = []
        
        torch.cuda.empty_cache()
