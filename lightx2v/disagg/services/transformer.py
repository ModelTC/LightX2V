import torch
import logging
from typing import Dict, Any, List, Optional

from lightx2v.disagg.services.base import BaseService
from lightx2v.disagg.utils import (
    load_wan_transformer,
    load_wan_vae_decoder,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import seed_all, save_to_video, wan_vae_to_comfy

class TransformerService(BaseService):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = None
        self.vae_decoder = None
        self.scheduler = None

        self.load_models()
        
        # Set global seed if present in config, though specific process calls might reuse it
        if "seed" in self.config:
            seed_all(self.config["seed"])

    def load_models(self):
        self.logger.info("Loading Transformer Models...")
        
        self.transformer = load_wan_transformer(self.config)
        self.vae_decoder = load_wan_vae_decoder(self.config)
        
        # Initialize scheduler
        self.scheduler = WanScheduler(self.config)
        self.transformer.set_scheduler(self.scheduler)
        
        self.logger.info("Transformer Models loaded successfully.")

    def process(self, inputs: Dict[str, Any]):
        """
        Executes the diffusion process and video decoding.
        
        Args:
            inputs: Dictionary containing 'text_encoder_output', 'image_encoder_output', and 'latent_shape'.
        """
        self.logger.info("Starting processing in TransformerService...")

        seed = self.config.get("seed")
        save_path = self.config.get("save_path")
        if seed is None:
            raise ValueError("seed is required in config.")
        if save_path is None:
            raise ValueError("save_path is required in config.")
        
        image_encoder_output = inputs.get("image_encoder_output")
        latent_shape = inputs.get("latent_shape")
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
