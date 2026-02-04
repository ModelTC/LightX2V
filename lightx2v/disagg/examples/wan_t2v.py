import os
import torch
import logging
from loguru import logger
from lightx2v.disagg.utils import (
    load_wan_text_encoder,
    load_wan_vae_decoder,
    load_wan_transformer,
    set_config,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.utils import seed_all, save_to_video, wan_vae_to_comfy
from lightx2v.utils.input_info import init_empty_input_info
from lightx2v.utils.envs import GET_DTYPE

# Setup basic logging
logging.basicConfig(level=logging.INFO)

def main():
    # 1. Configuration
    model_path = "/root/zht/LightX2V/models/Wan-AI/Wan2.1-T2V-1.3B"
    task = "t2v"
    model_cls = "wan2.1"
    save_result_path = "/root/zht/LightX2V/save_results/test_disagg.mp4"
    
    # Generation parameters
    seed = 42
    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    # Initialize configuration
    # Note: We pass generation parameters as kwargs to override defaults/config file settings
    config = set_config(
        model_path=model_path,
        task=task,
        model_cls=model_cls,
        # Configuration parameters from pipe.create_generator in original example
        attn_mode= "sage_attn2",
        infer_steps=50,
        target_height=480,
        target_width=832,
        target_video_length=81,
        sample_guide_scale=5.0,
        sample_shift=5.0,
        fps=16,
        # Default parameters usually set in LightX2VPipeline or config setup
        enable_cfg=True
    )
    
    logger.info(f"Config initialized for task: {task}")
    seed_all(seed)

    # 2. Load Models
    logger.info("Loading models...")
    
    # Text Encoder (T5)
    text_encoders = load_wan_text_encoder(config)
    text_encoder = text_encoders[0]
    
    # Transformer (WanModel)
    model = load_wan_transformer(config)
    
    # VAE Decoder
    vae_decoder = load_wan_vae_decoder(config)
    
    logger.info("Models loaded successfully.")

    # 3. Initialize Scheduler
    # Only supporting basic NoCaching scheduler for this simple example as per default config
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)
    
    # 4. Run Inference Pipeline
    
    # 4.1 Text Encoding
    logger.info("Running text encoding...")
    text_len = config.get("text_len", 512)
    
    # Context (Prompt)
    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])
    
    # Context Null (Negative Prompt) for CFG
    if config.get("enable_cfg", False):
        context_null = text_encoder.infer([negative_prompt])
        context_null = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context_null])
    else:
        context_null = None
        
    text_encoder_output = {
        "context": context,
        "context_null": context_null,
    }
    
    # 4.2 Prepare Inputs for Transformer
    # Wan T2V input construction
    # We need to construct the 'inputs' dictionary expected by model.infer
    
    # Calculate latent shape
    # Logic from DefaultRunner.get_latent_shape_with_target_hw
    latent_h = config["target_height"] // config["vae_stride"][1]
    latent_w = config["target_width"] // config["vae_stride"][2]
    latent_shape = [
        config.get("num_channels_latents", 16),
        (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
        latent_h,
        latent_w,
    ]
    
    inputs = {
        "text_encoder_output": text_encoder_output,
        "image_encoder_output": None # T2V usually doesn't need image encoder output unless specified
    }
    
    # 4.3 Scheduler Preparation
    logger.info("Preparing scheduler...")
    scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=None)
    
    # 4.4 Denoising Loop
    logger.info("Starting denoising loop...")
    infer_steps = scheduler.infer_steps
    
    for step_index in range(infer_steps):
        logger.info(f"Step {step_index + 1}/{infer_steps}")
        
        # Pre-step
        scheduler.step_pre(step_index=step_index)
        
        # Model Inference
        model.infer(inputs)
        
        # Post-step
        scheduler.step_post()
        
    latents = scheduler.latents
    
    # 4.5 VAE Decoding
    logger.info("Decoding latents...")
    # Decode latents to video frames
    # latents need to be cast to correct dtype usually
    gen_video = vae_decoder.decode(latents.to(GET_DTYPE()))
    
    # 5. Post-processing and Saving
    logger.info("Post-processing video...")
    gen_video_final = wan_vae_to_comfy(gen_video)
    
    logger.info(f"Saving video to {save_result_path}...")
    save_to_video(gen_video_final, save_result_path, fps=config.get("fps", 16), method="ffmpeg")
    logger.info("Done!")

if __name__ == "__main__":
    main()
