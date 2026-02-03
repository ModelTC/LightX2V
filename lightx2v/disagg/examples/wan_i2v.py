import logging

import numpy as np
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from PIL import Image

from lightx2v.disagg.utils import (
    load_wan_image_encoder,
    load_wan_text_encoder,
    load_wan_vae_decoder,
    load_wan_vae_encoder,
    load_wan_transformer,
    set_config,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import save_to_video, seed_all, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE

# Setup basic logging
logging.basicConfig(level=logging.INFO)


def read_image_input(image_path):
    img_ori = Image.open(image_path).convert("RGB")
    img = TF.to_tensor(img_ori).sub_(0.5).div_(0.5).unsqueeze(0).to(AI_DEVICE)
    return img, img_ori


def get_latent_shape_with_lat_hw(config, latent_h, latent_w):
    return [
        config.get("num_channels_latents", 16),
        (config["target_video_length"] - 1) // config["vae_stride"][0] + 1,
        latent_h,
        latent_w,
    ]


def compute_latent_shape_from_image(config, image_tensor):
    h, w = image_tensor.shape[2:]
    aspect_ratio = h / w
    max_area = config["target_height"] * config["target_width"]

    latent_h = round(
        np.sqrt(max_area * aspect_ratio)
        // config["vae_stride"][1]
        // config["patch_size"][1]
        * config["patch_size"][1]
    )
    latent_w = round(
        np.sqrt(max_area / aspect_ratio)
        // config["vae_stride"][2]
        // config["patch_size"][2]
        * config["patch_size"][2]
    )
    latent_shape = get_latent_shape_with_lat_hw(config, latent_h, latent_w)
    return latent_shape, latent_h, latent_w


def get_vae_encoder_output(vae_encoder, config, first_frame, latent_h, latent_w):
    h = latent_h * config["vae_stride"][1]
    w = latent_w * config["vae_stride"][2]

    msk = torch.ones(
        1,
        config["target_video_length"],
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
            torch.zeros(3, config["target_video_length"] - 1, h, w),
        ],
        dim=1,
    ).to(AI_DEVICE)

    vae_encoder_out = vae_encoder.encode(vae_input.unsqueeze(0).to(GET_DTYPE()))
    vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(GET_DTYPE())
    return vae_encoder_out


def main():
    # 1. Configuration
    model_path = "/root/LightX2V/models/Wan-AI/Wan2.2-I2V-A14B"
    task = "i2v"
    model_cls = "wan2.2_moe"

    # Generation parameters
    seed = 42
    prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
        "Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, "
        "and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if "
        "savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details "
        "and the refreshing atmosphere of the seaside."
    )
    negative_prompt = (
        "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
        "三条腿，背景人很多，倒着走"
    )
    image_path = "/root/LightX2V/models/Wan-AI/Wan2.2-I2V-A14B/examples/i2v_input.JPG"
    save_result_path = "/root/LightX2V/save_results/wan_i2v_A14B_disagg.mp4"

    # Initialize configuration
    config = set_config(
        model_path=model_path,
        task=task,
        model_cls=model_cls,
        attn_mode="sage_attn2",
        infer_steps=40,
        target_height=480,
        target_width=832,
        target_video_length=81,
        sample_guide_scale=[3.5, 3.5],
        sample_shift=5.0,
        fps=16,
        enable_cfg=True,
        use_image_encoder=False,
        cpu_offload=True,
        offload_granularity="block",
        text_encoder_offload=True,
        image_encoder_offload=False,
        vae_offload=False,
    )

    logger.info(f"Config initialized for task: {task}")
    seed_all(seed)

    # 2. Load Models
    logger.info("Loading models...")

    text_encoders = load_wan_text_encoder(config)
    text_encoder = text_encoders[0]

    image_encoder = load_wan_image_encoder(config)

    model = load_wan_transformer(config)

    vae_encoder = load_wan_vae_encoder(config)
    vae_decoder = load_wan_vae_decoder(config)

    logger.info("Models loaded successfully.")

    # 3. Initialize Scheduler
    scheduler = WanScheduler(config)
    model.set_scheduler(scheduler)

    # 4. Run Inference Pipeline

    # 4.1 Text Encoding
    logger.info("Running text encoding...")
    text_len = config.get("text_len", 512)

    context = text_encoder.infer([prompt])
    context = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context])

    if config.get("enable_cfg", False):
        context_null = text_encoder.infer([negative_prompt])
        context_null = torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context_null])
    else:
        context_null = None

    text_encoder_output = {
        "context": context,
        "context_null": context_null,
    }

    # 4.2 Image Encoding + VAE Encoding
    logger.info("Running image encoding...")
    img, _ = read_image_input(image_path)

    if image_encoder is not None:
        clip_encoder_out = image_encoder.visual([img]).squeeze(0).to(GET_DTYPE())
    else:
        clip_encoder_out = None

    if vae_encoder is None:
        raise RuntimeError("VAE encoder is required for i2v task but was not loaded.")

    latent_shape, latent_h, latent_w = compute_latent_shape_from_image(config, img)
    vae_encoder_out = get_vae_encoder_output(vae_encoder, config, img, latent_h, latent_w)

    image_encoder_output = {
        "clip_encoder_out": clip_encoder_out,
        "vae_encoder_out": vae_encoder_out,
    }

    inputs = {
        "text_encoder_output": text_encoder_output,
        "image_encoder_output": image_encoder_output,
    }

    # 4.3 Scheduler Preparation
    logger.info("Preparing scheduler...")
    scheduler.prepare(seed=seed, latent_shape=latent_shape, image_encoder_output=image_encoder_output)

    # 4.4 Denoising Loop
    logger.info("Starting denoising loop...")
    infer_steps = scheduler.infer_steps

    for step_index in range(infer_steps):
        logger.info(f"Step {step_index + 1}/{infer_steps}")
        scheduler.step_pre(step_index=step_index)
        model.infer(inputs)
        scheduler.step_post()

    latents = scheduler.latents

    # 4.5 VAE Decoding
    logger.info("Decoding latents...")
    gen_video = vae_decoder.decode(latents.to(GET_DTYPE()))

    # 5. Post-processing and Saving
    logger.info("Post-processing video...")
    gen_video_final = wan_vae_to_comfy(gen_video)

    logger.info(f"Saving video to {save_result_path}...")
    save_to_video(gen_video_final, save_result_path, fps=config.get("fps", 16), method="ffmpeg")
    logger.info("Done!")


if __name__ == "__main__":
    main()
