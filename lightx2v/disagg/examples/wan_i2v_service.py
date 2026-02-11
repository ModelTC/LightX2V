import logging
import os
import torch
from loguru import logger

from lightx2v.disagg.utils import set_config
from lightx2v.disagg.services.encoder import EncoderService
from lightx2v.disagg.services.transformer import TransformerService
from lightx2v.utils.utils import seed_all

# Setup basic logging
logging.basicConfig(level=logging.INFO)

def main():
    # 1. Configuration
    model_path = "/root/zht/LightX2V/models/Wan-AI/Wan2.2-I2V-A14B"
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
    image_path = "/root/zht/LightX2V/models/Wan-AI/Wan2.2-I2V-A14B/examples/i2v_input.JPG"
    save_result_path = "/root/zht/LightX2V/save_results/wan_i2v_A14B_disagg_service.mp4"

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
        data_bootstrap_addr="127.0.0.1",
        data_bootstrap_room=0,
    )

    config["image_path"] = image_path
    config["prompt"] = prompt
    config["negative_prompt"] = negative_prompt
    config["save_path"] = save_result_path

    logger.info(f"Config initialized for task: {task}")
    seed_all(seed)

    # 2. Add seed to config so services use it
    config["seed"] = seed

    # 3. Define service threads
    import threading

    def run_encoder():
        logger.info("Initializing Encoder Service...")
        encoder_service = EncoderService(config)
        logger.info("Running Encoder Service...")
        encoder_service.process()
        logger.info("Encoder Service completed.")
        encoder_service.release_memory()

    def run_transformer():
        logger.info("Initializing Transformer Service...")
        transformer_service = TransformerService(config)
        logger.info("Running Transformer Service...")
        result_path = transformer_service.process()
        logger.info(f"Video generation completed. Saved to: {result_path}")
        transformer_service.release_memory()

    # 4. Start threads
    encoder_thread = threading.Thread(target=run_encoder)
    transformer_thread = threading.Thread(target=run_transformer)

    logger.info("Starting services in separate threads...")
    encoder_thread.start()
    transformer_thread.start()

    # 5. Wait for completion
    encoder_thread.join()
    transformer_thread.join()
    logger.info("All services finished.")

if __name__ == "__main__":
    main()
