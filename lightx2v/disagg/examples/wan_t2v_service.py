import logging
import threading

from loguru import logger

from lightx2v.disagg.services.controller import ControllerService
from lightx2v.disagg.services.decoder import DecoderService
from lightx2v.disagg.services.encoder import EncoderService
from lightx2v.disagg.services.transformer import TransformerService
from lightx2v.disagg.utils import set_config
from lightx2v.utils.utils import seed_all

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
    negative_prompt = (
        "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    # Initialize configuration (same as wan_t2v.py)
    config = set_config(
        model_path=model_path,
        task=task,
        model_cls=model_cls,
        attn_mode="sage_attn2",
        infer_steps=50,
        target_height=480,
        target_width=832,
        target_video_length=81,
        sample_guide_scale=5.0,
        sample_shift=5.0,
        fps=16,
        enable_cfg=True,
        data_bootstrap_addr="127.0.0.1",
        data_bootstrap_room=0,
        encoder_engine_rank=0,
        transformer_engine_rank=1,
        decoder_engine_rank=2,
    )

    logger.info(f"Config initialized for task: {task}")
    seed_all(seed)

    # Add seed into config so services can use it if needed
    config["seed"] = seed
    config["prompt"] = prompt
    config["negative_prompt"] = negative_prompt
    config["save_path"] = save_result_path

    encoder_stop_event = threading.Event()
    transformer_stop_event = threading.Event()
    decoder_stop_event = threading.Event()
    
    # 2. Define service threads
    def run_encoder():
        logger.info("Initializing Encoder Service...")
        encoder_service = EncoderService()
        logger.info("Running Encoder Service...")
        encoder_service.exec_request(stop_event=encoder_stop_event)
        logger.info("Encoder Service completed.")

    def run_transformer():
        logger.info("Initializing Transformer Service...")
        transformer_service = TransformerService()
        logger.info("Running Transformer Service...")
        transformer_service.exec_request(stop_event=transformer_stop_event)
        logger.info("Transformer Service completed.")

    def run_decoder():
        logger.info("Initializing Decoder Service...")
        decoder_service = DecoderService()
        logger.info("Running Decoder Service...")
        decoder_service.exec_request(stop_event=decoder_stop_event)
        logger.info("Video generation completed.")

    def run_controller():
        logger.info("Initializing Controller Service...")
        controller_service = ControllerService()
        logger.info("Dispatching request to services...")
        controller_service.add_request(config)
        encoder_stop_event.set()
        transformer_stop_event.set()
        decoder_stop_event.set()
        logger.info("Controller Service completed.")

    # 3. Start threads
    encoder_thread = threading.Thread(target=run_encoder)
    transformer_thread = threading.Thread(target=run_transformer)
    decoder_thread = threading.Thread(target=run_decoder)
    controller_thread = threading.Thread(target=run_controller)

    logger.info("Starting services in separate threads...")
    encoder_thread.start()
    transformer_thread.start()
    decoder_thread.start()
    controller_thread.start()

    # 4. Wait for completion
    encoder_thread.join()
    transformer_thread.join()
    decoder_thread.join()
    controller_thread.join()
    logger.info("All services finished.")


if __name__ == "__main__":
    main()
