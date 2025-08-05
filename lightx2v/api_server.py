import argparse
import atexit
import signal
import sys
from pathlib import Path

import uvicorn
from loguru import logger

from lightx2v.server.api import ApiServer
from lightx2v.server.service import DistributedInferenceService


def create_signal_handler(inference_service: DistributedInferenceService):
    """Create unified signal handler function"""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, gracefully shutting down...")
        try:
            if inference_service.is_running:
                inference_service.stop_distributed_inference()
        except Exception as e:
            logger.error(f"Error occurred while shutting down distributed inference service: {str(e)}")
        finally:
            sys.exit(0)

    return signal_handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_cls",
        type=str,
        required=True,
        choices=[
            "wan2.1",
            "hunyuan",
            "wan2.1_distill",
            "wan2.1_causvid",
            "wan2.1_skyreels_v2_df",
            "wan2.1_audio",
            "wan2.2_moe",
            "wan2.2_moe_distill",
        ],
        default="wan2.1",
    )
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)

    parser.add_argument("--split", action="store_true")
    parser.add_argument("--lora_path", type=str, required=False, default=None)
    parser.add_argument("--lora_strength", type=float, default=1.0, help="The strength for the lora (default: 1.0)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node for distributed inference")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    cache_dir = Path(__file__).parent.parent / "server_cache"
    inference_service = DistributedInferenceService()

    api_server = ApiServer()
    api_server.initialize_services(cache_dir, inference_service)

    signal_handler = create_signal_handler(inference_service)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Starting distributed inference service...")
    success = inference_service.start_distributed_inference(args)
    if not success:
        logger.error("Failed to start distributed inference service, exiting program")
        sys.exit(1)

    atexit.register(inference_service.stop_distributed_inference)

    try:
        logger.info(f"Starting FastAPI server on port: {args.port}")
        uvicorn.run(
            api_server.get_app(),
            host="0.0.0.0",
            port=args.port,
            reload=False,
            workers=1,
        )
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down service...")
    except Exception as e:
        logger.error(f"Error occurred while running FastAPI server: {str(e)}")
    finally:
        inference_service.stop_distributed_inference()


if __name__ == "__main__":
    main()
