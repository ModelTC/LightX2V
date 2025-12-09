import os
import signal
import sys
from pathlib import Path

import uvicorn
from loguru import logger

from .api import ApiServer
from .config import server_config
from .services import DistributedInferenceService

_shutdown_requested = False


def run_server(args):
    global _shutdown_requested
    inference_service = None
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    def _signal_handler(signum, frame):
        global _shutdown_requested
        if _shutdown_requested:
            return
        _shutdown_requested = True
        logger.info(f"Server rank {rank} received shutdown signal")
        if inference_service:
            inference_service.stop_distributed_inference()
        os._exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        logger.info(f"Starting LightX2V server (Rank {rank}/{world_size})...")

        if hasattr(args, "host") and args.host:
            server_config.host = args.host
        if hasattr(args, "port") and args.port:
            server_config.port = args.port

        if not server_config.validate():
            raise RuntimeError("Invalid server configuration")

        inference_service = DistributedInferenceService()
        if not inference_service.start_distributed_inference(args):
            raise RuntimeError("Failed to start distributed inference service")
        logger.info(f"Rank {rank}: Inference service started successfully")

        if rank == 0:
            cache_dir = Path(server_config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            api_server = ApiServer(max_queue_size=server_config.max_queue_size)
            api_server.initialize_services(cache_dir, inference_service)

            app = api_server.get_app()

            logger.info(f"Starting FastAPI server on {server_config.host}:{server_config.port}")
            uvicorn.run(app, host=server_config.host, port=server_config.port, log_level="info")
        else:
            logger.info(f"Rank {rank}: Starting worker loop")
            import asyncio

            asyncio.run(inference_service.run_worker_loop())

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Server rank {rank} failed: {e}")
        sys.exit(1)
    finally:
        if not _shutdown_requested and inference_service:
            inference_service.stop_distributed_inference()
