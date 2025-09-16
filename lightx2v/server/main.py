import os
import sys
from pathlib import Path

import uvicorn
from loguru import logger

from .api import ApiServer
from .config import server_config
from .service import DistributedInferenceService


def run_server(args):
    """Run server with torchrun support"""
    inference_service = None
    try:
        # Get rank from environment (set by torchrun)
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        logger.info(f"Starting LightX2V server (Rank {rank}/{world_size})...")

        if hasattr(args, "host") and args.host:
            server_config.host = args.host
        if hasattr(args, "port") and args.port:
            server_config.port = args.port

        if not server_config.validate():
            raise RuntimeError("Invalid server configuration")

        # Initialize inference service
        inference_service = DistributedInferenceService()
        if not inference_service.start_distributed_inference(args):
            raise RuntimeError("Failed to start distributed inference service")
        logger.info(f"Rank {rank}: Inference service started successfully")

        if rank == 0:
            # Only rank 0 runs the FastAPI server
            cache_dir = Path(server_config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            api_server = ApiServer(max_queue_size=server_config.max_queue_size)
            api_server.initialize_services(cache_dir, inference_service)

            app = api_server.get_app()

            logger.info(f"Starting FastAPI server on {server_config.host}:{server_config.port}")
            uvicorn.run(app, host=server_config.host, port=server_config.port, log_level="info")
        else:
            # Non-rank-0 processes run the worker loop
            logger.info(f"Rank {rank}: Starting worker loop")
            import asyncio

            asyncio.run(inference_service.run_worker_loop())

    except KeyboardInterrupt:
        logger.info(f"Server rank {rank} interrupted by user")
        if inference_service:
            inference_service.stop_distributed_inference()
    except Exception as e:
        logger.error(f"Server rank {rank} failed: {e}")
        if inference_service:
            inference_service.stop_distributed_inference()
        sys.exit(1)
