import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_queue_size: int = 10

    task_timeout: int = 300
    task_history_limit: int = 1000

    http_timeout: int = 30
    http_max_retries: int = 3

    cache_dir: str = str(Path(__file__).parent.parent / "server_cache")
    max_upload_size: int = 500 * 1024 * 1024  # 500MB
    
    lora_dir: str = None  # Directory for LoRA files

    @classmethod
    def from_env(cls) -> "ServerConfig":
        config = cls()

        if env_host := os.environ.get("LIGHTX2V_HOST"):
            config.host = env_host

        if env_port := os.environ.get("LIGHTX2V_PORT"):
            try:
                config.port = int(env_port)
            except ValueError:
                logger.warning(f"Invalid port in environment: {env_port}")

        if env_queue_size := os.environ.get("LIGHTX2V_MAX_QUEUE_SIZE"):
            try:
                config.max_queue_size = int(env_queue_size)
            except ValueError:
                logger.warning(f"Invalid max queue size: {env_queue_size}")

        # MASTER_ADDR is now managed by torchrun, no need to set manually

        if env_cache_dir := os.environ.get("LIGHTX2V_CACHE_DIR"):
            config.cache_dir = env_cache_dir

        return config

    def validate(self) -> bool:
        valid = True

        if self.max_queue_size <= 0:
            logger.error("max_queue_size must be positive")
            valid = False

        if self.task_timeout <= 0:
            logger.error("task_timeout must be positive")
            valid = False

        return valid


server_config = ServerConfig.from_env()
