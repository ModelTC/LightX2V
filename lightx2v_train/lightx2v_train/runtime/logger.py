import os
import sys
from pathlib import Path

from loguru import logger

from lightx2v_train.runtime.distributed import (
    get_rank,
    get_world_size,
    is_main_process,
)


def setup_logger(config=None):
    config = config or {}
    logging_config = config.get("logging", {})
    rank_zero_only = logging_config.get("rank_zero_only", True)
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | rank={extra[rank]} | {name}:{function}:{line} - {message}"
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | rank={extra[rank]} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    def rank_filter(record):
        return (not rank_zero_only) or is_main_process()

    logger.remove()
    logger.configure(extra={"rank": get_rank()})
    logger.add(
        sys.stderr,
        format=console_format,
        filter=rank_filter,
    )

    output_dir = config.get("training", {}).get("output_dir")
    if not output_dir:
        return
    if rank_zero_only and not is_main_process():
        return

    log_dir = Path(os.path.expanduser(str(output_dir)))
    log_dir.mkdir(parents=True, exist_ok=True)
    file_name = str(logging_config.get("file_name", "train.log"))
    if not rank_zero_only and get_world_size() > 1:
        file_path = Path(file_name)
        file_name = f"{file_path.stem}.rank-{get_rank()}{file_path.suffix}"
    log_path = log_dir / file_name
    logger.add(
        str(log_path),
        format=log_format,
        filter=rank_filter,
        level=logging_config.get("level", "DEBUG"),
        mode="a",
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
    )
    logger.info("Writing training logs to {}", log_path)
