import sys

from loguru import logger

from lightx2v_train.runtime.distributed import get_rank, is_main_process


def setup_logger(config=None):
    logging_config = (config or {}).get("logging", {})
    rank_zero_only = logging_config.get("rank_zero_only", True)

    logger.remove()
    logger.configure(extra={"rank": get_rank()})
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | rank={extra[rank]} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        filter=lambda record: (not rank_zero_only) or is_main_process(),
    )
