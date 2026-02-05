

"""
Logging utility functions.
"""

import logging
import sys
from typing import Optional

from .distributed.basic import get_global_rank, get_local_rank, get_world_size

_default_handler = logging.StreamHandler(sys.stdout)
_default_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s "
        + (f"[Rank:{get_global_rank()}]" if get_world_size() > 1 else "")
        + (f"[LocalRank:{get_local_rank()}]" if get_world_size() > 1 else "")
        + "[%(threadName).12s][%(name)s][%(levelname).5s] "
        + "%(message)s"
    )
)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger.
    """
    logger = logging.getLogger(name)
    logger.addHandler(_default_handler)
    logger.setLevel(logging.INFO)
    return logger
