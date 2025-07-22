import os
from functools import lru_cache


@lru_cache(maxsize=None)
def CHECK_ENABLE_PROFILING_DEBUG():
    ENABLE_PROFILING_DEBUG = os.getenv("ENABLE_PROFILING_DEBUG", "false").lower() == "true"
    return ENABLE_PROFILING_DEBUG


@lru_cache(maxsize=None)
def CHECK_ENABLE_GRAPH_MODE():
    ENABLE_GRAPH_MODE = os.getenv("ENABLE_GRAPH_MODE", "false").lower() == "true"
    return ENABLE_GRAPH_MODE


@lru_cache(maxsize=None)
def GET_RUNNING_FLAG():
    RUNNING_FLAG = os.getenv("RUNNING_FLAG", "infer")
    return RUNNING_FLAG


@lru_cache(maxsize=None)
def GET_DTYPE():
    RUNNING_FLAG = os.getenv("DTYPE")
    return RUNNING_FLAG
