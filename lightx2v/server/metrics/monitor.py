# -*-coding=utf-8-*-
import threading

from .metrics import MetricsClient


class Monitor(MetricsClient):
    _instance = None
    _lock = threading.Lock()
    _initialized = False  # 添加初始化标志

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        if not self.__class__._initialized:
            super().__init__(*args, **kwargs)
            self.__class__._initialized = True
