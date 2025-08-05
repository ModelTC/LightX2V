import base64
import io
import signal
import sys
import threading
from datetime import datetime
from typing import List, Optional

import psutil
import torch
from PIL import Image
from loguru import logger
from pydantic import BaseModel


class ProcessManager:
    @staticmethod
    def kill_all_related_processes():
        """Kill the current process and all its child processes"""
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception as e:
                logger.info(f"Failed to kill child process {child.pid}: {e}")
        try:
            current_process.kill()
        except Exception as e:
            logger.info(f"Failed to kill main process: {e}")

    @staticmethod
    def signal_handler(sig, frame):
        logger.info("\nReceived Ctrl+C, shutting down all related processes...")
        ProcessManager.kill_all_related_processes()
        sys.exit(0)

    @staticmethod
    def register_signal_handler():
        """Register the signal handler for SIGINT"""
        signal.signal(signal.SIGINT, ProcessManager.signal_handler)


class TaskStatusMessage(BaseModel):
    task_id: str


class BaseServiceStatus:
    _lock = threading.Lock()
    _current_task = None
    _result_store = {}

    @classmethod
    def start_task(cls, message):
        with cls._lock:
            if cls._current_task is not None:
                raise RuntimeError("Service busy")
            if message.task_id_must_unique and message.task_id in cls._result_store:
                raise RuntimeError(f"Task ID {message.task_id} already exists")
            cls._current_task = {"message": message, "start_time": datetime.now()}
            return message.task_id

    @classmethod
    def complete_task(cls, message):
        with cls._lock:
            cls._result_store[message.task_id] = {"success": True, "message": message, "start_time": cls._current_task["start_time"], "completion_time": datetime.now()}
            cls._current_task = None

    @classmethod
    def record_failed_task(cls, message, error: Optional[str] = None):
        """Record a failed task with an error message."""
        with cls._lock:
            cls._result_store[message.task_id] = {"success": False, "message": message, "start_time": cls._current_task["start_time"], "error": error}
            cls._current_task = None

    @classmethod
    def clean_stopped_task(cls):
        with cls._lock:
            if cls._current_task:
                message = cls._current_task["message"]
                error = "Task stopped by user"
                cls._result_store[message.task_id] = {"success": False, "message": message, "start_time": cls._current_task["start_time"], "error": error}
                cls._current_task = None

    @classmethod
    def get_status_task_id(cls, task_id: str):
        with cls._lock:
            if cls._current_task and cls._current_task["message"].task_id == task_id:
                return {"task_status": "processing"}
            if task_id in cls._result_store:
                return {"task_status": "completed", **cls._result_store[task_id]}
            return {"task_status": "not_found"}

    @classmethod
    def get_status_service(cls):
        with cls._lock:
            if cls._current_task:
                return {"service_status": "busy", "task_id": cls._current_task["message"].task_id}
            return {"service_status": "idle"}

    @classmethod
    def get_all_tasks(cls):
        with cls._lock:
            return cls._result_store


class TensorTransporter:
    def __init__(self):
        self.buffer = io.BytesIO()

    def to_device(self, data, device):
        if isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    def prepare_tensor(self, data) -> bytes:
        self.buffer.seek(0)
        self.buffer.truncate()
        torch.save(self.to_device(data, "cpu"), self.buffer)
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_tensor(self, tensor_base64: str, device="cuda") -> torch.Tensor:
        tensor_bytes = base64.b64decode(tensor_base64)
        with io.BytesIO(tensor_bytes) as buffer:
            return self.to_device(torch.load(buffer), device)


class ImageTransporter:
    def __init__(self):
        self.buffer = io.BytesIO()

    def prepare_image(self, image: Image.Image) -> bytes:
        self.buffer.seek(0)
        self.buffer.truncate()
        image.save(self.buffer, format="PNG")
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_image(self, image_base64: bytes) -> Image.Image:
        image_bytes = base64.b64decode(image_base64)
        with io.BytesIO(image_bytes) as buffer:
            return Image.open(buffer).convert("RGB")
