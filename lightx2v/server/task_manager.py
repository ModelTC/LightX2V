import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from loguru import logger


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    task_id: str
    status: TaskStatus
    message: Any
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    save_result_path: Optional[str] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None


class TaskManager:
    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size

        self._tasks: OrderedDict[str, TaskInfo] = OrderedDict()
        self._lock = threading.RLock()

        self._processing_lock = threading.Lock()
        self._current_processing_task: Optional[str] = None

        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0

    def create_task(self, message: Any) -> str:
        with self._lock:
            if hasattr(message, "task_id") and message.task_id in self._tasks:
                raise RuntimeError(f"Task ID {message.task_id} already exists")

            active_tasks = sum(1 for t in self._tasks.values() if t.status in [TaskStatus.PENDING, TaskStatus.PROCESSING])
            if active_tasks >= self.max_queue_size:
                raise RuntimeError(f"Task queue is full (max {self.max_queue_size} tasks)")

            task_id = getattr(message, "task_id", str(uuid.uuid4()))
            task_info = TaskInfo(task_id=task_id, status=TaskStatus.PENDING, message=message, save_result_path=getattr(message, "save_result_path", None))

            self._tasks[task_id] = task_info
            self.total_tasks += 1

            self._cleanup_old_tasks()

            return task_id

    def start_task(self, task_id: str) -> TaskInfo:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task {task_id} not found")

            task = self._tasks[task_id]
            task.status = TaskStatus.PROCESSING
            task.start_time = datetime.now()

            self._tasks.move_to_end(task_id)

            return task

    def complete_task(self, task_id: str, save_result_path: Optional[str] = None):
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found for completion")
                return

            task = self._tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            if save_result_path:
                task.save_result_path = save_result_path

            self.completed_tasks += 1

    def fail_task(self, task_id: str, error: str):
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found for failure")
                return

            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            task.error = error

            self.failed_tasks += 1

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False

            task.stop_event.set()
            task.status = TaskStatus.CANCELLED
            task.end_time = datetime.now()
            task.error = "Task cancelled by user"

            if task.thread and task.thread.is_alive():
                task.thread.join(timeout=5)

            return True

    def cancel_all_tasks(self):
        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
                    self.cancel_task(task_id)

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.get_task(task_id)
        if not task:
            return None

        return {"task_id": task.task_id, "status": task.status.value, "start_time": task.start_time, "end_time": task.end_time, "error": task.error, "save_result_path": task.save_result_path}

    def get_all_tasks(self):
        with self._lock:
            return {task_id: self.get_task_status(task_id) for task_id in self._tasks}

    def get_active_task_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status in [TaskStatus.PENDING, TaskStatus.PROCESSING])

    def get_pending_task_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

    def is_processing(self) -> bool:
        with self._lock:
            return self._current_processing_task is not None

    def acquire_processing_lock(self, task_id: str, timeout: Optional[float] = None) -> bool:
        acquired = self._processing_lock.acquire(timeout=timeout if timeout else False)
        if acquired:
            with self._lock:
                self._current_processing_task = task_id
                logger.info(f"Task {task_id} acquired processing lock")
        return acquired

    def release_processing_lock(self, task_id: str):
        with self._lock:
            if self._current_processing_task == task_id:
                self._current_processing_task = None
                try:
                    self._processing_lock.release()
                    logger.info(f"Task {task_id} released processing lock")
                except RuntimeError as e:
                    logger.warning(f"Task {task_id} tried to release lock but failed: {e}")

    def get_next_pending_task(self) -> Optional[str]:
        with self._lock:
            for task_id, task in self._tasks.items():
                if task.status == TaskStatus.PENDING:
                    return task_id
        return None

    def get_service_status(self) -> Dict[str, Any]:
        with self._lock:
            active_tasks = [task_id for task_id, task in self._tasks.items() if task.status == TaskStatus.PROCESSING]

            pending_count = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

            return {
                "service_status": "busy" if self._current_processing_task else "idle",
                "current_task": self._current_processing_task,
                "active_tasks": active_tasks,
                "pending_tasks": pending_count,
                "queue_size": self.max_queue_size,
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
            }

    def _cleanup_old_tasks(self, keep_count: int = 1000):
        if len(self._tasks) <= keep_count:
            return

        completed_tasks = [(task_id, task) for task_id, task in self._tasks.items() if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]

        completed_tasks.sort(key=lambda x: x[1].end_time or x[1].start_time)

        remove_count = len(self._tasks) - keep_count
        for task_id, _ in completed_tasks[:remove_count]:
            del self._tasks[task_id]
            logger.debug(f"Cleaned up old task: {task_id}")


task_manager = TaskManager()
