from loguru import logger
from prometheus_client import Counter, Gauge, Summary, generate_latest
from prometheus_client.core import CollectorRegistry

from lightx2v.deploy.task_manager import ActiveStatus, FinishedStatus, TaskStatus

REGISTRY = CollectorRegistry()


class MetricMonitor:
    def __init__(self):
        self.task_all = Counter("task_all_total", "Total count of all tasks", ["task_type", "model_cls", "stage"], registry=REGISTRY)
        self.task_end = Counter("task_end_total", "Total count of ended tasks", ["task_type", "model_cls", "stage", "status"], registry=REGISTRY)
        self.task_active = Gauge("task_active_size", "Current count of active tasks", ["task_type", "model_cls", "stage"], registry=REGISTRY)
        self.task_elapse = Summary("task_elapse_seconds", "Elapse time of tasks", ["task_type", "model_cls", "stage", "end_status"], registry=REGISTRY)
        self.subtask_all = Counter("subtask_all_total", "Total count of all subtasks", ["queue"], registry=REGISTRY)
        self.subtask_end = Counter("subtask_end_total", "Total count of ended subtasks", ["queue", "status"], registry=REGISTRY)
        self.subtask_active = Gauge("subtask_active_size", "Current count of active subtasks", ["queue", "status"], registry=REGISTRY)
        self.subtask_elapse = Summary("subtask_elapse_seconds", "Elapse time of subtasks", ["queue", "elapse_key"], registry=REGISTRY)

    def record_task_start(self, task):
        self.task_all.labels(task["task_type"], task["model_cls"], task["stage"]).inc()
        self.task_active.labels(task["task_type"], task["model_cls"], task["stage"]).inc()
        logger.info(f"Metrics task_all + 1, task_active +1")

    def record_task_end(self, task, status, elapse):
        self.task_end.labels(task["task_type"], task["model_cls"], task["stage"], status.name).inc()
        self.task_active.labels(task["task_type"], task["model_cls"], task["stage"]).dec()
        self.task_elapse.labels(task["task_type"], task["model_cls"], task["stage"], status.name).observe(elapse)
        logger.info(f"Metrics task_end + 1, task_active -1, task_elapse observe {elapse}")

    def record_subtask_change(self, subtask, old_status, new_status, elapse_key, elapse):
        if old_status in ActiveStatus and new_status in FinishedStatus:
            self.subtask_end.labels(subtask["queue"], elapse_key).inc()
            logger.info(f"Metrics subtask_end + 1")
        if old_status in ActiveStatus:
            self.subtask_active.labels(subtask["queue"], old_status.name).dec()
            logger.info(f"Metrics subtask_active {old_status.name} -1")
        if new_status in ActiveStatus:
            self.subtask_active.labels(subtask["queue"], new_status.name).inc()
            logger.info(f"Metrics subtask_active {new_status.name} + 1")
        if new_status == TaskStatus.CREATED:
            self.subtask_all.labels(subtask["queue"]).inc()
            logger.info(f"Metrics subtask_all + 1")
        if elapse and elapse_key:
            self.subtask_elapse.labels(subtask["queue"], elapse_key).observe(elapse)
            logger.info(f"Metrics subtask_elapse observe {elapse}")

    # restart server, we should recover active tasks in data_manager
    def record_task_recover(self, tasks):
        for task in tasks:
            if task["status"] in ActiveStatus:
                self.record_task_start(task)

    # restart server, we should recover active tasks in data_manager
    def record_subtask_recover(self, subtasks):
        for subtask in subtasks:
            if subtask["status"] in ActiveStatus:
                self.subtask_all.labels(subtask["queue"]).inc()
                self.subtask_active.labels(subtask["queue"], subtask["status"].name).inc()
                logger.info(f"Metrics subtask_active {subtask['status'].name} + 1")
                logger.info(f"Metrics subtask_all + 1")

    def get_metrics(self):
        return generate_latest(REGISTRY)
