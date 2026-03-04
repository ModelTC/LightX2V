class BaseQueueManager:
    def __init__(self):
        pass

    async def init(self):
        pass

    async def close(self):
        pass

    async def put_subtask(self, subtask):
        raise NotImplementedError

    async def get_subtasks(self, queue, max_batch, timeout):
        raise NotImplementedError

    async def pending_num(self, queue):
        raise NotImplementedError


# Import queue manager implementations
from .local_queue_manager import LocalQueueManager  # noqa
from .rabbitmq_queue_manager import RabbitMQQueueManager  # noqa

__all__ = ["BaseQueueManager", "LocalQueueManager", "RabbitMQQueueManager"]
