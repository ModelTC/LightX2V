import asyncio
import json
import time

from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.deploy.server.monitor import ServerMonitor, WorkerStatus
from lightx2v.deploy.server.redis_client import RedisClient


class RedisServerMonitor(ServerMonitor):
    def __init__(self, model_pipelines, task_manager, queue_manager, redis_url):
        super().__init__(model_pipelines, task_manager, queue_manager)
        self.redis_url = redis_url
        self.redis_client = RedisClient(redis_url)
        self.last_correct = None
        self.correct_interval = 60 * 60 * 24

    async def init(self):
        await self.redis_client.init()
        await self.init_pending_subtasks()

    async def loop(self):
        while True:
            if self.stop:
                break
            if self.last_correct is None or time.time() - self.last_correct > self.correct_interval:
                self.last_correct = time.time()
                await self.correct_pending_info()
            await self.clean_workers()
            await self.clean_subtasks()
            await asyncio.sleep(self.interval)
        logger.info("RedisServerMonitor stopped")

    async def close(self):
        await super().close()
        await self.redis_client.close()

    @class_try_catch_async
    async def worker_update(self, queue, identity, status):
        status = status.name
        key = f"workers:{queue}:workers"
        infer_key = f"workers:{queue}:infer_cost"

        update_t = time.time()
        worker = await self.redis_client.hget(key, identity)
        if worker is None:
            worker = {"status": "", "fetched_t": 0, "update_t": update_t}
            await self.redis_client.hset(key, identity, json.dumps(worker))
        else:
            worker = json.loads(worker)

        pre_status = worker["status"]
        pre_fetched_t = float(worker["fetched_t"])
        worker["status"] = status
        worker["update_t"] = update_t

        if status == WorkerStatus.REPORT.name and pre_fetched_t > 0:
            cur_cost = update_t - pre_fetched_t
            worker["fetched_t"] = 0.0
            if cur_cost < self.subtask_run_timeouts[queue]:
                await self.redis_client.list_push(infer_key, max(cur_cost, 1), self.worker_avg_window)
                logger.info(f"Worker {identity} {queue} avg infer cost update: {cur_cost:.2f} s")

        elif status == WorkerStatus.FETCHED.name:
            worker["fetched_t"] = update_t

        await self.redis_client.hset(key, identity, json.dumps(worker))
        logger.info(f"Worker {identity} {queue} update [{status}]")

    @class_try_catch_async
    async def clean_workers(self):
        for queue in self.all_queues:
            key = f"workers:{queue}:workers"
            workers = await self.redis_client.hgetall(key)

            for identity, worker in workers.items():
                worker = json.loads(worker)
                fetched_t = float(worker["fetched_t"])
                update_t = float(worker["update_t"])
                status = worker["status"]
                # logger.warning(f"{queue} avg infer cost {infer_avg:.2f} s, worker: {worker}")

                # infer too long
                if fetched_t > 0:
                    elapse = time.time() - fetched_t
                    if elapse > self.subtask_run_timeouts[queue]:
                        logger.warning(f"Worker {identity} {queue} infer timeout2: {elapse:.2f} s")
                        await self.redis_client.hdel(key, identity)
                        continue

                elapse = time.time() - update_t
                # no ping too long
                if status in [WorkerStatus.FETCHED.name, WorkerStatus.PING.name]:
                    if elapse > self.ping_timeout:
                        logger.warning(f"Worker {identity} {queue} ping timeout: {elapse:.2f} s")
                        await self.redis_client.hdel(key, identity)
                        continue

                # offline too long
                elif status in [WorkerStatus.DISCONNECT.name, WorkerStatus.REPORT.name]:
                    if elapse > self.worker_offline_timeout:
                        logger.warning(f"Worker {identity} {queue} offline timeout2: {elapse:.2f} s")
                        await self.redis_client.hdel(key, identity)
                        continue

                # fetching too long
                elif status == WorkerStatus.FETCHING.name:
                    if elapse > self.fetching_timeout:
                        logger.warning(f"Worker {identity} {queue} fetching timeout: {elapse:.2f} s")
                        await self.redis_client.hdel(key, identity)
                        continue

    async def get_ready_worker_count(self, queue):
        key = f"workers:{queue}:workers"
        worker_count = await self.redis_client.hlen(key)
        return worker_count

    async def get_avg_worker_infer_cost(self, queue):
        infer_key = f"workers:{queue}:infer_cost"
        infer_cost = await self.redis_client.list_avg(infer_key, self.worker_avg_window)
        if infer_cost < 0:
            return self.subtask_run_timeouts[queue]
        return infer_cost

    async def correct_pending_info(self):
        for queue in self.all_queues:
            pending_num = await self.queue_manager.pending_num(queue)
            await self.redis_client.correct_pending_info(f"pendings:{queue}:info", pending_num)

    @class_try_catch_async
    async def init_pending_subtasks(self):
        await super().init_pending_subtasks()
        # save to redis if not exists
        for queue, v in self.pending_subtasks.items():
            subtasks = v.pop("subtasks", {})
            await self.redis_client.create_if_not_exists(f"pendings:{queue}:info", v)
            for task_id, order_id in subtasks.items():
                await self.redis_client.set(f"pendings:{queue}:subtasks:{task_id}", order_id, nx=True)
        self.pending_subtasks = None
        logger.info(f"Inited pending subtasks to redis")

    @class_try_catch_async
    async def pending_subtasks_add(self, queue, task_id):
        max_count = await self.redis_client.increment_and_get(f"pendings:{queue}:info", "max_count", 1)
        await self.redis_client.set(f"pendings:{queue}:subtasks:{task_id}", max_count)
        # logger.warning(f"Redis pending subtasks {queue} add {task_id}: {max_count}")

    @class_try_catch_async
    async def pending_subtasks_sub(self, queue, task_id):
        consume_count = await self.redis_client.increment_and_get(f"pendings:{queue}:info", "consume_count", 1)
        await self.redis_client.delete_key(f"pendings:{queue}:subtasks:{task_id}")
        # logger.warning(f"Redis pending subtasks {queue} sub {task_id}: {consume_count}")

    @class_try_catch_async
    async def pending_subtasks_get_order(self, queue, task_id):
        order = await self.redis_client.get(f"pendings:{queue}:subtasks:{task_id}")
        if order is None:
            return None
        consume = await self.redis_client.hget(f"pendings:{queue}:info", "consume_count")
        if consume is None:
            return None
        real_order = max(int(order) - int(consume), 1)
        # logger.warning(f"Redis pending subtasks {queue} get order {task_id}: real={real_order} order={order} consume={consume}")
        return real_order
