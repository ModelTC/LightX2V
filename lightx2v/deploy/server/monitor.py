import asyncio
import time
from enum import Enum

from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.deploy.task_manager import TaskStatus


class WorkerStatus(Enum):
    FETCHING = 1
    FETCHED = 2
    DISCONNECT = 3
    REPORT = 4
    PING = 5


class CostWindow:
    def __init__(self, window):
        self.window = window
        self.costs = []
        self.avg = None

    def append(self, cost):
        self.costs.append(cost)
        if len(self.costs) > self.window:
            self.costs.pop(0)
        self.avg = sum(self.costs) / len(self.costs)


class WorkerClient:
    def __init__(self, queue, identity, infer_timeout, offline_timeout, avg_window, ping_timeout, fetching_timeout):
        self.queue = queue
        self.identity = identity
        self.status = None
        self.update_t = time.time()
        self.fetched_t = None
        self.infer_cost = CostWindow(avg_window)
        self.offline_cost = CostWindow(avg_window)
        self.infer_timeout = infer_timeout
        self.offline_timeout = offline_timeout
        self.ping_timeout = ping_timeout
        self.fetching_timeout = fetching_timeout

    # FETCHING -> FETCHED -> PING * n -> REPORT -> FETCHING
    # FETCHING -> DISCONNECT -> FETCHING
    def update(self, status: WorkerStatus):
        pre_status = self.status
        pre_t = self.update_t
        self.status = status
        self.update_t = time.time()

        if status == WorkerStatus.FETCHING:
            if pre_status in [WorkerStatus.DISCONNECT, WorkerStatus.REPORT] and pre_t is not None:
                cur_cost = self.update_t - pre_t
                if cur_cost < self.offline_timeout:
                    self.offline_cost.append(max(cur_cost, 1))

        elif status == WorkerStatus.REPORT:
            if self.fetched_t is not None:
                cur_cost = self.update_t - self.fetched_t
                self.fetched_t = None
                if cur_cost < self.infer_timeout:
                    self.infer_cost.append(max(cur_cost, 1))
                    logger.info(f"Worker {self.identity} {self.queue} avg infer cost update: {self.infer_cost.avg:.2f} s")

        elif status == WorkerStatus.FETCHED:
            self.fetched_t = time.time()

    def check(self):
        # infer too long
        if self.fetched_t is not None:
            elapse = time.time() - self.fetched_t
            if self.infer_cost.avg is not None and elapse > self.infer_cost.avg * 5:
                logger.warning(f"Worker {self.identity} {self.queue} infer timeout: {elapse:.2f} s")
                return False
            if elapse > self.infer_timeout:
                logger.warning(f"Worker {self.identity} {self.queue} infer timeout2: {elapse:.2f} s")
                return False

        elapse = time.time() - self.update_t
        # no ping too long
        if self.status in [WorkerStatus.FETCHED, WorkerStatus.PING]:
            if elapse > self.ping_timeout:
                logger.warning(f"Worker {self.identity} {self.queue} ping timeout: {elapse:.2f} s")
                return False
        # offline too long
        elif self.status in [WorkerStatus.DISCONNECT, WorkerStatus.REPORT]:
            if self.offline_cost.avg is not None and elapse > self.offline_cost.avg * 5:
                logger.warning(f"Worker {self.identity} {self.queue} offline timeout: {elapse:.2f} s")
                return False
            if elapse > self.offline_timeout:
                logger.warning(f"Worker {self.identity} {self.queue} offline timeout2: {elapse:.2f} s")
                return False
        # fetching too long
        elif self.status == WorkerStatus.FETCHING:
            if elapse > self.fetching_timeout:
                logger.warning(f"Worker {self.identity} {self.queue} fetching timeout: {elapse:.2f} s")
                return False
        return True


class ServerMonitor:
    def __init__(self, model_pipelines, task_manager, queue_manager):
        self.model_pipelines = model_pipelines
        self.task_manager = task_manager
        self.queue_manager = queue_manager
        self.stop = False
        self.worker_clients = {}
        self.subtask_run_timeouts = {}
        self.pending_subtasks = {}

        self.all_queues = self.model_pipelines.get_queues()
        self.config = self.model_pipelines.get_monitor_config()
        self.interval = self.config.get("monitor_interval", 30)
        self.fetching_timeout = self.config.get("fetching_timeout", 1000)

        for queue in self.all_queues:
            self.subtask_run_timeouts[queue] = self.config["subtask_running_timeouts"].get(queue, 3600)
        self.subtask_created_timeout = self.config["subtask_created_timeout"]
        self.subtask_pending_timeout = self.config["subtask_pending_timeout"]
        self.worker_avg_window = self.config["worker_avg_window"]
        self.worker_offline_timeout = self.config["worker_offline_timeout"]
        self.worker_min_capacity = self.config["worker_min_capacity"]
        self.task_timeout = self.config["task_timeout"]
        self.ping_timeout = self.config["ping_timeout"]

        self.user_visits = {}  # user_id -> last_visit_t
        self.user_max_active_tasks = self.config["user_max_active_tasks"]
        self.user_max_daily_tasks = self.config["user_max_daily_tasks"]
        self.user_visit_frequency = self.config["user_visit_frequency"]

        assert self.worker_avg_window > 0
        assert self.worker_offline_timeout > 0
        assert self.worker_min_capacity > 0
        assert self.task_timeout > 0
        assert self.ping_timeout > 0
        assert self.user_max_active_tasks > 0
        assert self.user_max_daily_tasks > 0
        assert self.user_visit_frequency > 0

    async def init(self):
        await self.init_pending_subtasks()

    async def loop(self):
        while True:
            if self.stop:
                break
            await self.clean_workers()
            await self.clean_subtasks()
            await asyncio.sleep(self.interval)
        logger.info("ServerMonitor stopped")

    async def close(self):
        self.stop = True
        self.model_pipelines = None
        self.task_manager = None
        self.queue_manager = None
        self.worker_clients = None

    def init_worker(self, queue, identity):
        if queue not in self.worker_clients:
            self.worker_clients[queue] = {}
        if identity not in self.worker_clients[queue]:
            infer_timeout = self.subtask_run_timeouts[queue]
            self.worker_clients[queue][identity] = WorkerClient(queue, identity, infer_timeout, self.worker_offline_timeout, self.worker_avg_window, self.ping_timeout, self.fetching_timeout)
        return self.worker_clients[queue][identity]

    @class_try_catch_async
    async def worker_update(self, queue, identity, status):
        worker = self.init_worker(queue, identity)
        worker.update(status)
        logger.info(f"Worker {identity} {queue} update [{status}]")

    @class_try_catch_async
    async def clean_workers(self):
        qs = list(self.worker_clients.keys())
        for queue in qs:
            idens = list(self.worker_clients[queue].keys())
            for identity in idens:
                if not self.worker_clients[queue][identity].check():
                    self.worker_clients[queue].pop(identity)
                    logger.warning(f"Worker {queue} {identity} out of contact removed, remain {self.worker_clients[queue]}")

    @class_try_catch_async
    async def clean_subtasks(self):
        created_end_t = time.time() - self.subtask_created_timeout
        pending_end_t = time.time() - self.subtask_pending_timeout
        ping_end_t = time.time() - self.ping_timeout
        fails = set()

        created_tasks = await self.task_manager.list_tasks(status=TaskStatus.CREATED, subtasks=True, end_updated_t=created_end_t)
        pending_tasks = await self.task_manager.list_tasks(status=TaskStatus.PENDING, subtasks=True, end_updated_t=pending_end_t)

        def fmt_subtask(t):
            return f"({t['task_id']}, {t['worker_name']}, {t['queue']}, {t['worker_identity']})"

        for t in created_tasks + pending_tasks:
            if t["task_id"] in fails:
                continue
            elapse = time.time() - t["update_t"]
            logger.warning(f"Subtask {fmt_subtask(t)} CREATED / PENDING timeout: {elapse:.2f} s")
            await self.task_manager.finish_subtasks(t["task_id"], TaskStatus.FAILED, worker_name=t["worker_name"], fail_msg=f"CREATED / PENDING timeout: {elapse:.2f} s")
            fails.add(t["task_id"])

        running_tasks = await self.task_manager.list_tasks(status=TaskStatus.RUNNING, subtasks=True)

        for t in running_tasks:
            if t["task_id"] in fails:
                continue
            if t["ping_t"] > 0:
                ping_elapse = time.time() - t["ping_t"]
                if ping_elapse >= self.ping_timeout:
                    logger.warning(f"Subtask {fmt_subtask(t)} PING timeout: {ping_elapse:.2f} s")
                    await self.task_manager.finish_subtasks(t["task_id"], TaskStatus.FAILED, worker_name=t["worker_name"], fail_msg=f"PING timeout: {ping_elapse:.2f} s")
                    fails.add(t["task_id"])
            elapse = time.time() - t["update_t"]
            limit = self.subtask_run_timeouts[t["queue"]]
            if elapse >= limit:
                logger.warning(f"Subtask {fmt_subtask(t)} RUNNING timeout: {elapse:.2f} s")
                await self.task_manager.finish_subtasks(t["task_id"], TaskStatus.FAILED, worker_name=t["worker_name"], fail_msg=f"RUNNING timeout: {elapse:.2f} s")
                fails.add(t["task_id"])

    @class_try_catch_async
    async def get_avg_worker_infer_cost(self, queue):
        if queue not in self.worker_clients:
            self.worker_clients[queue] = {}
        infer_costs = []
        for _, client in self.worker_clients[queue].items():
            if client.infer_cost.avg is not None:
                infer_costs.append(client.infer_cost.avg)
        if len(infer_costs) <= 0:
            return self.subtask_run_timeouts[queue]
        return sum(infer_costs) / len(infer_costs)

    @class_try_catch_async
    async def check_user_busy(self, user_id, active_new_task=False):
        # check if user visit too frequently
        cur_t = time.time()
        if user_id in self.user_visits:
            elapse = cur_t - self.user_visits[user_id]
            if elapse <= self.user_visit_frequency:
                return f"User {user_id} visit too frequently, {elapse:.2f} s vs {self.user_visit_frequency:.2f} s"
        self.user_visits[user_id] = cur_t

        if active_new_task:
            # check if user has too many active tasks
            active_statuses = [TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.CREATED]
            active_tasks = await self.task_manager.list_tasks(status=active_statuses, user_id=user_id)
            if len(active_tasks) >= self.user_max_active_tasks:
                return f"User {user_id} has too many active tasks, {len(active_tasks)} vs {self.user_max_active_tasks}"

            # check if user has too many daily tasks
            daily_statuses = active_statuses + [TaskStatus.SUCCEED, TaskStatus.CANCEL, TaskStatus.FAILED]
            daily_tasks = await self.task_manager.list_tasks(status=daily_statuses, user_id=user_id, start_created_t=cur_t - 86400, include_delete=True)
            if len(daily_tasks) >= self.user_max_daily_tasks:
                return f"User {user_id} has too many daily tasks, {len(daily_tasks)} vs {self.user_max_daily_tasks}"

        return True

    # check if a task can be published to queues
    @class_try_catch_async
    async def check_queue_busy(self, keys, queues):
        wait_time = 0

        for queue in queues:
            avg_cost = await self.get_avg_worker_infer_cost(queue)
            worker_cnt = await self.get_ready_worker_count(queue)
            subtask_pending = await self.queue_manager.pending_num(queue)
            capacity = self.task_timeout * max(worker_cnt, 1) // avg_cost
            capacity = max(self.worker_min_capacity, capacity)

            if subtask_pending >= capacity:
                ss = f"pending={subtask_pending}, capacity={capacity}"
                logger.warning(f"Queue {queue} busy, {ss}, task {keys} cannot be publised!")
                return None
            wait_time += avg_cost * subtask_pending / max(worker_cnt, 1)
        return wait_time

    @class_try_catch_async
    async def init_pending_subtasks(self):
        # query all pending subtasks in task_manager
        subtasks = {}
        rows = await self.task_manager.list_tasks(status=TaskStatus.PENDING, subtasks=True, sort_by_update_t=True)
        for row in rows:
            if row["queue"] not in subtasks:
                subtasks[row["queue"]] = []
            subtasks[row["queue"]].append(row["task_id"])
        for queue in self.all_queues:
            if queue not in subtasks:
                subtasks[queue] = []

        # self.pending_subtasks = {queue: {"consume_count": int, "max_count": int, subtasks: {task_id: order}}
        for queue, task_ids in subtasks.items():
            pending_num = await self.queue_manager.pending_num(queue)
            self.pending_subtasks[queue] = {"consume_count": 0, "max_count": pending_num, "subtasks": {}}
            for i, task_id in enumerate(task_ids):
                self.pending_subtasks[queue]["subtasks"][task_id] = max(pending_num - i, 1)
        logger.info(f"Init pending subtasks: {self.pending_subtasks}")

    @class_try_catch_async
    async def pending_subtasks_add(self, queue, task_id):
        if queue not in self.pending_subtasks:
            logger.warning(f"Queue {queue} not found in self.pending_subtasks")
            return
        max_count = self.pending_subtasks[queue]["max_count"]
        self.pending_subtasks[queue]["subtasks"][task_id] = max_count + 1
        self.pending_subtasks[queue]["max_count"] = max_count + 1
        # logger.warning(f"Pending subtasks {queue} add {task_id}: {self.pending_subtasks[queue]}")

    @class_try_catch_async
    async def pending_subtasks_sub(self, queue, task_id):
        if queue not in self.pending_subtasks:
            logger.warning(f"Queue {queue} not found in self.pending_subtasks")
            return
        self.pending_subtasks[queue]["consume_count"] += 1
        if task_id in self.pending_subtasks[queue]["subtasks"]:
            self.pending_subtasks[queue]["subtasks"].pop(task_id)
        # logger.warning(f"Pending subtasks {queue} sub {task_id}: {self.pending_subtasks[queue]}")

    @class_try_catch_async
    async def pending_subtasks_get_order(self, queue, task_id):
        if queue not in self.pending_subtasks:
            logger.warning(f"Queue {queue} not found in self.pending_subtasks")
            return None
        if task_id not in self.pending_subtasks[queue]["subtasks"]:
            logger.warning(f"Task {task_id} not found in self.pending_subtasks[queue]['subtasks']")
            return None
        order = self.pending_subtasks[queue]["subtasks"][task_id]
        consume = self.pending_subtasks[queue]["consume_count"]
        real_order = max(order - consume, 1)
        # logger.warning(f"Pending subtasks {queue} get order {task_id}: real={real_order} order={order} consume={consume}")
        return real_order

    @class_try_catch_async
    async def get_ready_worker_count(self, queue):
        if queue not in self.worker_clients:
            self.worker_clients[queue] = {}
        return len(self.worker_clients[queue])

    @class_try_catch_async
    async def format_subtask(self, subtasks):
        ret = []
        for sub in subtasks:
            cur = {
                "status": sub["status"].name,
                "worker_name": sub["worker_name"],
                "fail_msg": None,
                "elapses": {},
                "estimated_pending_order": None,
                "estimated_pending_secs": None,
                "estimated_running_secs": None,
                "ready_worker_count": None,
            }
            if sub["status"] in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                cur["estimated_running_secs"] = await self.get_avg_worker_infer_cost(sub["queue"])
                cur["ready_worker_count"] = await self.get_ready_worker_count(sub["queue"])
                if sub["status"] == TaskStatus.PENDING:
                    order = await self.pending_subtasks_get_order(sub["queue"], sub["task_id"])
                    worker_count = max(cur["ready_worker_count"], 1e-7)
                    if order is not None:
                        cur["estimated_pending_order"] = order
                        wait_cycle = (order - 1) // worker_count + 1
                        cur["estimated_pending_secs"] = cur["estimated_running_secs"] * wait_cycle

            if isinstance(sub["extra_info"], dict):
                if "elapses" in sub["extra_info"]:
                    cur["elapses"] = sub["extra_info"]["elapses"]
                if "start_t" in sub["extra_info"]:
                    cur["elapses"][f"{cur['status']}-"] = time.time() - sub["extra_info"]["start_t"]
                if "fail_msg" in sub["extra_info"]:
                    cur["fail_msg"] = sub["extra_info"]["fail_msg"]
            ret.append(cur)
        return ret
