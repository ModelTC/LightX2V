import uuid
from enum import Enum
from re import T

from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch, current_time, data_name


class TaskStatus(Enum):
    CREATED = 1
    PENDING = 2
    RUNNING = 3
    SUCCEED = 4
    FAILED = 5
    CANCEL = 6


ActiveStatus = [TaskStatus.CREATED, TaskStatus.PENDING, TaskStatus.RUNNING]
FinishedStatus = [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]


class BaseTaskManager:
    def __init__(self):
        pass

    async def init(self):
        pass

    async def close(self):
        pass

    async def insert_user_if_not_exists(self, user_info):
        raise NotImplementedError

    async def query_user(self, user_id):
        raise NotImplementedError

    async def insert_task(self, task, subtasks):
        raise NotImplementedError

    async def list_tasks(self, **kwargs):
        raise NotImplementedError

    async def query_task(self, task_id, user_id=None, only_task=True):
        raise NotImplementedError

    async def next_subtasks(self, task_id):
        raise NotImplementedError

    async def run_subtasks(self, subtasks, worker_identity):
        raise NotImplementedError

    async def ping_subtask(self, task_id, worker_name, worker_identity):
        raise NotImplementedError

    async def finish_subtasks(self, task_id, status, worker_identity=None, worker_name=None, fail_msg=None, should_running=False):
        raise NotImplementedError

    async def cancel_task(self, task_id, user_id=None):
        raise NotImplementedError

    async def resume_task(self, task_id, all_subtask=False, user_id=None):
        raise NotImplementedError

    async def delete_task(self, task_id, user_id=None):
        raise NotImplementedError

    async def insert_share(self, share_info):
        raise NotImplementedError

    async def query_share(self, share_id):
        raise NotImplementedError

    def fmt_dict(self, data):
        for k in ["status"]:
            if k in data:
                data[k] = data[k].name

    def parse_dict(self, data):
        for k in ["status"]:
            if k in data:
                data[k] = TaskStatus[data[k]]

    async def create_share(self, task_id, user_id, share_type, valid_days, auth_type, auth_value):
        assert share_type in ["task", "template"], f"do not support {share_type} share type!"
        assert auth_type in ["public", "login", "user_id"], f"do not support {auth_type} auth type!"
        assert valid_days > 0, f"valid_days must be greater than 0!"
        share_id = str(uuid.uuid4())
        cur_t = current_time()
        share_info = {
            "share_id": share_id,
            "task_id": task_id,
            "user_id": user_id,
            "share_type": share_type,
            "create_t": cur_t,
            "update_t": cur_t,
            "valid_days": valid_days,
            "valid_t": cur_t + valid_days * 24 * 3600,
            "auth_type": auth_type,
            "auth_value": auth_value,
            "extra_info": "",
            "tag": "",
        }
        assert await self.insert_share(share_info), f"create share {share_info} failed"
        return share_id

    async def create_user(self, user_info):
        assert user_info["source"] in ["github", "google", "phone"], f"do not support {user_info['source']} user!"
        cur_t = current_time()
        user_id = f"{user_info['source']}_{user_info['id']}"
        data = {
            "user_id": user_id,
            "source": user_info["source"],
            "id": user_info["id"],
            "username": user_info["username"],
            "email": user_info["email"],
            "homepage": user_info["homepage"],
            "avatar_url": user_info["avatar_url"],
            "create_t": cur_t,
            "update_t": cur_t,
            "extra_info": "",
            "tag": "",
        }
        assert await self.insert_user_if_not_exists(data), f"create user {data} failed"
        return user_id

    async def create_task(self, worker_keys, workers, params, inputs, outputs, user_id):
        task_type, model_cls, stage = worker_keys
        cur_t = current_time()
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "task_type": task_type,
            "model_cls": model_cls,
            "stage": stage,
            "params": params,
            "create_t": cur_t,
            "update_t": cur_t,
            "status": TaskStatus.CREATED,
            "extra_info": "",
            "tag": "",
            "inputs": {x: data_name(x, task_id) for x in inputs},
            "outputs": {x: data_name(x, task_id) for x in outputs},
            "user_id": user_id,
        }
        records = []
        self.mark_task_start(records, task)
        subtasks = []
        for worker_name, worker_item in workers.items():
            subtasks.append(
                {
                    "task_id": task_id,
                    "worker_name": worker_name,
                    "inputs": {x: data_name(x, task_id) for x in worker_item["inputs"]},
                    "outputs": {x: data_name(x, task_id) for x in worker_item["outputs"]},
                    "queue": worker_item["queue"],
                    "previous": worker_item["previous"],
                    "status": TaskStatus.CREATED,
                    "worker_identity": "",
                    "result": "",
                    "fail_time": 0,
                    "extra_info": "",
                    "create_t": cur_t,
                    "update_t": cur_t,
                    "ping_t": 0.0,
                    "infer_cost": -1.0,
                }
            )
            self.mark_subtask_change(records, subtasks[-1], None, TaskStatus.CREATED)
        ret = await self.insert_task(task, subtasks)
        assert ret, f"create task {task_id} failed"
        self.metrics_commit(records)
        return task_id

    async def mark_server_restart(self):
        pass
        # only for start server with active tasks
        # if self.metrics_monitor:
        #     tasks = await self.list_tasks(status=ActiveStatus)
        #     subtasks = await self.list_tasks(status=ActiveStatus, subtasks=True)
        #     logger.warning(f"Mark system restart, {len(tasks)} tasks, {len(subtasks)} subtasks")
        #     self.metrics_monitor.record_task_recover(tasks)
        #     self.metrics_monitor.record_subtask_recover(subtasks)

    def mark_task_start(self, records, task):
        t = current_time()
        if not isinstance(task["extra_info"], dict):
            task["extra_info"] = {}
        if "active_elapse" in task["extra_info"]:
            del task["extra_info"]["active_elapse"]
        task["extra_info"]["start_t"] = t
        logger.info(f"Task {task['task_id']} active start")
        if self.metrics_monitor:
            records.append(
                [
                    self.metrics_monitor.record_task_start,
                    [task],
                ]
            )

    def mark_task_end(self, records, task, end_status):
        if "start_t" not in task["extra_info"]:
            logger.warning(f"Task {task} has no start time")
        else:
            elapse = current_time() - task["extra_info"]["start_t"]
            task["extra_info"]["active_elapse"] = elapse
            del task["extra_info"]["start_t"]

            logger.info(f"Task {task['task_id']} active end with [{end_status}], elapse: {elapse}")
            if self.metrics_monitor:
                records.append(
                    [
                        self.metrics_monitor.record_task_end,
                        [task, end_status, elapse],
                    ]
                )

    def mark_subtask_change(self, records, subtask, old_status, new_status, fail_msg=None):
        t = current_time()
        if not isinstance(subtask["extra_info"], dict):
            subtask["extra_info"] = {}
        if isinstance(fail_msg, str) and len(fail_msg) > 0:
            subtask["extra_info"]["fail_msg"] = fail_msg
        elif "fail_msg" in subtask["extra_info"]:
            del subtask["extra_info"]["fail_msg"]

        if old_status == new_status:
            logger.warning(f"Subtask {subtask} update same status: {old_status} vs {new_status}")
            return

        elapse, elapse_key = None, None
        if old_status in ActiveStatus:
            if "start_t" not in subtask["extra_info"]:
                logger.warning(f"Subtask {subtask} has no start time, status: {old_status}")
            else:
                elapse = t - subtask["extra_info"]["start_t"]
                elapse_key = f"{old_status.name}-{new_status.name}"
                if "elapses" not in subtask["extra_info"]:
                    subtask["extra_info"]["elapses"] = {}
                subtask["extra_info"]["elapses"][elapse_key] = elapse
                del subtask["extra_info"]["start_t"]

        if new_status in ActiveStatus:
            subtask["extra_info"]["start_t"] = t
        if new_status == TaskStatus.CREATED and "elapses" in subtask["extra_info"]:
            del subtask["extra_info"]["elapses"]

        logger.info(
            f"Subtask {subtask['task_id']} {subtask['worker_name']} status changed: \
            [{old_status}] -> [{new_status}], {elapse_key}: {elapse}, fail_msg: {fail_msg}"
        )

        if self.metrics_monitor:
            records.append(
                [
                    self.metrics_monitor.record_subtask_change,
                    [subtask, old_status, new_status, elapse_key, elapse],
                ]
            )

    @class_try_catch
    def metrics_commit(self, records):
        for func, args in records:
            func(*args)


# Import task manager implementations
from .local_task_manager import LocalTaskManager  # noqa
from .sql_task_manager import PostgresSQLTaskManager  # noqa

__all__ = ["BaseTaskManager", "LocalTaskManager", "PostgresSQLTaskManager"]
