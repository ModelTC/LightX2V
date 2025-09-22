import asyncio
import json
import os

from lightx2v.deploy.common.utils import class_try_catch_async, current_time, str2time, time2str
from lightx2v.deploy.task_manager import ActiveStatus, BaseTaskManager, FinishedStatus, TaskStatus


class LocalTaskManager(BaseTaskManager):
    def __init__(self, local_dir, metrics_monitor=None):
        self.local_dir = local_dir
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        self.metrics_monitor = metrics_monitor

    def get_task_filename(self, task_id):
        return os.path.join(self.local_dir, f"task_{task_id}.json")

    def get_user_filename(self, user_id):
        return os.path.join(self.local_dir, f"user_{user_id}.json")

    def fmt_dict(self, data):
        super().fmt_dict(data)
        for k in ["create_t", "update_t", "ping_t"]:
            if k in data:
                data[k] = time2str(data[k])

    def parse_dict(self, data):
        super().parse_dict(data)
        for k in ["create_t", "update_t", "ping_t"]:
            if k in data:
                data[k] = str2time(data[k])

    def save(self, task, subtasks, with_fmt=True):
        info = {"task": task, "subtasks": subtasks}
        if with_fmt:
            self.fmt_dict(info["task"])
            [self.fmt_dict(x) for x in info["subtasks"]]
        out_name = self.get_task_filename(task["task_id"])
        with open(out_name, "w") as fout:
            fout.write(json.dumps(info, indent=4, ensure_ascii=False))

    def load(self, task_id, user_id=None, only_task=False):
        fpath = self.get_task_filename(task_id)
        info = json.load(open(fpath))
        task, subtasks = info["task"], info["subtasks"]
        if user_id is not None and task["user_id"] != user_id:
            raise Exception(f"Task {task_id} is not belong to user {user_id}")
        self.parse_dict(task)
        if only_task:
            return task
        for sub in subtasks:
            self.parse_dict(sub)
        return task, subtasks

    @class_try_catch_async
    async def insert_task(self, task, subtasks):
        self.save(task, subtasks)
        return True

    @class_try_catch_async
    async def list_tasks(self, **kwargs):
        tasks = []
        fs = [os.path.join(self.local_dir, f) for f in os.listdir(self.local_dir)]
        for f in os.listdir(self.local_dir):
            if not f.startswith("task_"):
                continue
            fpath = os.path.join(self.local_dir, f)
            info = json.load(open(fpath))
            if kwargs.get("subtasks", False):
                items = info["subtasks"]
                assert "user_id" not in kwargs, "user_id is not allowed when subtasks is True"
            else:
                items = [info["task"]]
            for task in items:
                self.parse_dict(task)
                if "user_id" in kwargs and task["user_id"] != kwargs["user_id"]:
                    continue
                if "status" in kwargs:
                    if isinstance(kwargs["status"], list) and task["status"] not in kwargs["status"]:
                        continue
                    elif kwargs["status"] != task["status"]:
                        continue
                if "start_created_t" in kwargs and kwargs["start_created_t"] > task["create_t"]:
                    continue
                if "end_created_t" in kwargs and kwargs["end_created_t"] < task["create_t"]:
                    continue
                if "start_updated_t" in kwargs and kwargs["start_updated_t"] > task["update_t"]:
                    continue
                if "end_updated_t" in kwargs and kwargs["end_updated_t"] < task["update_t"]:
                    continue
                if "start_ping_t" in kwargs and kwargs["start_ping_t"] > task["ping_t"]:
                    continue
                if "end_ping_t" in kwargs and kwargs["end_ping_t"] < task["ping_t"]:
                    continue

                # 如果不是查询子任务，则添加子任务信息到任务中
                if not kwargs.get("subtasks", False):
                    task["subtasks"] = info.get("subtasks", [])

                tasks.append(task)
        if "count" in kwargs:
            return len(tasks)

        sort_key = "update_t" if kwargs.get("sort_by_update_t", False) else "create_t"
        tasks = sorted(tasks, key=lambda x: x[sort_key], reverse=True)

        if "offset" in kwargs:
            tasks = tasks[kwargs["offset"] :]
        if "limit" in kwargs:
            tasks = tasks[: kwargs["limit"]]
        return tasks

    @class_try_catch_async
    async def query_task(self, task_id, user_id=None, only_task=True):
        return self.load(task_id, user_id, only_task)

    @class_try_catch_async
    async def next_subtasks(self, task_id):
        records = []
        task, subtasks = self.load(task_id)
        if task["status"] not in ActiveStatus:
            return []
        succeeds = set()
        for sub in subtasks:
            if sub["status"] == TaskStatus.SUCCEED:
                succeeds.add(sub["worker_name"])
        nexts = []
        for sub in subtasks:
            if sub["status"] == TaskStatus.CREATED:
                dep_ok = True
                for prev in sub["previous"]:
                    if prev not in succeeds:
                        dep_ok = False
                        break
                if dep_ok:
                    self.mark_subtask_change(records, sub, sub["status"], TaskStatus.PENDING)
                    sub["params"] = task["params"]
                    sub["status"] = TaskStatus.PENDING
                    sub["update_t"] = current_time()
                    nexts.append(sub)
        if len(nexts) > 0:
            task["status"] = TaskStatus.PENDING
            task["update_t"] = current_time()
            self.save(task, subtasks)
        self.metrics_commit(records)
        return nexts

    @class_try_catch_async
    async def run_subtasks(self, cands, worker_identity):
        records = []
        valids = []
        for cand in cands:
            task_id = cand["task_id"]
            worker_name = cand["worker_name"]
            task, subtasks = self.load(task_id)
            if task["status"] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]:
                continue
            for sub in subtasks:
                if sub["worker_name"] == worker_name:
                    self.mark_subtask_change(records, sub, sub["status"], TaskStatus.RUNNING)
                    sub["status"] = TaskStatus.RUNNING
                    sub["worker_identity"] = worker_identity
                    sub["update_t"] = current_time()
                    task["status"] = TaskStatus.RUNNING
                    task["update_t"] = current_time()
                    task["ping_t"] = current_time()
                    self.save(task, subtasks)
                    valids.append(cand)
                    break
        self.metrics_commit(records)
        return valids

    @class_try_catch_async
    async def ping_subtask(self, task_id, worker_name, worker_identity):
        task, subtasks = self.load(task_id)
        for sub in subtasks:
            if sub["worker_name"] == worker_name:
                pre = sub["worker_identity"]
                assert pre == worker_identity, f"worker identity not matched: {pre} vs {worker_identity}"
                sub["ping_t"] = current_time()
                self.save(task, subtasks)
                return True
        return False

    @class_try_catch_async
    async def finish_subtasks(self, task_id, status, worker_identity=None, worker_name=None, fail_msg=None, should_running=False):
        records = []
        task, subtasks = self.load(task_id)
        subs = subtasks

        if worker_name:
            subs = [sub for sub in subtasks if sub["worker_name"] == worker_name]
        assert len(subs) >= 1, f"no worker task_id={task_id}, name={worker_name}"

        if worker_identity:
            pre = subs[0]["worker_identity"]
            assert pre == worker_identity, f"worker identity not matched: {pre} vs {worker_identity}"

        assert status in [TaskStatus.SUCCEED, TaskStatus.FAILED], f"invalid finish status: {status}"
        for sub in subs:
            if sub["status"] not in FinishedStatus:
                if should_running and sub["status"] != TaskStatus.RUNNING:
                    print(f"task {task_id} is not running, skip finish subtask: {sub}")
                    continue
                self.mark_subtask_change(records, sub, sub["status"], status, fail_msg=fail_msg)
                sub["status"] = status
                sub["update_t"] = current_time()

        if task["status"] == TaskStatus.CANCEL:
            self.save(task, subtasks)
            self.metrics_commit(records)
            return TaskStatus.CANCEL

        running_subs = []
        failed_sub = False
        for sub in subtasks:
            if sub["status"] not in FinishedStatus:
                running_subs.append(sub)
            if sub["status"] == TaskStatus.FAILED:
                failed_sub = True

        # some subtask failed, we should fail all other subtasks
        if failed_sub:
            if task["status"] != TaskStatus.FAILED:
                self.mark_task_end(records, task, TaskStatus.FAILED)
                task["status"] = TaskStatus.FAILED
                task["update_t"] = current_time()
            for sub in running_subs:
                self.mark_subtask_change(records, sub, sub["status"], TaskStatus.FAILED, fail_msg="other subtask failed")
                sub["status"] = TaskStatus.FAILED
                sub["update_t"] = current_time()
            self.save(task, subtasks)
            self.metrics_commit(records)
            return TaskStatus.FAILED

        # all subtasks finished and all succeed
        elif len(running_subs) == 0:
            if task["status"] != TaskStatus.SUCCEED:
                self.mark_task_end(records, task, TaskStatus.SUCCEED)
                task["status"] = TaskStatus.SUCCEED
                task["update_t"] = current_time()
            self.save(task, subtasks)
            self.metrics_commit(records)
            return TaskStatus.SUCCEED

        self.save(task, subtasks)
        self.metrics_commit(records)
        return None

    @class_try_catch_async
    async def cancel_task(self, task_id, user_id=None):
        records = []
        task, subtasks = self.load(task_id, user_id)
        if task["status"] not in ActiveStatus:
            return f"Task {task_id} is not in active status (current status: {task['status']}). Only tasks with status CREATED, PENDING, or RUNNING can be cancelled."

        for sub in subtasks:
            if sub["status"] not in FinishedStatus:
                self.mark_subtask_change(records, sub, sub["status"], TaskStatus.CANCEL)
                sub["status"] = TaskStatus.CANCEL
                sub["update_t"] = current_time()
        self.mark_task_end(records, task, TaskStatus.CANCEL)
        task["status"] = TaskStatus.CANCEL
        task["update_t"] = current_time()
        self.save(task, subtasks)
        self.metrics_commit(records)
        return True

    @class_try_catch_async
    async def resume_task(self, task_id, all_subtask=False, user_id=None):
        records = []
        task, subtasks = self.load(task_id, user_id)
        # the task is not finished
        if task["status"] not in FinishedStatus:
            return "Active task cannot be resumed"
        # the task is no need to resume
        if not all_subtask and task["status"] == TaskStatus.SUCCEED:
            return "Succeed task cannot be resumed"
        for sub in subtasks:
            if all_subtask or sub["status"] != TaskStatus.SUCCEED:
                self.mark_subtask_change(records, sub, None, TaskStatus.CREATED)
                sub["status"] = TaskStatus.CREATED
                sub["update_t"] = current_time()
                sub["ping_t"] = 0.0
        self.mark_task_start(records, task)
        task["status"] = TaskStatus.CREATED
        task["update_t"] = current_time()
        self.save(task, subtasks)
        self.metrics_commit(records)
        return True

    @class_try_catch_async
    async def delete_task(self, task_id, user_id=None):
        task = self.load(task_id, user_id, only_task=True)
        # only allow to delete finished tasks
        if task["status"] not in FinishedStatus:
            return False
        # delete task file
        task_file = self.get_task_filename(task_id)
        if os.path.exists(task_file):
            os.remove(task_file)
        return True

    @class_try_catch_async
    async def insert_user_if_not_exists(self, user_info):
        fpath = self.get_user_filename(user_info["user_id"])
        if os.path.exists(fpath):
            return True
        self.fmt_dict(user_info)
        with open(fpath, "w") as fout:
            fout.write(json.dumps(user_info, indent=4, ensure_ascii=False))
        return True

    @class_try_catch_async
    async def query_user(self, user_id):
        fpath = self.get_user_filename(user_id)
        if not os.path.exists(fpath):
            return None
        data = json.load(open(fpath))
        self.parse_dict(data)
        return data


async def test():
    from lightx2v.deploy.common.pipeline import Pipeline

    p = Pipeline("/data/nvme1/liuliang1/lightx2v/configs/model_pipeline.json")
    m = LocalTaskManager("/data/nvme1/liuliang1/lightx2v/local_task")
    await m.init()

    keys = ["t2v", "wan2.1", "multi_stage"]
    workers = p.get_workers(keys)
    inputs = p.get_inputs(keys)
    outputs = p.get_outputs(keys)
    params = {
        "prompt": "fake input prompts",
        "resolution": {
            "height": 233,
            "width": 456,
        },
    }

    user_info = {
        "source": "github",
        "id": "test-id-233",
        "username": "test-username-233",
        "email": "test-email-233@test.com",
        "homepage": "https://test.com",
        "avatar_url": "https://test.com/avatar.png",
    }
    user_id = await m.create_user(user_info)
    print(" - create_user:", user_id)

    user = await m.query_user(user_id)
    print(" - query_user:", user)

    task_id = await m.create_task(keys, workers, params, inputs, outputs, user_id)
    print(" - create_task:", task_id)

    tasks = await m.list_tasks()
    print(" - list_tasks:", tasks)

    task = await m.query_task(task_id)
    print(" - query_task:", task)

    subtasks = await m.next_subtasks(task_id)
    print(" - next_subtasks:", subtasks)

    await m.run_subtasks(subtasks, "fake-worker")
    await m.finish_subtasks(task_id, TaskStatus.FAILED)
    await m.cancel_task(task_id)
    await m.resume_task(task_id)
    for sub in subtasks:
        await m.finish_subtasks(sub["task_id"], TaskStatus.SUCCEED, worker_name=sub["worker_name"], worker_identity="fake-worker")

    subtasks = await m.next_subtasks(task_id)
    print(" - final next_subtasks:", subtasks)

    task = await m.query_task(task_id)
    print(" - final task:", task)

    await m.close()


if __name__ == "__main__":
    asyncio.run(test())
