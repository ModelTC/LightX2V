import asyncio
import json
import os
import time
import traceback

from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.deploy.queue_manager import BaseQueueManager


class LocalQueueManager(BaseQueueManager):
    def __init__(self, local_dir):
        self.local_dir = local_dir
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)

    async def get_conn(self):
        pass

    async def del_conn(self):
        pass

    async def declare_queue(self, queue):
        pass

    @class_try_catch_async
    async def put_subtask(self, subtask):
        out_name = self.get_filename(subtask["queue"])
        keys = ["queue", "task_id", "worker_name", "inputs", "outputs", "params"]
        msg = json.dumps({k: subtask[k] for k in keys}) + "\n"
        logger.info(f"Local published subtask: ({subtask['task_id']}, {subtask['worker_name']}) to {subtask['queue']}")
        with open(out_name, "a") as fout:
            fout.write(msg)
            return True

    def read_first_line(self, queue):
        out_name = self.get_filename(queue)
        if not os.path.exists(out_name):
            return None
        lines = []
        with open(out_name) as fin:
            lines = fin.readlines()
        if len(lines) <= 0:
            return None
        subtask = json.loads(lines[0])
        msgs = "".join(lines[1:])
        fout = open(out_name, "w")
        fout.write(msgs)
        fout.close()
        return subtask

    @class_try_catch_async
    async def get_subtasks(self, queue, max_batch, timeout):
        try:
            t0 = time.time()
            subtasks = []
            while True:
                subtask = self.read_first_line(queue)
                if subtask:
                    subtasks.append(subtask)
                    if len(subtasks) >= max_batch:
                        return subtasks
                    else:
                        continue
                else:
                    if len(subtasks) > 0:
                        return subtasks
                    if time.time() - t0 > timeout:
                        return None
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.warning(f"local queue get_subtasks for {queue} cancelled")
            return None
        except:  # noqa
            logger.warning(f"local queue get_subtasks for {queue} failed: {traceback.format_exc()}")
            return None

    def get_filename(self, queue):
        return os.path.join(self.local_dir, f"{queue}.jsonl")

    @class_try_catch_async
    async def pending_num(self, queue):
        out_name = self.get_filename(queue)
        if not os.path.exists(out_name):
            return 0
        lines = []
        with open(out_name) as fin:
            lines = fin.readlines()
        return len(lines)


async def test():
    q = LocalQueueManager("/data/nvme1/liuliang1/lightx2v/local_queue")
    await q.init()
    subtask = {
        "task_id": "test-subtask-id",
        "queue": "test_queue",
        "worker_name": "test_worker",
        "inputs": {},
        "outputs": {},
        "params": {},
    }
    await q.put_subtask(subtask)
    await asyncio.sleep(5)
    for i in range(2):
        subtask = await q.get_subtasks("test_queue", 3, 5)
        print("get subtask:", subtask)


if __name__ == "__main__":
    asyncio.run(test())
