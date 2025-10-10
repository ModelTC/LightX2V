import asyncio
import json
import traceback
from datetime import datetime

import asyncpg
from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.deploy.task_manager import ActiveStatus, BaseTaskManager, FinishedStatus, TaskStatus


class PostgresSQLTaskManager(BaseTaskManager):
    def __init__(self, db_url, metrics_monitor=None):
        self.db_url = db_url
        self.table_tasks = "tasks"
        self.table_subtasks = "subtasks"
        self.table_users = "users"
        self.table_versions = "versions"
        self.table_shares = "shares"
        self.pool = None
        self.metrics_monitor = metrics_monitor

    async def init(self):
        await self.upgrade_db()

    async def close(self):
        if self.pool:
            await self.pool.close()

    def fmt_dict(self, data):
        super().fmt_dict(data)
        for k in ["create_t", "update_t", "ping_t", "valid_t"]:
            if k in data and isinstance(data[k], float):
                data[k] = datetime.fromtimestamp(data[k])
        for k in ["params", "extra_info", "inputs", "outputs", "previous"]:
            if k in data:
                data[k] = json.dumps(data[k], ensure_ascii=False)

    def parse_dict(self, data):
        super().parse_dict(data)
        for k in ["params", "extra_info", "inputs", "outputs", "previous"]:
            if k in data:
                data[k] = json.loads(data[k])
        for k in ["create_t", "update_t", "ping_t", "valid_t"]:
            if k in data:
                data[k] = data[k].timestamp()

    async def get_conn(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.db_url)
        return await self.pool.acquire()

    async def release_conn(self, conn):
        await self.pool.release(conn)

    async def query_version(self):
        conn = await self.get_conn()
        try:
            row = await conn.fetchrow(f"SELECT version FROM {self.table_versions} ORDER BY create_t DESC LIMIT 1")
            row = dict(row)
            return row["version"] if row else 0
        except:  # noqa
            logger.error(f"query_version error: {traceback.format_exc()}")
            return 0
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def upgrade_db(self):
        versions = [
            (1, "Init tables", self.upgrade_v1),
            (2, "Add shares table", self.upgrade_v2),
        ]
        logger.info(f"upgrade_db: {self.db_url}")
        cur_ver = await self.query_version()
        for ver, description, func in versions:
            if cur_ver < ver:
                logger.info(f"Upgrade to version {ver}: {description}")
                if not await func(ver, description):
                    logger.error(f"Upgrade to version {ver}: {description} func failed")
                    return False
                cur_ver = ver
        logger.info(f"upgrade_db: {self.db_url} done")
        return True

    async def upgrade_v1(self, version, description):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                # create users table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_users} (
                        user_id VARCHAR(256) PRIMARY KEY,
                        source VARCHAR(32),
                        id VARCHAR(200),
                        username VARCHAR(256),
                        email VARCHAR(256),
                        homepage VARCHAR(256),
                        avatar_url VARCHAR(256),
                        create_t TIMESTAMPTZ,
                        update_t TIMESTAMPTZ,
                        extra_info JSONB,
                        tag VARCHAR(64)
                    )
                """)
                # create tasks table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_tasks} (
                        task_id VARCHAR(128) PRIMARY KEY,
                        task_type VARCHAR(64),
                        model_cls VARCHAR(64),
                        stage VARCHAR(64),
                        params JSONB,
                        create_t TIMESTAMPTZ,
                        update_t TIMESTAMPTZ,
                        status VARCHAR(64),
                        extra_info JSONB,
                        tag VARCHAR(64),
                        inputs JSONB,
                        outputs JSONB,
                        user_id VARCHAR(256),
                        FOREIGN KEY (user_id) REFERENCES {self.table_users}(user_id) ON DELETE CASCADE
                    )
                """)
                # create subtasks table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_subtasks} (
                        task_id VARCHAR(128),
                        worker_name VARCHAR(128),
                        inputs JSONB,
                        outputs JSONB,
                        queue VARCHAR(128),
                        previous JSONB,
                        status VARCHAR(64),
                        worker_identity VARCHAR(128),
                        result VARCHAR(128),
                        fail_time INTEGER,
                        extra_info JSONB,
                        create_t TIMESTAMPTZ,
                        update_t TIMESTAMPTZ,
                        ping_t TIMESTAMPTZ,
                        infer_cost FLOAT,
                        PRIMARY KEY (task_id, worker_name),
                        FOREIGN KEY (task_id) REFERENCES {self.table_tasks}(task_id) ON DELETE CASCADE
                    )
                """)
                # create versions table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_versions} (
                        version INTEGER PRIMARY KEY,
                        description VARCHAR(255),
                        create_t TIMESTAMPTZ NOT NULL
                    )
                """)
                # create indexes
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_users}_source ON {self.table_users}(source)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_users}_id ON {self.table_users}(id)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_tasks}_status ON {self.table_tasks}(status)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_tasks}_create_t ON {self.table_tasks}(create_t)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_tasks}_tag ON {self.table_tasks}(tag)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_subtasks}_task_id ON {self.table_subtasks}(task_id)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_subtasks}_worker_name ON {self.table_subtasks}(worker_name)")
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_subtasks}_status ON {self.table_subtasks}(status)")

                # update version
                await conn.execute(f"INSERT INTO {self.table_versions} (version, description, create_t) VALUES ($1, $2, $3)", version, description, datetime.now())
                return True
        except:  # noqa
            logger.error(f"upgrade_v1 error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    async def upgrade_v2(self, version, description):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                # create shares table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_shares} (
                        share_id VARCHAR(128) PRIMARY KEY,
                        task_id VARCHAR(128),
                        user_id VARCHAR(256),
                        share_type VARCHAR(32),
                        create_t TIMESTAMPTZ,
                        update_t TIMESTAMPTZ,
                        valid_days INTEGER,
                        valid_t TIMESTAMPTZ,
                        auth_type VARCHAR(32),
                        auth_value VARCHAR(256),
                        extra_info JSONB,
                        tag VARCHAR(64),
                        FOREIGN KEY (user_id) REFERENCES {self.table_users}(user_id) ON DELETE CASCADE
                    )
                """)

                # update version
                await conn.execute(f"INSERT INTO {self.table_versions} (version, description, create_t) VALUES ($1, $2, $3)", version, description, datetime.now())
                return True
        except:  # noqa
            logger.error(f"upgrade_v1 error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    async def load(self, conn, task_id, user_id=None, only_task=False, worker_name=None):
        query = f"SELECT * FROM {self.table_tasks} WHERE task_id = $1 AND tag != 'delete'"
        params = [task_id]
        if user_id is not None:
            query += " AND user_id = $2"
            params.append(user_id)
        row = await conn.fetchrow(query, *params)
        task = dict(row)
        assert task, f"query_task: task not found: {task_id} {user_id}"
        self.parse_dict(task)
        if only_task:
            return task
        query2 = f"SELECT * FROM {self.table_subtasks} WHERE task_id = $1"
        params2 = [task_id]
        if worker_name is not None:
            query2 += " AND worker_name = $2"
            params2.append(worker_name)
        rows = await conn.fetch(query2, *params2)
        subtasks = []
        for row in rows:
            sub = dict(row)
            self.parse_dict(sub)
            subtasks.append(sub)
        return task, subtasks

    def check_update_valid(self, ret, prefix, query, params):
        if ret.startswith("UPDATE "):
            count = int(ret.split(" ")[1])
            assert count > 0, f"{prefix}: no row changed: {query} {params}"
            return count
        else:
            logger.warning(f"parse postsql update ret failed: {ret}")
            return 0

    async def update_task(self, conn, task_id, **kwargs):
        query = f"UPDATE {self.table_tasks} SET "
        conds = ["update_t = $1"]
        params = [datetime.now()]
        param_idx = 1
        if "status" in kwargs:
            param_idx += 1
            conds.append(f"status = ${param_idx}")
            params.append(kwargs["status"].name)
        if "extra_info" in kwargs:
            param_idx += 1
            conds.append(f"extra_info = ${param_idx}")
            params.append(json.dumps(kwargs["extra_info"], ensure_ascii=False))

        limit_conds = [f"task_id = ${param_idx + 1}"]
        param_idx += 1
        params.append(task_id)

        if "src_status" in kwargs:
            param_idx += 1
            limit_conds.append(f"status = ${param_idx}")
            params.append(kwargs["src_status"].name)

        query += " ,".join(conds) + " WHERE " + " AND ".join(limit_conds)
        ret = await conn.execute(query, *params)
        return self.check_update_valid(ret, "update_task", query, params)

    async def update_subtask(self, conn, task_id, worker_name, **kwargs):
        query = f"UPDATE {self.table_subtasks} SET "
        conds = []
        params = []
        param_idx = 0
        if kwargs.get("update_t", True):
            param_idx += 1
            conds.append(f"update_t = ${param_idx}")
            params.append(datetime.now())
        if kwargs.get("ping_t", False):
            param_idx += 1
            conds.append(f"ping_t = ${param_idx}")
            params.append(datetime.now())
        if kwargs.get("reset_ping_t", False):
            param_idx += 1
            conds.append(f"ping_t = ${param_idx}")
            params.append(datetime.fromtimestamp(0))
        if "status" in kwargs:
            param_idx += 1
            conds.append(f"status = ${param_idx}")
            params.append(kwargs["status"].name)
        if "worker_identity" in kwargs:
            param_idx += 1
            conds.append(f"worker_identity = ${param_idx}")
            params.append(kwargs["worker_identity"])
        if "infer_cost" in kwargs:
            param_idx += 1
            conds.append(f"infer_cost = ${param_idx}")
            params.append(kwargs["infer_cost"])
        if "extra_info" in kwargs:
            param_idx += 1
            conds.append(f"extra_info = ${param_idx}")
            params.append(json.dumps(kwargs["extra_info"], ensure_ascii=False))

        limit_conds = [f"task_id = ${param_idx + 1}", f"worker_name = ${param_idx + 2}"]
        param_idx += 2
        params.extend([task_id, worker_name])

        if "src_status" in kwargs:
            param_idx += 1
            limit_conds.append(f"status = ${param_idx}")
            params.append(kwargs["src_status"].name)

        query += " ,".join(conds) + " WHERE " + " AND ".join(limit_conds)
        ret = await conn.execute(query, *params)
        return self.check_update_valid(ret, "update_subtask", query, params)

    @class_try_catch_async
    async def insert_task(self, task, subtasks):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                self.fmt_dict(task)
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_tasks}
                    (task_id, task_type, model_cls, stage, params, create_t,
                        update_t, status, extra_info, tag, inputs, outputs, user_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    task["task_id"],
                    task["task_type"],
                    task["model_cls"],
                    task["stage"],
                    task["params"],
                    task["create_t"],
                    task["update_t"],
                    task["status"],
                    task["extra_info"],
                    task["tag"],
                    task["inputs"],
                    task["outputs"],
                    task["user_id"],
                )
                for sub in subtasks:
                    self.fmt_dict(sub)
                    await conn.execute(
                        f"""
                        INSERT INTO {self.table_subtasks}
                        (task_id, worker_name, inputs, outputs, queue, previous, status,
                            worker_identity, result, fail_time, extra_info, create_t, update_t,
                            ping_t, infer_cost)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                        """,
                        sub["task_id"],
                        sub["worker_name"],
                        sub["inputs"],
                        sub["outputs"],
                        sub["queue"],
                        sub["previous"],
                        sub["status"],
                        sub["worker_identity"],
                        sub["result"],
                        sub["fail_time"],
                        sub["extra_info"],
                        sub["create_t"],
                        sub["update_t"],
                        sub["ping_t"],
                        sub["infer_cost"],
                    )
                return True
        except:  # noqa
            logger.error(f"insert_task error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def list_tasks(self, **kwargs):
        conn = await self.get_conn()
        try:
            count = kwargs.get("count", False)
            query = f"SELECT * FROM "
            if count:
                query = f"SELECT COUNT(*) FROM "
                assert "limit" not in kwargs, "limit is not allowed when count is True"
                assert "offset" not in kwargs, "offset is not allowed when count is True"
            params = []
            conds = []
            param_idx = 0
            if kwargs.get("subtasks", False):
                query += self.table_subtasks
                assert "user_id" not in kwargs, "user_id is not allowed when subtasks is True"
            else:
                query += self.table_tasks
                if not kwargs.get("include_delete", False):
                    param_idx += 1
                    conds.append(f"tag != ${param_idx}")
                    params.append("delete")

            if "status" in kwargs:
                param_idx += 1
                if isinstance(kwargs["status"], list):
                    next_idx = param_idx + len(kwargs["status"])
                    placeholders = ",".join([f"${i}" for i in range(param_idx, next_idx)])
                    conds.append(f"status IN ({placeholders})")
                    params.extend([x.name for x in kwargs["status"]])
                    param_idx = next_idx - 1
                else:
                    conds.append(f"status = ${param_idx}")
                    params.append(kwargs["status"].name)

            if "start_created_t" in kwargs:
                param_idx += 1
                conds.append(f"create_t >= ${param_idx}")
                params.append(datetime.fromtimestamp(kwargs["start_created_t"]))

            if "end_created_t" in kwargs:
                param_idx += 1
                conds.append(f"create_t <= ${param_idx}")
                params.append(datetime.fromtimestamp(kwargs["end_created_t"]))

            if "start_updated_t" in kwargs:
                param_idx += 1
                conds.append(f"update_t >= ${param_idx}")
                params.append(datetime.fromtimestamp(kwargs["start_updated_t"]))

            if "end_updated_t" in kwargs:
                param_idx += 1
                conds.append(f"update_t <= ${param_idx}")
                params.append(datetime.fromtimestamp(kwargs["end_updated_t"]))

            if "start_ping_t" in kwargs:
                param_idx += 1
                conds.append(f"ping_t >= ${param_idx}")
                params.append(datetime.fromtimestamp(kwargs["start_ping_t"]))

            if "end_ping_t" in kwargs:
                param_idx += 1
                conds.append(f"ping_t <= ${param_idx}")
                params.append(datetime.fromtimestamp(kwargs["end_ping_t"]))

            if "user_id" in kwargs:
                param_idx += 1
                conds.append(f"user_id = ${param_idx}")
                params.append(kwargs["user_id"])

            if conds:
                query += " WHERE " + " AND ".join(conds)

            if not count:
                sort_key = "update_t" if kwargs.get("sort_by_update_t", False) else "create_t"
                query += f" ORDER BY {sort_key} DESC"

            if "limit" in kwargs:
                param_idx += 1
                query += f" LIMIT ${param_idx}"
                params.append(kwargs["limit"])

            if "offset" in kwargs:
                param_idx += 1
                query += f" OFFSET ${param_idx}"
                params.append(kwargs["offset"])

            rows = await conn.fetch(query, *params)
            if count:
                return rows[0]["count"]

            # query subtasks with task
            subtasks = {}
            if not kwargs.get("subtasks", False):
                subtask_query = f"SELECT {self.table_subtasks}.* FROM ({query}) AS t \
                                JOIN {self.table_subtasks} ON t.task_id = {self.table_subtasks}.task_id"
                subtask_rows = await conn.fetch(subtask_query, *params)
                for row in subtask_rows:
                    sub = dict(row)
                    self.parse_dict(sub)
                    if sub["task_id"] not in subtasks:
                        subtasks[sub["task_id"]] = []
                    subtasks[sub["task_id"]].append(sub)

            tasks = []
            for row in rows:
                task = dict(row)
                self.parse_dict(task)
                if not kwargs.get("subtasks", False):
                    task["subtasks"] = subtasks.get(task["task_id"], [])
                tasks.append(task)
            return tasks
        except:  # noqa
            logger.error(f"list_tasks error: {traceback.format_exc()}")
            return []
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def query_task(self, task_id, user_id=None, only_task=True):
        conn = await self.get_conn()
        try:
            return await self.load(conn, task_id, user_id, only_task=only_task)
        except:  # noqa
            logger.error(f"query_task error: {traceback.format_exc()}")
            return None
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def next_subtasks(self, task_id):
        conn = await self.get_conn()
        records = []
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                task, subtasks = await self.load(conn, task_id)
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
                            sub["params"] = task["params"]
                            self.mark_subtask_change(records, sub, sub["status"], TaskStatus.PENDING)
                            await self.update_subtask(
                                conn,
                                task_id,
                                sub["worker_name"],
                                status=TaskStatus.PENDING,
                                extra_info=sub["extra_info"],
                                src_status=sub["status"],
                            )
                            nexts.append(sub)
                if len(nexts) > 0:
                    await self.update_task(conn, task_id, status=TaskStatus.PENDING)
                self.metrics_commit(records)
                return nexts
        except:  # noqa
            logger.error(f"next_subtasks error: {traceback.format_exc()}")
            return None
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def run_subtasks(self, cands, worker_identity):
        conn = await self.get_conn()
        records = []
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                valids = []
                for cand in cands:
                    task_id = cand["task_id"]
                    worker_name = cand["worker_name"]
                    task, subs = await self.load(conn, task_id, worker_name=worker_name)
                    assert len(subs) == 1, f"task {task_id} has multiple subtasks: {subs} with worker_name: {worker_name}"
                    if task["status"] in [TaskStatus.SUCCEED, TaskStatus.FAILED, TaskStatus.CANCEL]:
                        continue

                    self.mark_subtask_change(records, subs[0], subs[0]["status"], TaskStatus.RUNNING)
                    await self.update_subtask(
                        conn,
                        task_id,
                        worker_name,
                        status=TaskStatus.RUNNING,
                        worker_identity=worker_identity,
                        ping_t=True,
                        extra_info=subs[0]["extra_info"],
                        src_status=subs[0]["status"],
                    )
                    await self.update_task(conn, task_id, status=TaskStatus.RUNNING)
                    valids.append(cand)
                    break
                self.metrics_commit(records)
                return valids
        except:  # noqa
            logger.error(f"run_subtasks error: {traceback.format_exc()}")
            return []
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def ping_subtask(self, task_id, worker_name, worker_identity):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                task, subtasks = await self.load(conn, task_id)
                for sub in subtasks:
                    if sub["worker_name"] == worker_name:
                        pre = sub["worker_identity"]
                        assert pre == worker_identity, f"worker identity not matched: {pre} vs {worker_identity}"
                        await self.update_subtask(conn, task_id, worker_name, ping_t=True, update_t=False)
                        return True
                return False
        except:  # noqa
            logger.error(f"ping_subtask error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def finish_subtasks(self, task_id, status, worker_identity=None, worker_name=None, fail_msg=None, should_running=False):
        conn = await self.get_conn()
        records = []
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                task, subtasks = await self.load(conn, task_id)
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
                            logger.warning(f"task {task_id} is not running, skip finish subtask: {sub}")
                            continue
                        self.mark_subtask_change(records, sub, sub["status"], status, fail_msg=fail_msg)
                        await self.update_subtask(
                            conn,
                            task_id,
                            sub["worker_name"],
                            status=status,
                            extra_info=sub["extra_info"],
                            src_status=sub["status"],
                        )
                        sub["status"] = status

                if task["status"] == TaskStatus.CANCEL:
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
                        await self.update_task(
                            conn,
                            task_id,
                            status=TaskStatus.FAILED,
                            extra_info=task["extra_info"],
                            src_status=task["status"],
                        )
                    for sub in running_subs:
                        self.mark_subtask_change(records, sub, sub["status"], TaskStatus.FAILED, fail_msg="other subtask failed")
                        await self.update_subtask(
                            conn,
                            task_id,
                            sub["worker_name"],
                            status=TaskStatus.FAILED,
                            extra_info=sub["extra_info"],
                            src_status=sub["status"],
                        )
                    self.metrics_commit(records)
                    return TaskStatus.FAILED

                # all subtasks finished and all succeed
                elif len(running_subs) == 0:
                    if task["status"] != TaskStatus.SUCCEED:
                        self.mark_task_end(records, task, TaskStatus.SUCCEED)
                        await self.update_task(
                            conn,
                            task_id,
                            status=TaskStatus.SUCCEED,
                            extra_info=task["extra_info"],
                            src_status=task["status"],
                        )
                    self.metrics_commit(records)
                    return TaskStatus.SUCCEED

                self.metrics_commit(records)
                return None
        except:  # noqa
            logger.error(f"finish_subtasks error: {traceback.format_exc()}")
            return None
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def cancel_task(self, task_id, user_id=None):
        conn = await self.get_conn()
        records = []
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                task, subtasks = await self.load(conn, task_id, user_id)
                if task["status"] not in ActiveStatus:
                    return f"Task {task_id} is not in active status (current status: {task['status']}). \
                        Only tasks with status CREATED, PENDING, or RUNNING can be cancelled."

                for sub in subtasks:
                    if sub["status"] not in FinishedStatus:
                        self.mark_subtask_change(records, sub, sub["status"], TaskStatus.CANCEL)
                        await self.update_subtask(
                            conn,
                            task_id,
                            sub["worker_name"],
                            status=TaskStatus.CANCEL,
                            extra_info=sub["extra_info"],
                            src_status=sub["status"],
                        )

                self.mark_task_end(records, task, TaskStatus.CANCEL)
                await self.update_task(
                    conn,
                    task_id,
                    status=TaskStatus.CANCEL,
                    extra_info=task["extra_info"],
                    src_status=task["status"],
                )
                self.metrics_commit(records)
                return True
        except:  # noqa
            logger.error(f"cancel_task error: {traceback.format_exc()}")
            return "unknown cancel error"
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def resume_task(self, task_id, all_subtask=False, user_id=None):
        conn = await self.get_conn()
        records = []
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                task, subtasks = await self.load(conn, task_id, user_id)
                # the task is not finished
                if task["status"] not in FinishedStatus:
                    return "Active task cannot be resumed"
                # the task is no need to resume
                if not all_subtask and task["status"] == TaskStatus.SUCCEED:
                    return "Succeed task cannot be resumed"

                for sub in subtasks:
                    if all_subtask or sub["status"] != TaskStatus.SUCCEED:
                        self.mark_subtask_change(records, sub, None, TaskStatus.CREATED)
                        await self.update_subtask(
                            conn,
                            task_id,
                            sub["worker_name"],
                            status=TaskStatus.CREATED,
                            reset_ping_t=True,
                            extra_info=sub["extra_info"],
                            src_status=sub["status"],
                        )

                self.mark_task_start(records, task)
                await self.update_task(
                    conn,
                    task_id,
                    status=TaskStatus.CREATED,
                    extra_info=task["extra_info"],
                    src_status=task["status"],
                )
                self.metrics_commit(records)
                return True
        except:  # noqa
            logger.error(f"resume_task error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def delete_task(self, task_id, user_id=None):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                task = await self.load(conn, task_id, user_id, only_task=True)

                # only allow to delete finished tasks
                if task["status"] not in FinishedStatus:
                    logger.warning(f"Cannot delete task {task_id} with status {task['status']}, only finished tasks can be deleted")
                    return False

                # delete task record
                await conn.execute(f"UPDATE {self.table_tasks} SET tag = 'delete', update_t = $1 WHERE task_id = $2", datetime.now(), task_id)
                logger.info(f"Task {task_id} deleted successfully")
                return True

        except:  # noqa
            logger.error(f"delete_task error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def insert_share(self, share_info):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                self.fmt_dict(share_info)
                await conn.execute(
                    f"""INSERT INTO {self.table_shares}
                    (share_id, task_id, user_id, share_type, create_t, update_t,
                        valid_days, valid_t, auth_type, auth_value, extra_info, tag)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """,
                    share_info["share_id"],
                    share_info["task_id"],
                    share_info["user_id"],
                    share_info["share_type"],
                    share_info["create_t"],
                    share_info["update_t"],
                    share_info["valid_days"],
                    share_info["valid_t"],
                    share_info["auth_type"],
                    share_info["auth_value"],
                    share_info["extra_info"],
                    share_info["tag"],
                )
                return True
        except:  # noqa
            logger.error(f"create_share_link error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def query_share(self, share_id):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                row = await conn.fetchrow(f"SELECT * FROM {self.table_shares} WHERE share_id = $1 AND tag != 'delete' AND valid_t >= $2", share_id, datetime.now())
                share = dict(row)
                self.parse_dict(share)
                return share
        except:  # noqa
            logger.error(f"query_share error: {traceback.format_exc()}")
            return None
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def insert_user_if_not_exists(self, user_info):
        conn = await self.get_conn()
        try:
            async with conn.transaction(isolation="read_uncommitted"):
                row = await conn.fetchrow(f"SELECT * FROM {self.table_users} WHERE user_id = $1", user_info["user_id"])
                if row:
                    logger.info(f"user already exists: {user_info['user_id']}")
                    return True
                self.fmt_dict(user_info)
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_users}
                    (user_id, source, id, username, email, homepage,
                        avatar_url, create_t, update_t, extra_info, tag)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    user_info["user_id"],
                    user_info["source"],
                    user_info["id"],
                    user_info["username"],
                    user_info["email"],
                    user_info["homepage"],
                    user_info["avatar_url"],
                    user_info["create_t"],
                    user_info["update_t"],
                    user_info["extra_info"],
                    user_info["tag"],
                )
                return True
        except:  # noqa
            logger.error(f"insert_user_if_not_exists error: {traceback.format_exc()}")
            return False
        finally:
            await self.release_conn(conn)

    @class_try_catch_async
    async def query_user(self, user_id):
        conn = await self.get_conn()
        try:
            row = await conn.fetchrow(f"SELECT * FROM {self.table_users} WHERE user_id = $1", user_id)
            user = dict(row)
            self.parse_dict(user)
            return user
        except:  # noqa
            logger.error(f"query_user error: {traceback.format_exc()}")
            return None
        finally:
            await self.release_conn(conn)


async def test():
    from lightx2v.deploy.common.pipeline import Pipeline

    p = Pipeline("/data/nvme1/liuliang1/lightx2v/configs/model_pipeline.json")
    m = PostgresSQLTaskManager("postgresql://test:test@127.0.0.1:5432/lightx2v_test")
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
        "id": "4566",
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
