import argparse
import asyncio
import json
import mimetypes
import os
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from loguru import logger

from lightx2v.deploy.common.pipeline import Pipeline
from lightx2v.deploy.common.utils import check_params, data_name, load_inputs
from lightx2v.deploy.data_manager import LocalDataManager, S3DataManager
from lightx2v.deploy.queue_manager import LocalQueueManager, RabbitMQQueueManager
from lightx2v.deploy.server.auth import AuthManager
from lightx2v.deploy.server.metrics import MetricMonitor
from lightx2v.deploy.server.monitor import ServerMonitor, WorkerStatus
from lightx2v.deploy.server.redis_monitor import RedisServerMonitor
from lightx2v.deploy.task_manager import FinishedStatus, LocalTaskManager, PostgresSQLTaskManager, TaskStatus
from lightx2v.utils.service_utils import ProcessManager

# =========================
# FastAPI Related Code
# =========================

model_pipelines = None
task_manager = None
data_manager = None
queue_manager = None
server_monitor = None
auth_manager = None
metrics_monitor = MetricMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await task_manager.init()
    await task_manager.mark_server_restart()
    await data_manager.init()
    await queue_manager.init()
    await server_monitor.init()
    asyncio.create_task(server_monitor.loop())
    yield
    await server_monitor.close()
    await queue_manager.close()
    await data_manager.close()
    await task_manager.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail} for {request.url}")
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 添加assets目录的静态文件服务
assets_dir = os.path.join(os.path.dirname(__file__), "static", "assets")
app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
security = HTTPBearer()


async def verify_user_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = auth_manager.verify_jwt_token(token)
    user_id = payload.get("user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    user = await task_manager.query_user(user_id)
    # logger.info(f"Verfiy user access: {payload}")
    if user is None or user["user_id"] != user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    return user


async def verify_user_access_from_query(request: Request):
    """从查询参数中验证用户访问权限"""
    # 首先尝试从 Authorization 头部获取 token
    auth_header = request.headers.get("Authorization")
    token = None

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]  # 移除 "Bearer " 前缀
    else:
        # 如果没有 Authorization 头部，尝试从查询参数获取
        token = request.query_params.get("token")

    payload = auth_manager.verify_jwt_token(token)
    user_id = payload.get("user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    user = await task_manager.query_user(user_id)
    if user is None or user["user_id"] != user_id:
        raise HTTPException(status_code=401, detail="Invalid user")
    return user


async def verify_worker_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not auth_manager.verify_worker_token(token):
        raise HTTPException(status_code=403, detail="Invalid worker token")
    return True


def error_response(e, code):
    return JSONResponse({"message": f"error: {e}!"}, status_code=code)


def guess_file_type(name, default_type):
    content_type, _ = mimetypes.guess_type(name)
    if content_type is None:
        content_type = default_type
    return content_type


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/auth/login/github")
async def github_auth(request: Request):
    client_id = auth_manager.github_client_id
    redirect_uri = f"{request.base_url}"
    auth_url = f"https://github.com/login/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}"
    return {"auth_url": auth_url}


@app.get("/auth/callback/github")
async def github_callback(request: Request):
    try:
        code = request.query_params.get("code")
        if not code:
            return error_response("Missing authorization code", 400)
        user_info = await auth_manager.auth_github(code)
        user_id = await task_manager.create_user(user_info)
        user_info["user_id"] = user_id
        access_token = auth_manager.create_jwt_token(user_info)
        logger.info(f"GitHub callback: user_info: {user_info}, access_token: {access_token}")
        return {"access_token": access_token, "user_info": user_info}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/auth/login/google")
async def google_auth(request: Request):
    client_id = auth_manager.google_client_id
    redirect_uri = auth_manager.google_redirect_uri
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=openid%20email%20profile&access_type=offline"
    logger.info(f"Google auth: auth_url: {auth_url}")
    return {"auth_url": auth_url}


@app.get("/auth/callback/google")
async def google_callback(request: Request):
    try:
        code = request.query_params.get("code")
        if not code:
            return error_response("Missing authorization code", 400)
        user_info = await auth_manager.auth_google(code)
        user_id = await task_manager.create_user(user_info)
        user_info["user_id"] = user_id
        access_token = auth_manager.create_jwt_token(user_info)
        logger.info(f"Google callback: user_info: {user_info}, access_token: {access_token}")
        return {"access_token": access_token, "user_info": user_info}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/auth/login/sms")
async def sms_auth(request: Request):
    try:
        phone_number = request.query_params.get("phone_number")
        if not phone_number:
            return error_response("Missing phone number", 400)
        ok = await auth_manager.send_sms(phone_number)
        if not ok:
            return error_response("SMS send failed", 400)
        return {"msg": "SMS send successfully"}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/auth/callback/sms")
async def sms_callback(request: Request):
    try:
        phone_number = request.query_params.get("phone_number")
        verify_code = request.query_params.get("verify_code")
        if not phone_number or not verify_code:
            return error_response("Missing phone number or verify code", 400)
        user_info = await auth_manager.check_sms(phone_number, verify_code)
        if not user_info:
            return error_response("SMS verify failed", 400)

        user_id = await task_manager.create_user(user_info)
        user_info["user_id"] = user_id
        access_token = auth_manager.create_jwt_token(user_info)
        logger.info(f"SMS callback: user_info: {user_info}, access_token: {access_token}")
        return {"access_token": access_token, "user_info": user_info}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


async def prepare_subtasks(task_id):
    # schedule next subtasks and pend, put to message queue
    subtasks = await task_manager.next_subtasks(task_id)
    for sub in subtasks:
        logger.info(f"Prepare ready subtask: ({task_id}, {sub['worker_name']})")
        r = await queue_manager.put_subtask(sub)
        assert r, "put subtask to queue error"
        await server_monitor.pending_subtasks_add(sub["queue"], sub["task_id"])


def format_task(task):
    task["status"] = task["status"].name
    task["model_cls"] = model_pipelines.outer_model_name(task["model_cls"])


@app.get("/api/v1/model/list")
async def api_v1_model_list(user=Depends(verify_user_access)):
    try:
        return {"models": model_pipelines.get_model_lists()}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/task/submit")
async def api_v1_task_submit(request: Request, user=Depends(verify_user_access)):
    task_id = None
    try:
        msg = await server_monitor.check_user_busy(user["user_id"], active_new_task=True)
        if msg is not True:
            return error_response(msg, 400)
        params = await request.json()
        keys = [params.pop("task"), params.pop("model_cls"), params.pop("stage")]
        keys[1] = model_pipelines.inner_model_name(keys[1])
        assert len(params["prompt"]) > 0, "valid prompt is required"

        # get worker infos, model input names
        workers = model_pipelines.get_workers(keys)
        inputs = model_pipelines.get_inputs(keys)
        outputs = model_pipelines.get_outputs(keys)
        types = model_pipelines.get_types(keys)
        check_params(params, inputs, outputs, types)

        # check if task can be published to queues
        queues = [v["queue"] for v in workers.values()]
        wait_time = await server_monitor.check_queue_busy(keys, queues)
        if wait_time is None:
            return error_response(f"Queue busy, please try again later", 500)

        # process multimodal inputs data
        inputs_data = await load_inputs(params, inputs, types)

        # init task
        task_id = await task_manager.create_task(keys, workers, params, inputs, outputs, user["user_id"])
        logger.info(f"Submit task: {task_id} {params}")

        # save multimodal inputs data
        for inp, data in inputs_data.items():
            await data_manager.save_bytes(data, data_name(inp, task_id))

        await prepare_subtasks(task_id)
        return {"task_id": task_id, "workers": workers, "params": params, "wait_time": wait_time}

    except Exception as e:
        traceback.print_exc()
        if task_id:
            await task_manager.finish_subtasks(task_id, TaskStatus.FAILED, fail_msg=f"submit failed: {e}")
        return error_response(str(e), 500)


@app.get("/api/v1/task/query")
async def api_v1_task_query(request: Request, user=Depends(verify_user_access)):
    try:
        if "task_ids" in request.query_params:
            task_ids = request.query_params["task_ids"].split(",")
            tasks = []
            for task_id in task_ids:
                task_id = task_id.strip()
                if task_id:
                    task, subtasks = await task_manager.query_task(task_id, user["user_id"], only_task=False)
                    if task is not None:
                        task["subtasks"] = await server_monitor.format_subtask(subtasks)
                        format_task(task)
                        tasks.append(task)
            return {"tasks": tasks}

        # 单个任务查询
        task_id = request.query_params["task_id"]
        task, subtasks = await task_manager.query_task(task_id, user["user_id"], only_task=False)
        if task is None:
            return error_response(f"Task {task_id} not found", 404)
        task["subtasks"] = await server_monitor.format_subtask(subtasks)
        format_task(task)
        return task
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/list")
async def api_v1_task_list(request: Request, user=Depends(verify_user_access)):
    try:
        user_id = user["user_id"]
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 10))
        assert page > 0 and page_size > 0, "page and page_size must be greater than 0"
        status_filter = request.query_params.get("status", None)

        query_params = {"user_id": user_id}
        if status_filter and status_filter != "ALL":
            query_params["status"] = TaskStatus[status_filter.upper()]

        total_tasks = await task_manager.list_tasks(count=True, **query_params)
        total_pages = (total_tasks + page_size - 1) // page_size
        page_info = {"page": page, "page_size": page_size, "total": total_tasks, "total_pages": total_pages}
        if page > total_pages:
            return {"tasks": [], "pagination": page_info}

        query_params["offset"] = (page - 1) * page_size
        query_params["limit"] = page_size

        tasks = await task_manager.list_tasks(**query_params)
        for task in tasks:
            format_task(task)

        return {"tasks": tasks, "pagination": page_info}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/result_url")
async def api_v1_task_result_url(request: Request, user=Depends(verify_user_access)):
    try:
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert task["status"] == TaskStatus.SUCCEED, f"Task {task_id} not succeed"
        assert name in task["outputs"], f"Output {name} not found in task {task_id}"
        assert name not in task["params"], f"Output {name} is a stream"

        url = await data_manager.presign_url(task["outputs"][name])
        if url is None:
            url = f"./assets/task/result?task_id={task_id}&name={name}"
        return {"url": url}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/input_url")
async def api_v1_task_input_url(request: Request, user=Depends(verify_user_access)):
    try:
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert name in task["inputs"], f"Input {name} not found in task {task_id}"
        assert name not in task["params"], f"Input {name} is a stream"

        url = await data_manager.presign_url(task["inputs"][name])
        if url is None:
            url = f"./assets/task/input?task_id={task_id}&name={name}"
        return {"url": url}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/assets/task/result")
async def assets_task_result(request: Request, user=Depends(verify_user_access_from_query)):
    try:
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert task["status"] == TaskStatus.SUCCEED, f"Task {task_id} not succeed"
        assert name in task["outputs"], f"Output {name} not found in task {task_id}"
        assert name not in task["params"], f"Output {name} is a stream"
        data = await data_manager.load_bytes(task["outputs"][name])

        #  set correct Content-Type
        content_type = guess_file_type(name, "application/octet-stream")
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        headers["Cache-Control"] = "public, max-age=3600"
        return Response(content=data, media_type=content_type, headers=headers)

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/assets/task/input")
async def assets_task_input(request: Request, user=Depends(verify_user_access_from_query)):
    try:
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert name in task["inputs"], f"Input {name} not found in task {task_id}"
        assert name not in task["params"], f"Input {name} is a stream"
        data = await data_manager.load_bytes(task["inputs"][name])

        #  set correct Content-Type
        content_type = guess_file_type(name, "application/octet-stream")
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        headers["Cache-Control"] = "public, max-age=3600"
        return Response(content=data, media_type=content_type, headers=headers)

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/cancel")
async def api_v1_task_cancel(request: Request, user=Depends(verify_user_access)):
    try:
        task_id = request.query_params["task_id"]
        ret = await task_manager.cancel_task(task_id, user_id=user["user_id"])
        logger.warning(f"Task {task_id} cancelled: {ret}")
        if ret is True:
            return {"msg": "Task cancelled successfully"}
        else:
            return error_response({"error": f"Task {task_id} cancel failed: {ret}"}, 400)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/resume")
async def api_v1_task_resume(request: Request, user=Depends(verify_user_access)):
    try:
        task_id = request.query_params["task_id"]

        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        keys = [task["task_type"], task["model_cls"], task["stage"]]
        if not model_pipelines.check_item_by_keys(keys):
            return error_response(f"Model {keys} is not supported now, please submit a new task", 400)

        ret = await task_manager.resume_task(task_id, user_id=user["user_id"], all_subtask=False)
        if ret is True:
            await prepare_subtasks(task_id)
            return {"msg": "ok"}
        else:
            return error_response(f"Task {task_id} resume failed: {ret}", 400)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.delete("/api/v1/task/delete")
async def api_v1_task_delete(request: Request, user=Depends(verify_user_access)):
    try:
        task_id = request.query_params["task_id"]

        task = await task_manager.query_task(task_id, user["user_id"], only_task=True)
        if not task:
            return error_response("Task not found", 404)

        if task["status"] not in FinishedStatus:
            return error_response("Only finished tasks can be deleted", 400)

        # delete task record
        success = await task_manager.delete_task(task_id, user["user_id"])
        if success:
            logger.info(f"Task {task_id} deleted by user {user['user_id']}")
            return JSONResponse({"message": "Task deleted successfully"})
        else:
            return error_response("Failed to delete task", 400)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/worker/fetch")
async def api_v1_worker_fetch(request: Request, valid=Depends(verify_worker_access)):
    try:
        params = await request.json()
        logger.info(f"Worker fetching: {params}")
        keys = params.pop("worker_keys")
        identity = params.pop("worker_identity")
        max_batch = params.get("max_batch", 1)
        timeout = params.get("timeout", 5)

        # check client disconnected
        async def check_client(request, fetch_task, identity, queue):
            while True:
                msg = await request.receive()
                if msg["type"] == "http.disconnect":
                    logger.warning(f"Worker {identity} {queue} disconnected, req: {request.client}, {msg}")
                    fetch_task.cancel()
                    await server_monitor.worker_update(queue, identity, WorkerStatus.DISCONNECT)
                    return
                await asyncio.sleep(1)

        # get worker info
        worker = model_pipelines.get_worker(keys)
        await server_monitor.worker_update(worker["queue"], identity, WorkerStatus.FETCHING)

        fetch_task = asyncio.create_task(queue_manager.get_subtasks(worker["queue"], max_batch, timeout))
        check_task = asyncio.create_task(check_client(request, fetch_task, identity, worker["queue"]))
        try:
            subtasks = await asyncio.wait_for(fetch_task, timeout=timeout)
        except asyncio.TimeoutError:
            subtasks = []
            fetch_task.cancel()
        check_task.cancel()

        subtasks = [] if subtasks is None else subtasks
        for sub in subtasks:
            await server_monitor.pending_subtasks_sub(sub["queue"], sub["task_id"])
        valid_subtasks = await task_manager.run_subtasks(subtasks, identity)
        valids = [sub["task_id"] for sub in valid_subtasks]

        if len(valid_subtasks) > 0:
            await server_monitor.worker_update(worker["queue"], identity, WorkerStatus.FETCHED)
            logger.info(f"Worker {identity} {keys} {request.client} fetched {valids}")
        else:
            await server_monitor.worker_update(worker["queue"], identity, WorkerStatus.DISCONNECT)
        return {"subtasks": valid_subtasks}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/worker/report")
async def api_v1_worker_report(request: Request, valid=Depends(verify_worker_access)):
    try:
        params = await request.json()
        logger.info(f"{params}")
        task_id = params.pop("task_id")
        worker_name = params.pop("worker_name")
        status = TaskStatus[params.pop("status")]
        identity = params.pop("worker_identity")
        queue = params.pop("queue")
        fail_msg = params.pop("fail_msg", None)
        await server_monitor.worker_update(queue, identity, WorkerStatus.REPORT)

        ret = await task_manager.finish_subtasks(task_id, status, worker_identity=identity, worker_name=worker_name, fail_msg=fail_msg, should_running=True)

        # not all subtasks finished, prepare new ready subtasks
        if ret not in [TaskStatus.SUCCEED, TaskStatus.FAILED]:
            await prepare_subtasks(task_id)

        # all subtasks succeed, delete temp data
        elif ret == TaskStatus.SUCCEED:
            logger.info(f"Task {task_id} succeed")
            task = await task_manager.query_task(task_id)
            keys = [task["task_type"], task["model_cls"], task["stage"]]
            temps = model_pipelines.get_temps(keys)
            for temp in temps:
                type = model_pipelines.get_type(temp)
                name = data_name(temp, task_id)
                await data_manager.get_delete_func(type)(name)

        elif ret == TaskStatus.FAILED:
            logger.warning(f"Task {task_id} failed")

        return {"msg": "ok"}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/worker/ping/subtask")
async def api_v1_worker_ping_subtask(request: Request, valid=Depends(verify_worker_access)):
    try:
        params = await request.json()
        logger.info(f"{params}")
        task_id = params.pop("task_id")
        worker_name = params.pop("worker_name")
        identity = params.pop("worker_identity")
        queue = params.pop("queue")

        task = await task_manager.query_task(task_id)
        if task is None or task["status"] != TaskStatus.RUNNING:
            return {"msg": "delete"}

        assert await task_manager.ping_subtask(task_id, worker_name, identity)
        await server_monitor.worker_update(queue, identity, WorkerStatus.PING)
        return {"msg": "ok"}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/metrics")
async def api_v1_monitor_metrics():
    try:
        return Response(content=metrics_monitor.get_metrics(), media_type="text/plain")
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/template/asset_url/{template_type}/{filename}")
async def api_v1_template_asset_url(template_type: str, filename: str):
    """get template asset URL - no authentication required"""
    try:
        url = await data_manager.presign_template_url(template_type, filename)
        if url is None:
            url = f"./assets/template/{template_type}/{filename}"
        headers = {"Cache-Control": "public, max-age=3600"}
        return Response(content=json.dumps({"url": url}), media_type="application/json", headers=headers)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


# Template API endpoints
@app.get("/assets/template/{template_type}/{filename}")
async def assets_template(template_type: str, filename: str):
    """get template file - no authentication required"""
    try:
        if not await data_manager.template_file_exists(template_type, filename):
            return error_response(f"template file {template_type} {filename} not found", 404)
        data = await data_manager.load_template_file(template_type, filename)

        # set media type according to file type
        if template_type == "images":
            if filename.lower().endswith(".png"):
                media_type = "image/png"
            elif filename.lower().endswith((".jpg", ".jpeg")):
                media_type = "image/jpeg"
            else:
                media_type = "application/octet-stream"
        elif template_type == "audios":
            if filename.lower().endswith(".mp3"):
                media_type = "audio/mpeg"
            elif filename.lower().endswith(".wav"):
                media_type = "audio/wav"
            else:
                media_type = "application/octet-stream"
        elif template_type == "videos":
            if filename.lower().endswith(".mp4"):
                media_type = "video/mp4"
            elif filename.lower().endswith(".webm"):
                media_type = "video/webm"
            elif filename.lower().endswith(".avi"):
                media_type = "video/x-msvideo"
            else:
                media_type = "video/mp4"  # default to mp4
        else:
            media_type = "application/octet-stream"

        headers = {"Cache-Control": "public, max-age=3600"}
        return Response(content=data, media_type=media_type, headers=headers)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/template/list")
async def api_v1_template_list(request: Request):
    """get template file list (support pagination) - no authentication required"""
    try:
        # check page params
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 12))
        if page < 1 or page_size < 1:
            return error_response("page and page_size must be greater than 0", 400)
        # limit page size
        page_size = min(page_size, 100)

        all_images = await data_manager.list_template_files("images")
        all_audios = await data_manager.list_template_files("audios")
        all_videos = await data_manager.list_template_files("videos")
        all_images = [] if all_images is None else all_images
        all_audios = [] if all_audios is None else all_audios
        all_videos = [] if all_videos is None else all_videos

        # page info
        total_images = len(all_images)
        total_audios = len(all_audios)
        total_videos = len(all_videos)
        total_pages = (max(total_images, total_audios, total_videos) + page_size - 1) // page_size

        paginated_image_templates = []
        paginated_audio_templates = []
        paginated_video_templates = []

        if page <= total_pages:
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size

            async def handle_media(media_type, media_names, paginated_media_templates):
                media_names.sort(key=lambda x: x)
                for media_name in media_names[start_idx:end_idx]:
                    url = await data_manager.presign_template_url(media_type, media_name)
                    if url is None:
                        url = f"./assets/template/{media_type}/{media_name}"
                    paginated_media_templates.append({"filename": media_name, "url": url})

            await handle_media("images", all_images, paginated_image_templates)
            await handle_media("audios", all_audios, paginated_audio_templates)
            await handle_media("videos", all_videos, paginated_video_templates)

        return {
            "templates": {"images": paginated_image_templates, "audios": paginated_audio_templates, "videos": paginated_video_templates},
            "pagination": {"page": page, "page_size": page_size, "total": max(total_images, total_audios), "total_pages": total_pages},
        }
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/template/tasks")
async def api_v1_template_tasks(request: Request):
    """get template task list (support pagination) - no authentication required"""
    try:
        # check page params
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 12))
        category = request.query_params.get("category", None)
        search = request.query_params.get("search", None)
        if page < 1 or page_size < 1:
            return error_response("page and page_size must be greater than 0", 400)
        # limit page size
        page_size = min(page_size, 100)

        all_templates = []
        all_categories = set()
        template_files = await data_manager.list_template_files("tasks")
        template_files = [] if template_files is None else template_files

        for template_file in template_files:
            try:
                bytes_data = await data_manager.load_template_file("tasks", template_file)
                template_data = json.loads(bytes_data)
                template_data["task"]["model_cls"] = model_pipelines.outer_model_name(template_data["task"]["model_cls"])
                all_categories.update(template_data["task"]["tags"])
                if category and category not in template_data["task"]["tags"]:
                    continue
                if search and search not in template_data["task"]["params"]["prompt"] + template_data["task"]["params"]["negative_prompt"] + template_data["task"][
                    "model_cls"
                ] + template_data["task"]["stage"] + template_data["task"]["task_type"] + ",".join(template_data["task"]["tags"]):
                    continue
                all_templates.append(template_data["task"])
            except Exception as e:
                logger.warning(f"Failed to load template file {template_file}: {e}")

        # page info
        total_templates = len(all_templates)
        total_pages = (total_templates + page_size - 1) // page_size
        paginated_templates = []

        if page <= total_pages:
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_templates = all_templates[start_idx:end_idx]

        return {"templates": paginated_templates, "pagination": {"page": page, "page_size": page_size, "total": total_templates, "total_pages": total_pages}, "categories": list(all_categories)}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/template/{template_id}")
async def api_v1_template_get(template_id: str, user=None):
    try:
        template_files = await data_manager.list_template_files("tasks")
        template_files = [] if template_files is None else template_files

        for template_file in template_files:
            try:
                bytes_data = await data_manager.load_template_file("tasks", template_file)
                template_data = json.loads(bytes_data)
                if template_data["task"]["task_id"] == template_id:
                    return template_data["task"]
            except Exception as e:
                logger.warning(f"Failed to load template file {template_file}: {e}")
                continue
        return error_response("Template not found", 404)

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/share/create")
async def api_v1_share_create(request: Request, user=Depends(verify_user_access)):
    try:
        params = await request.json()
        task_id = params["task_id"]
        valid_days = params.get("valid_days", 7)
        auth_type = params.get("auth_type", "public")
        auth_value = params.get("auth_value", "")
        share_type = params.get("share_type", "task")
        assert auth_type == "public", "Only public share is supported now"

        if share_type == "template":
            template = await api_v1_template_get(task_id, user)
            assert isinstance(template, dict) and template["task_id"] == task_id, f"Template {task_id} not found"
        else:
            task = await task_manager.query_task(task_id, user["user_id"], only_task=True)
            assert task, f"Task {task_id} not found"

        if auth_type == "user_id":
            assert auth_value != "", "Target user is required for auth_type = user_id"
            target_user = await task_manager.query_user(auth_value)
            assert target_user and target_user["user_id"] == auth_value, f"Target user {auth_value} not found"

        share_id = await task_manager.create_share(task_id, user["user_id"], share_type, valid_days, auth_type, auth_value)
        return {"share_id": share_id, "share_url": f"/share/{share_id}"}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/share/{share_id}")
async def api_v1_share_get(share_id: str):
    try:
        share_data = await task_manager.query_share(share_id)
        assert share_data, f"Share {share_id} not found or expired or deleted"
        task_id = share_data["task_id"]
        share_type = share_data["share_type"]
        assert share_data["auth_type"] == "public", "Only public share is supported now"

        if share_type == "template":
            task = await api_v1_template_get(task_id, None)
            assert isinstance(task, dict) and task["task_id"] == task_id, f"Template {task_id} not found"
        else:
            task = await task_manager.query_task(task_id, only_task=True)
            assert task, f"Task {task_id} not found"

        user_info = await task_manager.query_user(share_data["user_id"])
        username = user_info.get("username", "用户") if user_info else "用户"

        share_info = {
            "task_id": task_id,
            "share_type": share_type,
            "user_id": share_data["user_id"],
            "username": username,
            "task_type": task["task_type"],
            "model_cls": task["model_cls"],
            "stage": task["stage"],
            "prompt": task["params"].get("prompt", ""),
            "negative_prompt": task["params"].get("negative_prompt", ""),
            "inputs": task["inputs"],
            "outputs": task["outputs"],
            "create_t": task["create_t"],
            "valid_days": share_data["valid_days"],
            "valid_t": share_data["valid_t"],
            "auth_type": share_data["auth_type"],
            "auth_value": share_data["auth_value"],
            "output_video_url": None,
            "input_urls": {},
        }

        for input_name, input_filename in task["inputs"].items():
            if share_type == "template":
                template_type = "images" if "image" in input_name else "audios"
                input_url = await data_manager.presign_template_url(template_type, input_filename)
            else:
                input_url = await data_manager.presign_url(input_filename)
            share_info["input_urls"][input_name] = input_url

        for output_name, output_filename in task["outputs"].items():
            if share_type == "template":
                assert "video" in output_name, "Only video output is supported for template share"
                output_url = await data_manager.presign_template_url("videos", output_filename)
            else:
                output_url = await data_manager.presign_url(output_filename)
            share_info["output_video_url"] = output_url

        return share_info

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)

# 所有未知路由 fallback 到 index.html (必须在所有API路由之后)
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def vue_fallback(full_path: str):
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    ProcessManager.register_signal_handler()
    parser = argparse.ArgumentParser()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(cur_dir, "../../.."))
    dft_pipeline_json = os.path.join(base_dir, "configs/model_pipeline.json")
    dft_task_url = os.path.join(base_dir, "local_task")
    dft_data_url = os.path.join(base_dir, "local_data")
    dft_queue_url = os.path.join(base_dir, "local_queue")

    parser.add_argument("--pipeline_json", type=str, default=dft_pipeline_json)
    parser.add_argument("--task_url", type=str, default=dft_task_url)
    parser.add_argument("--data_url", type=str, default=dft_data_url)
    parser.add_argument("--queue_url", type=str, default=dft_queue_url)
    parser.add_argument("--redis_url", type=str, default="")
    parser.add_argument("--template_dir", type=str, default="")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_pipelines = Pipeline(args.pipeline_json)
    auth_manager = AuthManager()
    if args.task_url.startswith("/"):
        task_manager = LocalTaskManager(args.task_url, metrics_monitor)
    elif args.task_url.startswith("postgresql://"):
        task_manager = PostgresSQLTaskManager(args.task_url, metrics_monitor)
    else:
        raise NotImplementedError
    if args.data_url.startswith("/"):
        data_manager = LocalDataManager(args.data_url, args.template_dir)
    elif args.data_url.startswith("{"):
        data_manager = S3DataManager(args.data_url, args.template_dir)
    else:
        raise NotImplementedError
    if args.queue_url.startswith("/"):
        queue_manager = LocalQueueManager(args.queue_url)
    elif args.queue_url.startswith("amqp://"):
        queue_manager = RabbitMQQueueManager(args.queue_url)
    else:
        raise NotImplementedError
    if args.redis_url:
        server_monitor = RedisServerMonitor(model_pipelines, task_manager, queue_manager, args.redis_url)
    else:
        server_monitor = ServerMonitor(model_pipelines, task_manager, queue_manager)

    uvicorn.run(app, host=args.ip, port=args.port, reload=False, workers=1)
