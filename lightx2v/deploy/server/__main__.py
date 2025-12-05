import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import tempfile
import traceback
import uuid
import copy
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from lightx2v.deploy.common.audio_separator import AudioSeparator
from lightx2v.deploy.common.face_detector import FaceDetector
from lightx2v.deploy.common.pipeline import Pipeline
from lightx2v.deploy.common.podcasts import VolcEnginePodcastClient
from lightx2v.deploy.common.utils import check_params, data_name, fetch_resource, format_image_data, load_inputs
from lightx2v.deploy.common.volcengine_tts import VolcEngineTTSClient
from lightx2v.deploy.data_manager import LocalDataManager, S3DataManager
from lightx2v.deploy.queue_manager import LocalQueueManager, RabbitMQQueueManager
from lightx2v.deploy.server.auth import AuthManager
from lightx2v.deploy.server.metrics import MetricMonitor
from lightx2v.deploy.server.monitor import ServerMonitor, WorkerStatus
from lightx2v.deploy.server.redis_monitor import RedisServerMonitor
from lightx2v.deploy.task_manager import FinishedStatus, LocalTaskManager, PostgresSQLTaskManager, TaskStatus
from lightx2v.utils.service_utils import ProcessManager

# =========================
# Pydantic Models
# =========================


class TTSRequest(BaseModel):
    text: str
    voice_type: str
    context_texts: str = ""
    emotion: str = ""
    emotion_scale: int = 3
    speech_rate: int = 0
    pitch: int = 0
    loudness_rate: int = 0
    resource_id: str = "seed-tts-1.0"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


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
volcengine_tts_client = None
volcengine_podcast_client = None
face_detector = None
audio_separator = None


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


def format_user_response(user):
    return {
        "user_id": user.get("user_id"),
        "id": user.get("id"),
        "source": user.get("source"),
        "username": user.get("username") or "",
        "email": user.get("email") or "",
        "homepage": user.get("homepage") or "",
        "avatar_url": user.get("avatar_url") or "",
    }


def guess_file_type(name, default_type):
    content_type, _ = mimetypes.guess_type(name)
    if content_type is None:
        content_type = default_type
    return content_type


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/sitemap.xml", response_class=HTMLResponse)
async def sitemap():
    with open(os.path.join(os.path.dirname(__file__), "frontend", "dist", "sitemap.xml"), "r", encoding="utf-8") as f:
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
        user_response = format_user_response(user_info)
        access_token, refresh_token = auth_manager.create_tokens(user_response)
        logger.info(f"GitHub callback: user_info: {user_response}, access token issued")
        return {"access_token": access_token, "refresh_token": refresh_token, "user_info": user_response}
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
        user_response = format_user_response(user_info)
        access_token, refresh_token = auth_manager.create_tokens(user_response)
        logger.info(f"Google callback: user_info: {user_response}, access token issued")
        return {"access_token": access_token, "refresh_token": refresh_token, "user_info": user_response}
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
        user_response = format_user_response(user_info)
        access_token, refresh_token = auth_manager.create_tokens(user_response)
        logger.info(f"SMS callback: user_info: {user_response}, access token issued")
        return {"access_token": access_token, "refresh_token": refresh_token, "user_info": user_response}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/auth/refresh")
async def refresh_access_token(request: RefreshTokenRequest):
    try:
        payload = auth_manager.verify_refresh_token(request.refresh_token)
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        user = await task_manager.query_user(user_id)
        if user is None or user.get("user_id") != user_id:
            raise HTTPException(status_code=401, detail="Invalid user")
        user_info = format_user_response(user)
        access_token, refresh_token = auth_manager.create_tokens(user_info)
        return {"access_token": access_token, "refresh_token": refresh_token, "user_info": user_info}
    except HTTPException as exc:
        raise exc
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

        # init task (we need task_id before preprocessing to save processed files)
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
        filename = request.query_params.get("filename", None)

        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert name in task["inputs"], f"Input {name} not found in task {task_id}"
        if name in task["params"]:
            return error_response(f"Input {name} is a stream", 400)

        # eg, multi person audio directory input
        if filename is not None:
            extra_inputs = task["params"]["extra_inputs"][name]
            name = f"{name}/{filename}"
            assert name in task["inputs"], f"Extra input {name} not found in task {task_id}"
            assert name in extra_inputs, f"Filename {filename} not found in extra inputs"

        url = await data_manager.presign_url(task["inputs"][name])
        if url is None:
            url = f"./assets/task/input?task_id={task_id}&name={name}"
            if filename is not None:
                url += f"&filename={filename}"
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
        filename = request.query_params.get("filename", None)

        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        assert task is not None, f"Task {task_id} not found"
        assert name in task["inputs"], f"Input {name} not found in task {task_id}"
        if name in task["params"]:
            return error_response(f"Input {name} is a stream", 400)

        # eg, multi person audio directory input
        if filename is not None:
            extra_inputs = task["params"]["extra_inputs"][name]
            name = f"{name}/{filename}"
            assert name in task["inputs"], f"Extra input {name} not found in task {task_id}"
            assert name in extra_inputs, f"Filename {filename} not found in extra inputs"
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

        # 创建图片文件名（不含扩展名）到图片信息的映射
        all_images_sorted = sorted(all_images)
        image_map = {}  # 文件名（不含扩展名） -> {"filename": 完整文件名, "url": URL}
        for img_name in all_images_sorted:
            img_name_without_ext = img_name.rsplit(".", 1)[0] if "." in img_name else img_name
            url = await data_manager.presign_template_url("images", img_name)
            if url is None:
                url = f"./assets/template/images/{img_name}"
            image_map[img_name_without_ext] = {"filename": img_name, "url": url}

        # 创建音频文件名（不含扩展名）到音频信息的映射
        all_audios_sorted = sorted(all_audios)
        audio_map = {}  # 文件名（不含扩展名） -> {"filename": 完整文件名, "url": URL}
        for audio_name in all_audios_sorted:
            audio_name_without_ext = audio_name.rsplit(".", 1)[0] if "." in audio_name else audio_name
            url = await data_manager.presign_template_url("audios", audio_name)
            if url is None:
                url = f"./assets/template/audios/{audio_name}"
            audio_map[audio_name_without_ext] = {"filename": audio_name, "url": url}

        # 合并音频和图片模板，基于文件名前缀匹配
        # 获取所有唯一的基础文件名（不含扩展名）
        all_base_names = set(list(image_map.keys()) + list(audio_map.keys()))
        all_base_names_sorted = sorted(all_base_names)

        # 构建合并后的模板列表
        merged_templates = []
        for base_name in all_base_names_sorted:
            template_item = {
                "id": base_name,  # 使用基础文件名作为ID
                "image": image_map.get(base_name),
                "audio": audio_map.get(base_name),
            }
            merged_templates.append(template_item)

        # 分页处理
        total = len(merged_templates)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1

        paginated_templates = []
        if page <= total_pages:
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_templates = merged_templates[start_idx:end_idx]

        # 为了保持向后兼容，仍然返回images和audios字段（但可能为空）
        # 同时添加新的merged字段
        return {
            "templates": {
                "images": [],  # 保持向后兼容，但设为空
                "audios": [],  # 保持向后兼容，但设为空
                "videos": [],  # 保持向后兼容
                "merged": paginated_templates,  # 新的合并列表
            },
            "pagination": {"page": page, "page_size": page_size, "total": total, "total_pages": total_pages},
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
                if search and search not in template_data["task"]["params"]["prompt"] + template_data["task"]["params"]["negative_prompt"] + template_data["task"]["model_cls"] + template_data["task"][
                    "stage"
                ] + template_data["task"]["task_type"] + ",".join(template_data["task"]["tags"]):
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


@app.get("/api/v1/voices/list")
async def api_v1_voices_list(request: Request):
    try:
        version = request.query_params.get("version", "all")
        if volcengine_tts_client is None:
            return error_response("Volcengine TTS client not loaded", 500)
        voices = volcengine_tts_client.get_voice_list()
        if voices is None:
            return error_response("No voice list found", 404)
        if version != "all":
            voices = copy.deepcopy(voices)
            voices["voices"] = [v for v in voices["voices"] if v["version"] == version]
        return voices
    except Exception as e:
        traceback.print_exc()
        return error_response("Failed to get voice list", 500)


@app.post("/api/v1/tts/generate")
async def api_v1_tts_generate(request: TTSRequest):
    """Generate TTS audio from text"""
    try:
        # Validate parameters
        if not request.text.strip():
            return JSONResponse({"error": "Text cannot be empty"}, status_code=400)

        if not request.voice_type:
            return JSONResponse({"error": "Voice type is required"}, status_code=400)

        # Generate unique output filename
        output_filename = f"tts_output_{uuid.uuid4().hex}.mp3"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # Generate TTS
        success = await volcengine_tts_client.tts_request(
            text=request.text,
            voice_type=request.voice_type,
            context_texts=request.context_texts,
            emotion=request.emotion,
            emotion_scale=request.emotion_scale,
            speech_rate=request.speech_rate,
            loudness_rate=request.loudness_rate,
            pitch=request.pitch,
            output=output_path,
            resource_id=request.resource_id,
        )

        if success and os.path.exists(output_path):
            # Return the audio file
            return FileResponse(output_path, media_type="audio/mpeg", filename=output_filename)
        else:
            return JSONResponse({"error": "TTS generation failed"}, status_code=500)

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return JSONResponse({"error": f"TTS generation failed: {str(e)}"}, status_code=500)


@app.websocket("/api/v1/podcast/generate")
async def api_v1_podcast_generate_ws(websocket: WebSocket):
    await websocket.accept()

    def ws_get_user_id():
        token = websocket.query_params.get("token")
        if not token:
            token = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
            if token and token.startswith("Bearer "):
                token = token[7:]
        payload = auth_manager.verify_jwt_token(token)
        user_id = payload["user_id"]
        return user_id

    async def safe_send_json(payload):
        try:
            await websocket.send_json(payload)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.warning(f"WebSocket send skipped: {e}")

    try:
        user_id = ws_get_user_id()
        data = await websocket.receive_text()
        request_data = json.loads(data)

        # stop request
        if request_data.get("type") == "stop":
            logger.info("Received stop signal from client")
            await safe_send_json({"type": "stopped"})
            return

        # user input prompt
        input_text = request_data.get("input", "")
        is_url = input_text.startswith(("http://", "https://"))
        if not input_text:
            await safe_send_json({"error": "输入不能为空"})
            return

        session_id = "session_" + str(uuid.uuid4())
        params = {
            "session_id": session_id,
            "data_manager": data_manager,
            "text": "" if is_url else input_text,
            "input_url": input_text if is_url else "",
            "action": 0,
            "use_head_music": False,
            "use_tail_music": False,
            "skip_round_audio_save": False,
        }
        logger.info(f"WebSocket generating podcast with params: {params}")

        # 使用回调函数实时推送音频
        async def on_round_complete(round_info):
            await safe_send_json({"type": "audio_update", "data": round_info})

        params["on_round_complete"] = on_round_complete

        # 创建一个任务来处理停止信号
        async def listen_for_stop(podcast_task):
            while True:
                try:
                    if podcast_task.done():
                        return
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    request = json.loads(data)
                    if request.get("type") == "stop":
                        logger.warning("Stop signal received during podcast generation")
                        podcast_task.cancel()
                        return
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.warning(f"Stop listener ended: {e}")
                    return

        podcast_task = asyncio.create_task(volcengine_podcast_client.podcast_request(**params))
        stop_listener_task = asyncio.create_task(listen_for_stop(podcast_task))
        podcast_info = None

        try:
            podcast_info = await podcast_task
        except asyncio.CancelledError:
            logger.warning("Podcast generation cancelled by user")
            await safe_send_json({"type": "stopped"})
            return
        finally:
            stop_listener_task.cancel()
        if podcast_info is None:
            await safe_send_json({"error": "播客生成失败，请稍后重试"})
            return

        audio_path = podcast_info["audio_name"]
        rounds = podcast_info["subtitles"]
        await task_manager.create_podcast(session_id, user_id, input_text, audio_path, rounds)
        audio_url = await data_manager.presign_podcast_output_url(audio_path)
        if not audio_url:
            audio_url = f"/api/v1/podcast/audio?session_id={session_id}&filename={audio_path}"
        logger.info(f"completed podcast generation (session: {session_id})")

        await safe_send_json(
            {
                "type": "complete",
                "data": {
                    "audio_url": audio_url,
                    "subtitles": podcast_info["subtitles"],
                    "session_id": session_id,
                    "user_id": user_id,
                },
            }
        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

    except Exception:
        logger.error(f"Error in websocket: {traceback.format_exc()}")
        await safe_send_json({"error": "websocket internal error, please try again later!"})


@app.get("/api/v1/podcast/audio")
async def api_v1_podcast_audio(request: Request, user=Depends(verify_user_access_from_query)):
    try:
        user_id = user["user_id"]
        session_id = request.query_params.get("session_id")
        filename = request.query_params.get("filename")
        if not session_id or not filename:
            return JSONResponse({"error": "session_id and filename are required"}, status_code=400)

        ext = os.path.splitext(filename)[1].lower()
        assert ext == ".mp3", f"Unsupported file extension: {ext}"

        # 解析 Range 头，格式：bytes=start-end 或 bytes=start-
        range_header = request.headers.get("Range")
        start_byte, end_byte = None, None
        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start_byte = int(match.group(1))
                end_byte = int(match.group(2)) + 1 if match.group(2) else None

        podcast_data = await task_manager.query_podcast(session_id, user_id)
        if podcast_data:
            # generate is finished and save info to database
            func = data_manager.load_podcast_output_file
            filename = podcast_data["audio_path"]
            func_args = (filename,)
        else:
            func = data_manager.load_podcast_temp_session_file
            func_args = (session_id, filename)

        logger.debug(f"Serving audio file from {func.__name__} with args: {func_args}, start_byte: {start_byte}, end_byte: {end_byte}")
        file_bytes = await func(*func_args)
        file_size = len(file_bytes)
        file_bytes = file_bytes[start_byte:end_byte]

        content_length = len(file_bytes)
        media_type = "audio/mpeg"
        status_code = 200
        headers = {"Content-Length": str(content_length), "Accept-Ranges": "bytes", "Content-Type": media_type, "Content-Disposition": f'attachment; filename="{filename}"'}

        if start_byte is not None and start_byte > 0:
            status_code = 206  # Partial Content
            headers["Content-Range"] = f"bytes {start_byte}-{start_byte + content_length - 1}/{file_size}"
        return Response(content=file_bytes, media_type=media_type, status_code=status_code, headers=headers)

    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v1/podcast/history")
async def api_v1_podcast_history(request: Request, user=Depends(verify_user_access)):
    try:
        user_id = user["user_id"]
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 10))
        assert page > 0 and page_size > 0, "page and page_size must be greater than 0"
        status = request.query_params.get("status", None)  # has_audio, no_audio

        query_params = {"user_id": user_id}
        if status == "has_audio":
            query_params["has_audio"] = True
        elif status == "no_audio":
            query_params["has_audio"] = False

        total_tasks = await task_manager.list_podcasts(count=True, **query_params)
        total_pages = (total_tasks + page_size - 1) // page_size
        page_info = {"page": page, "page_size": page_size, "total": total_tasks, "total_pages": total_pages}
        if page > total_pages:
            return {"sessions": [], "pagination": page_info}

        query_params["offset"] = (page - 1) * page_size
        query_params["limit"] = page_size
        sessions = await task_manager.list_podcasts(**query_params)
        return {"sessions": sessions, "pagination": page_info}

    except Exception as e:
        logger.error(f"Error getting podcast history: {e}")
        traceback.print_exc()
        return {"sessions": []}


@app.get("/api/v1/podcast/session/{session_id}/audio_url")
async def api_v1_podcast_session_audio_url(session_id: str, user=Depends(verify_user_access)):
    try:
        user_id = user["user_id"]
        podcast_data = await task_manager.query_podcast(session_id, user_id)
        if not podcast_data:
            return JSONResponse({"error": "Podcast session not found"}, status_code=404)

        audio_path = podcast_data["audio_path"]
        audio_url = await data_manager.presign_podcast_output_url(audio_path)
        if not audio_url:
            audio_url = f"/api/v1/podcast/audio?session_id={session_id}&filename={audio_path}"
        return {"audio_url": audio_url}

    except Exception as e:
        logger.error(f"Error getting podcast session audio URL: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


class FaceDetectRequest(BaseModel):
    image: str  # Base64 encoded image


class AudioSeparateRequest(BaseModel):
    audio: str  # Base64 encoded audio
    num_speakers: int = None  # Optional: number of speakers to separate


@app.post("/api/v1/face/detect")
async def api_v1_face_detect(request: FaceDetectRequest, user=Depends(verify_user_access)):
    """Detect faces in image (only detection, no cropping - cropping is done on frontend)
    Supports both base64 encoded images and URLs (blob URLs, http URLs, etc.)
    """
    try:
        if not face_detector:
            return error_response("Face detector not initialized", 500)

        # 验证输入
        if not request.image or not request.image.strip():
            logger.error("Face detection request: image is empty")
            return error_response("Image input is empty", 400)

        image_bytes = None
        try:
            # Check if input is a URL (blob:, http:, https:, or data: URL)
            if request.image.startswith(("http://", "https://")):
                timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
                image_bytes = await fetch_resource(request.image, timeout=timeout)
                logger.debug(f"Fetched image from URL for face detection: {request.image[:100]}... (size: {len(image_bytes)} bytes)")
            else:
                encoded = request.image
                # Data URL format: "data:image/png;base64,..."
                if encoded.startswith("data:image"):
                    _, encoded = encoded.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                logger.debug(f"Decoded base64 image: {request.image[:100]}... (size: {len(image_bytes)} bytes)")

            # Validate image format before passing to face detector
            image_bytes = await asyncio.to_thread(format_image_data, image_bytes)

        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}, image length: {len(request.image) if request.image else 0}")
            return error_response(f"Invalid image format: {str(e)}", 400)

        # Detect faces only (no cropping)
        result = await asyncio.to_thread(face_detector.detect_faces, image_bytes, return_image=False)
        faces_data = []
        for i, face in enumerate(result["faces"]):
            faces_data.append(
                {
                    "index": i,
                    "bbox": face["bbox"],  # [x1, y1, x2, y2] - absolute pixel coordinates in original image
                    "confidence": face["confidence"],
                    "class_id": face["class_id"],
                    "class_name": face["class_name"],
                    # Note: face_image is not included - frontend will crop it based on bbox
                }
            )
        return {"faces": faces_data, "total": len(faces_data)}

    except Exception as e:
        logger.error(f"Face detection error: {traceback.format_exc()}")
        return error_response(f"Face detection failed: {str(e)}", 500)


@app.post("/api/v1/audio/separate")
async def api_v1_audio_separate(request: AudioSeparateRequest, user=Depends(verify_user_access)):
    """Separate different speakers in audio"""
    try:
        if not audio_separator:
            return error_response("Audio separator not initialized", 500)
        audio_bytes = None
        try:
            encoded = request.audio
            if encoded.startswith("data:"):
                # Remove data URL prefix (e.g., "data:audio/mpeg;base64," or "data:application/octet-stream;base64,")
                _, encoded = encoded.split(",", 1)
            audio_bytes = await asyncio.to_thread(base64.b64decode, encoded, validate=True)
            logger.debug(f"Successfully decoded base64 audio, size: {len(audio_bytes)} bytes")

        except Exception as e:
            logger.error(f"Failed to decode base64 audio {request.audio[:100]}..., error: {str(e)}")
            return error_response(f"Invalid base64 audio data", 400)

        # Separate speakers
        result = await asyncio.to_thread(audio_separator.separate_speakers, audio_bytes, num_speakers=request.num_speakers)

        # Convert audio tensors to base64 strings (without saving to file)
        speakers_data = []
        for speaker in result["speakers"]:
            # Convert audio tensor directly to base64
            audio_base64 = await asyncio.to_thread(audio_separator.speaker_audio_to_base64, speaker["audio"], speaker["sample_rate"], format="wav")
            speakers_data.append(
                {
                    "speaker_id": speaker["speaker_id"],
                    "audio": audio_base64,  # Base64 encoded audio
                    "segments": speaker["segments"],
                    "sample_rate": speaker["sample_rate"],
                }
            )
        return {"speakers": speakers_data, "total": len(speakers_data), "method": result.get("method", "pyannote")}

    except Exception as e:
        logger.error(f"Audio separation error: {traceback.format_exc()}")
        return error_response(f"Audio separation failed: {str(e)}", 500)


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
    dft_volcengine_tts_list_json = os.path.join(base_dir, "configs/volcengine_voices_list.json")

    parser.add_argument("--pipeline_json", type=str, default=dft_pipeline_json)
    parser.add_argument("--task_url", type=str, default=dft_task_url)
    parser.add_argument("--data_url", type=str, default=dft_data_url)
    parser.add_argument("--queue_url", type=str, default=dft_queue_url)
    parser.add_argument("--redis_url", type=str, default="")
    parser.add_argument("--template_dir", type=str, default="")
    parser.add_argument("--volcengine_tts_list_json", type=str, default=dft_volcengine_tts_list_json)
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--face_detector_model_path", type=str, default=None)
    parser.add_argument("--audio_separator_model_path", type=str, default="")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_pipelines = Pipeline(args.pipeline_json)
    volcengine_tts_client = VolcEngineTTSClient(args.volcengine_tts_list_json)
    volcengine_podcast_client = VolcEnginePodcastClient()
    face_detector = FaceDetector(model_path=args.face_detector_model_path)
    audio_separator = AudioSeparator(model_path=args.audio_separator_model_path)
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
