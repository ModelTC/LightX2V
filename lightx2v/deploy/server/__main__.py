import argparse
import asyncio
import mimetypes
import os
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
from lightx2v.deploy.task_manager import LocalTaskManager, PostgresSQLTaskManager, TaskStatus
from lightx2v.utils.profiler import *
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
    asyncio.create_task(server_monitor.init())
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


async def prepare_subtasks(task_id):
    # schedule next subtasks and pend, put to message queue
    subtasks = await task_manager.next_subtasks(task_id)
    for sub in subtasks:
        logger.info(f"Prepare ready subtask: ({task_id}, {sub['worker_name']})")
        r = await queue_manager.put_subtask(sub)
        assert r, "put subtask to queue error"


@app.get("/api/v1/model/list")
async def api_v1_model_list(user=Depends(verify_user_access)):
    try:
        msg = await server_monitor.check_user_busy(user["user_id"])
        if msg is not True:
            return error_response(msg, 400)
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
        msg = await server_monitor.check_user_busy(user["user_id"])
        if msg is not True:
            return error_response(msg, 400)
        task_id = request.query_params["task_id"]
        task, subtasks = await task_manager.query_task(task_id, user["user_id"], only_task=False)
        if task is None:
            return error_response(f"Task {task_id} not found", 404)
        for sub in subtasks:
            sub["status"] = sub["status"].name
        task["subtasks"] = subtasks
        task["status"] = task["status"].name
        return task
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/list")
async def api_v1_task_list(request: Request, user=Depends(verify_user_access)):
    try:
        user_id = user["user_id"]
        msg = await server_monitor.check_user_busy(user_id)
        if msg is not True:
            return error_response(msg, 400)

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
            task["status"] = task["status"].name

        return {"tasks": tasks, "pagination": page_info}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/result")
async def api_v1_task_result(request: Request, user=Depends(verify_user_access_from_query)):
    try:
        msg = await server_monitor.check_user_busy(user["user_id"])
        if msg is not True:
            return error_response(msg, 400)
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        if task is None:
            return error_response(f"Task {task_id} not found", 404)
        if task["status"] != TaskStatus.SUCCEED:
            return error_response(f"Task {task_id} not succeed", 400)
        assert name in task["outputs"], f"Output {name} not found in task {task_id}"
        if name in task["params"]:
            return error_response(f"Output {name} is a stream", 400)
        data = await data_manager.load_bytes(task["outputs"][name])

        #  set correct Content-Type
        content_type, _ = mimetypes.guess_type(name)
        if content_type is None:
            content_type = "application/octet-stream"
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        return Response(content=data, media_type=content_type, headers=headers)

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/input")
async def api_v1_task_input(request: Request, user=Depends(verify_user_access_from_query)):
    try:
        # msg = await server_monitor.check_user_busy(user['user_id'])
        # if msg is not True:
        #     return error_response(msg, 400)
        name = request.query_params["name"]
        task_id = request.query_params["task_id"]
        task = await task_manager.query_task(task_id, user_id=user["user_id"])
        if task is None:
            return error_response(f"Task {task_id} not found", 404)
        if name not in task["inputs"]:
            return error_response(f"Input {name} not found in task {task_id}", 404)
        if name in task["params"]:
            return error_response(f"Input {name} is a stream", 400)
        data = await data_manager.load_bytes(task["inputs"][name])

        #  set correct Content-Type
        content_type, _ = mimetypes.guess_type(name)
        if content_type is None:
            content_type = "application/octet-stream"
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        return Response(content=data, media_type=content_type, headers=headers)

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/thumbnails")
async def api_v1_task_thumbnails(request: Request, user=Depends(verify_user_access)):
    """一次性获取所有任务的缩略图"""
    try:
        user_id = user["user_id"]
        msg = await server_monitor.check_user_busy(user_id)
        if msg is not True:
            return error_response(msg, 400)

        # 获取所有任务
        tasks = await task_manager.list_tasks(user_id=user_id, limit=1000)  # 限制最多1000个任务
        logger.info(f"获取到 {len(tasks)} 个任务")

        # 转换任务状态为字符串
        for task in tasks:
            task["status"] = task["status"].name

        thumbnails = {}

        for task in tasks:
            task_id = task["task_id"]
            if task.get("inputs"):
                # 查找输入中的图片文件
                image_inputs = []
                for key, value in task["inputs"].items():
                    if key.lower().find("image") != -1 or str(value).lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
                        image_inputs.append(key)

                if image_inputs:
                    # 使用第一个图片作为缩略图
                    first_image_key = image_inputs[0]
                    try:
                        # 获取图片数据
                        image_data = await data_manager.load_bytes(task["inputs"][first_image_key])

                        # 转换为base64
                        import base64

                        image_base64 = base64.b64encode(image_data).decode("utf-8")

                        # 根据文件扩展名确定MIME类型
                        content_type, _ = mimetypes.guess_type(first_image_key)
                        if content_type is None:
                            content_type = "image/jpeg"  # 默认类型

                        # 构建data URL
                        data_url = f"data:{content_type};base64,{image_base64}"
                        thumbnails[task_id] = data_url

                    except Exception as e:
                        logger.warning(f"Failed to load thumbnail for task {task_id}: {e}")
                        # 如果加载失败，不添加到结果中

        result = {"thumbnails": thumbnails, "total_tasks": len(tasks), "total_thumbnails": len(thumbnails)}
        return result

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/task/cancel")
async def api_v1_task_cancel(request: Request, user=Depends(verify_user_access)):
    try:
        msg = await server_monitor.check_user_busy(user["user_id"])
        if msg is not True:
            return error_response(msg, 400)
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
        msg = await server_monitor.check_user_busy(user["user_id"], active_new_task=True)
        if msg is not True:
            return error_response(msg, 400)
        task_id = request.query_params["task_id"]
        ret = await task_manager.resume_task(task_id, user_id=user["user_id"], all_subtask=True)
        if ret:
            await prepare_subtasks(task_id)
            return {"msg": "ok"}
        else:
            return error_response(f"Task {task_id} resume failed", 400)
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
        if task["status"] != TaskStatus.RUNNING:
            return {"msg": "delete"}

        assert await task_manager.ping_subtask(task_id, worker_name, identity)
        await server_monitor.worker_update(queue, identity, WorkerStatus.PING)
        return {"msg": "ok"}

    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.post("/api/v1/worker/ping/life")
async def api_v1_worker_ping_life(request: Request, valid=Depends(verify_worker_access)):
    try:
        params = await request.json()
        logger.info(f"{params}")
        identity = params.pop("worker_identity")
        keys = params.pop("worker_keys")
        worker = model_pipelines.get_worker(keys)

        # worker lost, init it again
        queue = server_monitor.identity_to_queue.get(identity, None)
        if queue is None:
            queue = worker["queue"]
            logger.warning(f"worker {identity} lost, refetching it")
            await server_monitor.worker_update(queue, identity, WorkerStatus.FETCHING)
        else:
            assert queue == worker["queue"], f"worker {identity} queue not matched: {queue} vs {worker['queue']}"

        metrics = await server_monitor.cal_metrics()
        ret = {"queue": queue, "metrics": metrics[queue]}
        if identity in metrics[queue]["del_worker_identities"]:
            ret["msg"] = "delete"
        else:
            ret["msg"] = "ok"
        return ret

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


# Template API endpoints
@app.get("/api/v1/template/{template_type}/{filename}")
async def api_v1_template(template_type: str, filename: str):
    """获取模板文件"""
    try:
        import os

        template_dir = os.path.join(os.path.dirname(__file__), "..", "template")
        file_path = os.path.join(template_dir, template_type, filename)

        # 安全检查：确保文件在template目录内
        if not os.path.exists(file_path) or not file_path.startswith(template_dir):
            return error_response(f"Template file not found", 404)

        with open(file_path, "rb") as f:
            data = f.read()

        # 根据文件类型设置媒体类型
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
        else:
            media_type = "application/octet-stream"

        return Response(content=data, media_type=media_type)
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


@app.get("/api/v1/template/list")
async def api_v1_template_list():
    """获取模板文件列表"""
    try:
        import glob
        import os

        template_dir = os.path.join(os.path.dirname(__file__), "..", "template")

        templates = {"images": [], "audios": []}

        # 获取图片模板
        image_dir = os.path.join(template_dir, "images")
        if os.path.exists(image_dir):
            for file_path in glob.glob(os.path.join(image_dir, "*")):
                if os.path.isfile(file_path):
                    filename = os.path.basename(file_path)
                    templates["images"].append({"filename": filename, "url": f"/api/v1/template/images/{filename}"})

        # 获取音频模板
        audio_dir = os.path.join(template_dir, "audios")
        if os.path.exists(audio_dir):
            for file_path in glob.glob(os.path.join(audio_dir, "*")):
                if os.path.isfile(file_path):
                    filename = os.path.basename(file_path)
                    templates["audios"].append({"filename": filename, "url": f"/api/v1/template/audios/{filename}"})

        return {"templates": templates}
    except Exception as e:
        traceback.print_exc()
        return error_response(str(e), 500)


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
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext4DebugL1("Init Server Cost"):
        model_pipelines = Pipeline(args.pipeline_json)
        auth_manager = AuthManager()
        if args.task_url.startswith("/"):
            task_manager = LocalTaskManager(args.task_url, metrics_monitor)
        elif args.task_url.startswith("postgresql://"):
            task_manager = PostgresSQLTaskManager(args.task_url, metrics_monitor)
        else:
            raise NotImplementedError
        if args.data_url.startswith("/"):
            data_manager = LocalDataManager(args.data_url)
        elif args.data_url.startswith("{"):
            data_manager = S3DataManager(args.data_url)
        else:
            raise NotImplementedError
        if args.queue_url.startswith("/"):
            queue_manager = LocalQueueManager(args.queue_url)
        elif args.queue_url.startswith("amqp://"):
            queue_manager = RabbitMQQueueManager(args.queue_url)
        else:
            raise NotImplementedError
        server_monitor = ServerMonitor(model_pipelines, task_manager, queue_manager)

    uvicorn.run(app, host=args.ip, port=args.port, reload=False, workers=1)
