import argparse
import asyncio
import json
import os
import signal
import sys
import traceback
import uuid

import aiohttp
import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.deploy.data_manager import LocalDataManager, S3DataManager
from lightx2v.deploy.task_manager import TaskStatus
from lightx2v.deploy.worker.hub import DiTWorker, ImageEncoderWorker, PipelineWorker, SegmentDiTWorker, TextEncoderWorker, VaeDecoderWorker, VaeEncoderWorker
from lightx2v.server.metrics import metrics

RUNNER_MAP = {
    "pipeline": PipelineWorker,
    "text_encoder": TextEncoderWorker,
    "image_encoder": ImageEncoderWorker,
    "vae_encoder": VaeEncoderWorker,
    "vae_decoder": VaeDecoderWorker,
    "dit": DiTWorker,
    "segment_dit": SegmentDiTWorker,
}

# {task_id: {"server": xx, "worker_name": xx, "identity": xx}}
RUNNING_SUBTASKS = {}
WORKER_SECRET_KEY = os.getenv("WORKER_SECRET_KEY", "worker-secret-key-change-in-production")
HEADERS = {"Authorization": f"Bearer {WORKER_SECRET_KEY}", "Content-Type": "application/json"}
STOPPED = False
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
TARGET_RANK = int(os.getenv("WORKER_RANK", "0")) % WORLD_SIZE


async def ping_life(server_url, worker_identity, keys):
    url = server_url + "/api/v1/worker/ping/life"
    params = {"worker_identity": worker_identity, "worker_keys": keys}
    while True:
        try:
            logger.info(f"{worker_identity} pinging life ...")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=json.dumps(params), headers=HEADERS) as ret:
                    if ret.status == 200:
                        ret = await ret.json()
                        logger.info(f"{worker_identity} ping life: {ret}")
                        if ret["msg"] == "delete":
                            logger.warning(f"{worker_identity} deleted")
                            # asyncio.create_task(shutdown(asyncio.get_event_loop()))
                            return
                        await asyncio.sleep(10)
                    else:
                        error_text = await ret.text()
                        raise Exception(f"{worker_identity} ping life fail: [{ret.status}], error: {error_text}")
        except asyncio.CancelledError:
            logger.warning("Ping life cancelled, shutting down...")
            raise asyncio.CancelledError
        except:  # noqa
            logger.warning(f"Ping life failed: {traceback.format_exc()}")
            await asyncio.sleep(10)


async def ping_subtask(server_url, worker_identity, task_id, worker_name, queue, running_task, ping_interval):
    url = server_url + "/api/v1/worker/ping/subtask"
    params = {
        "worker_identity": worker_identity,
        "task_id": task_id,
        "worker_name": worker_name,
        "queue": queue,
    }
    while True:
        try:
            logger.info(f"{worker_identity} pinging subtask {task_id} {worker_name} ...")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=json.dumps(params), headers=HEADERS) as ret:
                    if ret.status == 200:
                        ret = await ret.json()
                        logger.info(f"{worker_identity} ping subtask {task_id} {worker_name}: {ret}")
                        if ret["msg"] == "delete":
                            logger.warning(f"{worker_identity} subtask {task_id} {worker_name} deleted")
                            running_task.cancel()
                            return
                        await asyncio.sleep(ping_interval)
                    else:
                        error_text = await ret.text()
                        raise Exception(f"{worker_identity} ping subtask fail: [{ret.status}], error: {error_text}")
        except asyncio.CancelledError:
            logger.warning(f"Ping subtask {task_id} {worker_name} cancelled")
            raise asyncio.CancelledError
        except:  # noqa
            logger.warning(f"Ping subtask failed: {traceback.format_exc()}")
            await asyncio.sleep(10)


async def fetch_subtasks(server_url, worker_keys, worker_identity, max_batch, timeout):
    url = server_url + "/api/v1/worker/fetch"
    params = {
        "worker_keys": worker_keys,
        "worker_identity": worker_identity,
        "max_batch": max_batch,
        "timeout": timeout,
    }
    try:
        logger.info(f"{worker_identity} fetching {worker_keys} with timeout: {timeout}s ...")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=json.dumps(params), headers=HEADERS, timeout=timeout + 1) as ret:
                if ret.status == 200:
                    ret = await ret.json()
                    subtasks = ret["subtasks"]
                    for sub in subtasks:
                        sub["server_url"] = server_url
                        sub["worker_identity"] = worker_identity
                        RUNNING_SUBTASKS[sub["task_id"]] = sub
                    logger.info(f"{worker_identity} fetch {worker_keys} ok: {subtasks}")
                    return subtasks
                else:
                    error_text = await ret.text()
                    logger.warning(f"{worker_identity} fetch {worker_keys} fail: [{ret.status}], error: {error_text}")
                    return None
    except asyncio.CancelledError:
        logger.warning("Fetch subtasks cancelled, shutting down...")
        raise asyncio.CancelledError
    except:  # noqa
        logger.warning(f"Fetch subtasks failed: {traceback.format_exc()}")
        await asyncio.sleep(10)


async def report_task(server_url, task_id, worker_name, status, worker_identity, queue, **kwargs):
    url = server_url + "/api/v1/worker/report"
    params = {
        "task_id": task_id,
        "worker_name": worker_name,
        "status": status,
        "worker_identity": worker_identity,
        "queue": queue,
        "fail_msg": "" if status == TaskStatus.SUCCEED.name else "worker failed",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=json.dumps(params), headers=HEADERS) as ret:
                if ret.status == 200:
                    RUNNING_SUBTASKS.pop(task_id)
                    ret = await ret.json()
                    logger.info(f"{worker_identity} report {task_id} {worker_name} {status} ok")
                    return True
                else:
                    error_text = await ret.text()
                    logger.warning(f"{worker_identity} report {task_id} {worker_name} {status} fail: [{ret.status}], error: {error_text}")
                    return False
    except asyncio.CancelledError:
        logger.warning("Report task cancelled, shutting down...")
        raise asyncio.CancelledError
    except:  # noqa
        logger.warning(f"Report task failed: {traceback.format_exc()}")


async def boradcast_subtasks(subtasks):
    subtasks = [] if subtasks is None else subtasks
    if WORLD_SIZE <= 1:
        return subtasks
    try:
        if RANK == TARGET_RANK:
            subtasks_data = json.dumps(subtasks, ensure_ascii=False).encode("utf-8")
            subtasks_tensor = torch.frombuffer(bytearray(subtasks_data), dtype=torch.uint8).to(device="cuda")
            data_size = subtasks_tensor.shape[0]
            size_tensor = torch.tensor([data_size], dtype=torch.int32).to(device="cuda")
            logger.info(f"rank {RANK} send subtasks: {subtasks_tensor.shape}, {size_tensor}")
        else:
            size_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")

        dist.broadcast(size_tensor, src=TARGET_RANK)
        if RANK != TARGET_RANK:
            subtasks_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device="cuda")
        dist.broadcast(subtasks_tensor, src=TARGET_RANK)

        if RANK != TARGET_RANK:
            subtasks_data = subtasks_tensor.cpu().numpy().tobytes()
            subtasks = json.loads(subtasks_data.decode("utf-8"))
            logger.info(f"rank {RANK} recv subtasks: {subtasks}")
        return subtasks

    except:  # noqa
        logger.error(f"Broadcast subtasks failed: {traceback.format_exc()}")
        return []


async def sync_subtask():
    if WORLD_SIZE <= 1:
        return
    try:
        logger.info(f"Sync subtask {RANK}/{WORLD_SIZE} wait barrier")
        dist.barrier()
        logger.info(f"Sync subtask {RANK}/{WORLD_SIZE} ok")
    except:  # noqa
        logger.error(f"Sync subtask failed: {traceback.format_exc()}")


async def main(args):
    if args.model_name == "":
        args.model_name = args.model_cls
    if args.task_name == "":
        args.task_name = args.task
    worker_keys = [args.task_name, args.model_name, args.stage, args.worker]

    metrics.server_process(args.metric_port)

    data_manager = None
    if args.data_url.startswith("/"):
        data_manager = LocalDataManager(args.data_url, None)
    elif args.data_url.startswith("{"):
        data_manager = S3DataManager(args.data_url, None)
    else:
        raise NotImplementedError
    await data_manager.init()

    if WORLD_SIZE > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        logger.info(f"Distributed initialized: rank={RANK}, world_size={WORLD_SIZE}")

    runner = RUNNER_MAP[args.worker](args)
    if WORLD_SIZE > 1:
        dist.barrier()
    # asyncio.create_task(ping_life(args.server, args.identity, worker_keys))

    while True:
        subtasks = None
        if RANK == TARGET_RANK:
            subtasks = await fetch_subtasks(args.server, worker_keys, args.identity, args.max_batch, args.timeout)
        subtasks = await boradcast_subtasks(subtasks)

        for sub in subtasks:
            status = TaskStatus.FAILED.name
            ping_task = None
            try:
                run_task = asyncio.create_task(runner.run(sub["inputs"], sub["outputs"], sub["params"], data_manager))
                if RANK == TARGET_RANK:
                    ping_task = asyncio.create_task(ping_subtask(args.server, sub["worker_identity"], sub["task_id"], sub["worker_name"], sub["queue"], run_task, args.ping_interval))
                ret = await run_task
                if ret is True:
                    status = TaskStatus.SUCCEED.name

            except asyncio.CancelledError:
                if STOPPED:
                    logger.warning("Main loop cancelled, already stopped, should exit")
                    return
                logger.warning("Main loop cancelled, do not shut down")

            finally:
                try:
                    if ping_task:
                        ping_task.cancel()
                    await sync_subtask()
                except Exception:
                    logger.warning(f"Sync subtask failed: {traceback.format_exc()}")
                if RANK == TARGET_RANK and sub["task_id"] in RUNNING_SUBTASKS:
                    try:
                        await report_task(status=status, **sub)
                    except Exception:
                        logger.warning(f"Report failed: {traceback.format_exc()}")


async def shutdown(loop):
    logger.warning("Received kill signal")
    global STOPPED
    STOPPED = True

    for t in asyncio.all_tasks():
        if t is not asyncio.current_task():
            logger.warning(f"Cancel async task {t} ...")
            t.cancel()

    # Report remaining running subtasks failed
    if RANK == TARGET_RANK:
        task_ids = list(RUNNING_SUBTASKS.keys())
        for task_id in task_ids:
            try:
                s = RUNNING_SUBTASKS[task_id]
                logger.warning(f"Report {task_id} {s['worker_name']} {TaskStatus.FAILED.name} ...")
                await report_task(status=TaskStatus.FAILED.name, **s)
            except:  # noqa
                logger.warning(f"Report task {task_id} failed: {traceback.format_exc()}")

    if WORLD_SIZE > 1:
        dist.destroy_process_group()

    # Force exit after a short delay to ensure cleanup
    def force_exit():
        logger.warning("Force exiting process...")
        sys.exit(0)

    loop.call_later(2, force_exit)


# align args like infer.py
def align_args(args):
    args.seed = 42
    args.sf_model_path = args.sf_model_path if args.sf_model_path else ""
    args.use_prompt_enhancer = False
    args.prompt = ""
    args.negative_prompt = ""
    args.image_path = ""
    args.last_frame_path = ""
    args.audio_path = ""
    args.src_pose_path = None
    args.src_face_path = None
    args.src_bg_path = None
    args.src_mask_path = None
    args.src_ref_images = None
    args.src_video = None
    args.src_mask = None
    args.save_result_path = ""
    args.return_result_tensor = False
    args.is_live = True


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(cur_dir, "../../.."))
    dft_data_url = os.path.join(base_dir, "local_data")

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--model_cls", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--worker", type=str, required=True)
    parser.add_argument("--identity", type=str, default="")
    parser.add_argument("--max_batch", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--ping_interval", type=int, default=10)

    parser.add_argument("--metric_port", type=int, default=8001)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sf_model_path", type=str, default="")
    parser.add_argument("--config_json", type=str, required=True)

    parser.add_argument("--server", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--data_url", type=str, default=dft_data_url)

    args = parser.parse_args()
    align_args(args)
    if args.identity == "":
        # TODO: spec worker instance identity by k8s env
        args.identity = "worker-" + str(uuid.uuid4())[:8]
    logger.info(f"args: {args}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for s in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(s, lambda: asyncio.create_task(shutdown(loop)))

    try:
        loop.create_task(main(args), name="main")
        loop.run_forever()
    finally:
        loop.close()
        logger.warning("Event loop closed")
