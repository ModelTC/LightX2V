import argparse
import asyncio
import json
import os
import signal
import traceback
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

from lightx2v.infer import init_runner
from lightx2v.server.ws.protocol import ClientMessage, error_message
from lightx2v.server.ws.session import LiveSession, join_pipeline, launch_pipeline
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
TARGET_RANK = int(os.getenv("WORKER_RANK", "0")) % max(WORLD_SIZE, 1)
TASK_GROUP = None


def _dist_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return RANK, WORLD_SIZE


def init_task_group():
    global TASK_GROUP
    if TASK_GROUP is not None or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    TASK_GROUP = dist.new_group(backend="gloo", timeout=timedelta(days=30))
    logger.info(f"Rank {dist.get_rank()} created gloo group for websocket session commands")


async def broadcast_obj(obj):
    obj = {} if obj is None else obj
    rank, world_size = _dist_world()
    if world_size <= 1:
        return obj
    group = TASK_GROUP
    if rank == TARGET_RANK:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
        size = torch.tensor([tensor.numel()], dtype=torch.int32)
    else:
        size = torch.zeros(1, dtype=torch.int32)
    dist.broadcast(size, src=TARGET_RANK, group=group)
    if rank != TARGET_RANK:
        tensor = torch.zeros(int(size.item()), dtype=torch.uint8)
    dist.broadcast(tensor, src=TARGET_RANK, group=group)
    if rank != TARGET_RANK:
        obj = json.loads(bytes(tensor.numpy()).decode("utf-8"))
    return obj


async def sync_ranks():
    if _dist_world()[1] > 1:
        dist.barrier()
        logger.info(f"Rank {RANK} ranks synced")


def build_parser():
    parser = argparse.ArgumentParser(description="LightX2V websocket live inference (seko_talk_ar)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_cls", type=str, default="seko_talk_ar")
    parser.add_argument("--task", type=str, default="s2v")
    parser.add_argument("--support_tasks", type=str, nargs="+", default=[])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--ws_host", type=str, default="0.0.0.0")
    parser.add_argument("--ws_port", type=int, default=8765)
    return parser


async def follower_loop(runner, input_info):
    logger.info(f"Rank {RANK} waiting for websocket sessions")
    loop = asyncio.get_running_loop()
    while True:
        cmd = await broadcast_obj(None)
        op = cmd.get("op")
        if op == "shutdown":
            logger.info(f"Rank {RANK} shutdown")
            return
        if op != "run":
            logger.info(f"Rank {RANK} received {op} message, skipping")
            continue
        try:
            update_input_info_from_dict(
                input_info,
                {
                    "prompt": cmd["prompt"],
                    "negative_prompt": cmd["negative_prompt"],
                    # "seed": cmd["seed"],
                    "target_shape": cmd["target_shape"],
                    "image_path": cmd["image_path"],
                    "audio_path": {"type": "ws"},
                    "save_result_path": {"type": "ws"},
                },
            )
            logger.info(f"Rank {RANK} input_info: {input_info}")
            thread, future = launch_pipeline(runner, input_info, RANK, loop)
            await join_pipeline(thread, future, RANK)
        except Exception:
            logger.error(f"Rank {RANK} pipeline failed: {traceback.format_exc()}")
        await sync_ranks()


async def run_websocket_server(args, runner, input_info):
    app = FastAPI(title="LightX2V WS Live", version="0.1.0")
    session_lock = asyncio.Lock()
    state = SimpleNamespace(session=None)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_cls": args.model_cls, "busy": state.session is not None and state.session.is_busy()}

    @app.websocket("/v1/live")
    async def live_ws(websocket: WebSocket):
        await websocket.accept()
        session = None
        pump_task = None
        try:
            while True:
                data = await websocket.receive_bytes()
                msg = ClientMessage.parse(data)
                which = msg.which()
                if which == "start":
                    async with session_lock:
                        if state.session is not None and state.session.is_busy():
                            await websocket.send_bytes(error_message("busy", "another session is running", True))
                            continue
                        session = LiveSession(runner, input_info, RANK, asyncio.get_running_loop())
                        try:
                            cmd = session.apply_start(msg.start)
                        except Exception as e:
                            await websocket.send_bytes(error_message("invalid_start", str(e), False))
                            session.close()
                            session = None
                            continue
                        state.session = session
                        await broadcast_obj(cmd)
                    session.start_pipeline()
                    try:
                        await session.wait_started()
                    except Exception as e:
                        await websocket.send_bytes(error_message("start_failed", str(e), True))
                        await session.stop_and_join()
                        session.close()
                        state.session = None
                        session = None
                        continue
                    pump_task = asyncio.create_task(session.pump_video(websocket))
                    if session.future is not None:
                        session.future.add_done_callback(
                            lambda fut: session.mark_pipeline_done(*(fut.result() if not fut.cancelled() else (False, "cancelled")))
                        )
                    continue

                if session is None or session.audio_source is None:
                    await websocket.send_bytes(error_message("no_session", "send Start first", True))
                    continue
                if which is None:
                    await websocket.send_bytes(error_message("invalid_message", "empty ClientMessage", False))
                    continue
                try:
                    session.handle_client_body(msg)
                except Exception as e:
                    await websocket.send_bytes(error_message("bad_message", str(e), False))

        except WebSocketDisconnect:
            logger.warning(f"Rank {RANK} websocket disconnected")

        except Exception:
            logger.error(f"Rank {RANK} websocket handler failed: {traceback.format_exc()}")
            try:
                await websocket.send_bytes(error_message("internal", "websocket internal error", True))
            except Exception:
                pass
        finally:
            if session is not None:
                if pump_task is not None:
                    session.mark_pipeline_done(True, "")
                    pump_task.cancel()
                await session.stop_and_join()
                try:
                    await websocket.close()
                except Exception:
                    pass
                session.close()
                await sync_ranks()
                if state.session is session:
                    state.session = None

    config = uvicorn.Config(app, host=args.ws_host, port=args.ws_port, log_level="info")
    server = uvicorn.Server(config)

    def _stop(*_):
        logger.warning("received stop signal")
        server.should_exit = True
        if state.session is not None:
            state.session.request_stop()

    async def _idle_heartbeat():
        # Keep gloo wait alive so followers never hit the process-group timeout.
        while not server.should_exit:
            await asyncio.sleep(300)
            if server.should_exit:
                return
            async with session_lock:
                if state.session is not None:
                    continue
                await broadcast_obj({"op": "idle"})
                logger.info(f"Rank {RANK} sent idle message")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass
    heartbeat = asyncio.create_task(_idle_heartbeat()) if WORLD_SIZE > 1 else None
    logger.info(f"websocket live listening on ws://{args.ws_host}:{args.ws_port}/v1/live")
    try:
        await server.serve()
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
    if WORLD_SIZE > 1:
        await broadcast_obj({"op": "shutdown"})


def main():
    global RANK, WORLD_SIZE, TARGET_RANK
    parser = build_parser()
    args = parser.parse_args()
    seed_all(args.seed)
    config = set_config(args)

    if config.get("parallel") or WORLD_SIZE > 1:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        if platform_device is not None and not dist.is_initialized():
            platform_device.init_parallel_env()
        if config.get("parallel"):
            set_parallel_config(config)
        if dist.is_initialized():
            torch.cuda.set_device(dist.get_rank())

    if dist.is_initialized():
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        TARGET_RANK = int(os.getenv("WORKER_RANK", "0")) % WORLD_SIZE

    print_config(config)
    runner = init_runner(config)
    input_info = init_empty_input_info(args.task, args.support_tasks)
    init_task_group()

    if RANK != TARGET_RANK:
        asyncio.run(follower_loop(runner, input_info))
    else:
        asyncio.run(run_websocket_server(args, runner, input_info))

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
