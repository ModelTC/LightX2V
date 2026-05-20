import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from lightx2v.disagg.conn import MONITOR_POLLING_PORT, REQUEST_POLLING_PORT, ReqManager
from lightx2v.disagg.monitor import Monitor, Reporter
from lightx2v.disagg.workload import build_payload, current_stage, load_stage_specs, start_workload_clock


def _parse_gpus(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return [0]
    return [int(p) for p in parts]


def _parallel_world_size_from_config(config: dict[str, Any] | None) -> int:
    if not isinstance(config, dict):
        return 1
    parallel = config.get("parallel")
    if not isinstance(parallel, dict):
        return 1

    tensor_p_size = int(parallel.get("tensor_p_size", 1) or 1)
    if tensor_p_size > 1:
        return tensor_p_size

    cfg_p_size = int(parallel.get("cfg_p_size", 1) or 1)
    seq_p_size = int(parallel.get("seq_p_size", 1) or 1)
    return max(1, cfg_p_size * seq_p_size)


def _load_base_config_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch baseline infer requests and collect latency/GPU metrics")

    parser.add_argument("--mode", choices=["controller", "worker"], default="controller")

    # Controller mode
    parser.add_argument("--request_source", choices=["run_user", "generate"], default="run_user")
    parser.add_argument("--controller_request_port", type=int, default=REQUEST_POLLING_PORT - 2)
    parser.add_argument("--result_port", type=int, default=REQUEST_POLLING_PORT - 1)
    parser.add_argument("--worker_base_port", type=int, default=REQUEST_POLLING_PORT + 100)
    parser.add_argument("--worker_monitor_base_port", type=int, default=MONITOR_POLLING_PORT + 100)
    parser.add_argument("--monitor_poll_interval_s", type=float, default=2.0)
    parser.add_argument("--request_poll_sleep_s", type=float, default=0.02)
    parser.add_argument("--completion_timeout_s", type=float, default=7200.0)
    parser.add_argument("--dist_master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--dist_master_port", type=int, default=29600)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--python_executable", type=str, default=sys.executable)

    parser.add_argument("--model_cls", type=str, default="wan2.2_moe")
    parser.add_argument("--task", type=str, default="i2v")
    parser.add_argument("--model_path", type=str, default="/root/zht/LightX2V/models/Wan-AI/Wan2.2-I2V-A14B")
    parser.add_argument("--base_config_json", type=str, default="/root/zht/LightX2V/configs/disagg/baseline/wan22_moe_i2v_baseline.json")
    parser.add_argument("--save_dir", type=str, default="/root/zht/LightX2V/save_results")
    parser.add_argument("--save_result_path", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--keep_parallel_config", action="store_true", default=True)
    parser.add_argument("--drop_parallel_config", action="store_true", default=False)

    parser.add_argument("--generate_requests", type=int, default=10)
    parser.add_argument("--generate_interval_s", type=float, default=0.0)

    parser.add_argument("--metrics_output_json", type=str, default="/root/zht/LightX2V/save_results/baseline_controller_metrics.json")

    # Worker mode
    parser.add_argument("--worker_id", type=int, default=-1)
    parser.add_argument("--worker_recv_port", type=int, default=0)
    parser.add_argument("--worker_gpu", type=int, default=0)
    parser.add_argument("--worker_monitor_port", type=int, default=0)
    parser.add_argument("--worker_dist_rank", type=int, default=0)
    parser.add_argument("--worker_dist_world_size", type=int, default=1)
    parser.add_argument("--worker_cooperative_parallel", action="store_true", default=False)

    return parser


def _write_request_config(payload: dict[str, Any], request_id: int, keep_parallel_config: bool) -> tuple[str, str]:
    temp_dir = tempfile.mkdtemp(prefix=f"baseline_req_{request_id}_")
    config_path = str(Path(temp_dir) / "request_config.json")

    request_config = copy.deepcopy(payload)
    request_config.pop("workload_end", None)
    request_config.pop("request_metrics", None)
    if not keep_parallel_config:
        request_config.pop("parallel", None)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(request_config, f, ensure_ascii=False, indent=2)

    return temp_dir, config_path


def _effective_keep_parallel(args: argparse.Namespace) -> bool:
    return bool(args.keep_parallel_config) and not bool(args.drop_parallel_config)


def _run_infer_once(args: argparse.Namespace, payload: dict[str, Any], worker_id: int) -> tuple[int, str]:
    request_metrics = payload.get("request_metrics", {}) if isinstance(payload.get("request_metrics"), dict) else {}
    request_id = int(request_metrics.get("request_id", int(time.time() * 1000)))

    keep_parallel = _effective_keep_parallel(args)
    temp_dir, config_json = _write_request_config(payload, request_id, keep_parallel_config=keep_parallel)
    save_path = payload.get("save_path") or payload.get("save_result_path")
    if not save_path:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f"baseline_worker{worker_id}_req{request_id}.mp4")

    infer_argv = [
        "-m",
        "lightx2v.disagg.examples.infer",
        "--model_cls",
        str(payload.get("model_cls", args.model_cls)),
        "--task",
        str(payload.get("task", args.task)),
        "--model_path",
        str(payload.get("model_path", args.model_path)),
        "--config_json",
        config_json,
        "--prompt",
        str(payload.get("prompt", "")),
        "--negative_prompt",
        str(payload.get("negative_prompt", "")),
        "--save_result_path",
        str(save_path),
    ]

    image_path = payload.get("image_path")
    if image_path:
        infer_argv.extend(["--image_path", str(image_path)])

    seed = payload.get("seed")
    if seed is not None:
        infer_argv.extend(["--seed", str(int(seed))])

    # Keep execution path aligned with `lightx2v.disagg.examples.infer`:
    # one request uses one worker process, and when parallel is enabled this
    # single request is launched by torchrun on all visible devices.
    parallel_world_size = _parallel_world_size_from_config(payload) if keep_parallel else 1
    worker_world_size = int(args.worker_dist_world_size or 1)
    cooperative_parallel = bool(args.worker_cooperative_parallel) and worker_world_size > 1 and parallel_world_size > 1
    if cooperative_parallel:
        cmd = [args.python_executable, *infer_argv]
    elif parallel_world_size > 1:
        cmd = [
            args.python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={parallel_world_size}",
            *infer_argv,
        ]
    else:
        cmd = [args.python_executable, *infer_argv]

    env = os.environ.copy()
    if cooperative_parallel:
        env["MASTER_ADDR"] = str(args.dist_master_addr)
        env["MASTER_PORT"] = str(int(args.dist_master_port))
        env["RANK"] = str(int(args.worker_dist_rank))
        env["LOCAL_RANK"] = str(int(args.worker_dist_rank))
        env["NODE_RANK"] = "0"
        env["WORLD_SIZE"] = str(worker_world_size)

    result = subprocess.run(cmd, env=env, check=False)

    try:
        os.remove(config_json)
        os.rmdir(temp_dir)
    except OSError:
        pass

    return int(result.returncode), str(save_path)


def _worker_main(args: argparse.Namespace) -> None:
    req_mgr = ReqManager()

    reporter = Reporter(
        service_type=f"baseline_infer_worker_{args.worker_id}",
        gpu_id=int(args.worker_gpu),
        bind_address=f"tcp://*:{args.worker_monitor_port}",
    )
    reporter_thread = threading.Thread(target=reporter.serve_forever, name=f"worker-{args.worker_id}-reporter", daemon=True)
    reporter_thread.start()

    logger.info(
        "worker={} listening on port={} gpu={} monitor_port={}",
        args.worker_id,
        args.worker_recv_port,
        args.worker_gpu,
        args.worker_monitor_port,
    )

    while True:
        msg = req_mgr.receive(args.worker_recv_port)
        if isinstance(msg, dict) and msg.get("__control__") == "stop":
            break

        payload = msg if isinstance(msg, dict) else {}
        req_metrics = payload.get("request_metrics", {}) if isinstance(payload.get("request_metrics"), dict) else {}
        request_id = int(req_metrics.get("request_id", int(time.time() * 1000)))
        client_send_ts = float(req_metrics.get("client_send_ts", time.time()))

        start_ts = time.time()
        return_code, save_path = _run_infer_once(args, payload, args.worker_id)
        finish_ts = time.time()

        # In cooperative parallel mode, all workers are ranks of one request.
        # Only rank0 reports completion to avoid duplicated completion events.
        if not args.worker_cooperative_parallel or int(args.worker_dist_rank) == 0:
            req_mgr.send(
                "127.0.0.1",
                args.result_port,
                {
                    "request_id": request_id,
                    "worker_id": args.worker_id,
                    "start_ts": start_ts,
                    "finish_ts": finish_ts,
                    "client_send_ts": client_send_ts,
                    "e2e_latency_s": finish_ts - client_send_ts,
                    "return_code": return_code,
                    "save_path": save_path,
                },
            )

    reporter.stop()


def _launch_worker_processes(args: argparse.Namespace, gpus: list[int]) -> tuple[list[subprocess.Popen], int, bool]:
    procs: list[subprocess.Popen] = []
    keep_parallel = _effective_keep_parallel(args)
    base_config = _load_base_config_json(args.base_config_json)
    world_size = _parallel_world_size_from_config(base_config) if keep_parallel else 1
    if world_size < 1:
        world_size = 1

    requested_workers = int(args.num_workers)
    cooperative_parallel = bool(keep_parallel and world_size > 1)
    if cooperative_parallel:
        if len(gpus) < world_size:
            raise RuntimeError(f"cooperative model-parallel requires at least {world_size} gpus, got {len(gpus)} from --gpus={args.gpus}")
        if requested_workers != world_size:
            logger.warning(
                "parallel_world_size={} enabled; forcing num_workers {} -> {} (one worker per rank for one request)",
                world_size,
                requested_workers,
                world_size,
            )
        actual_workers = world_size
    else:
        capacity = max(1, len(gpus) // world_size)
        if requested_workers > capacity:
            logger.warning(
                "num_workers={} exceeds gpu capacity {} for parallel_world_size={}, auto-adjust to {}",
                requested_workers,
                capacity,
                world_size,
                capacity,
            )
        actual_workers = min(requested_workers, capacity)

    for worker_id in range(actual_workers):
        if cooperative_parallel:
            # In cooperative mode all ranks must see the full GPU list so legacy
            # rank->device mapping by dist.get_rank() stays valid.
            gpu_group = gpus[:actual_workers]
            gpu_id = gpus[worker_id]
        else:
            start = worker_id * world_size
            gpu_group = gpus[start : start + world_size]
            if len(gpu_group) < world_size:
                gpu_group = gpus[:world_size]
            gpu_id = gpu_group[0]
        recv_port = args.worker_base_port + worker_id
        monitor_port = args.worker_monitor_base_port + worker_id

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_group)

        cmd = [
            args.python_executable,
            "-m",
            "lightx2v.disagg.examples.run_controller",
            "--mode",
            "worker",
            "--worker_id",
            str(worker_id),
            "--worker_recv_port",
            str(recv_port),
            "--worker_gpu",
            str(gpu_id),
            "--worker_monitor_port",
            str(monitor_port),
            "--worker_dist_rank",
            str(worker_id),
            "--worker_dist_world_size",
            str(actual_workers),
            "--result_port",
            str(args.result_port),
            "--python_executable",
            args.python_executable,
            "--model_cls",
            args.model_cls,
            "--task",
            args.task,
            "--model_path",
            args.model_path,
            "--save_dir",
            args.save_dir,
        ]

        if keep_parallel:
            cmd.append("--keep_parallel_config")
        if args.drop_parallel_config:
            cmd.append("--drop_parallel_config")
        if cooperative_parallel:
            cmd.append("--worker_cooperative_parallel")
            cmd.extend(
                [
                    "--dist_master_addr",
                    args.dist_master_addr,
                    "--dist_master_port",
                    str(args.dist_master_port),
                ]
            )

        procs.append(subprocess.Popen(cmd, env=env))
    return procs, actual_workers, cooperative_parallel


def _build_generated_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = _load_base_config_json(args.base_config_json)

    base.setdefault("model_cls", args.model_cls)
    base.setdefault("task", args.task)
    base.setdefault("model_path", args.model_path)
    if args.save_result_path:
        base["save_path"] = args.save_result_path
    if args.prompt:
        base.setdefault("prompt", args.prompt)
    if args.negative_prompt:
        base.setdefault("negative_prompt", args.negative_prompt)
    if args.image_path:
        base.setdefault("image_path", args.image_path)
    if str(base.get("model_cls", args.model_cls)) == "wan2.2_moe":
        base.setdefault("boundary", 0.9)

    stages = load_stage_specs()
    start_workload_clock()

    requests: list[dict[str, Any]] = []
    for i in range(args.generate_requests):
        stage = current_stage(stages)
        req = build_payload(base, stage, i)
        req["model_path"] = req.get("model_path", args.model_path)
        req["model_cls"] = req.get("model_cls", args.model_cls)
        req["task"] = req.get("task", args.task)
        if str(req.get("model_cls", args.model_cls)) == "wan2.2_moe":
            req.setdefault("boundary", 0.9)
        requests.append(req)
    return requests


def _controller_main(args: argparse.Namespace) -> None:
    gpus = _parse_gpus(args.gpus)
    req_mgr = ReqManager()

    worker_procs, worker_count, cooperative_parallel = _launch_worker_processes(args, gpus)
    logger.info("launched {} workers", worker_count)
    if worker_count <= 0:
        raise RuntimeError("no workers launched, please check --num_workers / --gpus / parallel config")

    monitor_nodes = [f"tcp://127.0.0.1:{args.worker_monitor_base_port + i}" for i in range(worker_count)]
    monitor = Monitor(monitor_nodes)
    monitor_samples: list[dict[str, Any]] = []
    monitor_stop_event = threading.Event()
    global_first_send_ts: float | None = None

    def _monitor_cb(results: list[dict[str, Any]]) -> None:
        ts = time.time()
        for item in results:
            sample = dict(item)
            sample["sample_ts"] = ts
            sample["sample_ts_from_global_start_s"] = ts - global_first_send_ts if global_first_send_ts is not None else None
            monitor_samples.append(sample)
            status = sample.get("status")
            # if status == "ok":
            #     logger.info(
            #         "[monitor] {} gpu_util={} mem={}/{} MB",
            #         sample.get("address"),
            #         sample.get("gpu_utilization"),
            #         sample.get("gpu_memory_used_mb"),
            #         sample.get("gpu_memory_total_mb"),
            #     )

    monitor_thread = threading.Thread(
        target=monitor.run_forever,
        kwargs={
            "interval_seconds": args.monitor_poll_interval_s,
            "callback": _monitor_cb,
            "stop_event": monitor_stop_event,
        },
        daemon=True,
    )
    monitor_thread.start()

    dispatched: dict[int, dict[str, Any]] = {}
    completions: list[dict[str, Any]] = []

    next_worker = 0

    def _dispatch(payload: dict[str, Any]) -> None:
        nonlocal next_worker, global_first_send_ts
        req_metrics = payload.setdefault("request_metrics", {})
        request_id = int(req_metrics.get("request_id", len(dispatched)))
        req_metrics["request_id"] = request_id
        req_metrics.setdefault("client_send_ts", time.time())
        if global_first_send_ts is None:
            global_first_send_ts = time.time()
        req_metrics["global_first_send_ts"] = global_first_send_ts

        if cooperative_parallel:
            dispatch_ts = time.time()
            for worker_id in range(worker_count):
                recv_port = args.worker_base_port + worker_id
                req_mgr.send("127.0.0.1", recv_port, payload)
            dispatched[request_id] = {
                "worker_id": "cooperative_group",
                "dispatch_ts": dispatch_ts,
                "payload": payload,
            }
            logger.info("[dispatch] request_id={} -> cooperative workers [0..{}]", request_id, worker_count - 1)
            return

        worker_id = next_worker % worker_count
        next_worker += 1
        recv_port = args.worker_base_port + worker_id

        dispatch_ts = time.time()
        req_mgr.send("127.0.0.1", recv_port, payload)
        dispatched[request_id] = {
            "worker_id": worker_id,
            "dispatch_ts": dispatch_ts,
            "payload": payload,
        }
        logger.info("[dispatch] request_id={} -> worker={} port={}", request_id, worker_id, recv_port)

    if args.request_source == "generate":
        for payload in _build_generated_requests(args):
            _dispatch(payload)
            if args.generate_interval_s > 0:
                time.sleep(args.generate_interval_s)
    else:
        logger.info("waiting run_user requests on port={}", args.controller_request_port)
        workload_end = False
        while not workload_end:
            payload = req_mgr.receive_non_block(args.controller_request_port)
            if payload is None:
                time.sleep(args.request_poll_sleep_s)
                continue
            if isinstance(payload, dict) and payload.get("workload_end"):
                workload_end = True
                continue
            if isinstance(payload, dict):
                _dispatch(payload)

    pending_ids = set(dispatched.keys())
    wait_start = time.time()
    while pending_ids:
        msg = req_mgr.receive_non_block(args.result_port)
        if msg is None:
            if time.time() - wait_start > args.completion_timeout_s:
                logger.warning("timeout waiting completions, pending={}", sorted(pending_ids))
                break
            time.sleep(args.request_poll_sleep_s)
            continue

        if not isinstance(msg, dict):
            continue

        request_id = int(msg.get("request_id", -1))
        finish_ts = float(msg.get("finish_ts", 0.0))
        elapsed_from_global_start_s = finish_ts - global_first_send_ts if global_first_send_ts is not None else None
        if elapsed_from_global_start_s is not None:
            msg["elapsed_from_global_start_s"] = elapsed_from_global_start_s
        if request_id in pending_ids:
            pending_ids.remove(request_id)
        completions.append(msg)

        logger.info(
            "[done] request_id={} worker={} start_ts={:.6f} finish_ts={:.6f} e2e={:.3f}s elapsed_from_global_start={:.3f}s rc={}",
            request_id,
            msg.get("worker_id"),
            float(msg.get("start_ts", 0.0)),
            finish_ts,
            float(msg.get("e2e_latency_s", 0.0)),
            float(elapsed_from_global_start_s if elapsed_from_global_start_s is not None else -1.0),
            msg.get("return_code"),
        )

    for worker_id in range(worker_count):
        req_mgr.send("127.0.0.1", args.worker_base_port + worker_id, {"__control__": "stop"})

    for proc in worker_procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()

    monitor_stop_event.set()
    monitor_thread.join(timeout=2.0)

    summary = {
        "request_source": args.request_source,
        "dispatched_count": len(dispatched),
        "completed_count": len(completions),
        "requests": completions,
        "monitor_samples": monitor_samples,
        "generated_at": time.time(),
    }

    out_path = Path(args.metrics_output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("metrics saved to {}", out_path)


def main() -> None:
    args = _make_parser().parse_args()
    if args.mode == "worker":
        _worker_main(args)
    else:
        _controller_main(args)


if __name__ == "__main__":
    main()
