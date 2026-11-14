#!/usr/bin/env python3
"""Prepare data/cache, then run one independent 8-GPU service or offline smoke."""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from collect import collect, model_identity, validate_export, wait_ready
from common import atomic_json, file_digest, records
from prepare_selection import DEFAULT_CACHE, DEFAULT_MEDIA, DEFAULT_OUTPUT, DEFAULT_RAW, prepare

REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-DeCFG-score"


def verify_model(model_path):
    root = Path(model_path).resolve()
    manifest = json.loads((root / "decfg_merged.json").read_text())
    if manifest.get("format") != "lightx2v_h3_score_decfg" or manifest.get("complete") is not True or manifest.get("schema_version") != 1 or manifest.get("precision") != "fp32_merged_targets":
        raise ValueError("Model must be the completed FP32 merged DeCFG score checkpoint")
    component = manifest.get("components", {}).get("transformer_ref")
    if not isinstance(component, dict) or component.get("model_name") != "minimax_h3_ref2av" or component.get("adapted_weights", 0) <= 0:
        raise ValueError("Merged DeCFG manifest has no adapted transformer_ref component")
    transformer = root / "transformer_ref"
    for name, hash_key in (("config.json", "config_sha256"), ("diffusion_pytorch_model.safetensors.index.json", "index_sha256")):
        if file_digest(transformer / name) != component.get(hash_key):
            raise ValueError(f"Merged transformer_ref {name} does not match its manifest")
    shards = component.get("shards", [])
    weight_map = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    if not shards or {shard["path"] for shard in shards} != set(weight_map.values()):
        raise ValueError("Merged transformer_ref weight index/shards disagree")
    for shard in shards:
        path = transformer / shard["path"]
        if not path.is_file() or path.stat().st_size != shard["size_bytes"]:
            raise ValueError(f"Missing/truncated merged transformer_ref shard: {path}")
    return manifest


def verify_config(path):
    config = json.loads(Path(path).read_text())
    expected = json.loads((REPO / "configs/minimax_h3/minimax_h3_ref2av_teacher29_tp2_sp4.json").read_text())
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Teacher config {key} must be {value!r}")
    if config.get("lora_configs") or config.get("dit_original_ckpt"):
        raise ValueError("Use the already merged transformer_ref; no LoRA or checkpoint override")
    return config


def run_logged(command, log_path, env):
    print(shlex.join(command), flush=True)
    print(f"Log: {log_path}", flush=True)
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
        finally:
            stop_process(process)


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def smoke_command(row, args, export_dir, video_path):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        "-m",
        "lightx2v.infer",
        "--model_cls",
        "minimax_h3",
        "--model-variant",
        "ref2av",
        "--task",
        "ref2av",
        "--model_path",
        args.model_path,
        "--config_json",
        args.config_json,
        "--prompt",
        row["prompt"],
        "--image_path",
        ",".join(row["reference_images"]),
        "--size",
        str(row["target_height"]),
        str(row["target_width"]),
        "--seed",
        str(row["seed"]),
        "--save_result_path",
        str(video_path),
    ]
    if not args.no_export:
        command += ["--teacher_latents_dir", str(export_dir), "--teacher_sample_id", row["source_id"]]
    return command


def main():
    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("node", "smoke"))
    parser.add_argument("--node-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--raw-jsonl", default=os.getenv("H3_RAW_JSONL", DEFAULT_RAW))
    parser.add_argument("--cache-metadata", default=os.getenv("H3_CACHE_METADATA", DEFAULT_CACHE))
    parser.add_argument("--media-root", default=os.getenv("H3_MEDIA_ROOT", DEFAULT_MEDIA))
    parser.add_argument("--output-dir", default=os.getenv("H3_OUTPUT_DIR", DEFAULT_OUTPUT))
    parser.add_argument("--model-path", default=os.getenv("H3_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--config-json", default=os.getenv("H3_CONFIG_JSON", str(REPO / "configs/minimax_h3/minimax_h3_ref2av_teacher29_tp2_sp4.json")))
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based position in this node's selected shard")
    parser.add_argument("--no-export", action="store_true", help="Smoke: decode MP4 without also exporting the trajectory")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    config = verify_config(args.config_json)
    verify_model(args.model_path)
    devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(devices) != 8 or len(set(devices)) != 8 or any(not value.strip() for value in devices):
        raise ValueError("Each machine must expose exactly eight distinct GPUs")
    root = Path(args.output_dir).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    log_dir = root / "logs" / f"node{args.node_index}"
    log_dir.mkdir(parents=True, exist_ok=True)
    prepare(args.raw_jsonl, args.cache_metadata, args.media_root, root, args.node_index)
    atomic_json(log_dir / f"config.{stamp}.json", config)
    atomic_json(log_dir / f"model.{stamp}.json", model_identity(args.model_path, config))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "DTYPE": "BF16",
        "SENSITIVE_LAYER_DTYPE": "BF16",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "1"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": str(REPO) + os.pathsep + os.getenv("PYTHONPATH", ""),
    }
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "LOCAL_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        env.pop(key, None)
    # This invokes the existing builder's exact spec/weight fingerprint logic.
    cache_command = [sys.executable, str(SCRIPT_DIR / "ensure_adaln.py"), "--model-path", args.model_path, "--config-json", args.config_json, "--receipt", str(log_dir / f"adaln.{stamp}.json")]
    run_logged(cache_command, log_dir / f"adaln.{stamp}.log", {**env, "CUDA_VISIBLE_DEVICES": devices[0]})
    if args.mode == "smoke":
        rows = [row for _, row in records(root / "selection" / f"node{args.node_index}" / f"node{args.node_index}.jsonl")]
        row = rows[args.sample_index]
        directory = root / "smoke" / f"node{args.node_index}" / stamp
        directory.mkdir(parents=True)
        export_dir, video_path = directory / "trajectory", directory / "sample.mp4"
        atomic_json(directory / "sample.json", row)
        command = smoke_command(row, args, export_dir, video_path)
        atomic_json(directory / "command.json", command)
        run_logged(command, log_dir / f"smoke.{stamp}.log", env)
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise FileNotFoundError(f"Smoke did not produce MP4: {video_path}")
        if not args.no_export:
            validate_export(export_dir, row)
        print(f"Smoke succeeded: {video_path}", flush=True)
        return
    # Refuse to accidentally send this shard to a previously running service.
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is occupied; stop that service before launching this node")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        "-m",
        "lightx2v.server",
        "--model_cls",
        "minimax_h3",
        "--model-variant",
        "ref2av",
        "--model_path",
        args.model_path,
        "--config_json",
        args.config_json,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max_queue_size",
        "1",
    ]
    atomic_json(log_dir / f"server_command.{stamp}.json", command)
    print(shlex.join(command), flush=True)
    server_log = log_dir / f"server.{stamp}.log"
    print(f"Server log: {server_log}", flush=True)
    with server_log.open("w") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            url = f"http://127.0.0.1:{args.port}"
            wait_ready(url, process=process)
            collect(root, args.node_index, args.config_json, args.model_path, url)
        finally:
            stop_process(process)


if __name__ == "__main__":
    main()
