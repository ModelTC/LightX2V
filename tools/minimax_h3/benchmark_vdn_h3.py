"""Measure VDN and H3 Turbo through the same ordinary LightX2V request pipeline.

Use --output-dir DIR --warmup-runs 2 --measured-runs 3 -- <lightx2v.infer args>.
The first request is also the first warmup; all requests recompute conditioning
and save video. Model/LoRA initialization is reported separately. Compare two
summary.json files using --compare REFERENCE CANDIDATE --output-dir DIR.
"""

import argparse
import copy
import functools
import hashlib
import json
import os
import runpy
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMMON_CONFIG = (
    "model_variant",
    "infer_steps",
    "fps",
    "enable_cfg",
    "cpu_offload",
    "offload_granularity",
    "dit_prepost_resident",
    "use_adaln_cache",
    "text_encoder_cpu_offload",
    "text_encoder_offload_granularity",
    "text_encoder_host_pinned",
    "text_encoder_release_block_offload_buffers",
    "vae_cpu_offload",
    "vae_decode_parallel",
    "vae_use_compile",
    "vae_decode_tile_shapes",
    "attn_type",
    "refiner_attn_type",
    "rms_type",
    "rope_type",
    "use_fused_qkv",
    "use_fused_qkv_norm_rope",
    "qkv_norm_rope_type",
    "feature_caching",
    "use_compile",
    "parallel",
    "h3_noise_device",
    "h3_visual_reference_only",
    "empty_cache_min_free_gib",
    "empty_cache_min_reclaimable_gib",
    "video_codec_options",
    "dit_quantized",
    "text_encoder_quantized",
)
PHASES = (
    "pipeline",
    "input_encoder",
    "text_encoder",
    "vae_encode",
    "dit_prepare",
    "dit_loop",
    "dit_offload",
    "vae_decode",
    "video_vae_decode",
    "audio_vae_decode",
    "save",
)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def summarize_rank_records(records):
    """Reduce local synchronized walls by max; take the median after step maxima."""
    if not records or len({len(record["steps_seconds"]) for record in records}) != 1:
        raise ValueError("Ranks must provide the same nonempty request and step count")
    phases = {name: max(record["phases_seconds"].get(name, 0.0) for record in records) for name in PHASES}
    steps = [max(record["steps_seconds"][index] for record in records) for index in range(len(records[0]["steps_seconds"]))]
    if not steps:
        raise ValueError("No complete denoising steps were measured")
    for record in records:
        if record["calls"].get("text_encoder") != 1:
            raise ValueError("Each request must execute the text encoder exactly once on every rank")
        if any(record["calls"].get(name) != 1 for name in ("pipeline", "input_encoder", "dit_prepare", "dit_loop", "vae_decode", "save")):
            raise ValueError("Each request must execute one complete common pipeline including decode and save")
        if len(record["steps_seconds"]) != record["infer_steps"]:
            raise ValueError("Captured step count differs from the native scheduler")
    phases.update(
        text_conditioning=max(record["phases_seconds"].get("input_encoder", 0.0) - record["phases_seconds"].get("vae_encode", 0.0) for record in records),
        dit_stage=max(sum(record["phases_seconds"].get(name, 0.0) for name in ("dit_prepare", "dit_loop", "dit_offload")) for record in records),
        step_median=statistics.median(steps),
        step_first=steps[0],
        step_sum=sum(steps),
    )
    return {
        "phase": records[0]["phase"],
        "index": records[0]["index"],
        "metrics_seconds": phases,
        "steps_seconds_rank_max": steps,
        "peak_allocated_bytes_rank_max": max(record["peak_allocated_bytes"] for record in records),
        "peak_reserved_bytes_rank_max": max(record["peak_reserved_bytes"] for record in records),
        "model_load_seconds_rank_max": max(record["model_load_seconds"] for record in records),
        "output_path": records[0]["output_path"],
        "output_bytes": records[0].get("output_bytes"),
        "conditioning": records[0]["conditioning"],
    }


def summarize_runs(runs):
    measured = [run for run in runs if run["phase"] == "measured"]
    if not measured:
        raise ValueError("At least one measured request is required")
    return {
        name: {"median": statistics.median(values), "min": min(values), "max": max(values), "values": values}
        for name in measured[0]["metrics_seconds"]
        for values in [[run["metrics_seconds"][name] for run in measured]]
    }


def compare_summaries(args):
    reference, candidate = [json.loads(Path(path).read_text()) for path in args.compare]
    if not reference.get("complete") or not candidate.get("complete"):
        raise ValueError("Both benchmarks must complete their warmup and measured requests")
    differences = {}
    for group in ("environment", "common_config", "request", "protocol"):
        left, right = reference["metadata"][group], candidate["metadata"][group]
        if left != right:
            differences[group] = {"reference": left, "candidate": right}
    metrics = {}
    for key, value in reference["measured"].items():
        a, b = value["median"], candidate["measured"][key]["median"]
        metrics[key] = {"reference_median_seconds": a, "candidate_median_seconds": b, "reference_speedup_vs_candidate": b / a if a > 0 else None}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", {"protocol_matches": not differences, "differences": differences, "reference": args.compare[0], "candidate": args.compare[1], "metrics": metrics})
    print(f"Common protocol matches: {not differences}; comparison: {output / 'comparison.json'}")
    if differences:
        raise SystemExit(1)


class Timings:
    def __init__(self, runner, synchronize):
        self.runner, self.synchronize = runner, synchronize
        self.current = None
        self.step_started = None
        for method, phase in (
            ("run_pipeline", "pipeline"),
            ("run_input_encoder", "input_encoder"),
            ("run_text_encoder", "text_encoder"),
            ("_encode_keyframes", "vae_encode"),
            ("_encode_references", "vae_encode"),
            ("init_run", "dit_prepare"),
            ("run_segment", "dit_loop"),
            ("_offload_transformer", "dit_offload"),
            ("run_vae_decoder", "vae_decode"),
            ("process_images_after_vae_decoder", "save"),
        ):
            self.wrap(runner, method, phase)
        self.wrap(runner.video_vae, "decode", "video_vae_decode")
        self.wrap(runner.audio_vae, "decode", "audio_vae_decode")
        self.install_step_hooks()

    def wrap(self, owner, method, phase):
        original = getattr(owner, method)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            self.synchronize()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            self.synchronize()
            elapsed = time.perf_counter() - started
            self.current["phases_seconds"][phase] = self.current["phases_seconds"].get(phase, 0.0) + elapsed
            self.current["calls"][phase] = self.current["calls"].get(phase, 0) + 1
            if phase == "dit_prepare":
                scheduler = self.runner.scheduler
                self.current["conditioning"] = {
                    "video_reference_rows": scheduler.num_condition_video_rows,
                    "audio_reference_rows": scheduler.num_condition_audio_rows,
                    "qwen_rows": int(self.runner.inputs["text_encoder_output"]["prompt_embeds"].shape[0]),
                }
            return result

        setattr(owner, method, measured)

    def install_step_hooks(self):
        scheduler = self.runner.scheduler
        step_pre, step_post = scheduler.step_pre, scheduler.step_post

        def before(*args, **kwargs):
            self.synchronize()
            self.step_started = time.perf_counter()
            return step_pre(*args, **kwargs)

        def after(*args, **kwargs):
            result = step_post(*args, **kwargs)
            self.synchronize()
            self.current["steps_seconds"].append(time.perf_counter() - self.step_started)
            self.step_started = None
            return result

        scheduler.step_pre, scheduler.step_post = before, after


def environment_metadata(torch):
    from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "gpu": device.name,
        "gpu_memory_bytes": device.total_memory,
        "compute_capability": [device.major, device.minor],
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "dtype": str(GET_DTYPE()),
        "sensitive_dtype": str(GET_SENSITIVE_DTYPE()),
        "torch_num_threads": torch.get_num_threads(),
        "profiling_debug_level": os.environ["PROFILING_DEBUG_LEVEL"],
        "recorder_mode": os.environ["RECORDER_MODE"],
    }


def file_identity(path):
    path = Path(path).resolve()
    record = {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    if path.suffix == ".safetensors":
        with path.open("rb") as handle:
            header = handle.read(int.from_bytes(handle.read(8), "little"))
        record["header_sha256"] = hashlib.sha256(header).hexdigest()
        record["safetensors_metadata"] = json.loads(header).get("__metadata__", {})
    else:
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def model_metadata(runner):
    config = runner.config
    record = {key: config.get(key) for key in ("model_path", "vdn_checkpoint", "lora_configs", "lora_dynamic_apply", "video_flow_shift", "audio_flow_shift", "h3_step_update", "vdn_linear_use_tf32")}
    record["lightx2v_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    sources = (
        "models/runners/minimax_h3/minimax_h3_runner.py",
        "models/networks/minimax_h3/model.py",
        "models/networks/minimax_h3/infer/transformer_infer.py",
        "models/networks/minimax_h3/infer/vdn_attention.py",
        "models/networks/minimax_h3/infer/vdn_linear.py",
        "models/networks/minimax_h3/infer/vdn_kernels.py",
        "models/networks/minimax_h3/weights/vdn.py",
        "models/networks/minimax_h3/lora.py",
        "models/networks/minimax_h3/adaln_cache.py",
        "models/schedulers/minimax_h3/scheduler.py",
    )
    record["source_sha256"] = {name: hashlib.sha256((REPO / "lightx2v" / name).read_bytes()).hexdigest() for name in sources}
    if config.get("vdn_checkpoint"):
        checkpoint = Path(config["vdn_checkpoint"])
        record["model_spec"] = json.loads((checkpoint / "model_spec.json").read_text())
        record["release_metadata"] = json.loads((checkpoint / "metadata.json").read_text())
        names = [adapter["config"].get("name", "default") for adapter in record["model_spec"]["adapters"]]
        record["adapter_files"] = [file_identity(checkpoint / "adapters" / name / "adapter_model.safetensors") for name in names]
        record["lora_application"] = {"mode": "load_time_merge", "order": names, "scale": 1.0, "arithmetic": "CPU FP32 B@A; cast delta to base dtype; ordered in-place add"}
    else:
        record["adapter_files"] = [file_identity(lora["path"]) for lora in config.get("lora_configs", [])]
        record["lora_application"] = {
            "mode": "dynamic" if config.get("lora_dynamic_apply", False) else "load_time_merge",
            "implementation": "MiniMaxH3LoraAdapter",
            "merge_device": config.get("lora_merge_device", "auto"),
        }
    return record


def request_metadata(input_info, runner):
    images = []
    image_path = getattr(input_info, "image_path", None)
    if image_path:
        paths = image_path.split(",") if isinstance(image_path, str) else image_path
        for path in paths:
            images.append({"sha256": hashlib.sha256(Path(path.strip()).read_bytes()).hexdigest()})
    visual_only = bool(runner.config.get("h3_visual_reference_only", False))
    if input_info.task not in ("t2av", "ref2av") or (input_info.task == "ref2av" and not visual_only):
        raise ValueError("Benchmark requests must be T2AV or explicit h3_visual_reference_only image conditioning")
    return {
        "task": input_info.task,
        "conditioning": "qwen_only_images" if input_info.task == "ref2av" else "text_only",
        "prompt_sha256": hashlib.sha256(input_info.prompt.encode()).hexdigest(),
        "images": images,
        "size": list(input_info.size),
        "num_frames": input_info.num_frames,
        "seed": input_info.seed,
    }


def run_benchmark(args):
    # Set these before importing LightX2V: profiler decorators resolve at import.
    os.environ["PROFILING_DEBUG_LEVEL"] = "0"
    os.environ["RECORDER_MODE"] = "0"
    sys.path.insert(0, str(REPO))
    import torch
    import torch.distributed as dist

    from lightx2v.models.runners import runner_factory
    from lightx2v.utils.profiler import no_sync_profiling

    output = Path(args.output_dir).resolve()
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "videos").mkdir()
    original_build = runner_factory.build_runner
    all_runs, local_runs = [], []
    metadata = {
        "protocol": {
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "first_request_is_warmup_0": True,
            "text_encoder": "recomputed_every_request",
            "save_every_request": True,
            "timing": "synchronized_local_wall_then_rank_max",
            "step_boundary": "scheduler.step_pre through step_post",
        }
    }

    def build(config):
        if config["model_cls"] != "minimax_h3" or config.get("model_variant") != "fl2av":
            raise ValueError("This benchmark requires model_cls=minimax_h3 and model_variant=fl2av")
        if config.get("warmup", False) or config.get("enable_reuse", False):
            raise ValueError("Set config warmup=false and enable_reuse=false; this harness owns repeats and recomputes TE")
        torch.cuda.synchronize()
        started = time.perf_counter()
        runner = original_build(config)
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - started
        metadata["environment"] = environment_metadata(torch)
        metadata["common_config"] = {key: runner.config.get(key) for key in COMMON_CONFIG}
        metadata["model"] = model_metadata(runner)
        timer = Timings(runner, torch.cuda.synchronize)
        original_request = runner.run_request

        def requests(input_info):
            metadata["request"] = request_metadata(input_info, runner)
            total = args.warmup_runs + args.measured_runs
            for index in range(total):
                phase = "warmup" if index < args.warmup_runs else "measured"
                phase_index = index if phase == "warmup" else index - args.warmup_runs
                request = copy.deepcopy(input_info)
                request.return_result_tensor = False
                request.save_result_path = str(output / "videos" / f"{phase}_{phase_index:02d}.mp4")
                timer.current = {
                    "rank": rank,
                    "phase": phase,
                    "index": phase_index,
                    "infer_steps": runner.scheduler.infer_steps,
                    "model_load_seconds": model_load_seconds,
                    "phases_seconds": {},
                    "steps_seconds": [],
                    "calls": {},
                    "output_path": request.save_result_path,
                }
                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                try:
                    with no_sync_profiling():
                        original_request(request)
                    torch.cuda.synchronize()
                    timer.current["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    timer.current["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    metadata["environment"] = environment_metadata(torch)
                    timer.current["environment"] = metadata["environment"]
                    if rank == 0:
                        timer.current["output_bytes"] = Path(request.save_result_path).stat().st_size
                    local_runs.append(timer.current)
                    vdn_attention = getattr(runner.model.transformer_infer, "vdn_attention", None)
                    metadata["window_backend"] = vdn_attention.window.backend if vdn_attention is not None else None
                    metadata["vdn_linear_use_tf32"] = vdn_attention.linear_use_tf32 if vdn_attention is not None else None
                    metadata["window_block_stats"] = vdn_attention.window.block_stats if vdn_attention is not None else None
                    write_json(output / f"timing_rank{rank}.json", {"complete": index + 1 == total, "model_load_seconds": model_load_seconds, "metadata": metadata, "runs": local_runs})
                    if dist.is_initialized():
                        gathered = [None] * dist.get_world_size()
                        dist.all_gather_object(gathered, timer.current)
                    else:
                        gathered = [timer.current]
                    reduced = summarize_rank_records(gathered)
                    if any(reduced["conditioning"][key] for key in ("video_reference_rows", "audio_reference_rows")):
                        raise ValueError("T2AV/Qwen-only benchmark unexpectedly contains VAE reference rows")
                    all_runs.append(reduced)
                    if rank == 0:
                        summary = {
                            "complete": index + 1 == total,
                            "metadata": metadata,
                            "first_request": all_runs[0],
                            "runs": all_runs,
                            "model_load_seconds_rank_max": reduced["model_load_seconds_rank_max"],
                            "measured": summarize_runs(all_runs) if phase == "measured" else None,
                        }
                        write_json(output / "summary.json", summary)
                        print(
                            f"{phase} {phase_index}: pipeline={reduced['metrics_seconds']['pipeline']:.3f}s, "
                            f"DiT={reduced['metrics_seconds']['dit_loop']:.3f}s, step median={reduced['metrics_seconds']['step_median']:.3f}s",
                            flush=True,
                        )
                except BaseException:
                    write_json(output / f"failed_rank{rank}.json", {"metadata": metadata, "completed_runs": local_runs, "current": timer.current})
                    raise
            return {"video": None, "audio": None}

        runner.run_request = requests
        return runner

    runner_factory.build_runner = build
    cli = args.infer_args[1:] if args.infer_args and args.infer_args[0] == "--" else args.infer_args
    sys.argv = ["lightx2v.infer", *cli]
    try:
        runpy.run_module("lightx2v.infer", run_name="__main__")
    finally:
        runner_factory.build_runner = original_build
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--compare", nargs=2, metavar=("REFERENCE", "CANDIDATE"))
    parser.add_argument("infer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.compare:
        compare_summaries(args)
    else:
        if args.warmup_runs < 1 or args.measured_runs < 1:
            parser.error("At least one warmup and one measured request are required")
        run_benchmark(args)


if __name__ == "__main__":
    main()
