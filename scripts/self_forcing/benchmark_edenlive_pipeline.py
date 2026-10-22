"""Run the real EdenLive cryo-crossing case with production warmup and inference."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path

os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "BF16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROFILING_DEBUG_LEVEL", "0")

import torch

from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO
from lightx2v.models.networks.wan.edenlive_model import EdenLiveARModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.wan.edenlive_runner import EdenLiveRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.pipeline import LightX2VPipeline
from lightx2v.utils.utils import save_to_video


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attention-backend", choices=("flash_attn3", "sage_attn2"), default="flash_attn3")
    parser.add_argument("--execution", choices=("compiled_graph", "eager"), default="compiled_graph")
    parser.add_argument("--prompt-mode", choices=("static", "dynamic"), default="dynamic")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=249)
    parser.add_argument("--seed", type=int, default=2026092901)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--recompile-limit", type=int)
    parser.add_argument("--compare-first-infer", action="store_true", help="Diagnostic run: compare the first real denoise prediction with eager using the same input and weights")
    return parser.parse_args()


def compile_counters():
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    return {**{name: dict(values) for name, values in counters.items() if values}, "generated_kernel_count": metrics.generated_kernel_count}


def fallback_counters():
    from torch._dynamo.utils import counters

    return {name: dict(counters[name]) for name in ("unimplemented", "graph_break")}


def numeric_delta(after, before):
    result = {}
    for key, value in after.items():
        if isinstance(value, dict):
            result[key] = numeric_delta(value, before.get(key, {}))
        else:
            result[key] = value - before.get(key, 0)
    return result


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def graph_counts(graph):
    return (0, 0) if graph is None else (graph.capture_count, graph.replay_count)


class Measurements:
    def __init__(self, compare_first_infer=False):
        self.phase = "startup"
        self.records = []
        self.dit_calls = []
        self.latents = None
        self.warmup_seconds = 0.0
        self.compare_first_infer = compare_first_infer
        self.first_infer = None

    def wrap(self, owner, name, stage):
        original = getattr(owner, name)
        measurements = self

        @wraps(original)
        def measured(instance, *args, **kwargs):
            previous = measurements.phase
            if stage == "warmup":
                measurements.phase = "warmup"
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            begin = time.perf_counter()
            try:
                return original(instance, *args, **kwargs)
            finally:
                end.record()
                wall = time.perf_counter() - begin
                measurements.records.append((measurements.phase, stage, wall, start, end))
                if stage == "warmup":
                    torch.cuda.synchronize()
                    measurements.warmup_seconds += time.perf_counter() - begin
                    measurements.phase = previous

        setattr(owner, name, measured)

    def install(self):
        self.wrap(EdenLiveRunner, "_run_warmup", "warmup")
        self.wrap(EdenLiveRunner, "_run_input_encoder_local_i2v", "input_encode")
        self.wrap(EdenLiveRunner, "_encode_inputs", "condition_encode")
        self.wrap(WanRunner, "run_text_encoder", "text_encode")
        self.wrap(ProxyDINO, "infer", "proxy")
        self.wrap(DefaultRunner, "run_vae_decoder", "vae_decode")
        self.wrap(EdenLiveARModel, "infer", "dit")
        original_infer = EdenLiveARModel.infer
        original_end = EdenLiveRunner.end_run
        measurements = self

        @wraps(original_infer)
        def record_dit(model, inputs):
            infer = model.transformer_infer
            graph = infer.cuda_graph_stack
            before = graph_counts(graph)
            result = original_infer(model, inputs)
            after = graph_counts(graph)
            if measurements.compare_first_infer and measurements.first_infer is None and measurements.phase.startswith("request") and not model.scheduler.cache_update:
                scheduler = model.scheduler
                prediction = scheduler.noise_pred.detach().clone()
                first_input = scheduler.latents_input.detach().clone()
                previous_phase = measurements.phase
                previous_compile = infer.use_compile
                previous_graph = infer.cuda_graph_stack
                try:
                    measurements.phase = "teacher_forced_eager"
                    infer.use_compile = False
                    infer.cuda_graph_stack = None
                    original_infer(model, inputs)
                    measurements.first_infer = {"latents_input": first_input, "compiled_noise_pred": prediction, "eager_noise_pred": scheduler.noise_pred.detach().clone()}
                finally:
                    infer.use_compile = previous_compile
                    infer.cuda_graph_stack = previous_graph
                    scheduler.noise_pred = prediction
                    measurements.phase = previous_phase
            measurements.dit_calls.append(
                {
                    "phase": measurements.phase,
                    "chunk_start": model.scheduler.chunk_start,
                    "step": model.scheduler.step_index,
                    "clean_commit": model.scheduler.cache_update,
                    "captures": after[0] - before[0],
                    "replays": after[1] - before[1],
                    "cross_generation": infer.cross_attn_kv_cache.generation,
                    "graph_cross_generation": infer._graph_cross_generation,
                    "cache_update_only": infer.cache_update_only,
                    "cache_update_done": infer.cache_update_done,
                }
            )
            return result

        @wraps(original_end)
        def retain_latents(runner):
            if measurements.phase.startswith("request"):
                measurements.latents = runner.scheduler.latents.detach()
            return original_end(runner)

        EdenLiveARModel.infer = record_dit
        EdenLiveRunner.end_run = retain_latents

    def summarize(self, phase):
        rows = defaultdict(lambda: {"calls": 0, "wall_seconds": 0.0, "cuda_ms": 0.0})
        for recorded_phase, stage, wall, start, end in self.records:
            if recorded_phase == phase:
                row = rows[stage]
                row["calls"] += 1
                row["wall_seconds"] += wall
                row["cuda_ms"] += start.elapsed_time(end)
        return dict(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@torch.no_grad()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_prompts = json.loads((args.case_dir / "prompts.json").read_text())
    config = json.loads((Path(__file__).resolve().parents[2] / "configs/edenlive/wan22_5b_few_step.json").read_text())
    optimized = args.execution == "compiled_graph"
    config.update(
        size=[args.height, args.width],
        num_frames=args.frames,
        self_attn_1_type=args.attention_backend,
        cross_attn_1_type=args.attention_backend,
        cross_attn_2_type=args.attention_backend,
        use_compile=optimized,
        use_cuda_graph=optimized,
        warmup=True,
    )
    if args.prompt_mode == "dynamic":
        config["prompt_travel"] = {
            "prompt_travel_text": [raw_prompts["staticPrompt"] + " " + item["text"] for item in raw_prompts["dynamicPrompts"]],
            "prompt_travel_start_frames": [round(item["startMs"] * config["fps"] / 1000) for item in raw_prompts["dynamicPrompts"]],
        }
    config_path = args.output_dir / "config.json"
    write_json(config_path, config)
    write_json(args.output_dir / "original_prompts.json", raw_prompts)
    if args.recompile_limit is not None:
        torch._dynamo.config.recompile_limit = args.recompile_limit
    measurements = Measurements(compare_first_infer=args.compare_first_infer)
    measurements.install()
    report = {
        "execution": args.execution,
        "command": sys.argv,
        "environment": {name: os.getenv(name) for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "DTYPE", "SENSITIVE_LAYER_DTYPE", "PROFILING_DEBUG_LEVEL", "TORCH_LOGS")},
        "model_path": str(args.model_path),
        "case_dir": str(args.case_dir),
        "source": "https://github.com/VAST-AI-Research/EdenLive/tree/main/examples/samples/cryo-crossing",
        "source_duration_seconds": raw_prompts["durationMs"] / 1000,
        "input_sha256": {name: hashlib.sha256((args.case_dir / name).read_bytes()).hexdigest() for name in ("cover.png", "recording.mp4", "prompts.json")},
        "requested_frames": args.frames,
        "size": [args.height, args.width],
        "seed": args.seed,
        "prompt_mode": args.prompt_mode,
        "recompile_limit": torch._dynamo.config.recompile_limit,
        "profiling_debug_level": os.environ["PROFILING_DEBUG_LEVEL"],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "timing_scope": "Pipeline includes input decoding/encoding, DiT, ProxyDINO, VAE decode and result CPU transfer; MP4/tensor serialization is outside. Stage timings are nested and must not be summed; CUDA events include intervening host idle time. Stage wall time measures the enclosing Python call without an added barrier.",
        "teacher_forced_diagnostic": args.compare_first_infer,
        "requests": [],
    }
    before = compile_counters()
    fallback_before = fallback_counters()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pipeline = LightX2VPipeline(task="i2v", model_path=str(args.model_path), model_cls="wan22_edenlive_ar")
    try:
        pipeline.create_generator(config_json=str(config_path))
    except BaseException as error:
        report.update(
            status="startup_failed_or_interrupted",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_startup_seconds=time.perf_counter() - start,
            compile_delta=numeric_delta(compile_counters(), before),
            fallback_delta=numeric_delta(fallback_counters(), fallback_before),
        )
        write_json(args.output_dir / "report.json", report)
        raise
    torch.cuda.synchronize()
    startup = time.perf_counter() - start
    infer = pipeline.runner.model.transformer_infer
    graph = infer.cuda_graph_stack
    after_warmup = compile_counters()
    report.update(
        startup_seconds=startup,
        warmup_seconds=measurements.warmup_seconds,
        loading_seconds=startup - measurements.warmup_seconds,
        warmup_stages=measurements.summarize("warmup"),
        warmup_compile_delta=numeric_delta(after_warmup, before),
        warmup_graph={"captures": graph_counts(graph)[0], "replays": graph_counts(graph)[1], "retained": graph is not None and graph.graph is not None},
        warmup_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        flags={"compile": infer.use_compile, "backend": infer.compile_backend, "cuda_graph": type(graph).__name__, "warmup_done": pipeline.runner._warmup_done},
        actual_weight_dtype=str(pipeline.runner.model.transformer_weights.blocks[0].compute_phases[0].self_attn_q.weight.dtype),
    )
    workspace = getattr(infer.self_attn_backend, "workspace", None)
    history = workspace.history if workspace is not None else {}
    report["attention_kv_cache"] = {
        "plugin": type(infer.self_attn_backend).__name__,
        "history_layers": len(history),
        "history_storage_bytes": sum(tensor.numel() * tensor.element_size() for _, key, value in history.values() for tensor in (key, value)),
    }
    report["warmup_fallback_delta"] = numeric_delta(fallback_counters(), fallback_before)
    report["compiled_callables_count"] = len(infer.compiled_blocks)
    report["compiled_kv_block_count"] = len(infer.compiled_kv_blocks)
    report["warmup_resolutions"] = [list(shape) for shape in pipeline.runner.get_warmup_resolutions()]
    write_json(args.output_dir / "report.json", report)
    for request in range(args.requests):
        measurements.phase = f"request_{request}"
        before = compile_counters()
        fallback_before = fallback_counters()
        graph_before = graph_counts(graph)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = pipeline.generate(
            seed=args.seed,
            prompt=raw_prompts["staticPrompt"],
            image_path=str(args.case_dir / "cover.png"),
            video_path=str(args.case_dir / "recording.mp4"),
            num_frames=args.frames,
            return_result_tensor=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        after = compile_counters()
        video = result["video"]
        latents = measurements.latents.cpu()
        torch.save(latents, args.output_dir / f"request_{request}_latents.pt")
        torch.save(video, args.output_dir / f"request_{request}_video.pt")
        save_start = time.perf_counter()
        save_to_video(video, str(args.output_dir / f"request_{request}.mp4"), fps=config["fps"], method="ffmpeg")
        mp4_seconds = time.perf_counter() - save_start
        row = {
            "pipeline_seconds": elapsed,
            "startup_plus_pipeline_seconds": startup + elapsed,
            "stages": measurements.summarize(measurements.phase),
            "compile_delta": numeric_delta(after, before),
            "graph_delta": {"captures": graph_counts(graph)[0] - graph_before[0], "replays": graph_counts(graph)[1] - graph_before[1]},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "video_shape": list(video.shape),
            "video_sha256": tensor_digest(video),
            "latent_shape": list(latents.shape),
            "latent_sha256": tensor_digest(latents),
            "mp4_save_seconds": mp4_seconds,
        }
        row["fallback_delta"] = numeric_delta(fallback_counters(), fallback_before)
        row["compiled_callables_count"] = len(infer.compiled_blocks)
        row["compiled_kv_block_count"] = len(infer.compiled_kv_blocks)
        if measurements.first_infer is not None:
            tensors = {name: tensor.cpu() for name, tensor in measurements.first_infer.items()}
            torch.save(tensors, args.output_dir / "first_infer_comparison.pt")
            difference = tensors["compiled_noise_pred"].float() - tensors["eager_noise_pred"].float()
            report["first_infer_comparison"] = {
                "max_abs": difference.abs().max().item(),
                "rmse": difference.square().mean().sqrt().item(),
                "relative_rmse": (difference.square().mean() / tensors["eager_noise_pred"].float().square().mean()).sqrt().item(),
                "input_sha256": tensor_digest(tensors["latents_input"]),
            }
        report["requests"].append(row)
        report["dit_calls"] = measurements.dit_calls
        write_json(args.output_dir / "report.json", report)
        print(json.dumps({"request": request, **row}), flush=True)
        measurements.latents = None
        del video, latents, result


if __name__ == "__main__":
    main()
