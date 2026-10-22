"""Compare dense rolling-cache execution with the complete Self-Forcing DiT.

Each measured chunk includes all denoising steps and the clean context rerun.
Text encoding, VAE decoding, model loading, and the first request are separate.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
import triton
from loguru import logger
from torch._dynamo.utils import counters

from lightx2v.pipeline import LightX2VPipeline
from lightx2v.utils.utils import save_to_video, seed_all, wan_vae_to_comfy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--size", type=int, nargs=2, default=(480, 832))
    parser.add_argument("--chunks", type=int, default=12)
    parser.add_argument("--local-frames", type=int, default=12)
    parser.add_argument("--sink-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--compile", dest="use_compile", action="store_true")
    parser.add_argument("--compile-options", type=json.loads)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--reference-latents", type=Path)
    parser.add_argument("--require-exact-reference", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--prompt", default="A golden retriever runs along a sunny beach, waves rolling onto the sand, cinematic tracking shot.")
    args = parser.parse_args()
    if args.require_exact_reference and args.reference_latents is None:
        parser.error("--require-exact-reference requires --reference-latents")
    return args


def compile_counters():
    return {group: dict(values) for group, values in counters.items() if values}


def counter_delta(after, before):
    return {
        group: {name: value - before.get(group, {}).get(name, 0) for name, value in values.items() if value != before.get(group, {}).get(name, 0)}
        for group, values in after.items()
        if values != before.get(group, {})
    }


def graph_counts(graph):
    return {"captures": graph.capture_count, "replays": graph.replay_count} if graph is not None else {"captures": 0, "replays": 0}


@torch.no_grad()
def decode_latents(runner, latents, output_dir):
    started = time.perf_counter()
    chunks = latents.to("cuda").split(3, dim=1)
    decoded = []
    for index, chunk in enumerate(chunks):
        images = runner.vae_decoder.cached_decode_withflag(chunk, index == 0, index == len(chunks) - 1)
        decoded.append(wan_vae_to_comfy(images).cpu())
    frames = torch.cat(decoded)
    decode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    video_path = output_dir / f"video_{frames.shape[0]}frames.mp4"
    save_to_video(frames, str(video_path), fps=16, method="ffmpeg")
    preview_path = output_dir / "preview_10s.mp4"
    save_to_video(frames[:160], str(preview_path), fps=16, method="ffmpeg")
    return {
        "path": str(video_path),
        "preview_path": str(preview_path),
        "shape": list(frames.shape),
        "frames": frames.shape[0],
        "fps": 16,
        "duration_seconds": frames.shape[0] / 16,
        "decode_seconds": decode_seconds,
        "video_encode_seconds": time.perf_counter() - started,
        "cached_decode_chunk_frames": [chunk.shape[0] for chunk in decoded],
    }


def run_request(runner, seed):
    seed_all(seed)
    runner.init_run()
    records = []
    for chunk in range(runner.video_segment_num):
        torch.cuda.synchronize()
        chunk_start = time.perf_counter()
        events = []
        for step in range(runner.scheduler.infer_steps):
            runner.scheduler.step_pre(seg_index=chunk, step_index=step, is_rerun=False)
            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before.record()
            runner.model.infer(runner.inputs)
            after.record()
            events.append((before, after))
            runner.scheduler.step_post()

        runner.scheduler.step_pre(seg_index=chunk, step_index=runner.scheduler.infer_steps - 1, is_rerun=True)
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record()
        runner.model.infer(runner.inputs)
        after.record()
        torch.cuda.synchronize()
        records.append(
            {
                "chunk": chunk,
                "wall_ms": (time.perf_counter() - chunk_start) * 1000,
                "denoise_ms": [start.elapsed_time(end) for start, end in events],
                "clean_commit_ms": before.elapsed_time(after),
            }
        )
        graph = getattr(runner.model.transformer_infer, "cuda_graph_stack", None)
        if graph is not None:
            records[-1]["graph_captures"] = graph.capture_count
            records[-1]["graph_replays"] = graph.replay_count
    return records, runner.scheduler.latents.detach().cpu().clone()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "num_frames": args.chunks * 3 * 4 - 3,
        "text_len": 512,
        "size": list(args.size),
        "self_attn_1_type": "flash_attn3",
        "cross_attn_1_type": "flash_attn3",
        "sample_shift": 5.0,
        "enable_cfg": False,
        "cpu_offload": False,
        "dit_original_ckpt": args.checkpoint,
        "causal_rope_type": "wan_causal_rope",
        "use_cuda_graph": args.mode == "graph",
        "use_compile": args.use_compile,
        "warmup": args.warmup,
        "fps": 16,
        "ar_config": {
            "local_attn_size": args.local_frames,
            "sink_size": args.sink_frames,
            "num_frame_per_chunk": 3,
            "timesteps_index": [0, 179, 358, 679],
            "kv_offload": False,
            "async_vae_decode": False,
        },
    }
    if args.num_layers is not None:
        config["num_layers"] = args.num_layers
    if args.compile_options is not None:
        config["compile_options"] = args.compile_options
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    logger.remove()
    logger.add(str(args.output_dir / "run.log"), level="INFO")
    load_start = time.perf_counter()
    pipeline = LightX2VPipeline(task="t2v", model_path=args.model_path, model_cls="wan2.1_sf")
    pipeline.create_generator(config_json=str(config_path))
    runner = pipeline.runner
    load_seconds = time.perf_counter() - load_start
    warmup_seconds = getattr(runner, "warmup_time_seconds", 0.0)
    transformer_infer = runner.model.transformer_infer
    graph = getattr(transformer_infer, "cuda_graph_stack", None)
    startup_graph_captures = graph.capture_count if graph is not None else 0
    startup_compiled_blocks = len(transformer_infer.compiled_blocks)
    startup_compile_counters = compile_counters()
    print(
        json.dumps(
            {
                "phase": "model_loaded_and_warmup_complete",
                "mode": args.mode,
                "load_seconds": load_seconds,
                "warmup_seconds": warmup_seconds,
                "compiled_blocks": startup_compiled_blocks,
                "graph": graph_counts(graph),
            }
        ),
        flush=True,
    )
    runner.input_info = runner.prepare_request({"prompt": args.prompt, "seed": args.seed, "return_result_tensor": True})
    encode_start = time.perf_counter()
    runner.inputs = runner.run_input_encoder()
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_start
    torch.cuda.reset_peak_memory_stats()

    first_start = time.perf_counter()
    first_counters_before = compile_counters()
    first_graph_before = graph_counts(graph)
    first_records, first_latents = run_request(runner, args.seed)
    first_seconds = time.perf_counter() - first_start
    first_counters_after = compile_counters()
    first_graph_after = graph_counts(graph)
    print(
        json.dumps({"phase": "first_request_complete", "seconds": first_seconds, "compile_delta": counter_delta(first_counters_after, first_counters_before), "graph": first_graph_after}), flush=True
    )
    if args.require_exact_reference:
        reference = torch.load(args.reference_latents, map_location="cpu", weights_only=True)
        first_exact = torch.equal(first_latents.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
        first_error = (first_latents.float() - reference.float()).abs().max().item()
        print(json.dumps({"phase": "first_request_reference_check", "bitwise_equal": first_exact, "max_abs": first_error}), flush=True)
        if not first_exact:
            torch.save(first_latents, args.output_dir / "failed_first_latents.pt")
            failure = {"phase": "first_request_reference_check", "reference": str(args.reference_latents), "max_abs": first_error, "bitwise_equal": False, "config": config}
            (args.output_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise AssertionError("First request differs from the required exact reference; formal repeats skipped")
    measurements = []
    repeat_comparisons = []
    request_counters = []
    previous_latents = first_latents
    for repeat in range(args.repeats):
        counters_before = compile_counters()
        graph_before = graph_counts(graph)
        request_started = time.perf_counter()
        records, latents = run_request(runner, args.seed)
        request_counters.append(
            {
                "repeat": repeat,
                "request_seconds": time.perf_counter() - request_started,
                "compile_before": counters_before,
                "compile_after": compile_counters(),
                "compile_delta": counter_delta(compile_counters(), counters_before),
                "graph_before": graph_before,
                "graph_after": graph_counts(graph),
            }
        )
        measurements.append(records)
        repeat_comparisons.append({"repeat": repeat, "previous_request_max_abs": (latents.float() - previous_latents.float()).abs().max().item()})
        previous_latents = latents
        print(f"repeat={repeat} total_ms={sum(row['wall_ms'] for row in records):.3f}", flush=True)
    torch.save(latents, args.output_dir / "latents.pt")
    full_window_chunk = args.local_frames // 3 - 1
    steady = [row for records in measurements for row in records if row["chunk"] >= full_window_chunk]
    report = {
        "mode": args.mode,
        "config": config,
        "prompt": args.prompt,
        "seed": args.seed,
        "repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "lightx2v_path": str(Path(__import__("lightx2v").__file__).parent),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "checkpoint": {"path": args.checkpoint, "bytes": Path(args.checkpoint).stat().st_size},
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "model_load_without_warmup_seconds": load_seconds - warmup_seconds,
        "startup_graph_captures": startup_graph_captures,
        "startup_compiled_blocks": startup_compiled_blocks,
        "compile_counters_after_startup_warmup": startup_compile_counters,
        "first_request_compile_delta": counter_delta(first_counters_after, first_counters_before),
        "first_request_graph_before": first_graph_before,
        "first_request_graph_after": first_graph_after,
        "formal_request_counters": request_counters,
        "compiled_blocks": len(transformer_infer.compiled_blocks),
        "compile_executed": args.use_compile and compile_counters().get("stats", {}).get("unique_graphs", 0) > 0 and bool(transformer_infer.compiled_blocks),
        "cuda_graph_executed": graph is not None and graph.replay_count > 0,
        "compile_backend": transformer_infer.compile_backend,
        "effective_compile_options": transformer_infer.compile_options,
        "num_layers": runner.config["num_layers"],
        "text_encode_seconds": encode_seconds,
        "first_request_seconds": first_seconds,
        "first_request": first_records,
        "measurements": measurements,
        "repeat_comparisons": repeat_comparisons,
        "steady_start_chunk": full_window_chunk,
        "steady_chunk_median_ms": statistics.median(row["wall_ms"] for row in steady),
        "steady_denoise_median_ms": statistics.median(sum(row["denoise_ms"]) for row in steady),
        "steady_clean_commit_median_ms": statistics.median(row["clean_commit_ms"] for row in steady),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "first_vs_repeat_max_abs": (first_latents.float() - latents.float()).abs().max().item(),
        "latent_sha256": hashlib.sha256(latents.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        "latents_finite": bool(torch.isfinite(latents).all()),
        "latent_shape": list(latents.shape),
        "expected_output_frames": latents.shape[1] * 4 - 3,
        "expected_duration_seconds": (latents.shape[1] * 4 - 3) / 16,
        "timing_scope": "Chunk wall time covers four denoise calls, scheduler steps and the clean-context rerun; excludes text encode, VAE decode and video encode. Request seconds additionally include request cache reset and the final latent CPU copy. Startup load includes warmup. Shared disk compiler caches are retained.",
    }
    if args.reference_latents is not None:
        reference_raw = torch.load(args.reference_latents, map_location="cpu", weights_only=True)
        reference = reference_raw.float()
        actual = latents.float()
        difference = actual - reference
        report["reference_comparison"] = {
            "path": str(args.reference_latents),
            "bitwise_equal": torch.equal(latents.contiguous().view(torch.uint8), reference_raw.contiguous().view(torch.uint8)),
            "max_abs": difference.abs().max().item(),
            "rmse": difference.square().mean().sqrt().item(),
            "relative_rmse": (difference.square().mean().sqrt() / reference.square().mean().sqrt()).item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(actual.double().flatten(), reference.double().flatten(), dim=0).item(),
            "chunk_max_abs": [part.abs().max().item() for part in difference.split(3, dim=1)],
        }
    if args.decode:
        report["video"] = decode_latents(runner, latents, args.output_dir)
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.require_exact_reference:
        assert report["reference_comparison"]["bitwise_equal"], "Final repeated request differs from the required exact reference"
    print(json.dumps({key: report[key] for key in ("mode", "steady_chunk_median_ms", "steady_denoise_median_ms", "steady_clean_commit_median_ms", "first_vs_repeat_max_abs")}), flush=True)


if __name__ == "__main__":
    main()
