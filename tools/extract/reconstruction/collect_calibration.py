"""Single-GPU calibration worker invoked by lora_reconstructor.py.

Run target inference and save every request's denoising trajectory for TFMQ.
The coordinator supplies model/config paths and selected manifest indices.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = ROOT / "lightx2v_train"
TASKS = ("t2i", "i2i_single", "i2i_multi")


def resolve_inference_config(config, model_path):
    """Pass saved target sampling nodes to the unmodified native scheduler."""
    config = dict(config)
    model_path = Path(model_path)
    scheduler = config.get("scheduler_config") or {}
    scheduler_path = model_path / "scheduler/scheduler_config.json"
    if not scheduler and scheduler_path.is_file():
        scheduler = json.loads(scheduler_path.read_text())
        config["scheduler_config"] = scheduler
    if config.get("sigmas") is not None or scheduler.get("use_dynamic_shifting") is not False:
        return config
    index_path = model_path / "model_index.json"
    sigmas = json.loads(index_path.read_text()).get("sample_sigmas") if index_path.is_file() else None
    if sigmas is None:
        return config
    if scheduler.get("shift", 1.0) != 1.0 or scheduler.get("shift_terminal") is not None:
        raise ValueError("Saved Qwen-Image-2.1 sample_sigmas require shift=1 and shift_terminal=null")
    if (
        not isinstance(sigmas, list)
        or len(sigmas) != config["infer_steps"]
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1 for value in sigmas)
        or any(left <= right for left, right in zip(sigmas, sigmas[1:]))
    ):
        raise ValueError("Qwen-Image-2.1 sample_sigmas must contain infer_steps finite, strictly decreasing values in (0, 1]")
    # These nodes already include shifting; explicit user sigmas above keep
    # their original do_time_shift behavior.
    config.update(sigmas=sigmas, do_time_shift=False)
    return config


def task_group(record):
    count = len(record.get("source_images", []))
    return "t2i" if count == 0 else "i2i_single" if count == 1 else "i2i_multi"


def build_requests(manifest, output_dir, per_task, indices, seed):
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if indices is None:
        counts = dict.fromkeys(TASKS, 0)
        indices = []
        for index, record in enumerate(records):
            group = task_group(record)
            if counts[group] < per_task:
                indices.append(index)
                counts[group] += 1
        if any(count < per_task for count in counts.values()):
            raise ValueError(f"Requested {per_task} per task, but found {counts}; use --indices for explicit selection.")
    if any(index < 0 or index >= len(records) for index in indices):
        raise ValueError(f"Manifest indices must be in [0, {len(records) - 1}].")

    jobs = []
    for index in dict.fromkeys(indices):
        record = records[index]
        group = task_group(record)
        height, width = int(record["target_height"]), int(record["target_width"])
        if min(height, width) < 32 or height % 32 or width % 32:
            raise ValueError(f"Sample {index} has an invalid output size: {height}x{width}.")
        refs = []
        for source in record.get("source_images", []):
            path = Path(source).expanduser()
            if not path.is_absolute():
                path = manifest.parent / path
            if not path.is_file():
                raise FileNotFoundError(f"Sample {index}: missing reference {path}")
            if "," in str(path):
                raise ValueError(f"LightX2V's image_path separator cannot represent a comma in {path}")
            refs.append(str(path))
        request = {
            "task": "t2i" if group == "t2i" else "i2i",
            "prompt": record["prompt"],
            "image_path": ",".join(refs) if refs else None,
            "size": [height, width],
            "seed": seed + index,
            "save_result_path": str(output_dir / f"{index:05d}_{group}.png"),
        }
        jobs.append({"validation_index": index, "sample_id": record.get("id"), "group": group, "request": request})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True, help="Zero-based manifest indices selected by the calibration coordinator.")
    parser.add_argument("--seed", type=int, default=42, help="Actual seed is this base plus the manifest index.")
    parser.add_argument("--dry-run", action="store_true", help="Check paths and print requests without importing models or using a GPU.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    # Cluster shells can export training ranks even for a plain python process.
    # This entry point is explicitly single-process; normalize before importing
    # LightX2V so inherited LOCAL_RANK cannot select an invisible GPU either.
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1")
    for key in ("model_path", "config_json", "manifest", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().absolute())
    config = resolve_inference_config(json.loads(args.config_json.read_text()), args.model_path)
    if config.get("parallel"):
        parser.error("Calibration collection requires parallel=false in --config-json.")
    if config.get("enable_cfg", False):
        parser.error("Calibration collection requires enable_cfg=false in --config-json.")
    for component in ("transformer", "text_encoder", "processor", "vae", "scheduler"):
        if not (args.model_path / component).is_dir():
            parser.error(f"Missing model component: {args.model_path / component}")
    jobs = build_requests(args.manifest, args.output_dir, per_task=1, indices=args.indices, seed=args.seed)
    if args.dry_run:
        print(json.dumps({"model_path": str(args.model_path), "config": config, "samples": jobs}, ensure_ascii=False, indent=2))
        return

    # Match the native inference launch environment before importing LightX2V.
    os.environ.setdefault("DTYPE", "BF16")
    os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "None")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PROFILING_DEBUG_LEVEL", "1")
    sys.path.insert(0, str(ROOT))
    from lightx2v.pipeline import LightX2VPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_config = args.output_dir / "inference_config.json"
    effective_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    run_info = {
        "model_path": str(args.model_path),
        "manifest": str(args.manifest),
        "config_json": str(args.config_json),
        "effective_config_json": str(effective_config),
        "config": config,
        "dtype": os.environ["DTYPE"],
        "samples": jobs,
        "noise_note": "Native LightX2V CPU noise; the same seed does not reproduce training's CUDA noise tensor.",
    }
    from tools.extract.reconstruction.calibration_trace import FORMAT_VERSION, capture_calibration_trace, save_calibration_trace

    run_info["trace"] = {
        "format_version": FORMAT_VERSION,
        "file_pattern": "<PNG stem>.pt",
        "tensor_layout": "latents/velocity [1, tokens, channels]; condition masks contain the prefix only",
        "loading": "torch.load(path, map_location='cpu', weights_only=True)",
    }
    (args.output_dir / "requests.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n")
    # task=None allows one runner to serve both T2I and I2I requests.
    pipe = LightX2VPipeline(task=None, model_path=str(args.model_path), model_cls="qwen_image_21")
    pipe.create_generator(config_json=str(effective_config))
    for position, job in enumerate(jobs, start=1):
        request = job["request"]
        print(f"[{position}/{len(jobs)}] sample={job['validation_index']} {job['group']} size={request['size']} seed={request['seed']}", flush=True)
        with capture_calibration_trace(pipe.runner, job, args.model_path) as trace:
            pipe.generate(**request)
        save_calibration_trace(trace, Path(request["save_result_path"]).with_suffix(".pt"))
        if not Path(request["save_result_path"]).is_file():
            raise RuntimeError(f"Native runner did not save {request['save_result_path']}")
    print(f"Saved {len(jobs)} calibration traces and images to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
