"""Run training validation samples through the native LightX2V Qwen-Image-2.1 runner.

Defaults to the first three T2I, single-reference I2I and multi-reference I2I
samples. Model/config/manifest/output defaults are relative to this repository,
independent of the working directory. Use python on one GPU, not torchrun.
Cluster-provided rank variables are normalized for this single-process preview.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
        raise ValueError(f"Validation indices must be in [0, {len(records) - 1}].")

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
    parser.add_argument("--model-path", type=Path, default=TRAIN_ROOT / "models/Qwen-Image-2.1")
    parser.add_argument("--config-json", type=Path, default=ROOT / "configs/qwen_image_21/qwen_image_21_validation.json")
    parser.add_argument("--manifest", type=Path, default=TRAIN_ROOT / "data/opengpt4o_qwen_image21/val_preview.jsonl")
    parser.add_argument("--output-dir", type=Path, default=TRAIN_ROOT / "output_infer/qwen_image21_lightx2v_teacher40_preview")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--per-task", type=int, default=3)
    selection.add_argument("--indices", type=int, nargs="+", help="Zero-based validation indices, e.g. 0 2 7 32 40 64 70.")
    parser.add_argument("--seed", type=int, default=42, help="Actual seed is this base plus the validation index.")
    parser.add_argument("--dry-run", action="store_true", help="Check paths and print requests without importing models or using a GPU.")
    parser.add_argument("--dump-trace", action="store_true", help="Save CPU conditioning/latent/velocity tensors beside each PNG for exact replay debugging; adds CPU copies.")
    args = parser.parse_args()
    if args.per_task <= 0 or args.seed < 0:
        parser.error("--per-task must be positive and --seed must be non-negative.")
    # Cluster shells can export training ranks even for a plain python process.
    # This entry point is explicitly single-process; normalize before importing
    # LightX2V so inherited LOCAL_RANK cannot select an invisible GPU either.
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1")
    for key in ("model_path", "config_json", "manifest", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().absolute())
    config = resolve_inference_config(json.loads(args.config_json.read_text()), args.model_path)
    if config.get("parallel"):
        parser.error("This single-GPU preview requires parallel=false in --config-json.")
    if args.dump_trace and config.get("enable_cfg", False):
        parser.error("--dump-trace currently requires enable_cfg=false in --config-json.")
    for component in ("transformer", "text_encoder", "processor", "vae", "scheduler"):
        if not (args.model_path / component).is_dir():
            parser.error(f"Missing model component: {args.model_path / component}")
    jobs = build_requests(args.manifest, args.output_dir, args.per_task, args.indices, args.seed)
    if args.dry_run:
        print(json.dumps({"model_path": str(args.model_path), "config": config, "samples": jobs, "dump_trace": args.dump_trace}, ensure_ascii=False, indent=2))
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
    if args.dump_trace:
        from tools.extract.validation_trace import FORMAT_VERSION, capture_validation_trace, save_validation_trace

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
        print(f"[{position}/{len(jobs)}] validation={job['validation_index']} {job['group']} size={request['size']} seed={request['seed']}", flush=True)
        if args.dump_trace:
            with capture_validation_trace(pipe.runner, job, args.model_path) as trace:
                pipe.generate(**request)
            save_validation_trace(trace, Path(request["save_result_path"]).with_suffix(".pt"))
        else:
            pipe.generate(**request)
        if not Path(request["save_result_path"]).is_file():
            raise RuntimeError(f"Native runner did not save {request['save_result_path']}")
    print(f"Saved {len(jobs)} images to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
