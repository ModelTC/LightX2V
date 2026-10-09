#!/usr/bin/env python3
"""Read-only DMAD launch preflight; never imports torch or opens tensor payloads."""

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path


def _load_helper(relative, name):
    # Import by file: data/runtime package initializers import torch.
    path = Path(__file__).resolve().parents[1] / "lightx2v_train" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check_sampler(rows, sampler):
    if not sampler:
        return
    cells = Counter()
    allowed = sampler.get("image_counts")
    cost_key = sampler.get("cost_key", "packed_sequence_tokens_124")
    for index, row in enumerate(rows, start=1):
        try:
            images = int(row.get("reference_image_count", row.get("ref_image_count")))
            videos, audios = int(row["reference_video_count"]), int(row["reference_audio_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"DMAD row {index}: reference image/video/audio count metadata is required by the cost sampler.") from error
        if images < 1 or (allowed is not None and images not in allowed):
            raise ValueError(f"DMAD row {index}: image count {images} is outside the configured image_counts={allowed}.")
        if sampler.get("require_image_only", True) and (videos or audios):
            raise ValueError(f"DMAD row {index}: this sampler requires image-only references.")
        if sampler.get("require_compute_cost", True):
            try:
                cost = int(row[cost_key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"DMAD row {index}: positive {cost_key} metadata is required by the cost sampler.") from error
            if cost < 1:
                raise ValueError(f"DMAD row {index}: {cost_key} must be positive.")
        orientation = "landscape" if int(row["target_width"]) > int(row["target_height"]) else "portrait"
        cells[(images, orientation)] += 1
    observed = {count for count, _ in cells}
    if allowed is not None and sampler.get("require_all_image_counts", True) and set(allowed) - observed:
        raise ValueError(f"DMAD sampler is missing image counts {sorted(set(allowed) - observed)}.")
    if sampler.get("batch_mode", "cost_local") == "cost_local" and sampler.get("balance_image_counts", False):
        deficient = {f"{count}/{orientation}": cells[(count, orientation)] for count in sorted(observed) for orientation in ("landscape", "portrait") if cells[(count, orientation)] < 16}
        if deficient:
            raise ValueError(
                f"FSDP32 cost-local image-count/orientation balancing needs at least 16 rows in every observed cell: {deficient}. Customize reference_cost_sampler in H3_DMAD_CONFIG for a different dataset."
            )


def validate_launch(config):
    training = config["training"]
    dmd = training["dmd"]
    model = config["model"]
    data = config["data"]["train"]
    distributed = config["distributed"]
    matching = model.get("capabilities", {}).get("distribution_matching", {})
    if training.get("method") != "dmad" or model.get("name") != "minimax_h3_ref2av":
        raise ValueError("This launcher requires method=dmad and model.name=minimax_h3_ref2av; old DMD/PDMD/HEAD configs are not accepted.")
    if matching.get("projected_dmd", False) or dmd.get("residual_head", {}).get("enabled", False):
        raise ValueError("DMAD is independent of PDMD and residual HEAD; disable both options.")
    if dmd.get("update_order", "student_first") != "student_first" or dmd.get("random_schedule", {}).get("enabled", False):
        raise ValueError("DMAD requires student_first and its own stochastic re-noise schedule; disable dmd.random_schedule.")
    if training.get("student", {}).get("ema", {}).get("enabled", False):
        raise ValueError("DMAD owns its power-function EMA banks; disable training.student.ema.")
    if not training.get("dmad", {}).get("gap_sync", True):
        raise ValueError("Distributed DMAD requires training.dmad.gap_sync=true.")
    if config.get("inference", {}).get("infer_every_iters"):
        raise ValueError("DMAD training requires inference.method=none / infer_every_iters=null; the DMD Euler inferencer is incompatible.")
    if data.get("name") != "minimax_h3_dmad_dataset":
        raise ValueError("DMAD requires minimax_h3_dmad_dataset with paired real and teacher latents; a condition-only dataset is insufficient.")
    fsdp = distributed.get("fsdp2", {})
    sequence = distributed.get("sequence_parallel", {})
    if not fsdp.get("enabled") or int(fsdp.get("size", 0)) != 32 or sequence.get("enabled", False) or int(sequence.get("size", 1)) != 1:
        raise ValueError("This ACP launcher requires FSDP32 and sequence parallel disabled (size 1).")
    if int(data.get("batch_size", 1)) != 1 or int(training.get("gradient_accumulation_iters", 1)) != 1:
        raise ValueError("The packed H3 DMAD recipe requires batch_size=1 and gradient_accumulation_iters=1.")
    if int(dmd.get("num_inference_steps", 0)) != 8:
        raise ValueError("This DMAD8 launcher requires num_inference_steps=8.")
    if int(training.get("max_train_iters", 0)) <= 0:
        raise ValueError("H3_DMAD_MAX_ITERS / training.max_train_iters must be positive.")
    model_path = Path(model["pretrained_model_name_or_path"])
    if not (model_path / "transformer_ref/config.json").is_file():
        raise FileNotFoundError(f"Missing Ref2AV transformer config: {model_path / 'transformer_ref/config.json'}")
    manifest = Path(data["data_path"])
    for name, actual in (("H3_DMAD_CACHE", manifest), ("H3_MODEL_PATH", model_path), ("H3_DMAD_OUTPUT", Path(training["output_dir"]))):
        if name in os.environ and actual.resolve() != Path(os.environ[name]).resolve():
            raise ValueError(f"Resolved config disagrees with {name}: {actual}")
    rows = _load_helper("data/minimax_h3_dmad_manifest.py", "_h3_dmad_manifest_preflight").validate_manifest(manifest)
    allowed_resolutions = matching.get("allowed_resolutions")
    fixed_frames = matching.get("fixed_num_frames")
    for index, row in enumerate(rows, start=1):
        resolution = [int(row["target_height"]), int(row["target_width"])]
        if allowed_resolutions and resolution not in allowed_resolutions:
            raise ValueError(f"DMAD row {index}: resolution {resolution} is outside the configured allowed_resolutions.")
        if fixed_frames and int(row["target_num_frames"]) != int(fixed_frames):
            raise ValueError(f"DMAD row {index}: target_num_frames={row['target_num_frames']} differs from configured fixed_num_frames={fixed_frames}.")
    _check_sampler(rows, data.get("reference_cost_sampler"))
    return rows


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_minimax_h3_dmad_launch.py CONFIG.yaml")
    try:
        config = _load_helper("runtime/config.py", "_h3_dmad_config_preflight").load_config(sys.argv[1])
        rows = validate_launch(config)
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise SystemExit(f"DMAD preflight failed: {error}") from error
    training = config["training"]
    dmd = training["dmd"]
    print(
        f"DMAD Ref2AV: 4 x 8 GPUs, paired_rows={len(rows)}, steps={dmd['num_inference_steps']}, iters={training['max_train_iters']}, update_order={dmd.get('update_order', 'student_first')}, fake_update_ratio={dmd['fake_update_ratio']}"
    )
    print("No online teacher; training uses paired offline real/teacher AV latents and stochastic re-noise rollouts.")
    print("Preflight checked metadata identity, geometry and paths only; tensor shape/normalization/finite checks run when the dataset loads each sample.")
    print(f"config={sys.argv[1]}\ndata={config['data']['train']['data_path']}\noutput={training['output_dir']}")
    print(f"sampler={json.dumps(config['data']['train'].get('reference_cost_sampler'), sort_keys=True)}")
    for role in ("student", "fake"):
        model = config["model"] if role == "student" else {**config["model"], **config["model"].get("fake", {})}
        precision = dict(config["distributed"]["fsdp2"]["mixed_precision"])
        if role == "fake":
            precision.update(model.get("distributed", {}).get("fsdp2", {}).get("mixed_precision", {}))
        print(f"precision {role}: master={model['transformer_param_dtype']}, compute={precision.get('param_dtype')}, reduce={precision.get('reduce_dtype')}")


if __name__ == "__main__":
    main()
