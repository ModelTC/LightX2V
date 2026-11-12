"""Check local Qwen assets, portable data and reward services before Ray starts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from omegaconf import OmegaConf

from lightx2v_rl.deployment.health import direct_urlopen as urlopen
from lightx2v_rl.paths import repository_path, resolve_runtime_paths
from lightx2v_rl.rewards.endpoint_pool import normalize_endpoints
from lightx2v_rl.rewards.preflight import check_reward_scoring
from lightx2v_rl.training.config import load_recipe, validate_recipe


def check_snapshot(root):
    root = Path(root)
    manifest = json.loads((root / "download-manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError(f"Download incomplete: {root}")
    for item in manifest["files"]:
        path = root / item["rfilename"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Missing or truncated model file: {path}")
    return {"repository": manifest["repository"], "revision": manifest["revision"]}


def _check_tensor_file(path):
    """Validate safetensors metadata and lengths without loading tensor payloads."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if not 2 <= header_size <= min(100_000_000, size - 8):
            raise ValueError(f"Invalid or truncated safetensors header: {path}")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict) or not header:
        raise ValueError(f"Invalid safetensors header: {path}")
    item_sizes = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ValueError(f"Empty safetensors shard: {path}")
    intervals = []
    for key, value in tensors.items():
        if not isinstance(value, dict) or not value:
            raise ValueError(f"Invalid safetensors tensor metadata: {path}: {key}")
        shape, offsets, dtype = value.get("shape"), value.get("data_offsets"), value.get("dtype")
        if (
            not isinstance(dtype, str)
            or dtype not in item_sizes
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * item_sizes[dtype]
        ):
            raise ValueError(f"Invalid safetensors tensor layout: {path}: {key}")
        intervals.append(tuple(offsets))
    expected = 0
    for start, end in sorted(intervals):
        if start != expected:
            raise ValueError(f"Overlapping or missing safetensors payload: {path}")
        expected = end
    if size != 8 + header_size + expected:
        raise ValueError(f"Missing or truncated safetensors payload: {path}")
    return tensors, expected


def _check_sharded_component(root, index_name):
    index = json.loads((root / index_name).read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Missing safetensors weight map: {root / index_name}")
    shards = {}
    for key, name in weight_map.items():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Invalid safetensors shard name: {name!r}")
        shards.setdefault(name, set()).add(key)
    total_size, tensors = 0, {}
    for name, expected_keys in shards.items():
        found, payload_size = _check_tensor_file(root / name)
        if set(found) != expected_keys:
            raise ValueError(f"Shard tensor names differ from index: {root / name}")
        total_size += payload_size
        tensors.update(found)
    declared_size = index.get("metadata", {}).get("total_size")
    if type(declared_size) is not int or total_size != declared_size:
        raise ValueError(f"Shard sizes differ from index total_size: {root / index_name}")
    return tensors, total_size


def check_qwen_snapshot(root, *, num_steps):
    """Accept downloaded pipelines or verified full student exports, never LoRAs."""
    from lightx2v_rl.models.qwen_image.rollout import validate_checkpoint_schedule

    root = Path(root)
    if (root / "download-manifest.json").is_file():
        report = check_snapshot(root)
    else:
        manifest = json.loads((root / "export_manifest.json").read_text())
        iteration = manifest.get("iteration")
        checkpoint = re.search(r"(?:^|/)checkpoint-(\d+)/?$", str(manifest.get("source_checkpoint", "")))
        if (
            manifest.get("role") != "student"
            or manifest.get("format") != "diffusers safetensors"
            or type(iteration) is not int
            or iteration < 0
            or checkpoint is None
            or int(checkpoint[1]) != iteration
        ):
            raise ValueError(f"Invalid student export format or iteration: {root}")
        for filename in (
            "transformer/config.json",
            "text_encoder/config.json",
            "vae/config.json",
            "scheduler/scheduler_config.json",
            "processor/preprocessor_config.json",
            "processor/tokenizer_config.json",
            "processor/tokenizer.json",
            "processor/chat_template.jinja",
        ):
            path = root / filename
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing exported pipeline component: {path}")
            if path.suffix == ".json" and not isinstance(json.loads(path.read_text()), dict):
                raise ValueError(f"Invalid exported component configuration: {path}")
        scheduler = json.loads((root / "scheduler/scheduler_config.json").read_text())
        if scheduler.get("use_dynamic_shifting", False) or scheduler.get("shift", 1.0) != 1.0 or scheduler.get("shift_terminal") is not None or scheduler.get("stochastic_sampling", False):
            raise ValueError(f"Export scheduler must preserve its saved sigma grid with deterministic Euler steps: {root}")
        tensors, total_size = _check_sharded_component(root / "transformer", "diffusion_pytorch_model.safetensors.index.json")
        if manifest.get("transformer_tensor_count") != len(tensors) or manifest.get("transformer_total_size") != total_size:
            raise ValueError(f"Transformer tensor count/size differs from export manifest: {root}")
        declared_dtype = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}.get(manifest.get("dtype"))
        if declared_dtype is None or any(value["dtype"] != declared_dtype for value in tensors.values()):
            raise ValueError(f"Transformer dtype differs from export manifest: {root}")
        if "sha256_bf16_tensors" in manifest and set(manifest["sha256_bf16_tensors"]) != set(tensors):
            raise ValueError(f"Transformer tensor names differ from export manifest: {root}")
        _check_sharded_component(root / "text_encoder", "model.safetensors.index.json")
        _check_tensor_file(root / "vae/diffusion_pytorch_model.safetensors")
        if manifest.get("terminal_sigma") != 0 or manifest.get("guidance_scale") != 1:
            raise ValueError(f"Few-step export requires terminal sigma 0 and guidance 1: {root}")
        report = {"source": "local_student_export", "iteration": iteration, "tensor_count": len(tensors), "tensor_bytes": total_size}
    pipeline = json.loads((root / "model_index.json").read_text())
    if pipeline.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError(f"Expected an exported QwenImage21Pipeline: {root}")
    sigmas = validate_checkpoint_schedule(pipeline.get("sample_sigmas"), num_steps=num_steps)
    if report.get("source") == "local_student_export":
        if manifest.get("sigmas") != sigmas:
            raise ValueError(f"Saved sigma grid differs from export manifest: {root}")
        for component in ("transformer", "text_encoder", "vae", "processor", "scheduler"):
            specification = pipeline.get(component)
            if not isinstance(specification, list) or len(specification) != 2 or not all(isinstance(value, str) and value for value in specification):
                raise ValueError(f"Missing pipeline component registration: {component}")
    report["sample_sigmas"] = sigmas
    return report


def check_reward_services(settings):
    """Check every configured replica from the calling node."""
    report = {}
    for name, weight, suffix in (
        ("hps", max(settings.get("hps_weight", 0), settings.get("hps_weight_i2i", 0)), "/health"),
        ("unifiedreward", settings.get("unifiedreward_weight", 0), "/v1/models"),
    ):
        if weight <= 0:
            continue
        services = {}
        for endpoint in normalize_endpoints(settings[name + "_endpoint"], name=name + "_endpoint"):
            with urlopen(endpoint.rstrip("/") + suffix, timeout=10) as response:
                health = json.load(response)
            if name == "hps":
                ready = health.get("status") == "ready" and "hpsv3_general" in health.get("metrics", [])
            else:
                ready = any(model.get("id") == "UnifiedReward" for model in health.get("data", []))
            if not ready:
                raise ValueError(f"Expected {name} model is not ready at {endpoint}.")
            services[endpoint] = health
        report[name + "_services"] = services
    return report


def check(config_name, *, skip_cuda=False, skip_services=False, overrides=None):
    from lightx2v_rl.data.opengpt4o import AlternatingTaskSampler, OpenGPT4oDataset
    from lightx2v_rl.rewards.assets import resolve_reward_assets

    overrides = overrides or []
    if any("=" not in item or item.startswith(("+", "~")) for item in overrides):
        raise ValueError("Use plain key=value preflight overrides.")
    config = resolve_runtime_paths(OmegaConf.merge(load_recipe(config_name), OmegaConf.from_dotlist(overrides)))
    validate_recipe(config, require_inputs=True)
    if config.actor_rollout_ref.model.architecture != "QwenImage21Pipeline":
        raise ValueError("This preflight is for the QwenImage21 recipes.")
    pipeline = config.actor_rollout_ref.rollout.pipeline
    report = {"model": check_qwen_snapshot(config.actor_rollout_ref.model.path, num_steps=pipeline.num_inference_steps)}
    settings = config.reward.custom_reward_function.reward_kwargs
    if settings.get("unifiedreward_weight", 0) > 0:
        report["unifiedreward"] = check_snapshot(repository_path(os.environ.get("LIGHTX2V_UNIFIEDREWARD_MODEL") or "lightx2v_rl/models/rewards/unifiedreward"))
    if max(settings.get("hps_weight", 0), settings.get("hps_weight_i2i", 0)) > 0:
        resolve_reward_assets(models=("hpsv3",))
    train = OpenGPT4oDataset(config.data.train_files, config=config.data)
    validation = OpenGPT4oDataset(config.data.val_files, config=config.data)
    AlternatingTaskSampler(train, data_config=config.data)
    if config.trainer.get("validation_preview_file"):
        from lightx2v_rl.training.validation_media import TaskValidationMedia

        report["validation_preview"] = TaskValidationMedia(config, 0).validate_dataset(validation)
    report["data"] = {}
    reference_capacity = int(pipeline.get("max_reference_images", 1))
    max_source_count = 1
    for name, dataset in (("train", train), ("validation", validation)):
        counts = {"t2i": 0, "i2i": 0, "i2i_single": 0, "i2i_multi": 0, "max_source_images": 0}
        for index in range(len(dataset)):
            row = dataset[index]
            counts[row["task_type"]] += 1
            source_count = len(row["source_images"])
            if source_count:
                counts["i2i_single" if source_count == 1 else "i2i_multi"] += 1
            counts["max_source_images"] = max(counts["max_source_images"], source_count)
            max_source_count = max(max_source_count, source_count)
            if name == "train" and source_count > reference_capacity:
                raise ValueError(f"Training sample needs {source_count} source images but replay capacity is {reference_capacity}; increase actor_rollout_ref.rollout.pipeline.max_reference_images.")
            for filename in row["source_images"]:
                if not Path(filename).is_file():
                    raise ValueError(f"Missing reference image: {filename}")
        if not counts["t2i"] or not counts["i2i"]:
            raise ValueError(f"Both tasks must be present in {name}.")
        report["data"][name] = counts
    from diffusers import QwenImage21Pipeline  # noqa: F401

    if not skip_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable on this host.")
        report["cuda_devices"] = torch.cuda.device_count()
    if not skip_services:
        report.update(check_reward_services(settings))
        report.update(check_reward_scoring(settings, width=pipeline.width, height=pipeline.height, i2i_source_count=max_source_count))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="qwen_image21_turbo_geo")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-services", action="store_true")
    parser.add_argument("overrides", nargs="*", help="The same plain key=value overrides used for training")
    args = parser.parse_args(argv)
    try:
        report = check(args.config, skip_cuda=args.skip_cuda, skip_services=args.skip_services, overrides=args.overrides)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Preflight failed: {error}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
