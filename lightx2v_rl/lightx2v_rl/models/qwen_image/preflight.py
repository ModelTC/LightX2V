"""Check local Qwen assets, portable data and reward services before Ray starts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen

from omegaconf import OmegaConf

from lightx2v_rl.paths import repository_path, resolve_runtime_paths
from lightx2v_rl.rewards.endpoint_pool import normalize_endpoints
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


def check_reward_services(settings):
    """Check every configured replica from the calling node."""
    report = {}
    for name, weight, suffix in (
        ("hps", "hps_weight", "/health"),
        ("unifiedreward", "unifiedreward_weight", "/v1/models"),
    ):
        if settings.get(weight, 0) <= 0:
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
    from lightx2v_rl.data.opengpt4o import OpenGPT4oDataset
    from lightx2v_rl.rewards.assets import resolve_reward_assets

    overrides = overrides or []
    if any("=" not in item or item.startswith(("+", "~")) for item in overrides):
        raise ValueError("Use plain key=value preflight overrides.")
    config = resolve_runtime_paths(OmegaConf.merge(load_recipe(config_name), OmegaConf.from_dotlist(overrides)))
    validate_recipe(config, require_inputs=True)
    if config.actor_rollout_ref.model.architecture != "QwenImage21Pipeline":
        raise ValueError("This preflight is for the QwenImage21 recipes.")
    report = {"model": check_snapshot(config.actor_rollout_ref.model.path)}
    report["unifiedreward"] = check_snapshot(repository_path(
        os.environ.get("LIGHTX2V_UNIFIEDREWARD_MODEL") or "lightx2v_rl/models/rewards/unifiedreward"
    ))
    resolve_reward_assets()
    train = OpenGPT4oDataset(config.data.train_files)
    validation = OpenGPT4oDataset(config.data.val_files)
    report["data"] = {}
    for name, dataset in (("train", train), ("validation", validation)):
        counts = {"t2i": 0, "i2i": 0}
        for index in range(len(dataset)):
            row = dataset[index]
            counts[row["task_type"]] += 1
            for filename in row["source_images"]:
                if not Path(filename).is_file():
                    raise ValueError(f"Missing reference image: {filename}")
        if not all(counts.values()):
            raise ValueError(f"Both tasks must be present in {name}.")
        report["data"][name] = counts
    from diffusers import QwenImage21Pipeline  # noqa: F401
    if not skip_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable on this host.")
        report["cuda_devices"] = torch.cuda.device_count()
    if not skip_services:
        settings = config.reward.custom_reward_function.reward_kwargs
        report.update(check_reward_services(settings))
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
