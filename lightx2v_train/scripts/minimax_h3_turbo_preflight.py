"""Read-only, CPU-only launch checks for the H3 Turbo curriculum recipes."""

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path


def require(passed, description):
    if not passed:
        raise SystemExit(f"H3 Turbo launcher requires {description}.")


def load_config(path):
    source = Path(__file__).resolve().parents[1] / "lightx2v_train/runtime/config.py"
    spec = importlib.util.spec_from_file_location("_h3_turbo_config", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_config(str(path))


def load_turbo_config(config):
    # Import this leaf module directly: importing the trainers package can load
    # Torch and model registries, which metadata-only preflight must avoid.
    source = Path(__file__).resolve().parents[1] / "lightx2v_train/trainers/turbo/config.py"
    spec = importlib.util.spec_from_file_location("_h3_turbo_schema", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        return module.TurboConfig.from_mapping(config)
    except ValueError as error:
        raise SystemExit(f"Invalid H3 Turbo configuration: {error}") from error


def check_payload_paths(row, manifest):
    for key in ("condition_path", "video_latent_path", "audio_latent_path"):
        value = row.get(key)
        require(isinstance(value, str) and bool(value.strip()), f"cached clean video/audio and conditioning: missing {key} in {manifest}")
        path = Path(value)
        if not path.is_absolute():
            path = manifest.parent / path
        require(path.is_file(), f"a readable cached {key}: {path}")


def check_ref_cache(data):
    sampler = data["reference_cost_sampler"]
    require(
        data["name"] == "minimax_h3_ref_cache_dataset"
        and sampler.get("require_image_only") is True
        and sampler.get("batch_mode") == "count_random"
        and sampler.get("image_counts") == list(range(1, 7))
        and sampler.get("balance_image_counts") is True
        and sampler.get("balance_orientation") is False
        and sampler.get("strict_full_epoch") is False
        and sampler.get("remainder_policy") == "rotating_drop",
        "the existing image-only Ref cache and balanced image-count sampling",
    )
    manifest = Path(data["data_path"])
    require(manifest.is_file(), f"a nonempty Ref cache manifest: {manifest}")
    receipt_path = manifest.with_suffix(".complete.json")
    require(receipt_path.is_file(), f"a completed cache merge receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    counts = [receipt.get(name) for name in ("completed_count", "failed_count", "input_total_rows")]
    require(
        all(type(value) is int and value >= 0 for value in counts) and counts[0] > 0 and counts[0] + counts[1] == counts[2],
        "valid completed/failed/source accounting in the merge receipt",
    )
    require(bool(receipt.get("preprocess_fingerprint")), "a preprocessing fingerprint in the merge receipt")
    digest, rows = hashlib.sha256(), 0
    with manifest.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            height, width = int(row["target_height"]), int(row["target_width"])
            frames = int(row.get("target_num_frames", row.get("num_frames", 0)))
            image_count = int(row.get("reference_image_count", row.get("ref_image_count", 0)))
            require(
                frames == 124
                and [height, width] in [[768, 1344], [1344, 768]]
                and image_count in sampler["image_counts"]
                and int(row.get("reference_video_count", -1)) == 0
                and int(row.get("reference_audio_count", -1)) == 0
                and int(row.get(sampler["cost_key"], 0)) > 0,
                f"124-frame 768p image-only Ref rows with positive compute cost: {manifest}",
            )
            check_payload_paths(row, manifest)
            rows += 1
    require(
        digest.hexdigest() == receipt.get("manifest_sha256") and rows == counts[0],
        "a cache manifest matching its complete receipt",
    )
    print(f"Verified Ref cache: completed={counts[0]}, failed={counts[1]}, input_rows={counts[2]}; clean video/audio files checked for every row")
    print("Sampling: one reference image count per global batch; balanced counts 1..6, natural orientations, rotating subsets.")


def check_fl2av_cache(data):
    tasks = ["t2av", "i2av", "fl2av"]
    require(
        data["name"] == "minimax_h3_cache_dataset"
        and data.get("tasks") == tasks
        and data.get("task_order") == tasks
        and data.get("buckets") == ["landscape", "portrait"]
        and data.get("route_mode") == "stratified"
        and data.get("balance_orientation") is False
        and data.get("balance_route_loss") is True
        and data.get("require_all_routes_per_optimizer_step") is True,
        "the T2AV/I2AV/FL2AV three-task cache with natural orientations and balanced route losses",
    )
    root = Path(data["data_path"])
    manifests = {}
    for task in tasks:
        candidates = [root / task / "metadata.jsonl"]
        for bucket in data["buckets"]:
            directory = root / task / bucket
            candidates.append(directory / "metadata.jsonl")
            candidates.extend(sorted(directory.glob("*/metadata.jsonl")))
        for path in candidates:
            if path.is_file():
                manifests[path.resolve()] = task
    counts = Counter()
    for manifest, task in manifests.items():
        with manifest.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                height, width = int(row["target_height"]), int(row["target_width"])
                frames = int(row.get("target_num_frames", row.get("num_frames", 0)))
                require(
                    row.get("task") == task and frames == 124 and [height, width] in [[768, 1344], [1344, 768]],
                    f"{task.upper()} 124-frame 768p cache rows: {manifest}",
                )
                check_payload_paths(row, manifest)
                counts[(task, "landscape" if width > height else "portrait")] += 1
    summary = {task: {bucket: counts[(task, bucket)] for bucket in data["buckets"]} for task in tasks}
    require(
        all(counts[(task, bucket)] > 0 for task in tasks for bucket in data["buckets"]) and all(sum(summary[task].values()) >= 32 for task in tasks),
        f"both orientations and at least 32 cached rows per task: {summary}",
    )
    print(f"Verified three-task cache: {json.dumps(summary, sort_keys=True)}, total={sum(counts.values())}")
    print("Task rows rotate 11/11/10 across fresh batches; route loss weights balance T2AV/I2AV/FL2AV.")


def validate(task, config):
    require(task in {"ref2av", "fl2av"}, "task ref2av or fl2av")
    model, training, data = config["model"], config["training"], config["data"]["train"]
    turbo = load_turbo_config(config)
    require(turbo.num_inference_steps == 8, "the 8-step Turbo recipe")
    stages = [turbo.dcm_steps, turbo.scm_steps, turbo.joint_steps]
    distributed = config["distributed"]
    require(
        distributed["fsdp2"]["enabled"] is True and distributed["fsdp2"]["size"] == 32 and distributed["sequence_parallel"] == {"enabled": False, "size": 1},
        "FSDP32 without sequence parallelism",
    )
    require(data["batch_size"] == 1 and training["gradient_accumulation_iters"] == 1, "global batch32 (per-rank batch1, accumulation1)")
    matching = model["capabilities"]["distribution_matching"]
    expected_shifts = (12.0, 3.0) if task == "ref2av" else (6.0, 3.0)
    require((matching["video_flow_shift"], matching["audio_flow_shift"]) == expected_shifts, f"video/audio shifts {expected_shifts[0]:g}/{expected_shifts[1]:g}")
    require((turbo.video_shift, turbo.audio_shift) == expected_shifts, "matching Turbo training and inference shifts")
    require(matching.get("geometry_from_metadata") is True, "metadata-controlled generation geometry")
    lora = training["student"]["lora"]
    patterns = {"adaln_proj.linear": 16, "norm_out.linear": 16}
    targets = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", *patterns]
    require(
        training["student"]["train_type"] == "lora"
        and (lora.get("rank"), lora.get("alpha"), lora.get("param_dtype")) == (128, 128, "fp32")
        and lora.get("target_modules") == targets
        and lora.get("rank_pattern") == patterns
        and lora.get("alpha_pattern") == patterns,
        "FP32 LoRA rank/alpha128 plus block and output AdaLN rank/alpha16",
    )
    require(training["fake"]["train_type"] == "full" and model["fake"]["transformer_param_dtype"] == "fp32", "a full fake model with FP32 master parameters")
    require(
        model["transformer_param_dtype"] == "bf16"
        and model["teacher"]["transformer_param_dtype"] == "bf16"
        and model["running_dtype"] == "bf16"
        and model["use_autocast"] is False
        and distributed["fsdp2"]["mixed_precision"]["param_dtype"] == "bf16"
        and distributed["fsdp2"]["mixed_precision"]["reduce_dtype"] == "fp32",
        "the existing BF16 base/teacher/compute and FP32 gradient reduction",
    )
    model_root = Path(model["pretrained_model_name_or_path"])
    component = "transformer_ref" if task == "ref2av" else "transformer"
    require(model["name"] == ("minimax_h3_ref2av" if task == "ref2av" else "minimax_h3_t2av"), f"the {task} model wrapper")
    require((model_root / component / "config.json").is_file(), f"H3_MODEL_PATH/{component}/config.json")
    prefix = "H3_REF2AV" if task == "ref2av" else "H3_FL2AV"
    for variable, path in (("H3_MODEL_PATH", model_root), (prefix + "_CACHE", Path(data["data_path"])), (prefix + "_TURBO_OUTPUT", Path(training["output_dir"]))):
        require(path.resolve() == Path(os.environ[variable]).resolve(), f"the config path to match {variable}: {path}")
    (check_ref_cache if task == "ref2av" else check_fl2av_cache)(data)
    print(f"H3 Turbo {task}: 4 x 8 GPUs, steps=8, global_batch=32")
    print(f"Student updates: dCM={stages[0]}, sCM={stages[1]}, joint={stages[2]}, total={sum(stages)}; joint fake_update_ratio={turbo.fake_update_ratio}")
    print("max_train_iters counts STUDENT optimizer updates; joint fake updates are counted separately.")
    print(f"Shifts: video={expected_shifts[0]}, audio={expected_shifts[1]}; direct conditional teacher, no CFG.")
    print("LoRA: regular rank=128 alpha=128; block/output AdaLN rank=16 alpha=16; all adapters FP32.")
    print(f"Student base={component}/ BF16; teacher BF16; fake full FP32; FSDP BF16 compute / FP32 reduction.")
    print(f"Checkpoint: every {training['save_every_iters']} student updates, keep {training['save_total_limit']}.")
    print(f"config={config.get('config_path')}\ndata={data['data_path']}\noutput={training['output_dir']}")
    print("Use a NEW output directory; previous PDMD checkpoints do not contain the Turbo curriculum state.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: minimax_h3_turbo_preflight.py {ref2av|fl2av} CONFIG")
    validate(sys.argv[1], load_config(Path(sys.argv[2])))
