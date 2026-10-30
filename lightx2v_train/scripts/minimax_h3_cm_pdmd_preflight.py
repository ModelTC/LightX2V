"""Read-only CPU checks for dCM-1500 -> official PDMD with the same DeCFG preset."""

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def _leaf(relative, name):
    spec = importlib.util.spec_from_file_location(name, TRAIN_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config(path):
    return _leaf("lightx2v_train/runtime/config.py", "_h3_staged_config").load_config(str(path))


def _receipt(manifest, rows):
    path = manifest.with_suffix(".complete.json")
    require(path.is_file(), f"New cache is not fully merged: missing {path}")
    record = json.loads(path.read_text())
    require(record.get("manifest_sha256") == hashlib.sha256(manifest.read_bytes()).hexdigest(), f"Manifest differs from its completion receipt: {manifest}")
    require(
        record.get("completed_count") == rows and record.get("failed_count") == 0 and record.get("input_total_rows") == rows and bool(record.get("preprocess_fingerprint")),
        f"New cache completion accounting differs: {manifest}",
    )


def _condition_path(row, manifest):
    value = row.get("condition_path")
    require(isinstance(value, str) and value.strip(), f"Missing condition_path in {manifest}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def _identity(row, manifest):
    return (
        str(row["source_id"]),
        row["cache_fingerprint"],
        str(_condition_path(row, manifest)),
        int(row["target_height"]),
        int(row["target_width"]),
        int(row.get("target_num_frames", row.get("num_frames", 0))),
    )


def _require_image_only_ref(row, manifest, phase):
    images = row.get("reference_image_count", row.get("ref_image_count", 0))
    videos, audios = row.get("reference_video_count", 0), row.get("reference_audio_count", 0)
    message = f"{phase} Ref cache must contain image-only references, with no video/audio references: {manifest}"
    require(all(type(count) is int for count in (images, videos, audios)) and images > 0 and videos == audios == 0, message)
    references = row.get("references")
    if references is not None:
        require(
            isinstance(references, list)
            and bool(references)
            and all(isinstance(reference, dict) and str(reference.get("kind", reference.get("type", ""))).lower() == "image" for reference in references),
            message,
        )


def _condition_source(source, task, phase):
    helper = _leaf("scripts/minimax_h3_curriculum_preflight.py", "_h3_staged_curriculum_paths")
    manifests = helper.manifest_paths(source["data_path"])
    counts, identities, paths, total = Counter(), set(), set(), 0
    for manifest in manifests:
        rows = 0
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_task = str(row.get("task", "")).lower()
            expected = {"t2av"} if task == "fl2av" and phase == "new" else {"t2av", "i2av", "fl2av"} if task == "fl2av" else {"ref2av", "ref2va"}
            require(row_task in expected, f"Unexpected {phase} task={row_task!r} in {manifest}")
            height, width = int(row["target_height"]), int(row["target_width"])
            frames = int(row.get("target_num_frames", row.get("num_frames", 0)))
            require(frames == 124 and min(height, width) > 0 and height % 32 == 0 and width % 32 == 0, f"Expected native 32-aligned 124-frame cache geometry: {manifest}")
            condition = _condition_path(row, manifest)
            require(str(condition) not in paths, f"Duplicate condition cache path: {condition}")
            paths.add(str(condition))
            # Every new condition is checked by the clean-AV loader below.
            # For old manifests check one payload; runtime validates every row.
            if rows == 0:
                require(condition.is_file(), f"Missing cached condition: {condition}")
            if task == "ref2av":
                images = int(row.get("reference_image_count", row.get("ref_image_count", 0)))
                videos, audios = int(row.get("reference_video_count", 0)), int(row.get("reference_audio_count", 0))
                cost = row.get(source.get("cost_key", "packed_sequence_tokens_124"))
                require(min(images, videos, audios) >= 0 and images + videos > 0, f"Missing visual Ref condition: {manifest}")
                require(isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost > 0, f"Ref cache needs a positive packed sequence cost: {manifest}")
                if phase == "old":
                    require(images in source.get("image_counts", [1, 2, 3, 4, 5, 6]) and videos == audios == 0, f"Old Ref cache must use image-only counts 1..6: {manifest}")
                else:
                    _require_image_only_ref(row, manifest, "NEW")
            if phase == "new":
                identities.add(_identity(row, manifest))
            counts[row_task] += 1
            total += 1
            rows += 1
        require(rows > 0, f"Empty {phase} cache manifest: {manifest}")
        if phase == "new":
            _receipt(manifest, rows)
    require(total > 0, f"Empty {phase} cache")
    if task == "fl2av" and phase == "old":
        require(set(counts) == {"t2av", "i2av", "fl2av"}, "Old FL2AV data must contain all three tasks")
    return {"rows": total, "tasks": dict(counts), "manifests": [str(path) for path in manifests]}, identities, paths


def validate(task, cm, pdmd, *, check_files=True, check_student_init=False):
    require(task in {"fl2av", "ref2av"}, "Task must be fl2av or ref2av")
    expected_model = "minimax_h3_t2av" if task == "fl2av" else "minimax_h3_ref2av"
    turbo = _leaf("lightx2v_train/trainers/turbo/config.py", "_h3_staged_turbo_config").TurboConfig.from_mapping(cm)
    require(
        (turbo.dcm_steps, turbo.scm_steps, turbo.joint_steps, turbo.total_steps, turbo.num_inference_steps) == (1500, 0, 0, 1500, 8),
        "CM must contain exactly 1500 dCM updates, no sCM/joint stage, and eight inference steps",
    )
    require((turbo.video_shift, turbo.audio_shift) == (12.0, 3.0), "CM shifts must be video12/audio3")
    patterns = {"adaln_proj.linear": 16, "norm_out.linear": 16}
    targets = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", *patterns]
    for config in (cm, pdmd):
        model, training, distributed = config["model"], config["training"], config["distributed"]
        require(model["name"] == expected_model, f"Wrong model for task={task}")
        require(training["gradient_accumulation_iters"] == 1 and config["data"]["train"]["batch_size"] == 1, "Both stages use batch1/rank, accumulation1")
        require(
            distributed["fsdp2"]["enabled"] is True
            and distributed["fsdp2"]["size"] == 32
            and distributed["fsdp2"].get("stream_load_pretrained") is True
            and distributed["sequence_parallel"] == {"enabled": False, "size": 1},
            "Both stages require streamed FSDP32 and SP1",
        )
        matching = model["capabilities"]["distribution_matching"]
        require(
            (matching["video_flow_shift"], matching["audio_flow_shift"]) == (12.0, 3.0) and matching["geometry_from_metadata"] is True and matching["allowed_resolutions"] == [],
            "Keep video12/audio3 and every native cached resolution",
        )
        require(config["scheduler"]["time_shift_settings"]["do_time_shift"] is False, "Physical training sigmas must not be shifted twice")
        lora = training["student"]["lora"]
        require(
            training["student"]["train_type"] == "lora"
            and (lora["rank"], lora["alpha"], lora["param_dtype"]) == (128, 128, "fp32")
            and lora["rank_pattern"] == lora["alpha_pattern"] == patterns
            and lora["target_modules"] == targets,
            "Use FP32 regular128/128 plus block/output AdaLN16/16 LoRA",
        )
        require(training["fake"]["train_type"] == "full" and model["fake"]["transformer_param_dtype"] == "fp32", "Keep the full FP32-master fake model")
        precision = distributed["fsdp2"]["mixed_precision"]
        require(
            model["transformer_param_dtype"] == model["teacher"]["transformer_param_dtype"] == model["running_dtype"] == precision["param_dtype"] == "bf16"
            and precision["reduce_dtype"] == "fp32"
            and model["use_autocast"] is False,
            "Keep BF16 base/teacher/compute, FP32 LoRA/fake masters and gradient reduction",
        )
        require(training["teacher"]["guidance_scale"] == 1.0, "H3 uses a direct conditional score teacher with the DeCFG preset")
        adapter = model["training_adapter"]
        require(
            adapter["enabled"] is True and adapter["roles"] == ["fake", "teacher"] and adapter["qkv_layout"] == "comfy" and math.isfinite(float(adapter["scale"])) and float(adapter["scale"]) > 0,
            "Enable the same positive-scale DeCFG preset on fake/teacher scores only; student rollout and inference must exclude it",
        )
        known_presets = {"model_for_comfy_dit.safetensors": "fl2av", "model_ref2va_for_comfy_dit.safetensors": "ref2av"}
        preset_name = Path(adapter["path"]).name
        require(preset_name not in {"model.safetensors", "model_ref2va.safetensors"} and known_presets.get(preset_name, task) == task, "Select the matching task-specific Comfy DeCFG preset")
    require(cm["model"]["training_adapter"] == pdmd["model"]["training_adapter"], "The DeCFG preset identity must stay the same across stages")
    require(cm["model"]["pretrained_model_name_or_path"] == pdmd["model"]["pretrained_model_name_or_path"], "The H3 base must stay the same across stages")
    require(cm["training"]["student"]["lora"] == pdmd["training"]["student"]["lora"], "The LoRA architecture must stay the same across stages")
    cm_data, pdmd_data = cm["data"]["train"], pdmd["data"]["train"]
    require(
        cm_data["name"] == "minimax_h3_turbo_cm_dataset"
        and cm_data["tasks"] == (["t2av"] if task == "fl2av" else ["ref2av"])
        and cm_data.get("native_geometry") is True
        and cm_data["shuffle"] is True
        and cm_data["drop_last"] is False
        and cm_data["defer_latent_loading"] is True
        and "cm" not in cm["data"],
        "dCM must use only this task's complete NEW clean-AV cache",
    )
    require(cm_data.get("max_samples") is None and cm_data.get("dataset_repeat", 1) == 1, "dCM must not truncate or artificially repeat the NEW clean-AV cache")
    require(
        pdmd_data["name"] == "minimax_h3_curriculum_cache_dataset" and pdmd_data["kind"] == task and pdmd_data["drop_last"] is False and set(pdmd_data["sources"]) == {"new", "old"},
        "PDMD must use complete NEW -> OLD -> repeat passes",
    )
    require(
        pdmd_data.get("max_samples") is None
        and pdmd_data.get("dataset_repeat", 1) == 1
        and all(source.get("max_samples") is None and source.get("dataset_repeat", 1) == 1 for source in pdmd_data["sources"].values()),
        "PDMD must not truncate either source or override curriculum-owned repetition",
    )
    options = pdmd["training"]["dmd"]
    require(
        pdmd["training"]["method"] == "dmd"
        and options["official_pdmd"] is True
        and pdmd["model"]["capabilities"]["distribution_matching"]["official_pdmd"] is True
        and pdmd["model"]["capabilities"]["distribution_matching"]["projected_dmd"] is True,
        "Use the released official PDMD loop and objective",
    )
    require(
        (options["num_inference_steps"], options["fake_update_ratio"], options["update_order"], options["model_mode"], options["official_rollout_video_shift"], options["official_rollout_audio_shift"])
        == (8, 5, "fake_first", "eval", 12.0, 3.0),
        "PDMD must use eight steps, fake5->student1, eval, video12/audio3",
    )
    require(pdmd["training"]["max_train_iters"] == 1500000 and pdmd["training"]["save_every_iters"] == 6000, "PDMD counts 1500000 role updates and saves every 6000")
    require(
        cm["training"]["student"]["ema"]["enabled"] is True and cm["training"]["student"]["ema"]["decay"] == 0.999 and pdmd["training"]["student"]["ema"]["enabled"] is False,
        "Export CM EMA with decay=0.999 into the live PDMD student; PDMD has no new EMA",
    )
    cm_output, pdmd_output = Path(cm["training"]["output_dir"]).resolve(), Path(pdmd["training"]["output_dir"]).resolve()
    require(cm_output != pdmd_output, "CM and PDMD must use separate outputs")
    student_init = cm_output / "checkpoint-000001500/student_ema"
    require(Path(pdmd["training"]["student"]["checkpoint_path"]).resolve() == student_init, "PDMD must load this task's dCM1500 EMA, not a stale checkpoint")
    if check_student_init:
        require((student_init / "pytorch_lora_weights.safetensors").is_file(), f"Missing dCM1500 EMA adapter: {student_init}")
    summary = {
        "task": task,
        "cm_total_steps": 1500,
        "cm_output": str(cm_output),
        "pdmd_output": str(pdmd_output),
        "student_init": str(student_init),
        "cm_rows": 0,
        "pdmd_new_rows": 0,
        "pdmd_old_rows": 0,
        "pdmd_total_iters": 1500000,
        "pdmd_student_updates": 250000,
        "pdmd_fake_updates": 1250000,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "cm_ema_decay": 0.999,
        "ref_new_image_only": task == "ref2av",
        "decfg_enabled": True,
        "decfg_roles": ["fake", "teacher"],
        "student_uses_decfg": False,
        "inference_requires_preset": False,
    }
    if check_files:
        component = "transformer" if task == "fl2av" else "transformer_ref"
        require((Path(cm["model"]["pretrained_model_name_or_path"]) / component / "config.json").is_file(), f"H3 root must contain {component}/config.json")
        preset = Path(cm["model"]["training_adapter"]["path"])
        require(preset.is_file(), f"Missing DeCFG preset: {preset}; run download_minimax_h3_training_adapter.sh first")
        paths = _leaf("lightx2v_train/data/minimax_h3_turbo_paths.py", "_h3_staged_cm_paths")
        manifest, rows, fingerprint = paths.load_cm_records(cm_data["data_path"], path_mapping=cm_data.get("path_mapping"), tasks=cm_data["tasks"], check_files=True, native_geometry=True)
        _receipt(manifest, len(rows))
        if task == "ref2av":
            for row in rows:
                _require_image_only_ref(row, manifest, "CM NEW")
        new, new_identities, new_paths = _condition_source(pdmd_data["sources"]["new"], task, "new")
        old, _, old_paths = _condition_source(pdmd_data["sources"]["old"], task, "old")
        require(
            {_identity(row, manifest) for row in rows} == new_identities and len(rows) == new["rows"], "CM and PDMD NEW sources must contain exactly the same complete rows and condition identities"
        )
        require(not new_paths.intersection(old_paths), "NEW and OLD sources must not repeat condition paths")
        summary.update(cm_rows=len(rows), cm_manifest=str(manifest), cm_fingerprint=fingerprint, pdmd_new_rows=new["rows"], pdmd_old_rows=old["rows"], pdmd_new=new, pdmd_old=old)
    print(f"H3 {task}: dCM1500 -> official PDMD1500000, 4 x 8 GPUs, 8 steps, shifts12/3, fake5->student1.")
    print(f"Data: dCM NEW full AV rows={summary['cm_rows']}; PDMD NEW={summary['pdmd_new_rows']} -> OLD={summary['pdmd_old_rows']} -> repeat; native geometry retained.")
    if task == "ref2av":
        print("Ref NEW data: image-only references for both dCM and PDMD; video/audio references excluded; full target AV latents retained.")
    print("CM student EMA decay=0.999.")
    print("PDMD budget: student_updates=250000, fake_updates=1250000; checkpoint every6000 role updates (1000 student updates).")
    print(f"DeCFG ON: {cm['model']['training_adapter']['path']}; fake/teacher scores only; student rollout and inference exclude DeCFG.")
    print(f"Warm start after dCM: {student_init}; fresh PDMD optimizers/fake/data cursor.")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: minimax_h3_cm_pdmd_preflight.py {fl2av|ref2av} CM_CONFIG PDMD_CONFIG")
    try:
        validate(sys.argv[1], load_config(sys.argv[2]), load_config(sys.argv[3]))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"H3 dCM->PDMD preflight failed: {error}") from error
