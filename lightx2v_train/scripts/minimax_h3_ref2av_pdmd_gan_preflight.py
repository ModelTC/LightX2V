#!/usr/bin/env python3
"""Check Ref PDMD weights, offline DeCFG bases and paired self-generated data."""

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def leaf(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_recipe(config):
    model, training, data = config["model"], config["training"], config["data"]["train"]
    dmd, matching = training["dmd"], model["capabilities"]["distribution_matching"]
    require(model["name"] == "minimax_h3_ref2av" and training["method"] == "dmd", "Use the Ref2AV DMD model")
    require(dmd["official_pdmd"] and matching["official_pdmd"] and matching["projected_dmd"], "Use official projected DMD")
    require((dmd["num_inference_steps"], dmd["fake_update_ratio"], dmd["update_order"]) == (8, 5, "fake_first"), "Expected eight steps and fake5 -> student1")
    require((dmd["official_rollout_video_shift"], dmd["official_rollout_audio_shift"]) == (12, 3), "Rollout shifts must be 12/3")
    require((matching["video_flow_shift"], matching["audio_flow_shift"]) == (12, 3), "Inference shifts must be 12/3")
    require(matching["geometry_from_metadata"] and matching["allowed_resolutions"] == [] and dmd["generation_shapes"] is None, "Keep every cached native target geometry")
    require(not config["scheduler"]["time_shift_settings"]["do_time_shift"], "Do not apply physical sigma shifts twice")
    require(dmd["gan"]["enabled"] and int(training["max_train_iters"]) >= 6, "Enable GAN and at least one complete fake5/student1 cycle")
    for key in ("generator_weight", "discriminator_weight"):
        value = float(dmd["gan"][key])
        require(math.isfinite(value) and value > 0, f"GAN {key} must be positive")
    require(0 < float(dmd["gan"]["sigma_max"]) < 1, "GAN sigma_max must be in (0,1)")
    student, fake = training["student"], training["fake"]
    require(student["train_type"] == "lora" and fake["train_type"] == "full" and not student["ema"]["enabled"], "Use live LoRA student and full fake")
    require(student.get("checkpoint_path") and student["checkpoint_path"] == fake.get("checkpoint_path"), "Warm-start student AND fake from the same PDMD checkpoint")
    require(student["checkpoint_strict"] and fake["checkpoint_strict"], "Warm-start both roles strictly")
    adapter = leaf("scripts/minimax_h3_pdmd_from_turbo_preflight.py", "_ref_gan_lora")
    lora = student["lora"]
    require(
        (lora["rank"], lora["alpha"], lora["param_dtype"]) == (128, 128, "fp32")
        and lora["rank_pattern"] == lora["alpha_pattern"] == adapter.PATTERNS
        and set(lora["target_modules"]) == adapter.TARGETS,
        "Keep regular128/128 and block/output AdaLN16/16",
    )
    require(float(student["optimizer"]["learning_rate"]) == 2e-5 and float(fake["optimizer"]["learning_rate"]) == 4e-6, "Preserve student/fake LR 2e-5/4e-6")
    require((training["save_every_iters"], training["save_total_limit"]) == (100, 2), "Save every100 role updates and retain2")
    require(training["gradient_accumulation_iters"] == data["batch_size"] == 1, "Use batch1/rank and accumulation1")
    distributed = config["distributed"]
    require(
        distributed["fsdp2"]["enabled"] and distributed["fsdp2"]["size"] == 32 and distributed["fsdp2"]["stream_load_pretrained"] and distributed["sequence_parallel"] == {"enabled": False, "size": 1},
        "Use streamed FSDP32 and SP1",
    )
    require(
        model["running_dtype"] == model["transformer_param_dtype"] == model["teacher"]["transformer_param_dtype"] == distributed["fsdp2"]["mixed_precision"]["param_dtype"] == "bf16"
        and model["fake"]["transformer_param_dtype"] == "fp32",
        "Keep BF16 student/teacher/compute and FP32 fake master weights",
    )
    require(
        data["name"] == "minimax_h3_ref_cache_dataset" and Path(data["data_path"]).resolve() == Path(dmd["gan"]["real_cache"]).resolve(),
        "Use the same self-generated Ref manifest for generation and GAN positives",
    )
    sampler = data["reference_cost_sampler"]
    require(
        sampler["require_image_only"]
        and not sampler["balance_image_counts"]
        and not sampler["balance_orientation"]
        and not sampler["strict_full_epoch"]
        and sampler["remainder_policy"] == "rotating_drop",
        "Keep image-only Ref data without count/orientation oversampling",
    )
    source = Path(student["checkpoint_path"]).expanduser().resolve()
    output = Path(training["output_dir"]).expanduser().resolve()
    require(output != source.parent and not output.is_relative_to(source) and not source.is_relative_to(output), "Select a new output directory separate from the source run")
    return source, adapter


def validate_models(config):
    model = config["model"]
    merged = leaf("lightx2v_train/model_zoo/minimax_h3/premerged_decfg.py", "_ref_gan_decfg")
    original = Path(model["pretrained_model_name_or_path"]).resolve()
    require(merged.validate_premerged_score_base(original, component_name="transformer_ref") is None, "Student must use the original Ref base without DeCFG")
    require(model["training_adapter"] == {"enabled": False}, "Disable online DeCFG on the student")
    require(model["fake"]["pretrained_model_name_or_path"] == model["teacher"]["pretrained_model_name_or_path"], "Use the same premerged score base for fake and teacher")
    for role in ("teacher", "fake"):
        override = model[role]
        require(override["training_adapter"] == {"enabled": False}, f"Disable online DeCFG on premerged {role}")
        record = merged.validate_premerged_score_base(override["pretrained_model_name_or_path"], component_name="transformer_ref", model_name=model["name"], required=True)
        require(Path(record["manifest"]["source_model_path"]).resolve() == original, "Score DeCFG base must come from the original student base")
    for component in ("transformer_ref", "text_encoder", "vae", "audio_vae"):
        require((original / component / "config.json").is_file(), f"Missing original model component: {original / component}")


def validate_source(config, source, adapter):
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    iteration = int((source / "official_pdmd.complete").read_text().strip())
    require(source.name == f"checkpoint-{iteration:09d}", "Source completion marker differs from its checkpoint directory")
    weights, count = adapter.validate_source(source, config["training"]["student"]["lora"])
    student_api = leaf("lightx2v_train/trainers/dmd/student_warmstart.py", "_ref_gan_student")
    student_api.validate_student_adapter_preset(weights, expected_preset={"enabled": False}, model_name="minimax_h3_ref2av")
    for rank in range(32):
        require((source / f"official_pdmd.rank{rank:05d}.pt").is_file(), f"Source checkpoint is missing rank {rank}")
    state = torch.load(source / "official_pdmd.rank00000.pt", map_location="cpu", weights_only=False)
    require((state["iteration"], state["world_size"], state["rank"]) == (iteration, 32, 0), "Source must be a complete 32-rank PDMD checkpoint")
    old = state["recipe"]
    require(old["model"]["name"] == config["model"]["name"], "Source checkpoint must be Ref2AV, not FL2AV")
    require(old["training"]["student"]["lora"] == config["training"]["student"]["lora"] and old["training"]["fake"]["train_type"] == "full", "Source LoRA architecture/full fake differ")
    for role in (None, "teacher", "fake"):
        saved = old["model"] if role is None else old["model"][role]
        current = config["model"] if role is None else config["model"][role]
        require(Path(saved["pretrained_model_name_or_path"]).resolve() == Path(current["pretrained_model_name_or_path"]).resolve(), f"Keep the source {role or 'student'} base identity")
        require(saved.get("training_adapter", {"enabled": False}) == current["training_adapter"] == {"enabled": False}, "Source must use offline DeCFG score bases and an unmodified student base")
    metadata = FileSystemReader(str(source / "dist_state")).read_metadata()
    require(any(key.startswith("fake_model.") for key in metadata.state_dict_metadata), "Source dist_state has no trained fake_model weights")
    return {"checkpoint": str(source), "iteration": iteration, "lora_tensors": count, "weights_only": ["student", "fake"]}


def validate_data(config):
    data_api = leaf("lightx2v_train/trainers/dmd/dmd2_gan_data.py", "_ref_gan_pairs")
    manifest, rows, fingerprint = data_api.load_gan_real_records(config["training"]["dmd"]["gan"]["real_cache"], task="ref2av", geometry=None)
    raw = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    require(len(rows) == len(raw), "The manifest must contain only eligible 124-frame image-only Ref AV pairs")
    require(len(rows) >= 32, "Self-generated Ref data needs at least32 rows for DP32")
    for row in rows:
        cost = row.get("packed_sequence_tokens_124")
        require(isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost > 0, "Ref data needs positive packed_sequence_tokens_124")
    receipt = leaf("scripts/minimax_h3_cm_pdmd_preflight.py", "_ref_gan_receipt")
    receipt._receipt(manifest, len(raw))
    geometry_counts = Counter(tuple(row[key] for key in ("target_num_frames", "target_height", "target_width")) for row in rows)
    return {"manifest": str(manifest), "rows": len(rows), "fingerprint": fingerprint, "geometry_rows": [{"geometry": key, "rows": value} for key, value in sorted(geometry_counts.items())]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args(argv)
    config = leaf("lightx2v_train/runtime/config.py", "_ref_gan_config").load_config(args.config)
    source, adapter = validate_recipe(config)
    validate_models(config)
    data = validate_data(config)
    resume_api = leaf("scripts/minimax_h3_pdmd_gan_preflight.py", "_ref_gan_resume")
    resume = resume_api.own_stage_resume(config, gan_enabled=True)
    source_info = validate_source(config, source, adapter) if resume is None else {"own_stage_resume": str(resume)}
    updates = int(config["training"]["max_train_iters"])
    print(
        json.dumps(
            {
                "task": "ref2av",
                "source": source_info,
                "data": data,
                "steps": 8,
                "shifts": [12, 3],
                "fake_per_student": 5,
                "role_updates": updates,
                "student_updates": updates // 6,
                "fake_updates": updates - updates // 6,
                "save_every": 100,
                "keep": 2,
                "student_uses_decfg": False,
                "output": config["training"]["output_dir"],
            },
            indent=2,
        ),
        flush=True,
    )
    if resume is None:
        print("[ref-gan] Student and trained fake weights only; fresh optimizers, GAN head and data cursors.", flush=True)


if __name__ == "__main__":
    main()
