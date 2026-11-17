#!/usr/bin/env python3
"""Validate CM1500 student-only transfer to base-teacher Ref6k PDMD + GAN."""

import json
import math
import sys
from collections import Counter
from pathlib import Path

from minimax_h3_ref2av_pdmd_gan_preflight import leaf, require
from run_minimax_h3_pdmd_from_cm_premerged import verify_cm_export


def validate_recipe(config):
    model, training, data = config["model"], config["training"], config["data"]["train"]
    dmd, matching = training["dmd"], model["capabilities"]["distribution_matching"]
    require(model["name"] == "minimax_h3_ref2av" and training["method"] == "dmd", "Use the Ref2AV DMD model")
    require(dmd["official_pdmd"] and matching["official_pdmd"] and matching["projected_dmd"], "Use official projected DMD")
    require(dmd["num_inference_steps"] in (4, 8), "Expected four or eight rollout steps")
    require((dmd["fake_update_ratio"], dmd["update_order"]) == (5, "fake_first"), "Expected fake5 -> student1")
    require((dmd["official_rollout_video_shift"], dmd["official_rollout_audio_shift"]) == (12, 3), "Rollout shifts must stay 12/3")
    require((matching["video_flow_shift"], matching["audio_flow_shift"]) == (12, 3), "Inference shifts must stay 12/3")
    require(matching["geometry_from_metadata"] and matching["allowed_resolutions"] == [] and dmd["generation_shapes"] is None, "Use cached target geometry")
    require(not config["scheduler"]["time_shift_settings"]["do_time_shift"], "Do not shift physical sigmas twice")
    require(dmd["score_sampling"] == {"type": "continuous_uniform", "min_sigma": 0.0, "max_sigma": 1.0}, "Keep official interval-uniform score sampling")
    require(int(training["max_train_iters"]) >= 6, "Include at least one fake5/student1 cycle")
    gan = dmd["gan"]
    require(gan["enabled"] and (gan["generator_weight"], gan["discriminator_weight"]) == (0.005, 0.01), "Use GAN G/D weights 0.005/0.01")
    require((gan["feature_block"], gan["head_hidden_dim"], gan["head_temporal_bins"]) == (49, 1024, 4), "Use block49 features and head1024/bins4")
    require(0 < float(gan["sigma_max"]) < 1, "GAN sigma_max must be in (0,1)")
    student, fake = training["student"], training["fake"]
    require(student["train_type"] == "lora" and fake["train_type"] == "full" and not student["ema"]["enabled"], "Use live LoRA student and full fake")
    require(student.get("checkpoint_path") and student["checkpoint_strict"], "Load the CM student EMA strictly")
    require(not fake.get("checkpoint_path"), "Fresh fake must initialize from the original base, not a checkpoint")
    require(not training["teacher"].get("checkpoint_path") and training["teacher"]["guidance_scale"] == 1, "Use frozen base teacher without CFG or checkpoint transfer")
    adapter = leaf("scripts/minimax_h3_pdmd_from_turbo_preflight.py", "_ref6k_lora")
    lora = student["lora"]
    require(
        (lora["rank"], lora["alpha"], lora["param_dtype"]) == (128, 128, "fp32")
        and lora["rank_pattern"] == lora["alpha_pattern"] == adapter.PATTERNS
        and set(lora["target_modules"]) == adapter.TARGETS,
        "Keep regular128/128 and block/output AdaLN16/16",
    )
    require(float(student["optimizer"]["learning_rate"]) == 5e-5 and float(fake["optimizer"]["learning_rate"]) == 4e-6, "Use student/fake LR 5e-5/4e-6")
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
        data["name"] == "minimax_h3_ref_cache_dataset" and Path(data["data_path"]).resolve() == Path(gan["real_cache"]).resolve(),
        "Use the unified Ref6k manifest for both generation and GAN positives",
    )
    sampler = data["reference_cost_sampler"]
    require(
        sampler["require_image_only"]
        and sampler["image_counts"] == [1, 2, 3, 4, 5, 6]
        and sampler["require_all_image_counts"]
        and not sampler["balance_image_counts"]
        and not sampler["balance_orientation"]
        and not sampler["strict_full_epoch"]
        and sampler["remainder_policy"] == "rotating_drop",
        "Cover image counts1..6 without count/orientation oversampling",
    )
    source = Path(student["checkpoint_path"]).expanduser().resolve()
    require(source.name == "student_ema" and source.parent.name == "checkpoint-000001500", "H3_STUDENT_INIT must explicitly select checkpoint-000001500/student_ema")
    output = Path(training["output_dir"]).expanduser().resolve()
    require(not output.is_relative_to(source.parent.parent) and not source.is_relative_to(output), "Use a new output directory outside the CM source run")
    require(not config.get("resume", {}).get("resume_from_checkpoint"), "Resume only this new run through auto_resume")
    return source, adapter


def validate_models(config):
    model = config["model"]
    merged = leaf("lightx2v_train/model_zoo/minimax_h3/premerged_decfg.py", "_ref6k_decfg")
    original = Path(model["pretrained_model_name_or_path"]).expanduser().resolve()
    for role in ("student", "teacher", "fake"):
        entry = model if role == "student" else model[role]
        require(Path(entry["pretrained_model_name_or_path"]).expanduser().resolve() == original, f"{role} must use the same original Ref base")
        require(entry["training_adapter"] == {"enabled": False}, f"Disable online DeCFG on {role}")
    require(merged.validate_premerged_score_base(original, component_name="transformer_ref") is None, "All roles require an original base without a premerged DeCFG receipt")
    for component in ("transformer_ref", "text_encoder", "vae", "audio_vae"):
        require((original / component / "config.json").is_file(), f"Missing original model component: {original / component}")


def validate_source(config, source, adapter):
    handoff, score_preset = verify_cm_export(config, task="ref2av")
    weights, count = adapter.validate_source(source, config["training"]["student"]["lora"])
    student_api = leaf("lightx2v_train/trainers/dmd/student_warmstart.py", "_ref6k_student")
    student_api.validate_student_adapter_preset(weights, expected_preset={"enabled": False}, model_name="minimax_h3_ref2av")
    return {"student_init": handoff["student_init"], "lora_tensors": count, "weights_only": ["student"], "cm_score_preset_roles": score_preset["roles"], "new_score_uses_decfg": False}


def validate_data(config):
    data_api = leaf("lightx2v_train/trainers/dmd/dmd2_gan_data.py", "_ref6k_pairs")
    manifest, rows, fingerprint = data_api.load_gan_real_records(config["training"]["dmd"]["gan"]["real_cache"], task="ref2av", geometry=None)
    raw_count = sum(bool(line.strip()) for line in manifest.read_text().splitlines())
    require(raw_count == len(rows) == 6000, "Ref6k requires exactly6000 eligible full-AV image-only pairs")
    receipt_api = leaf("scripts/minimax_h3_cm_pdmd_preflight.py", "_ref6k_receipt")
    receipt_api._receipt(manifest, raw_count)
    receipt = json.loads(manifest.with_suffix(".complete.json").read_text())
    require(receipt.get("schema") == "h3_ref6k_v1" and receipt.get("schema_version") == 1 and receipt.get("teacher_weights") == "base", "Expected a base-teacher Ref6k completion receipt")
    require(Path(receipt["teacher_model_path"]).resolve() == Path(config["model"]["pretrained_model_name_or_path"]).resolve(), "GAN teacher data must match the original training base")
    counts = Counter(row["reference_image_count"] for row in rows)
    expected = {index: 1000 for index in range(1, 7)}
    require(counts == expected and receipt["reference_image_counts"] == {str(key): value for key, value in expected.items()}, "Ref6k requires1000 rows for each image count1..6")
    sources = Counter(row.get("ref6k_source") for row in rows)
    require(sources == receipt["source_counts"] == {"omni_teacher_base": 5000, "selfgen_single_image": 1000}, "Expected5000 Omni teacher pairs and1000 self-generated single-image pairs")
    require(len({row["condition_path"] for row in rows}) == len(rows), "Duplicate Ref6k condition paths")
    for row in rows:
        require(row["ref6k_source"] == ("selfgen_single_image" if row["reference_image_count"] == 1 else "omni_teacher_base"), "Source and reference-image count disagree")
        require(sorted((row["target_height"], row["target_width"])) == [768, 1344], "Use124-frame 768p cached canvases")
        cost = row.get("packed_sequence_tokens_124")
        require(isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost > 0, "Ref data requires positive packed_sequence_tokens_124")
    return {"manifest": str(manifest), "rows": len(rows), "fingerprint": fingerprint, "reference_image_counts": dict(sorted(counts.items())), "sources": dict(sources)}


def main(config_path):
    config = leaf("lightx2v_train/runtime/config.py", "_ref6k_config").load_config(str(config_path))
    source, adapter = validate_recipe(config)
    validate_models(config)
    data = validate_data(config)
    resume_api = leaf("scripts/minimax_h3_pdmd_gan_preflight.py", "_ref6k_resume")
    resume = resume_api.own_stage_resume(config, gan_enabled=True)
    source_info = validate_source(config, source, adapter) if resume is None else {"own_stage_resume": str(resume)}
    updates = int(config["training"]["max_train_iters"])
    print(
        json.dumps(
            {
                "task": "ref2av",
                "source": source_info,
                "data": data,
                "steps": config["training"]["dmd"]["num_inference_steps"],
                "shifts": [12, 3],
                "fake_per_student": 5,
                "role_updates": updates,
                "student_updates": updates // 6,
                "fake_updates": updates - updates // 6,
                "save_every": 100,
                "keep": 2,
                "decfg_enabled": False,
                "output": config["training"]["output_dir"],
            },
            indent=2,
        ),
        flush=True,
    )
    if resume is None:
        print("[ref6k-gan] Student=CM1500 EMA adapter only; teacher/fake=original base; fresh optimizers, fake, GAN head and data cursors. CM score-only DeCFG is not carried forward.", flush=True)
    else:
        print("[ref6k-gan] Resume this run's student/fake/GAN weights, optimizers and cursors; original CM export is not required.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
