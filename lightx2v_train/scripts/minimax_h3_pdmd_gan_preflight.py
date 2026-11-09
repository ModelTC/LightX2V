#!/usr/bin/env python3
"""CPU preflight for the action-prompt H3 PDMD + GAN stage."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_prompts(path, expected=2000):
    reader = load_module("_h3_prompt_records", ROOT / "data_process/minimax_h3/prompt_records.py")
    rows = reader.read_json_rows(path)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} prompts, found {len(rows)}: {path}")
    for index, row in enumerate(rows):
        if row.get("task", "t2av") not in {"t2av", "t2va", "t2v"} or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Prompt row {index} requires task=t2av and nonempty prompt")
        if int(row.get("num_frames", 124)) != 124 or float(row.get("fps", 24)) != 24:
            raise ValueError(f"Prompt row {index} must use 124 frames at 24 fps")
    return rows


def validate_cache(root, prompt_rows, world_size=32):
    root = Path(root)
    receipt = json.loads((root / "prompt_cache.complete.json").read_text())
    manifest = root / "metadata.jsonl"
    content = manifest.read_bytes()
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    namespace = json.loads((root / "preprocess_config.json").read_text())
    if (
        receipt.get("prompt_count") != len(prompt_rows)
        or receipt.get("world_size") != world_size
        or receipt.get("manifest_sha256") != hashlib.sha256(content).hexdigest()
        or receipt.get("preprocess_fingerprint") != namespace.get("preprocess_fingerprint")
        or len(rows) != len(prompt_rows)
    ):
        raise ValueError("Prompt cache receipt does not match its manifest/namespace")
    indices = set()
    for row in rows:
        index = row.get("source_index")
        if type(index) is not int or not 0 <= index < len(prompt_rows) or index in indices:
            raise ValueError(f"Prompt cache has unexpected/duplicate source_index={index!r}")
        relative = f"conditions/condition_{index:08d}.pt"
        if row.get("condition_path") != relative or row.get("caption") != prompt_rows[index]["prompt"].strip():
            raise ValueError(f"Prompt cache row {index} does not match its source prompt")
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
        indices.add(index)
    return receipt


def validate_recipe(config, gan_enabled=None):
    training, model = config["training"], config["model"]
    dmd, data = training["dmd"], config["data"]["train"]
    matching = model["capabilities"]["distribution_matching"]
    if model["name"] != "minimax_h3_t2av" or training.get("method") != "dmd":
        raise ValueError("This launcher requires the H3 T2AV DMD model")
    actual_gan = bool(dmd.get("gan", {}).get("enabled", False))
    if gan_enabled is not None and actual_gan != gan_enabled:
        raise ValueError("Unexpected GAN enabled state for this phase")
    if not dmd.get("official_pdmd") or not matching.get("official_pdmd"):
        raise ValueError("This launcher requires official PDMD")
    if (dmd.get("num_inference_steps"), dmd.get("fake_update_ratio"), dmd.get("update_order")) != (8, 5, "fake_first"):
        raise ValueError("Expected eight rollout steps and five fake updates before each student update")
    if (dmd.get("official_rollout_video_shift"), dmd.get("official_rollout_audio_shift")) != (12.0, 3.0):
        raise ValueError("Expected physical rollout shifts 12/3")
    if (matching.get("video_flow_shift"), matching.get("audio_flow_shift")) != (12.0, 3.0):
        raise ValueError("Inference shifts must match rollout shifts 12/3")
    if data.get("tasks") != ["t2av"] or data.get("buckets") != ["landscape"] or data.get("route_mode") != "stratified":
        raise ValueError("Expected stratified landscape T2AV prompt data")
    if config["distributed"]["fsdp2"].get("size") != 32 or config["distributed"]["sequence_parallel"].get("enabled"):
        raise ValueError("Expected FSDP32 without sequence parallelism")
    student, fake = training["student"], training["fake"]
    if not student.get("checkpoint_path") or student["checkpoint_path"] != fake.get("checkpoint_path"):
        raise ValueError("Student and fake must initialize from the same prior PDMD checkpoint")
    if student.get("train_type") != "lora" or fake.get("train_type") != "full" or student.get("ema", {}).get("enabled"):
        raise ValueError("Expected live LoRA student and full fake without EMA")
    lora = student["lora"]
    adapter = load_module("_h3_gan_adapter_preflight", ROOT / "scripts/minimax_h3_pdmd_from_turbo_preflight.py")
    if (
        (lora.get("rank"), lora.get("alpha")) != (128, 128)
        or lora.get("rank_pattern") != adapter.PATTERNS
        or lora.get("alpha_pattern") != adapter.PATTERNS
        or set(lora.get("target_modules", [])) != adapter.TARGETS
    ):
        raise ValueError("Expected the source regular128/128 + block/output AdaLN16/16 adapter architecture")
    if student.get("checkpoint_strict") is not True:
        raise ValueError("Student checkpoint must load strictly")
    if int(training["max_train_iters"]) < 6:
        raise ValueError("H3_GAN_MAX_ITERS must include at least one complete 5-fake/1-student cycle")
    source = Path(student["checkpoint_path"]).expanduser().resolve()
    output = Path(training["output_dir"]).expanduser().resolve()
    if output in (source, source.parent) or source in output.parents:
        raise ValueError("GAN stage needs a new output directory outside the source checkpoint")
    return source, adapter


def own_stage_resume(config, gan_enabled=None):
    """Use the trainer's exact candidate selection; never hide a broken latest save."""
    if not config.get("resume", {}).get("auto_resume", False):
        return None
    if gan_enabled is None:
        gan_enabled = bool(config["training"]["dmd"].get("gan", {}).get("enabled", False))
    checkpoint_api = load_module("_h3_gan_checkpoint_preflight", ROOT / "lightx2v_train/runtime/checkpoint.py")
    candidate, iteration = checkpoint_api.find_latest_checkpoint(config["training"]["output_dir"])
    if candidate is None:
        return None
    checkpoint = Path(candidate)
    required = ["official_pdmd.complete", "dist_state/.metadata", "trainer_state.pt"]
    required.extend(f"official_pdmd.rank{rank:05d}.pt" for rank in range(32))
    missing = [relative for relative in required if not (checkpoint / relative).is_file()]
    if missing:
        raise ValueError(f"Latest own-stage checkpoint is incomplete: {checkpoint}; missing {missing}. Refusing source warm-start or an older-checkpoint fallback.")
    if (checkpoint / "official_pdmd.complete").read_text().strip() != str(iteration):
        raise ValueError(f"Latest own-stage completion marker has the wrong iteration: {checkpoint}")
    import torch

    state = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("iteration") != iteration or state.get("world_size") != 32 or bool(state.get("dmd2_gan_enabled", False)) != gan_enabled:
        raise ValueError(f"Latest own-stage checkpoint is not a matching 32-GPU training state: {checkpoint}")
    if not gan_enabled:
        return checkpoint
    for key in ("dmd2_gan_config", "dmd2_gan_head", "dmd2_gan_optimizer", "dmd2_gan_lr_scheduler"):
        if key not in state:
            raise ValueError(f"Latest own-stage checkpoint is missing {key}: {checkpoint}")
    if not isinstance(state["dmd2_gan_config"], dict) or state["dmd2_gan_config"].get("options") != config["training"]["dmd"]["gan"]:
        raise ValueError(f"Latest own-stage checkpoint GAN recipe differs from this configuration: {checkpoint}")
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--check-cache", action="store_true")
    args = parser.parse_args(argv)
    runtime = load_module("_h3_gan_runtime_config", ROOT / "lightx2v_train/runtime/config.py")
    config = runtime.load_config(args.config)
    source, adapter = validate_recipe(config)
    rows = validate_prompts(args.prompts)
    resume = own_stage_resume(config)
    if resume is None:
        for relative in ("official_pdmd.complete", "dist_state/.metadata"):
            if not (source / relative).is_file():
                raise FileNotFoundError(f"Source must be a complete official-PDMD checkpoint: {source / relative}")
        weights, count = adapter.validate_source(source, config["training"]["student"]["lora"])
    model = Path(config["model"]["pretrained_model_name_or_path"])
    for component in ("transformer/config.json", "text_encoder/config.json", "vae/config.json"):
        if not (model / component).is_file():
            raise FileNotFoundError(model / component)
    real_helper = load_module("_h3_gan_real_preflight", ROOT / "lightx2v_train/trainers/dmd/dmd2_gan_data.py")
    manifest, real_rows, fingerprint = real_helper.load_gan_real_records(config["training"]["dmd"]["gan"]["real_cache"])
    if args.check_cache:
        receipt = validate_cache(config["data"]["train"]["data_path"], rows)
        print(f"[preflight] prompt-cache complete={receipt['prompt_count']} global-shards={receipt['world_size']}")
    print(f"[preflight] prompts={len(rows)}; task=t2av; target=124x768x1344@24fps; encode_global_shards=32")
    print(f"[preflight] GAN real manifest={manifest}; eligible_full_AV_rows={len(real_rows)}; fingerprint={fingerprint}")
    if resume is None:
        print(f"[preflight] student={weights} ({count} LoRA tensors); fake={source / 'dist_state'} weights only")
        print("[preflight] New stage: fresh student/fake optimizers, data cursors and GAN head; own-stage auto-resume enabled.")
    else:
        print(f"[preflight] own-stage resume={resume}; original warm-start files are not required")
        print("[preflight] Trainer will validate the full saved recipe and restore student/fake/GAN weights, optimizers and cursors.")
    print(f"[preflight] 4x8 FSDP32; rollout=8; shifts=12/3; fake:student=5:1; optimizer_updates={config['training']['max_train_iters']}; save_every=100; keep=2")
    print(f"[preflight] config={args.config}; output={config['training']['output_dir']}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"H3 PDMD + GAN preflight failed: {error}") from error
