import argparse
import os

import torch
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import load_config
from lightx2v_train.trainers.fastwam_action_consistency.config import FastWAMActionConsistencyConfig
from lightx2v_train.trainers.fastwam_action_dmd.checkpoint import load_role_state_dict
from lightx2v_train.trainers.fastwam_action_dmd.config import FastWAMActionDmdConfig
from lightx2v_train.trainers.fastwam_action_dmd.roles import (
    DEFAULT_LORA_TARGETS,
    DEFAULT_MODULES_TO_SAVE,
    attach_video_role,
    configure_action_role,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export FastWAM action DMD or consistency weights into a native checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--weights", choices=("auto", "student", "ema"), default="auto",
        help="auto selects EMA for consistency and student for DMD.",
    )
    parser.add_argument("--lora-output", help="Also save PEFT adapters under this directory (action/ and optional video/).")
    return parser.parse_args()


def parse_training_config(config):
    if config["training"].get("method") == "fastwam_action_consistency":
        return FastWAMActionConsistencyConfig.from_mapping(config)
    return FastWAMActionDmdConfig.from_mapping(config)


def resolve_weights(parsed, weights):
    consistency = isinstance(parsed, FastWAMActionConsistencyConfig)
    if weights == "auto":
        return "ema" if consistency else "student"
    if weights == "ema" and not consistency:
        raise ValueError("EMA weights are only available for consistency checkpoints.")
    return weights


def infer_step_from_checkpoint_path(checkpoint_path):
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_path))
    prefix = "checkpoint-"
    if not checkpoint_name.startswith(prefix):
        return None
    suffix = checkpoint_name[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _role_architecture(role):
    """Return the PEFT architecture fields needed for a safe state restore."""
    if role is None:
        return None
    if role.train_type == "full":
        return ("full",)
    lora = role.lora or {}
    return (
        "lora",
        int(lora["rank"]),
        int(lora.get("alpha", lora["rank"])),
        float(lora.get("dropout", 0.0)),
        tuple(sorted(str(item) for item in lora.get("target_modules", DEFAULT_LORA_TARGETS))),
        tuple(sorted(str(item) for item in lora.get("modules_to_save", DEFAULT_MODULES_TO_SAVE))),
    )


def main():
    args = parse_args()
    step = infer_step_from_checkpoint_path(args.checkpoint)
    config = load_config(args.config)
    parsed = parse_training_config(config)
    weights = resolve_weights(parsed, args.weights)
    unfreeze_video = getattr(parsed, "unfreeze_video", False)
    checkpoint_config_path = os.path.join(args.checkpoint, "config.yaml")
    if os.path.isfile(checkpoint_config_path):
        checkpoint_config = load_config(checkpoint_config_path)
        checkpoint_parsed = parse_training_config(checkpoint_config)
        if type(checkpoint_parsed) is not type(parsed):
            raise RuntimeError("Export config and checkpoint disagree about the training method.")
        if getattr(checkpoint_parsed, "unfreeze_video", False) != unfreeze_video:
            raise RuntimeError(
                "Export config and checkpoint disagree about video training mode: "
                f"config={unfreeze_video}, checkpoint={getattr(checkpoint_parsed, 'unfreeze_video', False)}."
            )
        for role_name in ("student", "fake"):
            if _role_architecture(getattr(checkpoint_parsed, role_name, None)) != _role_architecture(getattr(parsed, role_name, None)):
                raise RuntimeError(f"Export config and checkpoint disagree about {role_name} LoRA architecture.")
        if unfreeze_video and _role_architecture(checkpoint_parsed.video) != _role_architecture(parsed.video):
            raise RuntimeError("Export config and checkpoint disagree about video LoRA architecture.")
    model = build_model(config)
    model.load_components()
    module = model.unwrap_module()
    student = configure_action_role(module.action_expert, parsed.student)
    student_state = torch.load(
        os.path.join(args.checkpoint, f"{weights}_action.pt"),
        map_location="cpu",
        weights_only=True,
    )
    load_role_state_dict(student, parsed.student.train_type, student_state)
    if parsed.student.train_type == "lora":
        if args.lora_output:
            student.save_pretrained(os.path.join(args.lora_output, "action"), safe_serialization=True)
        student = student.merge_and_unload(safe_merge=True)
    module.action_expert = student
    module.mot.mixtures["action"] = student
    if unfreeze_video:
        video_state_path = os.path.join(args.checkpoint, "video.pt")
        if not os.path.isfile(video_state_path):
            raise FileNotFoundError(
                "Video-unfreeze checkpoint is missing video.pt: "
                f"{video_state_path}"
            )
        video = attach_video_role(module, parsed.video)
        video_state = torch.load(video_state_path, map_location="cpu", weights_only=True)
        load_role_state_dict(video, parsed.video.train_type, video_state)
        if parsed.video.train_type == "lora":
            if args.lora_output:
                video.save_pretrained(os.path.join(args.lora_output, "video"), safe_serialization=True)
            video = video.merge_and_unload(safe_merge=True)
        module.video_expert = video
        module.mot.mixtures["video"] = video
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    module.save_checkpoint(args.output, step=step)
    print(f"Exported {weights} weights from {args.checkpoint} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
