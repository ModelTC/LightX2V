"""Full-feature CLI for HY-WorldMirror-2.0 through the LightX2V runner.

Mirrors the original ``python -m hyworld2.worldrecon.pipeline`` entry point:
all CLI flags, multi-GPU via ``torchrun`` (WorldMirror's own sequence-
parallel path — no FSDP), the ``>>>`` interactive loop, optional Gaussian-
splat flythrough video rendering, prior camera/depth inputs, and per-head
disable.

Examples
--------

Single-GPU, exported-model dir (default subfolder)::

    python examples/worldmirror/run_worldmirror.py \\
        --input_path /workspace/HY-World-2.0/examples/worldrecon/realistic/Workspace \\
        --pretrained_model_name_or_path /data/nvme1/models/HY-World-2.0 \\
        --no_interactive

Multi-GPU (sequence-parallel) + bf16::

    torchrun --nproc_per_node=2 examples/worldmirror/run_worldmirror.py \\
        --input_path /path/to/images \\
        --pretrained_model_name_or_path /data/nvme1/models/HY-World-2.0 \\
        --enable_bf16 --no_interactive

Training-output format (separate yaml + ckpt)::

    python examples/worldmirror/run_worldmirror.py \\
        --input_path /path/to/images \\
        --config_path /path/to/train_config.yaml \\
        --ckpt_path   /path/to/checkpoint.safetensors
"""

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs/worldmirror/worldmirror_recon.json")


def build_parser():
    p = argparse.ArgumentParser(description="HY-WorldMirror-2.0 via LightX2V")

    # --- Input / output ---
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, default="inference_output")
    p.add_argument("--strict_output_path", type=str, default=None, help="If set, save results directly to this path (no subdir/timestamp).")

    # --- Model loading ---
    p.add_argument("--pretrained_model_name_or_path", type=str, default="/data/nvme1/models/HY-World-2.0", help="Local directory containing HY-WorldMirror-2.0 weights.")
    p.add_argument("--subfolder", type=str, default="HY-WorldMirror-2.0", help="Subfolder inside the model directory.")
    p.add_argument("--config_path", type=str, default=None, help="Optional training YAML; used with --ckpt_path.")
    p.add_argument("--ckpt_path", type=str, default=None, help="Optional .ckpt/.safetensors; used with --config_path.")
    p.add_argument("--lightx2v_config", type=str, default=DEFAULT_CONFIG_PATH, help="LightX2V JSON config (saving/mask/render defaults).")

    # --- Execution mode ---
    p.add_argument("--enable_bf16", action="store_true", default=False)
    p.add_argument("--disable_heads", type=str, nargs="*", default=None, help="Heads to disable: camera depth normal points gs")

    # --- Inference params ---
    p.add_argument("--target_size", type=int, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--video_strategy", type=str, default=None, choices=["old", "new"])
    p.add_argument("--video_min_frames", type=int, default=None)
    p.add_argument("--video_max_frames", type=int, default=None)

    # --- Save toggles ---
    p.add_argument("--no_save_depth", action="store_true")
    p.add_argument("--no_save_normal", action="store_true")
    p.add_argument("--no_save_gs", action="store_true")
    p.add_argument("--no_save_camera", action="store_true")
    p.add_argument("--no_save_points", action="store_true")
    p.add_argument("--save_colmap", action="store_true", default=False)
    p.add_argument("--save_conf", action="store_true", default=False)
    p.add_argument("--save_sky_mask", action="store_true", default=False)

    # --- Mask params ---
    p.add_argument("--apply_sky_mask", action="store_true", default=None)
    p.add_argument("--no_sky_mask", dest="apply_sky_mask", action="store_false")
    p.add_argument("--apply_edge_mask", action="store_true", default=None)
    p.add_argument("--no_edge_mask", dest="apply_edge_mask", action="store_false")
    p.add_argument("--apply_confidence_mask", action="store_true", default=None)
    p.add_argument("--sky_mask_source", type=str, default=None, choices=["auto", "model", "onnx"])
    p.add_argument("--model_sky_threshold", type=float, default=None)
    p.add_argument("--confidence_percentile", type=float, default=None)
    p.add_argument("--edge_normal_threshold", type=float, default=None)
    p.add_argument("--edge_depth_threshold", type=float, default=None)

    # --- Compression ---
    p.add_argument("--compress_pts", action="store_true", default=None)
    p.add_argument("--no_compress_pts", dest="compress_pts", action="store_false")
    p.add_argument("--compress_pts_max_points", type=int, default=None)
    p.add_argument("--compress_pts_voxel_size", type=float, default=None)
    p.add_argument("--max_resolution", type=int, default=None)
    p.add_argument("--compress_gs_max_points", type=int, default=None)

    # --- Priors ---
    p.add_argument("--prior_cam_path", type=str, default=None)
    p.add_argument("--prior_depth_path", type=str, default=None)

    # --- Rendering ---
    p.add_argument("--save_rendered", action="store_true", default=False)
    p.add_argument("--render_interp_per_pair", type=int, default=None)
    p.add_argument("--render_depth", action="store_true", default=False)

    # --- Misc ---
    p.add_argument("--log_time", action="store_true", default=None)
    p.add_argument("--no_log_time", dest="log_time", action="store_false")
    p.add_argument("--no_interactive", action="store_true")

    return p


def build_config(args):
    with open(args.lightx2v_config, "r") as f:
        config_dict = json.load(f)

    config_dict["model_path"] = args.pretrained_model_name_or_path
    config_dict["subfolder"] = args.subfolder
    config_dict["config_path"] = args.config_path
    config_dict["ckpt_path"] = args.ckpt_path
    config_dict["enable_bf16"] = args.enable_bf16
    config_dict["disable_heads"] = args.disable_heads
    config_dict["save_rendered"] = args.save_rendered
    config_dict["render_depth"] = args.render_depth

    # Save toggles (negative flags — only override when user passed them).
    if args.no_save_depth:
        config_dict["save_depth"] = False
    if args.no_save_normal:
        config_dict["save_normal"] = False
    if args.no_save_gs:
        config_dict["save_gs"] = False
    if args.no_save_camera:
        config_dict["save_camera"] = False
    if args.no_save_points:
        config_dict["save_points"] = False
    if args.save_colmap:
        config_dict["save_colmap"] = True
    if args.save_conf:
        config_dict["save_conf"] = True
    if args.save_sky_mask:
        config_dict["save_sky_mask"] = True

    # Positive overrides (only if explicitly set).
    for key in (
        "target_size",
        "fps",
        "video_strategy",
        "video_min_frames",
        "video_max_frames",
        "apply_sky_mask",
        "apply_edge_mask",
        "apply_confidence_mask",
        "sky_mask_source",
        "model_sky_threshold",
        "confidence_percentile",
        "edge_normal_threshold",
        "edge_depth_threshold",
        "compress_pts",
        "compress_pts_max_points",
        "compress_pts_voxel_size",
        "max_resolution",
        "compress_gs_max_points",
        "render_interp_per_pair",
        "log_time",
    ):
        val = getattr(args, key, None)
        if val is not None:
            config_dict[key] = val

    # Passed along so rank-0 can reconstruct the per-case output dir.
    config_dict["output_path"] = args.output_path

    return config_dict


def run_one(runner, input_path, strict_output_path, output_path, task_name="recon"):
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict

    input_info = init_empty_input_info(task_name)
    update_input_info_from_dict(
        input_info,
        {
            "input_path": input_path,
            "save_result_path": output_path,
            "strict_output_path": strict_output_path,
            "return_result_tensor": False,
        },
    )
    return runner.run_pipeline(input_info)


def main():
    args = build_parser().parse_args()

    # Import only the runner we need (avoids dragging in diffusers/flash-attn conflicts).
    import torch
    import torch.distributed as dist

    from lightx2v.models.runners.worldmirror.worldmirror_runner import (
        WorldMirrorRunner,  # noqa: F401  — registers "worldmirror"
        _broadcast_string,
    )
    from lightx2v.utils.lockable_dict import LockableDict
    from lightx2v.utils.registry_factory import RUNNER_REGISTER

    config_dict = build_config(args)
    if args.strict_output_path is None:
        os.makedirs(args.output_path, exist_ok=True)

    config = LockableDict(config_dict)

    runner = RUNNER_REGISTER[config["model_cls"]](config)
    runner.init_modules()

    is_distributed = runner.is_distributed
    rank = runner.rank

    # Cache the NCCL backend handle once so we can bump its timeout while
    # waiting on stdin without reaching into c10d private state every iter.
    cuda_backend = None
    if is_distributed:
        cuda_backend = dist.distributed_c10d._get_default_group()._get_backend(torch.device("cuda"))

    try:
        run_one(runner, args.input_path, args.strict_output_path, args.output_path)

        if args.no_interactive:
            return

        if rank == 0:
            print("\n[Interactive] Enter new input paths. Type 'quit' to stop.\n")

        _INF_TIMEOUT = timedelta(days=365)
        _DEF_TIMEOUT = timedelta(minutes=10)

        while True:
            if cuda_backend is not None:
                cuda_backend.options._timeout = _INF_TIMEOUT

            new_input = ""
            if rank == 0:
                try:
                    new_input = input(">>> ").strip()
                except (EOFError, KeyboardInterrupt):
                    new_input = "quit"

            if is_distributed:
                new_input = _broadcast_string(new_input, rank, src=0)
                cuda_backend.options._timeout = _DEF_TIMEOUT

            if not new_input or new_input.lower() in ("quit", "exit", "q"):
                break

            if rank == 0 and not (Path(new_input).is_dir() or Path(new_input).is_file()):
                print(f"  Invalid path: {new_input}")
                continue

            # No need to re-.to(device)/.eval() — the model state is stable
            # across cases because run_pipeline doesn't mutate training mode
            # or device placement.
            run_one(runner, new_input, args.strict_output_path, args.output_path)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
