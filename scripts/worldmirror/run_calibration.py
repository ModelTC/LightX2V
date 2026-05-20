"""Run one fp32 forward pass per scene to collect per-MM-leaf input absmax.

Swaps every MM leaf in the WorldMirror ViT backbone + CameraHead refine_net
for the ``Calib`` scheme (numerically identical to ``Default`` except it
records ``max|input|`` into ``CALIB["absmax"][<weight_name>]``). After the
sweep, writes the resulting per-layer ``input_scale = absmax / 448`` to a
safetensors that the fp8-pertensor auto-quant path consumes via the runner's
``input_scale_file`` config entry.

Usage::

    python scripts/worldmirror/run_calibration.py \
        --scenes Workspace Desk Park Statue_Face \
        --output /tmp/wm_calib_input_scales.safetensors

Runs single-GPU only (SP would slice tokens and bias the absmax stats).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger

# Make sure the LightX2V repo root is on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SCENES = ["Workspace", "Desk", "Park", "Statue_Face"]
DEFAULT_SCENE_ROOT = "/workspace/HY-World-2.0/examples/worldrecon/realistic"
DEFAULT_MODEL_PATH = "/data/nvme1/models/HY-World-2.0"
DEFAULT_CONFIG_JSON = str(REPO_ROOT / "configs/worldmirror/worldmirror_recon.json")
DEFAULT_OUTPUT = str(REPO_ROOT / "configs/worldmirror/worldmirror_input_scales.safetensors")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    ap.add_argument("--scene_root", default=DEFAULT_SCENE_ROOT)
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--config_json", default=DEFAULT_CONFIG_JSON)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--target_size", type=int, default=952)
    return ap.parse_args()


def main():
    args = parse_args()

    # Force single-GPU for calibration (SP would shard tokens and skew absmax).
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(k, None)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # Load the base config but flip it into calibration mode.
    from lightx2v.utils.lockable_dict import LockableDict  # noqa: E402

    with open(args.config_json) as f:
        cfg = json.load(f)
    cfg["model_path"] = args.model_path
    cfg["run_calib"] = True
    # Calibration must run the numerically reference path, so force:
    #   * Default MM apply (Calib inherits from MMWeight, uses fp32 mm)
    #   * no cpu_offload/lazy_load (not wired for Calib scheme)
    #   * fp32 (no bf16 — calibration should match the precision our
    #     downstream quant tries to match to).
    cfg["cpu_offload"] = False
    cfg["lazy_load"] = False
    cfg["enable_bf16"] = False
    cfg["weight_auto_quant"] = False
    cfg["dit_quant_scheme"] = "Default"
    # Disable expensive post-processing we don't need.
    cfg["save_depth"] = False
    cfg["save_normal"] = False
    cfg["save_gs"] = False
    cfg["save_camera"] = False
    cfg["save_points"] = False
    cfg["save_colmap"] = False
    cfg["save_conf"] = False
    cfg["save_sky_mask"] = False
    cfg["apply_sky_mask"] = False
    cfg["apply_edge_mask"] = False
    cfg["apply_confidence_mask"] = False

    config = LockableDict(cfg)
    config["target_size"] = args.target_size

    # Build runner and force-init modules (loads model + installs adapters).
    from lightx2v.models.runners.worldmirror.worldmirror_runner import WorldMirrorRunner  # noqa: E402
    from lightx2v.utils.input_info import init_empty_input_info  # noqa: E402

    logger.info("[calib] Building WorldMirrorRunner...")
    runner = WorldMirrorRunner(config)
    runner.init_modules()
    mm_type = runner.model.transformer_weights._mm_type
    logger.info(f"[calib] mm_type in use: {mm_type}")
    if mm_type not in ("Calib", "CalibMax"):
        logger.error(f"[calib] mm_type={mm_type!r} — run_calib didn't reach the WeightModule. Check runtime_cfg plumbing.")
        sys.exit(2)

    # Iterate scenes. Any scene that 404s is skipped with a warning — do not
    # hard-fail because the caller may pass a trimmed-down list for a quick
    # sanity run.
    for i, scene in enumerate(args.scenes):
        scene_path = os.path.join(args.scene_root, scene)
        if not os.path.isdir(scene_path):
            logger.warning(f"[calib] Skipping missing scene: {scene_path}")
            continue
        logger.info(f"[calib] ({i + 1}/{len(args.scenes)}) scene={scene}")

        input_info = init_empty_input_info("recon")
        input_info.input_path = scene_path
        # Send output into a throwaway tmp dir — save_* are all off above
        # but the runner still wants a writable path.
        input_info.save_result_path = "/tmp/wm_calib_output"

        t0 = time.perf_counter()
        runner.run_pipeline(input_info)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(f"[calib]   done in {time.perf_counter() - t0:.1f}s")

    # Snapshot CALIB → safetensors.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_written = runner.model.transformer_weights.export_calibration(args.output)
    logger.info(f"[calib] Wrote {n_written} input_scale entries to {args.output}")

    # Report summary stats so we know calibration converged on something
    # non-degenerate.
    from lightx2v.utils.global_paras import CALIB  # noqa: E402

    vals = [float(v) for v in CALIB.get("absmax", {}).values()]
    if vals:
        logger.info(f"[calib] absmax stats: n={len(vals)} min={min(vals):.3e} max={max(vals):.3e} mean={sum(vals) / len(vals):.3e}")


if __name__ == "__main__":
    main()
