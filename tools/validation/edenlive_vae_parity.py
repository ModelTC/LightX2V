"""Compare native and upstream VAE pixels from the same saved latent fixture."""

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan

from lightx2v.models.runners.wan import edenlive_runner  # noqa: F401
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import build_startup_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.upstream)
    from edenlive.inference.io import _save_video_array
    from edenlive.models.adapters.wan22_diffusers.adapter import Wan22DiffusersAdapter

    torch.set_grad_enabled(False)
    variant = "5b" if "5b" in args.profile else "14b"
    component = Path(args.model_path) / f"vae-{variant}"
    weight = component / f"Wan2.{2 if variant == '5b' else 1}_VAE.pth"
    reference = Wan22DiffusersAdapter._load_vae_from_original(AutoencoderKLWan, weight, component, torch.bfloat16).cuda().eval()
    model_cls = args.profile.split("_")[0] + "_edenlive" + ("_ar" if args.profile.endswith("_few_step") else "")
    config = build_startup_config(dict(model_cls=model_cls, task="i2v", model_path=args.model_path, config_json=str(Path(__file__).resolve().parents[2] / "configs/edenlive" / f"{args.profile}.json")))
    config["vae_path"] = str(weight)
    runner = RUNNER_REGISTER[model_cls](config)
    native = runner.load_vae_decoder()
    fixture = torch.load(args.fixture, weights_only=True)["decode_video_latents"][0]
    latents = fixture[None] if fixture.ndim == 4 else fixture
    latents = latents.cuda().bfloat16()
    mean = latents.new_tensor(reference.config.latents_mean).view(1, -1, 1, 1, 1)
    inv_std = 1 / latents.new_tensor(reference.config.latents_std).view(1, -1, 1, 1, 1)
    expected = reference.decode(latents / inv_std + mean).sample.clamp(-1, 1)
    actual = native.decode(latents[0])
    difference = (actual.float() - expected.float()).abs()
    metrics = dict(max_abs=difference.max().item(), mean_abs=difference.mean().item())
    print(json.dumps(metrics), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "vae_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _save_video_array(actual[0], output / "native_matched_save.mp4", fps=config["fps"])
    _save_video_array(expected[0], output / "reference_matched_save.mp4", fps=config["fps"])
    from lightx2v.utils.utils import save_to_video, wan_vae_to_comfy

    save_to_video(wan_vae_to_comfy(actual.float().clone()), str(output / "native_pipeline_save.mp4"), fps=config["fps"], method="ffmpeg")
    torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)


if __name__ == "__main__":
    main()
