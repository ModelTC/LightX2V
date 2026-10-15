import argparse
import sys
from pathlib import Path

import torch
from diffusers import UniPCMultistepScheduler

from lightx2v.models.runners.wan import edenlive_runner  # noqa: F401
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import build_startup_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.upstream)
    from edenlive.schedulers.flow_matching import FlowMatchingScheduler

    torch.set_grad_enabled(False)
    for variant, mode in [("5b", "few_step"), ("14b", "few_step"), ("5b", "bidir"), ("14b", "bidir")]:
        profile = f"wan{22 if variant == '5b' else 21}_{variant}_{mode}"
        model_cls = f"wan{22 if variant == '5b' else 21}_edenlive" + ("_ar" if mode == "few_step" else "")
        config = build_startup_config(dict(model_cls=model_cls, task="i2v", model_path=args.model_path, config_json=str(Path(__file__).resolve().parents[2] / "configs/edenlive" / f"{profile}.json")))
        native = RUNNER_REGISTER[model_cls](config).scheduler
        native.prepare(42, [config["out_dim"], 3, 2, 2], {"clean_first_frame": torch.zeros(config["out_dim"], 1, 2, 2, device="cuda")})
        sample = native.latents.clone()[None]
        if mode == "bidir" and variant == "5b":
            source = UniPCMultistepScheduler(num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5)
            source.set_timesteps(config["infer_steps"], device="cuda")
            source.sigmas = source.sigmas.cuda()
            torch.testing.assert_close(native.sigmas, source.sigmas, atol=0, rtol=0)
            torch.testing.assert_close(native.timesteps, source.timesteps, atol=0, rtol=0)
        else:
            source = FlowMatchingScheduler()
            schedule = source.build_schedule(
                config["infer_steps"],
                device="cuda",
                config=dict(timesteps=config["ar_config"]["denoising_step_list"], stochastic_sampling=True) if mode == "few_step" else dict(schedule="shifted_linear", shift=3),
            )
        generator = torch.Generator(device="cuda").manual_seed(42)
        # The upstream generator has already drawn the initial trajectory.
        torch.randn(sample.shape, generator=generator, device="cuda", dtype=sample.dtype)
        for step in range(config["infer_steps"]):
            native.step_pre(step)
            if mode == "few_step" or variant == "5b":
                sample[:, :, :1] = 0
            if mode == "few_step":
                torch.testing.assert_close(native.timesteps[step], source.model_timestep(schedule, step, device="cuda", dtype=torch.float32), atol=0, rtol=0)
            prediction = torch.randn_like(sample)
            native.noise_pred = prediction[0]
            expected = (
                source.step(prediction, source.timesteps[step], sample, return_dict=False)[0]
                if mode == "bidir" and variant == "5b"
                else source.step(sample, prediction, schedule, step, generator=generator)
            )
            native.step_post()
            if mode == "few_step" or variant == "5b":
                expected[:, :, :1] = 0
            difference = (native.latents.float() - expected[0].float()).abs()
            print(profile, step, difference.max().item(), difference.mean().item(), flush=True)
            torch.testing.assert_close(native.latents, expected[0], rtol=0, atol=0)
            sample = expected


if __name__ == "__main__":
    main()
