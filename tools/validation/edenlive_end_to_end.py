"""Save upstream/native inference fixtures and videos for parity inspection."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["upstream", "native"], required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--converted-path", required=True)
    parser.add_argument("--wan-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="wan22_5b_few_step")
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--capture-dit", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, args.upstream)
    root = Path(args.upstream)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    captures = {}
    if args.sp:
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    if args.backend == "upstream":
        from edenlive.core.config import load_config
        from edenlive.inference.offline import run_offline_inference
        from edenlive.models.adapters.wan22_diffusers.adapter import Wan22DiffusersAdapter

        config = load_config(root / "edenlive/configs" / f"{args.profile.replace('5b_', '5b_720p_').replace('14b_', '14b_480p_')}.yaml")
        config["model"]["pretrained_dir"] = args.model_path
        config["data"]["dataset"]["params"]["max_frames"] = args.frames
        config["inference"]["max_batches"] = 1
        config["inference"]["seed"] = 2026092901 if "few_step" in args.profile else 42
        config["inference"]["output_dir"] = str(output)
        config["runtime"]["sequence_parallel"]["size"] = 1
        config["runtime"]["fsdp2"]["enabled"] = False
        # Use SDPA for the bidirectional validation fixture without requiring FA4.
        if not config["inference"]["strategy"] == "chunk_causal":
            config["model"]["teacher_forcing_attention_backend"] = "triton"
            if config["model"].get("dynamic_prompt", {}).get("enabled"):
                config["model"]["dynamic_prompt"]["enabled"] = False
                config["data"]["dataset"]["params"]["text_mode"] = "global"
        for name in ["encode_prompts", "encode_clip_image_embeds", "encode_image_condition", "encode_proxy_dino_condition", "prepare_generation_latents", "decode_video_latents"]:
            original = getattr(Wan22DiffusersAdapter, name)

            def capture(self, *values, _name=name, _original=original, **kwargs):
                result = _original(self, *values, **kwargs)
                tensor = values[0] if _name == "decode_video_latents" else result
                if isinstance(tensor, torch.Tensor):
                    captures.setdefault(_name, []).append(tensor.detach().cpu())
                return result

            setattr(Wan22DiffusersAdapter, name, capture)
        if args.capture_dit:
            original_forward = Wan22DiffusersAdapter.forward_dit_causal_cached

            def forward(self, latents, timesteps, *values, **kwargs):
                result = original_forward(self, latents, timesteps, *values, **kwargs)
                if len(captures.get("dit_prediction", [])) < 5:
                    for key, tensor in (("dit_input", latents), ("dit_timestep", timesteps), ("dit_prediction", result)):
                        captures.setdefault(key, []).append(tensor.detach().cpu())
                return result

            Wan22DiffusersAdapter.forward_dit_causal_cached = forward
        os.chdir(root)
        run_offline_inference(config)
    else:
        from lightx2v.models.runners.runner_factory import build_runner
        from lightx2v.models.runners.wan.edenlive_runner import EdenLiveRunner
        from lightx2v.utils.set_config import build_startup_config

        if args.capture_dit:
            from lightx2v.models.networks.wan.edenlive_model import EdenLiveModel

            original_forward = EdenLiveModel._infer_cond_uncond

            def forward(self, inputs, infer_condition=True):
                result = original_forward(self, inputs, infer_condition)
                if len(captures.get("dit_prediction", [])) < 5:
                    for key, tensor in (("dit_input", self.scheduler.latents_input[None]), ("dit_timestep", self.scheduler.timestep_input), ("dit_prediction", result[None])):
                        captures.setdefault(key, []).append(tensor.detach().cpu())
                return result

            EdenLiveModel._infer_cond_uncond = forward
        original = EdenLiveRunner.run_vae_decoder

        def decode(self, latent):
            captures["decode_video_latents"] = [latent.detach().cpu()]
            return original(self, latent)

        EdenLiveRunner.run_vae_decoder = decode
        original_text = EdenLiveRunner.run_text_encoder

        def text(self, info):
            result = original_text(self, info)
            captures.setdefault("encode_prompts", []).append(result["context"].detach().cpu())
            return result

        EdenLiveRunner.run_text_encoder = text
        original_prepare = EdenLiveRunner._prepare_chunk_inputs

        def chunk(self, start, frames):
            original_prepare(self, start, frames)
            captures.setdefault("encode_proxy_dino_condition", []).append(self.inputs["proxy_features"].detach().cpu())

        EdenLiveRunner._prepare_chunk_inputs = chunk
        original_input = EdenLiveRunner._run_input_encoder_local_i2v

        def encode(self):
            result = original_input(self)
            condition = result["image_encoder_output"]
            image = condition["vae_encoder_out"] if condition["vae_encoder_out"] is not None else condition["clean_first_frame"]
            captures["encode_image_condition"] = [image.detach().cpu()[None]]
            if condition["clip_encoder_out"] is not None:
                captures["encode_clip_image_embeds"] = [condition["clip_encoder_out"].detach().cpu()[None]]
            return result

        EdenLiveRunner._run_input_encoder_local_i2v = encode
        variant = "5b" if "5b" in args.profile else "14b"
        model_cls = args.profile.split("_")[0] + "_edenlive" + ("_ar" if args.profile.endswith("_few_step") else "")
        config = build_startup_config(
            dict(
                model_cls=model_cls,
                task="i2v",
                model_path=args.model_path,
                config_json=str(Path(__file__).resolve().parents[2] / "configs/edenlive" / f"{args.profile}.json"),
            )
        )
        config.update(
            dict(
                dit_original_ckpt=args.converted_path,
                t5_original_ckpt=str(Path(args.wan_path) / "models_t5_umt5-xxl-enc-bf16.pth"),
                t5_tokenizer_path=str(Path(args.model_path) / "umt5/tokenizer"),
                vae_path=str(Path(args.model_path) / f"vae-{variant}" / f"Wan2.{2 if variant == '5b' else 1}_VAE.pth"),
                clip_original_ckpt=str(Path(args.wan_path) / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth") if variant == "14b" else None,
            )
        )
        if args.sp:
            from lightx2v.utils.set_config import init_parallel

            config["parallel"] = {"seq_p_size": dist.get_world_size(), "vae_parallel": False}
            init_parallel(config)
        runner = build_runner(config)
        sample = root / "examples/samples/cryo-crossing"
        metadata = json.loads((sample / "prompts.json").read_text())
        prompt = str(sample / "prompts.json") if "few_step" in args.profile else metadata["staticPrompt"]
        request = dict(
            task="i2v",
            prompt=prompt,
            image_path=str(sample / "cover.png"),
            video_path=str(sample / "recording.mp4"),
            num_frames=args.frames,
            seed=2026092901 if "few_step" in args.profile else 42,
            save_result_path=str(output / "native.mp4"),
        )
        if config["enable_cfg"]:
            import yaml

            profile = yaml.safe_load((root / "edenlive/configs" / f"{args.profile.replace('5b_', '5b_720p_').replace('14b_', '14b_480p_')}.yaml").read_text())
            request["negative_prompt"] = profile["inference"].get("negative_prompt", profile["model"].get("negative_prompt", ""))
        runner.run_request(runner.prepare_request(request))
    if not args.sp or dist.get_rank() == 0:
        torch.save(captures, output / "boundaries.pt")
    if args.sp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
