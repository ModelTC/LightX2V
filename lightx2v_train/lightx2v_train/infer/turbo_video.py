"""Fixed-seed Wan Turbo previews with the training sampler and online VAE."""

import html
import json
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from loguru import logger

from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.runtime.distributed import get_world_size, is_main_process
from lightx2v_train.trainers.turbo.sampler import sample_turbo_latents
from lightx2v_train.utils.constants import WAN_NEGATIVE_PROMPT
from lightx2v_train.utils.registry import INFERENCER_REGISTER

from .base import BaseInferencer


def _write_gallery(directory, records):
    cards = []
    for record in records:
        caption = html.escape(record["prompt"])
        filename = html.escape(Path(record["video"]).name)
        cards.append(f'<article><video controls loop preload="metadata" src="{filename}"></video><p>seed={record["seed"]}</p><p>{caption}</p></article>')
    (directory / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Wan Turbo preview</title>'
        "<style>body{font:16px system-ui;background:#161616;color:#eee;margin:24px}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}video{width:100%}p{white-space:pre-wrap}</style>"
        "<h1>Wan2.1 1.3B Turbo preview</h1><main>" + "".join(cards) + "</main>",
        encoding="utf-8",
    )


@INFERENCER_REGISTER("wan_turbo_infer")
class WanTurboInferencer(BaseInferencer):
    """Native FP32 Wan inference; trainer owns the EMA weight context."""

    @torch.no_grad()
    def _teacher_baseline(self, capability, noise, positive, negative, steps, shift):
        from lightx2v_train.model_zoo.native.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        solver = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        solver.set_timesteps(steps, device=noise.device, shift=shift)
        latent = noise
        for index, timestep in enumerate(solver.timesteps):
            sigma = solver.sigmas[index].to(device=noise.device).reshape(1)
            velocity = capability.predict_guided_velocity(latent, sigma, positive, negative, 5.0, "none")
            latent = solver.step(velocity, timestep, latent, return_dict=False)[0]
        return latent

    @torch.no_grad()
    def infer(self):
        if self.model.running_dtype != torch.float32 or self.model.vae_dtype != torch.float32:
            raise ValueError("Wan FP32 previews require FP32 model compute and VAE.")
        capability = self.model.ensure_capabilities().require(DistributionMatchingCapability)
        self.model.set_denoiser_eval()
        directory = Path(self.output_infer_dir)
        directory.mkdir(parents=True, exist_ok=True)
        steps = int(self.infer_config.get("num_inference_steps", 8))
        shift = float(self.config["training"]["turbo"]["video_shift"])
        height = int(self.infer_config.get("default_height", 480))
        width = int(self.infer_config.get("default_width", 832))
        fps = int(self.infer_config.get("fps", 16))
        baseline = bool(self.infer_config.get("teacher_baseline_at_zero", False)) and directory.name == "iter-000000000"
        device = self.model.device
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        records = []
        saved_paths = []
        for index, sample in enumerate(self.dataloader_val.dataset.samples):
            prompt = sample["prompt"]
            seed = int(self.infer_config.get("seed", 42)) + index
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.default_generator.manual_seed(seed)
                for cuda_index in cuda_devices:
                    torch.cuda.default_generators[cuda_index].manual_seed(seed)
                positive = self.model.encode_prompt_condition(prompt)
                initial = self.model.prepare_infer_latents(height, width)
                logger.info("[turbo-infer] sample={}/{} seed={} steps={} shift={} fp32 output={}", index + 1, len(self.dataloader_val.dataset.samples), seed, steps, shift, directory)
                latent = sample_turbo_latents(capability, initial, positive, num_steps=steps, video_shift=shift, with_grad=False)
                # FSDP ranks execute every prompt and denoising call together.
                # Decode locally on all ranks; only rank zero publishes files.
                video = self.model.decode_latent(latent)[0]
                destination = directory / f"{index:05d}.mp4"
                if is_main_process():
                    export_to_video(video, str(destination), fps=fps, quality=6, macro_block_size=16)
                    saved_paths.append(str(destination))
                records.append(
                    {
                        "video": str(destination),
                        "prompt": prompt,
                        "seed": seed,
                        "steps": steps,
                        "shift": shift,
                        "fps": fps,
                        "num_frames": int(self.infer_config.get("num_frames", 81)),
                        "height": height,
                        "width": width,
                        "sampler": "turbo_trigflow_stochastic_x0_renoise",
                        "guidance_scale": 1.0,
                        "weights": "base" if directory.name == "iter-000000000" else "student_ema",
                        "model_dtype": "fp32",
                        "vae_dtype": "fp32",
                        "t5_dtype": "fp32",
                        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                        "sampling_world_size": get_world_size(),
                    }
                )
                if baseline:
                    negative = self.model.encode_prompt_condition(WAN_NEGATIVE_PROMPT)
                    teacher = self._teacher_baseline(capability, initial, positive, negative, 50, shift)
                    teacher_path = directory / f"{index:05d}_teacher50_cfg5.mp4"
                    teacher_video = self.model.decode_latent(teacher)[0]
                    if is_main_process():
                        export_to_video(teacher_video, str(teacher_path), fps=fps, quality=6, macro_block_size=16)
                        saved_paths.append(str(teacher_path))
                    records.append({**records[-1], "video": str(teacher_path), "steps": 50, "guidance_scale": 5.0, "weights": "base", "sampler": "flow_unipc"})
                del positive, initial, latent, video
        if is_main_process():
            (directory / "manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _write_gallery(directory, records)
        logger.info("[turbo-infer] finished saved={} gallery={}", len(saved_paths), directory / "index.html")
        return saved_paths
