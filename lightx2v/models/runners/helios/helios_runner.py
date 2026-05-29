import gc
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.input_encoders.hf.helios import HeliosTextEncoder
from lightx2v.models.networks.helios import HeliosModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.helios import HeliosDistilledScheduler
from lightx2v.models.video_encoders.hf.helios import HeliosVAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_RECORDER_MODE
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def randn_tensor(shape, generator=None, device=None, dtype=None):
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
    if isinstance(generator, list):
        shape1 = (1,) + tuple(shape[1:])
        return torch.cat([torch.randn(shape1, generator=g, device=device, dtype=dtype) for g in generator], dim=0)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def _apply_image_condition_noise(
    image_latents,
    fake_image_latents,
    generator,
    device,
    image_noise_sigma_min,
    image_noise_sigma_max,
    video_noise_sigma_min,
    video_noise_sigma_max,
):
    image_noise_sigma = torch.rand(1, device=device, generator=generator) * (image_noise_sigma_max - image_noise_sigma_min) + image_noise_sigma_min
    image_latents = image_noise_sigma * torch.randn(image_latents.shape, generator=generator, device=device) + (1 - image_noise_sigma) * image_latents
    fake_image_noise_sigma = torch.rand(1, device=device, generator=generator) * (video_noise_sigma_max - video_noise_sigma_min) + video_noise_sigma_min
    fake_image_latents = fake_image_noise_sigma * torch.randn(fake_image_latents.shape, generator=generator, device=device) + (1 - fake_image_noise_sigma) * fake_image_latents
    return image_latents, fake_image_latents


def _trim_generated_frames(frame_count, temporal_scale_factor):
    return ((frame_count - 1) // temporal_scale_factor) * temporal_scale_factor + 1


def _finalize_video_output(history_video, video_processor, temporal_scale_factor, output_type="pt"):
    generated_frames = _trim_generated_frames(history_video.size(2), temporal_scale_factor)
    history_video = history_video[:, :, :generated_frames]
    return video_processor.postprocess_video(history_video, output_type=output_type)


def _pt_video_output_to_frames(video):
    if video.dim() != 5:
        raise ValueError(f"Expected [B, T, C, H, W] tensor, got shape {tuple(video.shape)}")
    return video.permute(0, 1, 3, 4, 2).flatten(0, 1).cpu()


@RUNNER_REGISTER("helios_distilled")
class HeliosRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        self.keep_first_frame = self.config.get("keep_first_frame", True)
        self.request_generator = None

    def set_inputs(self, inputs):
        self.request_generator = None
        super().set_inputs(inputs)

    def end_run(self):
        self.request_generator = None
        super().end_run()

    def get_request_generator(self):
        if self.request_generator is None:
            self.request_generator = torch.Generator(device=AI_DEVICE).manual_seed(self.input_info.seed)
        return self.request_generator

    def init_scheduler(self):
        self.scheduler = HeliosDistilledScheduler(self.config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae_encoder, self.vae_decoder = self.load_vae()

    def load_transformer(self):
        return HeliosModel(self.config["model_path"], self.config, self.init_device)

    def load_text_encoder(self):
        return [HeliosTextEncoder(self.config)]

    def load_image_encoder(self):
        return None

    def load_vae(self):
        vae = HeliosVAE(self.config)
        return vae, vae

    def init_modules(self):
        if self.config["task"] not in ["t2v", "i2v"]:
            raise NotImplementedError(f"HeliosRunner only supports t2v/i2v, got {self.config['task']}")
        if self.config.get("lazy_load"):
            raise NotImplementedError("Helios native integration does not support lazy_load.")
        if self.config.get("unload_modules"):
            raise NotImplementedError("Helios native integration does not support unload_modules.")
        if self.config.get("cpu_offload"):
            raise NotImplementedError("Helios native integration does not support generic cpu_offload.")
        if self.config.get("compile"):
            raise NotImplementedError("Helios native integration does not support compile yet.")
        if self.config.get("enable_low_vram_mode"):
            raise NotImplementedError("Helios native integration does not support group offload yet.")
        if self.config.get("enable_parallelism"):
            raise NotImplementedError("Helios native integration does not support context parallelism yet.")
        super().init_modules()

    def get_latent_shape_with_target_hw(self):
        target_height = self.input_info.target_shape[0] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_height"]
        target_width = self.input_info.target_shape[1] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_width"]
        return [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            int(target_height) // self.config["vae_stride"][1],
            int(target_width) // self.config["vae_stride"][2],
        ]

    @ProfilingContext4DebugL1(
        "Run Text Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_text_encode_duration,
        metrics_labels=["HeliosRunner"],
    )
    def run_text_encoder(self, input_info):
        prompt = input_info.prompt_enhanced if self.config["use_prompt_enhancer"] else input_info.prompt
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_prompt_len.observe(len(prompt))
        prompt_embeds, _ = self.text_encoders[0].infer([prompt], max_sequence_length=self.config.get("max_sequence_length", 512))
        negative_prompt_embeds = None
        if self.config.get("enable_cfg", False):
            negative_prompt_embeds, _ = self.text_encoders[0].infer(
                [input_info.negative_prompt or ""],
                max_sequence_length=self.config.get("max_sequence_length", 512),
            )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2v(self):
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2v(self):
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
        text_encoder_output = self.run_text_encoder(self.input_info)
        generator = self.get_request_generator()
        image_latents, fake_image_latents = self.vae_encoder.prepare_image_latents(
            self.input_info.image_path,
            generator=generator,
            num_latent_frames_per_chunk=self.config.get("num_latent_frames_per_chunk", 9),
            height=self.config["target_height"],
            width=self.config["target_width"],
            dtype=torch.float32,
        )
        image_latents, fake_image_latents = _apply_image_condition_noise(
            image_latents=image_latents,
            fake_image_latents=fake_image_latents,
            generator=generator,
            device=torch.device(AI_DEVICE),
            image_noise_sigma_min=self.config.get("image_noise_sigma_min", 0.111),
            image_noise_sigma_max=self.config.get("image_noise_sigma_max", 0.135),
            video_noise_sigma_min=self.config.get("video_noise_sigma_min", 0.111),
            video_noise_sigma_max=self.config.get("video_noise_sigma_max", 0.135),
        )
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "image_latents": image_latents,
                "fake_image_latents": fake_image_latents,
            },
        }

    def sample_block_noise(self, batch_size, channel, num_frames, height, width, patch_size, device, generator):
        gamma = self.scheduler.inner.config.gamma
        _, ph, pw = patch_size
        block_size = ph * pw
        cov = torch.eye(block_size, device=device) * (1 + gamma) - torch.ones(block_size, block_size, device=device) * gamma
        cov += torch.eye(block_size, device=device) * 1e-8
        L = torch.linalg.cholesky(cov.float())
        block_number = batch_size * channel * num_frames * (height // ph) * (width // pw)
        z = torch.randn(block_number, block_size, generator=generator, device=device)
        noise = z @ L.T
        noise = noise.view(batch_size, channel, num_frames, height // ph, width // pw, ph, pw)
        return noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)

    def _prepare_latents(self, batch_size, num_channels_latents, height, width, num_frames, generator, dtype, device):
        num_latent_frames = (num_frames - 1) // self.vae_encoder.vae_scale_factor_temporal + 1
        shape = (batch_size, num_channels_latents, num_latent_frames, int(height) // self.vae_encoder.vae_scale_factor_spatial, int(width) // self.vae_encoder.vae_scale_factor_spatial)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.get_video_segment_num()
        self.model.set_scheduler(self.scheduler)
        self.scheduler.prepare(
            seed=self.input_info.seed,
            latent_shape=self.input_info.latent_shape,
            image_encoder_output=self.inputs["image_encoder_output"],
            generator=self.get_request_generator(),
        )

        prompt_embeds = self.inputs["text_encoder_output"]["prompt_embeds"].to(self.model.dtype)
        negative_prompt_embeds = self.inputs["text_encoder_output"].get("negative_prompt_embeds")
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.model.dtype)

        image_latents = None
        fake_image_latents = None
        if self.inputs["image_encoder_output"] is not None:
            image_latents = self.inputs["image_encoder_output"]["image_latents"].to(torch.float32)
            fake_image_latents = self.inputs["image_encoder_output"]["fake_image_latents"].to(torch.float32)

        batch_size = prompt_embeds.shape[0]
        device = torch.device(AI_DEVICE)
        transformer_dtype = self.model.dtype
        num_channels_latents = self.model.transformer.config.in_channels
        height = self.config["target_height"]
        width = self.config["target_width"]
        target_video_length = self.config["target_video_length"]
        history_sizes = sorted(self.config.get("history_sizes", [16, 2, 1]), reverse=True)
        num_latent_frames_per_chunk = self.config.get("num_latent_frames_per_chunk", 9)
        pyramid_num_inference_steps_list = self.config.get("pyramid_num_inference_steps_list", [2, 2, 2])
        guidance_scale = self.config.get("sample_guide_scale", 1.0)
        use_zero_init = self.config.get("use_zero_init", False)
        zero_steps = self.config.get("zero_steps", 1)
        is_skip_first_chunk = self.config.get("is_skip_first_chunk", False)
        is_amplify_first_chunk = self.config.get("is_amplify_first_chunk", False)
        attention_kwargs = None

        window_num_frames = (num_latent_frames_per_chunk - 1) * self.vae_encoder.vae_scale_factor_temporal + 1
        num_latent_chunk = max(1, (target_video_length + window_num_frames - 1) // window_num_frames)
        num_history_latent_frames = sum(history_sizes)
        history_video = None
        total_generated_latent_frames = 0

        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            num_history_latent_frames,
            height // self.vae_encoder.vae_scale_factor_spatial,
            width // self.vae_encoder.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
        if fake_image_latents is not None:
            history_latents = torch.cat([history_latents[:, :, :-1, :, :], fake_image_latents.to(device)], dim=2)
            total_generated_latent_frames += 1

        if self.keep_first_frame:
            indices = torch.arange(0, sum([1, *history_sizes, num_latent_frames_per_chunk]), device=device)
            (
                indices_prefix,
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_1x,
                indices_hidden_states,
            ) = indices.split([1, *history_sizes, num_latent_frames_per_chunk], dim=0)
            indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)
        else:
            indices = torch.arange(0, sum([*history_sizes, num_latent_frames_per_chunk]), device=device)
            (
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_short,
                indices_hidden_states,
            ) = indices.split([*history_sizes, num_latent_frames_per_chunk], dim=0)
        indices_hidden_states = indices_hidden_states.unsqueeze(0)
        indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
        indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
        indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

        for chunk_idx in range(num_latent_chunk):
            is_first_chunk = chunk_idx == 0
            is_second_chunk = chunk_idx == 1
            if self.keep_first_frame:
                latents_history_long, latents_history_mid, latents_history_1x = history_latents[:, :, -num_history_latent_frames:].split(history_sizes, dim=2)
                if image_latents is None and is_first_chunk:
                    latents_prefix = torch.zeros((batch_size, num_channels_latents, 1, latents_history_1x.shape[-2], latents_history_1x.shape[-1]), device=device, dtype=latents_history_1x.dtype)
                else:
                    latents_prefix = image_latents.to(device)
                latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                latents_history_long, latents_history_mid, latents_history_short = history_latents[:, :, -num_history_latent_frames:].split(history_sizes, dim=2)

            latents = self._prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=window_num_frames,
                generator=self.scheduler.generator,
                dtype=torch.float32,
                device=device,
            )
            num_inference_steps = sum(pyramid_num_inference_steps_list) * 2 if is_amplify_first_chunk and is_first_chunk else sum(pyramid_num_inference_steps_list)
            _, _, _, pyramid_height, pyramid_width = latents.shape
            latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_latent_frames_per_chunk, num_channels_latents, pyramid_height, pyramid_width)
            for _ in range(len(pyramid_num_inference_steps_list) - 1):
                pyramid_height //= 2
                pyramid_width //= 2
                latents = F.interpolate(latents, size=(pyramid_height, pyramid_width), mode="bilinear") * 2
            latents = latents.reshape(batch_size, num_latent_frames_per_chunk, num_channels_latents, pyramid_height, pyramid_width).permute(0, 2, 1, 3, 4)
            start_point_list = [latents]
            completed_steps = 0

            for stage_idx, stage_steps in enumerate(pyramid_num_inference_steps_list):
                patch_size = self.model.transformer.config.patch_size
                image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (patch_size[0] * patch_size[1] * patch_size[2])
                mu = calculate_shift(
                    image_seq_len,
                    self.scheduler.inner.config.get("base_image_seq_len", 256),
                    self.scheduler.inner.config.get("max_image_seq_len", 4096),
                    self.scheduler.inner.config.get("base_shift", 0.5),
                    self.scheduler.inner.config.get("max_shift", 1.15),
                )
                self.scheduler.set_timesteps(stage_steps, stage_idx, device=device, mu=mu, is_amplify_first_chunk=is_amplify_first_chunk and is_first_chunk)
                timesteps = self.scheduler.timesteps

                if stage_idx > 0:
                    pyramid_height *= 2
                    pyramid_width *= 2
                    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_latent_frames_per_chunk, num_channels_latents, pyramid_height // 2, pyramid_width // 2)
                    latents = F.interpolate(latents, size=(pyramid_height, pyramid_width), mode="nearest")
                    latents = latents.reshape(batch_size, num_latent_frames_per_chunk, num_channels_latents, pyramid_height, pyramid_width).permute(0, 2, 1, 3, 4)
                    ori_sigma = 1 - self.scheduler.inner.ori_start_sigmas[stage_idx]
                    gamma = self.scheduler.inner.config.gamma
                    alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                    beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)
                    noise = self.sample_block_noise(batch_size, num_channels_latents, latents.shape[2], pyramid_height, pyramid_width, patch_size, device, self.scheduler.generator).to(
                        dtype=transformer_dtype
                    )
                    latents = alpha * latents + beta * noise
                    start_point_list.append(latents)

                for step_idx, timestep_scalar in enumerate(timesteps):
                    timestep = timestep_scalar.expand(latents.shape[0]).to(torch.int64)
                    history_inputs = {
                        "indices_hidden_states": indices_hidden_states,
                        "indices_latents_history_short": indices_latents_history_short,
                        "indices_latents_history_mid": indices_latents_history_mid,
                        "indices_latents_history_long": indices_latents_history_long,
                        "latents_history_short": latents_history_short.to(transformer_dtype),
                        "latents_history_mid": latents_history_mid.to(transformer_dtype),
                        "latents_history_long": latents_history_long.to(transformer_dtype),
                    }
                    noise_pred = self.model.infer_cfg(
                        latents.to(transformer_dtype),
                        timestep,
                        prompt_embeds,
                        negative_prompt_embeds,
                        history_inputs,
                        guidance_scale=guidance_scale,
                        attention_kwargs=attention_kwargs,
                        is_cfg_zero_star=self.config.get("is_cfg_zero_star", False),
                        use_zero_init=use_zero_init,
                        zero_steps=zero_steps,
                        stage_idx=stage_idx,
                        step_idx=step_idx,
                    )
                    latents = self.scheduler.step(
                        noise_pred,
                        timestep_scalar,
                        latents,
                        generator=self.scheduler.generator,
                        return_dict=False,
                        cur_sampling_step=step_idx,
                        dmd_noisy_tensor=start_point_list[stage_idx],
                        dmd_sigmas=self.scheduler.sigmas,
                        dmd_timesteps=self.scheduler.timesteps,
                        all_timesteps=timesteps,
                    )[0]
                    completed_steps += 1
                    if self.progress_callback:
                        self.progress_callback((completed_steps / num_inference_steps) * 100, 100)

            if self.keep_first_frame and ((is_first_chunk and image_latents is None) or (is_skip_first_chunk and is_second_chunk)):
                image_latents = latents[:, :, 0:1, :, :]

            total_generated_latent_frames += latents.shape[2]
            history_latents = torch.cat([history_latents, latents], dim=2)
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:]
            current_latents = real_history_latents[:, :, -num_latent_frames_per_chunk:]
            current_video = self.vae_decoder.decode(current_latents)
            history_video = current_video if history_video is None else torch.cat([history_video, current_video], dim=2)

        self.gen_video = history_video
        self.gen_video_final = _pt_video_output_to_frames(
            _finalize_video_output(
                history_video=self.gen_video,
                video_processor=self.vae_decoder.video_processor,
                temporal_scale_factor=self.vae_decoder.vae_scale_factor_temporal,
                output_type="pt",
            )
        )
        result = self.process_images_after_vae_decoder_helios()
        self.end_run()
        return result

    def process_images_after_vae_decoder_helios(self):
        if "video_frame_interpolation" in self.config:
            assert self.vfi_model is not None and self.config["video_frame_interpolation"].get("target_fps", None) is not None
            target_fps = self.config["video_frame_interpolation"]["target_fps"]
            logger.info(f"Interpolating frames from {self.config.get('fps', 16)} to {target_fps}")
            self.gen_video_final = self.vfi_model.interpolate_frames(
                self.gen_video_final,
                source_fps=self.config.get("fps", 16),
                target_fps=target_fps,
            )

        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final}
        elif self.input_info.save_result_path is not None:
            fps = (
                self.config["video_frame_interpolation"]["target_fps"]
                if "video_frame_interpolation" in self.config and self.config["video_frame_interpolation"].get("target_fps")
                else self.config.get("fps", 16)
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                out_path = self.input_info.save_result_path
                logger.info("🎬 Start to save video 🎬")
                from lightx2v.utils.utils import save_to_video

                save_to_video(self.gen_video_final, out_path, fps=fps, method="ffmpeg")
                logger.info(f"✅ Video saved successfully to: {out_path} ✅")
            return {"video": None}
