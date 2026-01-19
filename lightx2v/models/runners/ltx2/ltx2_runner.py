import gc

import torch

from lightx2v.models.input_encoders.hf.ltx2.model import LTX2TextEncoder
from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.ltx2.scheduler import LTX2Scheduler
from lightx2v.models.video_encoders.hf.ltx2.model import LTX2AudioVAE, LTX2VideoVAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.ltx2_media_io import encode_video as save_video
from lightx2v.utils.ltx2_media_io import load_image_conditioning
from lightx2v.utils.memory_profiler import peak_memory_decorator
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("ltx2")
class LTX2Runner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    def init_scheduler(self):
        self.scheduler = LTX2Scheduler(self.config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.video_vae, self.audio_vae = self.load_vae()

    def load_transformer(self):
        wan_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = LTX2Model(**wan_model_kwargs)
        else:
            raise NotImplementedError
        return model

    def load_text_encoder(self):
        # offload config
        text_encoder_offload = self.config.get("gemma_cpu_offload", self.config.get("cpu_offload", False))
        if text_encoder_offload:
            text_encoder_device = torch.device("cpu")
        else:
            text_encoder_device = torch.device(AI_DEVICE)

        if "dit_original_ckpt" in self.config:
            ckpt_path = self.config["dit_original_ckpt"]
        elif "dit_quantized_ckpt" in self.config:
            ckpt_path = self.config["dit_quantized_ckpt"]
        else:
            ckpt_path = os.path.join(self.config["model_path"], "transformer")

        if "gemma_original_ckpt" in self.config:
            gemma_ckpt = self.config["gemma_original_ckpt"]
        else:
            gemma_ckpt = self.config["model_path"]

        text_encoder = LTX2TextEncoder(
            checkpoint_path=ckpt_path,
            gemma_root=gemma_ckpt,
            device=text_encoder_device,
            dtype=torch.bfloat16,
        )
        text_encoders = [text_encoder]
        return text_encoders

    def load_vae(self):
        """Load video and audio VAE decoders."""
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device(AI_DEVICE)

        if "dit_original_ckpt" in self.config:
            ckpt_path = self.config["dit_original_ckpt"]
        elif "dit_quantized_ckpt" in self.config:
            ckpt_path = self.config["dit_quantized_ckpt"]
        else:
            ckpt_path = os.path.join(self.config["model_path"], "transformer")

        # Video VAE
        video_vae = LTX2VideoVAE(checkpoint_path=ckpt_path, device=vae_device, dtype=GET_DTYPE(), load_encoder=self.config["task"] == "i2av")

        # Audio VAE
        audio_vae = LTX2AudioVAE(
            checkpoint_path=ckpt_path,
            device=vae_device,
            dtype=GET_DTYPE(),
        )

        return video_vae, audio_vae

    def get_latent_shape_with_target_hw(self):
        target_height = self.input_info.target_shape[0] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_height"]
        target_width = self.input_info.target_shape[1] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_width"]
        video_latent_shape = (
            self.config.get("num_channels_latents", 128),
            (self.config["target_video_length"] - 1) // self.config["vae_scale_factors"][0] + 1,
            int(target_height) // self.config["vae_scale_factors"][1],
            int(target_width) // self.config["vae_scale_factors"][2],
        )

        duration = float(self.config["target_video_length"]) / float(self.config["fps"])
        latents_per_second = float(self.config["audio_sampling_rate"]) / float(self.config["audio_hop_length"]) / float(self.config["audio_scale_factor"])
        audio_frames = round(duration * latents_per_second)

        audio_latent_shape = (
            8,
            audio_frames,
            self.config["audio_mel_bins"],
        )

        return video_latent_shape, audio_latent_shape

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2av(self):
        self.video_denoise_mask = None
        self.initial_video_latent = None
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()  # Important: set latent_shape in input_info
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch_device_module.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i2av(self):
        self.input_info.video_latent_shape, self.input_info.audio_latent_shape = self.get_latent_shape_with_target_hw()
        text_encoder_output = self.run_text_encoder(self.input_info)
        # Prepare image conditioning if provided
        logger.info(f"🖼️  I2AV mode: processing {len(self.input_info.images)} image conditioning(s)")
        self.video_denoise_mask, self.initial_video_latent = self._prepare_image_conditioning()
        torch_device_module.empty_cache()
        gc.collect()

        return {
            "text_encoder_output": text_encoder_output,
        }

    def _prepare_image_conditioning(self):
        """
        Prepare image conditioning by loading images and encoding them to latents.

        Returns:
            tuple: (video_denoise_mask, initial_video_latent)
                - video_denoise_mask: Mask indicating which frames to denoise (unpatchified, shape [1, F, H, W])
                - initial_video_latent: Initial latent with conditioned frames (unpatchified, shape [C, F, H, W])
        """
        logger.info(f"🖼️  Preparing {len(self.input_info.images)} image conditioning(s)")

        # Get latent shape
        C, F, H, W = self.input_info.video_latent_shape
        target_height = self.input_info.target_shape[0] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_height"]
        target_width = self.input_info.target_shape[1] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_width"]

        # Initialize denoise mask (1 = denoise, 0 = keep original)
        # Shape: [1, F, H, W]
        video_denoise_mask = torch.ones(
            1,
            F,
            H,
            W,
            dtype=torch.float32,
            device=AI_DEVICE,
        )

        # Initialize initial latent as zeros
        initial_video_latent = torch.zeros(
            C,
            F,
            H,
            W,
            dtype=GET_DTYPE(),
            device=AI_DEVICE,
        )

        # Process each image conditioning
        for image_path, frame_idx, strength in self.input_info.images:
            logger.info(f"  📷 Loading image: {image_path} for frame {frame_idx} with strength {strength}")

            # Load and preprocess image
            image = load_image_conditioning(
                image_path=image_path,
                height=target_height,
                width=target_width,
                dtype=GET_DTYPE(),
                device=AI_DEVICE,
            )

            # Encode image to latent space
            # image shape: [1, C, 1, H, W]
            with torch.no_grad():
                encoded_latent = self.video_vae.encode(image)

            # Remove batch dimension: [1, C, 1, H_latent, W_latent] -> [C, 1, H_latent, W_latent]
            encoded_latent = encoded_latent.squeeze(0)

            # Verify frame index is valid
            if frame_idx < 0 or frame_idx >= F:
                logger.warning(f"⚠️  Frame index {frame_idx} out of range [0, {F - 1}], skipping")
                continue

            # Get the latent frame index by converting pixel frame to latent frame
            # For LTX2, temporal compression is 8x, so latent_frame_idx = (frame_idx - 1) // 8 + 1 for frame_idx > 0
            # or 0 for frame_idx == 0
            if frame_idx == 0:
                latent_frame_idx = 0
            else:
                latent_frame_idx = (frame_idx - 1) // self.config["vae_scale_factors"][0] + 1

            if latent_frame_idx >= F:
                logger.warning(f"⚠️  Latent frame index {latent_frame_idx} out of range [0, {F - 1}], skipping")
                continue

            # Set the latent at the specified frame
            # encoded_latent shape: [C, 1, H_latent, W_latent]
            initial_video_latent[:, latent_frame_idx : latent_frame_idx + 1, :, :] = encoded_latent

            # Update denoise mask based on strength
            # strength = 1.0 means keep original (don't denoise)
            # strength = 0.0 means fully denoise
            video_denoise_mask[:, latent_frame_idx, :, :] = 1.0 - strength

            logger.info(f"  ✓ Encoded image to latent frame {latent_frame_idx}")

        torch_device_module.empty_cache()
        gc.collect()

        logger.info(f"✓ Image conditioning prepared successfully")

        return video_denoise_mask, initial_video_latent

    @ProfilingContext4DebugL1(
        "Run Text Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_text_encode_duration,
        metrics_labels=["WanRunner"],
    )
    def run_text_encoder(self, input_info):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()

        prompt = input_info.prompt
        neg_prompt = input_info.negative_prompt

        v_context_p, a_context_p, v_context_n, a_context_n = self.text_encoders[0].infer(
            prompt=prompt,
            negative_prompt=neg_prompt,
        )

        text_encoder_output = {
            "v_context_p": v_context_p,
            "a_context_p": a_context_p,
            "v_context_n": v_context_n,
            "a_context_n": a_context_n,
        }

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.text_encoders[0]
            torch_device_module.empty_cache()
            gc.collect()

        return text_encoder_output

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["LTX2Runner"],
    )
    def run_vae_decoder(self, v_latent, a_latent):
        """Decode video and audio latents to frames and waveform."""
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.video_vae, self.audio_vae = self.load_vae()

        # Decode video latents (returns iterator)
        video = self.video_vae.decode(v_latent.unsqueeze(0).to(GET_DTYPE()))
        # Decode audio latents
        audio = self.audio_vae.decode(a_latent.unsqueeze(0).to(GET_DTYPE()))

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.video_vae
            del self.audio_vae
            torch_device_module.empty_cache()
            gc.collect()

        return video, audio

    def init_run(self):
        self.gen_video_final = None
        self.get_video_segment_num()

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        # Image conditioning (if any) is already prepared in run_input_encoder
        # and stored in self.video_denoise_mask and self.initial_video_latent
        self.model.scheduler.prepare(
            seed=self.input_info.seed,
            video_latent_shape=self.input_info.video_latent_shape,
            audio_latent_shape=self.input_info.audio_latent_shape,
            initial_video_latent=self.initial_video_latent,
            video_denoise_mask=self.video_denoise_mask,
        )

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        if self.config.get("compile", False) and hasattr(self.model, "comple"):
            self.model.select_graph_for_compile(self.input_info)
        for segment_idx in range(self.video_segment_num):
            logger.info(f"🔄 start segment {segment_idx + 1}/{self.video_segment_num}")
            with ProfilingContext4DebugL1(
                f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                metrics_labels=["DefaultRunner"],
            ):
                self.check_stop()
                # 1. default do nothing
                self.init_run_segment(segment_idx)
                # 2. main inference loop
                v_latent, a_latent = self.run_segment(segment_idx)
                # 3. vae decoder
                self.gen_video, self.gen_audio = self.run_vae_decoder(v_latent, a_latent)

                # 4. default do nothing
                self.end_run_segment(segment_idx)
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    def end_run_segment(self, segment_idx=None):
        self.gen_video_final = self.gen_video
        self.gen_audio_final = self.gen_audio

    def process_images_after_vae_decoder(self):
        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final, "audio": self.gen_audio_final}
        elif self.input_info.save_result_path is not None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(f"🎬 Start to save video 🎬")
                save_video(
                    video=self.gen_video_final,
                    fps=self.config.get("fps", 24),
                    audio=self.gen_audio_final,
                    audio_sample_rate=self.config.get("audio_fps", 24000),
                    output_path=self.input_info.save_result_path,
                    video_chunks_number=1,
                )
                logger.info(f"✅ Video saved successfully to: {self.input_info.save_result_path} ✅")
            return {"video": None}

    @peak_memory_decorator
    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps

        for step_index in range(infer_steps):
            # only for single segment, check stop signal every step
            with ProfilingContext4DebugL1(
                f"Run Dit every step",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_per_step_dit_duration,
                metrics_labels=[step_index + 1, infer_steps],
            ):
                if self.video_segment_num == 1:
                    self.check_stop()
                logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")

                with ProfilingContext4DebugL1("step_pre"):
                    self.model.scheduler.step_pre(step_index=step_index)

                with ProfilingContext4DebugL1("🚀 infer_main"):
                    self.model.infer(self.inputs)

                with ProfilingContext4DebugL1("step_post"):
                    self.model.scheduler.step_post()

                if self.progress_callback:
                    current_step = segment_idx * infer_steps + step_index + 1
                    total_all_steps = self.video_segment_num * infer_steps
                    self.progress_callback((current_step / total_all_steps) * 100, 100)

        if segment_idx is not None and segment_idx == self.video_segment_num - 1:
            del self.inputs
            torch_device_module.empty_cache()

        return self.model.scheduler.video_latent_state.latent, self.model.scheduler.audio_latent_state.latent
