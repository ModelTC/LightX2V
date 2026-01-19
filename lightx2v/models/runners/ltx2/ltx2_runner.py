import gc

import torch

from lightx2v.models.input_encoders.hf.ltx2.model import LTX2TextEncoder
from lightx2v.models.networks.ltx2.model import LTX2Model
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.ltx2.scheduler import LTX2Scheduler
from lightx2v.models.video_encoders.hf.ltx2.model import LTX2AudioVAE, LTX2VideoVAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.ltx2_media_io import encode_video as save_video
from lightx2v.utils.memory_profiler import peak_memory_decorator
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("ltx2")
class LTX2Runner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.video_vae, self.audio_vae = self.load_vae()
        # self.image_encoder = self.load_image_encoder()
        # self.vfi_model = self.load_vfi_model() if "video_frame_interpolation" in self.config else None
        # self.vsr_model = self.load_vsr_model() if "video_super_resolution" in self.config else None

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
            pass
        return model

    def load_text_encoder(self):
        text_encoder = LTX2TextEncoder(
            checkpoint_path=self.config["dit_original_ckpt"],
            gemma_root=self.config["gemma_original_ckpt"],
            device="cuda",
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

        # Video VAE
        video_vae = LTX2VideoVAE(
            checkpoint_path=self.config["dit_original_ckpt"],
            device=vae_device,
            dtype=GET_DTYPE(),
        )

        # Audio VAE
        audio_vae = LTX2AudioVAE(
            checkpoint_path=self.config["dit_original_ckpt"],
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

    def init_scheduler(self):
        self.scheduler = LTX2Scheduler(self.config)

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

    def init_run(self):
        self.gen_video_final = None
        self.get_video_segment_num()

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)

        self.model.scheduler.prepare(
            seed=self.input_info.seed,
            video_latent_shape=self.input_info.video_latent_shape,
            audio_latent_shape=self.input_info.audio_latent_shape,
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
