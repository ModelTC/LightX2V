import gc
import math

import torch
from loguru import logger

from lightx2v.models.runners.autoregressive_runner import AutoregressiveRunnerMixin
from lightx2v.models.runners.wan.streaming_vae import WanStreamingVAERunnerMixin
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_RECORDER_MODE
from lightx2v.utils.input_info import T2VInputInfo
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanAutoregressiveRunnerMixin(WanStreamingVAERunnerMixin, AutoregressiveRunnerMixin):
    """Chunk denoising, clean cache commits and warmup for Wan AR models."""

    _WARMUP_RESOLUTIONS = ()
    vae_decode_in_worker = True

    def validate_warmup(self):
        if not type(self).__dict__.get("_SUPPORTS_AUTOREGRESSIVE_WARMUP", False) or self.config["task"] not in self._WARMUP_TASKS:
            raise NotImplementedError(f"Autoregressive warmup is not supported for {type(self).__name__} {self.config['task']}")

    def _run_warmup(self):
        self.scheduler.generator = None
        text_output = self.run_text_encoder(T2VInputInfo(prompt="warmup"))
        chunk_frames = self.scheduler.num_frame_per_chunk
        local_frames = self.config.get("ar_config", {}).get("local_attn_size", -1)
        for height, width in self.get_warmup_resolutions():
            latent_shape = self.get_warmup_latent_shape(height, width)
            chunks = math.ceil(local_frames / chunk_frames) + 1 if local_frames > 0 else math.ceil(latent_shape[1] / chunk_frames)
            latent_shape[1] = max(2, chunks) * chunk_frames
            self.prepare_warmup_inputs(height, width, latent_shape, text_output)
            logger.info("[Warmup] {} {}x{}, {} chunks", type(self).__name__, height, width, chunks)
            self.init_run()
            for chunk in range(self.video_segment_num):
                latents = self.run_segment(chunk)
                self.run_cache_commit(chunk)
                decoded = self.decode_segment_latents(chunk, latents)
                del decoded, latents
            torch_device_module.synchronize()
            self.clear_warmup_state()

    def clear_warmup_state(self):
        super().clear_warmup_state()
        self.reset_model_request_state()
        self.model.transformer_infer.reset_kv_cache()
        self.input_info = None
        if hasattr(self, "inputs"):
            del self.inputs

    def get_video_segment_num(self):
        self.video_segment_num = self.scheduler.num_chunks

    def init_run(self):
        output_frames = self.model.transformer_infer.initialize_kv_cache(self.input_info.latent_shape)
        self.input_info.latent_shape[1] = output_frames
        self.scheduler.num_output_frames = output_frames
        self.scheduler.num_chunks = output_frames // self.scheduler.num_frame_per_chunk
        super().init_run()

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        for step_index in range(infer_steps):
            # only for single segment, check stop signal every step
            if self.video_segment_num == 1:
                self.check_stop()
            logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")

            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(seg_index=segment_idx, step_index=step_index, is_rerun=False)

            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)

            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()

            if self.progress_callback:
                current_step = segment_idx * infer_steps + step_index + 1
                total_all_steps = self.video_segment_num * infer_steps
                self.progress_callback((current_step / total_all_steps) * 100, 100)

        return self.model.scheduler.stream_output

    def run_cache_commit(self, segment_idx):
        with ProfilingContext4DebugL1("step_pre_in_rerun"):
            self.model.scheduler.step_pre(seg_index=segment_idx, step_index=self.model.scheduler.infer_steps - 1, is_rerun=True)
        with ProfilingContext4DebugL1("🚀 infer_main_in_rerun"):
            self.model.infer(self.inputs)

    @ProfilingContext4DebugL1("End run segment")
    def end_run_segment(self, segment_idx=None):
        self.run_cache_commit(segment_idx)
        self.maybe_empty_cache()

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self, total_steps=None):
        self.init_run()
        lazy_vae = self.config.get("lazy_load", False) or self.config.get("unload_modules", False)
        if lazy_vae:
            self.vae_decoder = self.load_vae_decoder()
        try:
            with self.streaming_decode(dispatch_in_worker=self.vae_decode_in_worker) as vae_decoder:
                with ProfilingContext4DebugL1(
                    f"AR chunk total {self.video_segment_num} chunks",
                    recorder_mode=GET_RECORDER_MODE(),
                    metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                    metrics_labels=["DefaultRunner"],
                ):
                    for segment_idx in range(self.video_segment_num):
                        logger.info(f"start chunk {segment_idx + 1}/{self.video_segment_num}")
                        with ProfilingContext4DebugL1(
                            f"chunk end2end {segment_idx + 1}/{self.video_segment_num}",
                            recorder_mode=GET_RECORDER_MODE(),
                            metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                            metrics_labels=["DefaultRunner"],
                        ):
                            self.check_stop()
                            self.init_run_segment(segment_idx)
                            latents = self.run_segment(segment_idx)

                            self.run_cache_commit(segment_idx)

                        vae_decoder.submit(self.decode_segment_latents, segment_idx, latents)
                        self.maybe_empty_cache()
                    decoded_chunks = vae_decoder.finish()
        finally:
            if lazy_vae:
                del self.vae_decoder
                torch_device_module.empty_cache()
                gc.collect()

        self.gen_video = None if self.streaming_video_only else torch.cat(decoded_chunks, dim=2)
        self.gen_video_final = self.gen_video
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final
