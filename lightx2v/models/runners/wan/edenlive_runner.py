from contextlib import nullcontext
from functools import partial

import torch
from loguru import logger

from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO
from lightx2v.models.input_encoders.edenlive.recording import RecordingReader, frame_prompts, prepare_cover, prepare_recording
from lightx2v.models.networks.wan.edenlive_model import EdenLiveARModel, EdenLiveModel
from lightx2v.models.runners.lifecycle import RunnerLifecycleMixin
from lightx2v.models.runners.wan.autoregressive_runner import AutoregressiveRunnerMixin
from lightx2v.models.runners.wan.streaming_vae import WanStreamingVAERunnerMixin
from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner, WanRunner
from lightx2v.models.schedulers.wan.edenlive.scheduler import EdenLiveARScheduler, EdenLiveScheduler
from lightx2v.utils.async_vae import AsyncVAEChunkDecoder
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import EdenLiveInputInfo
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2, no_sync_profiling
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class EdenLiveRunner(RunnerLifecycleMixin):
    """Combine EdenLive inference with the Wan component loaders in concrete runners."""

    model_class = EdenLiveModel
    scheduler_class = EdenLiveScheduler
    _WARMUP_TASKS = ("i2v",)
    _WARMUP_RESOLUTIONS = ((480, 480), (480, 832))
    input_info_cls_by_task = {"i2v": EdenLiveInputInfo}
    supported_request_fields_by_task = {"i2v": WanRunner.supported_request_fields_by_task["i2v"] | {"video_path"}}

    def __init__(self, config):
        config.setdefault("proxy_resize_mode", "bilinear")
        config.setdefault("proxy_condition_rmsnorm", False)
        config.setdefault("proxy_unconditional_control", "same")
        super().__init__(config)
        self.is_wan21 = config["model_cls"] in ["wan21_edenlive", "wan21_edenlive_ar"]
        self.chunk_callback = None
        self.recording_reader = None

    def run_pipeline(self, input_info):
        # Keep only the current request's output. The caller may still own its
        # previous result, but the runner must not retain it during input encode
        # or while the next full video is being assembled.
        self.gen_video = None
        self.gen_video_final = None
        try:
            return super().run_pipeline(input_info)
        finally:
            self.reset_model_request_state()
            if self.recording_reader is not None:
                self.recording_reader.close()
                self.recording_reader = None

    def init_scheduler(self):
        self.scheduler = self.scheduler_class(self.config)

    def load_transformer(self):
        return self.model_class(self.config["model_path"], self.config, self.init_device)

    def load_model(self):
        super().load_model()
        self.proxy_encoder = self.load_proxy_encoder()

    def load_proxy_encoder(self):
        return ProxyDINO(
            self.config["model_path"],
            self.config["proxy_resize_mode"],
            use_cuda_graph=self.config.get("use_cuda_graph", False),
            attention_type="torch_sdpa",
            quantized_ckpt=self.config.get("proxy_quantized_ckpt"),
        )

    def load_vae(self):
        if self.is_wan21 or not self.config.get("use_lightvae", False):
            return super().load_vae()
        self.vae_encoder = self.load_vae_encoder()
        return self.vae_encoder, self.load_vae_decoder()

    def load_vae_decoder(self):
        if self.is_wan21 or not self.config.get("use_lightvae", False):
            return super().load_vae_decoder()
        from lightx2v.models.video_encoders.hf.wan.lightvae import WanLightVAEDecoder

        return WanLightVAEDecoder(
            vae_path=self.config["lightvae_path"],
            family="wan22",
            scale=self.vae_encoder.scale,
            dtype=GET_DTYPE(),
            device=AI_DEVICE,
            cpu_offload=self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False)),
        )

    def encode_image_condition(self, image, frames):
        if not self.is_wan21:
            return {"vae_encoder_out": None, "clip_encoder_out": None}
        height, width = image.shape[-2:]
        pixels = image.new_zeros(1, 3, frames, height, width)
        pixels[0, :, 0] = image
        image_latent = self.vae_encoder.encode(pixels.to(GET_DTYPE()))
        image_latent = image_latent[0] if image_latent.ndim == 5 else image_latent
        mask = image_latent.new_zeros(4, *image_latent.shape[1:])
        mask[:, 0] = 1
        return {"vae_encoder_out": torch.cat((mask, image_latent)), "clip_encoder_out": self.run_image_encoder(image[None])}

    def prompt_schedule(self, chunk_size=1):
        prompt_travel = self.config.get("prompt_travel")
        if prompt_travel is None:
            return [self.input_info.prompt]
        if self.config["model_cls"] == "wan21_edenlive":
            raise ValueError("Wan2.1 bidirectional EdenLive does not support prompt_travel")
        logger.info("Prompt travel enabled: start_frames={}", prompt_travel["prompt_travel_start_frames"])
        return frame_prompts(prompt_travel, self.input_info.latent_shape[1], chunk_size, self.config["vae_stride"][0])

    def _run_input_encoder_local_i2v(self):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder, self.vae_decoder = self.load_vae()
            self.image_encoder = self.load_image_encoder()
            self.proxy_encoder = self.load_proxy_encoder()
        height, width = self.get_target_size()
        fps = self.config["fps"]
        proxy, image, frames = self.load_recording(height, width, fps)
        self.input_info.num_frames = frames
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
        self.proxy_pixels = proxy
        return self._encode_inputs(image, frames)

    def load_recording(self, height, width, fps):
        return prepare_recording(self.input_info.video_path, self.input_info.image_path, height, width, fps, self.input_info.num_frames)

    def _encode_inputs(self, image, frames):
        image = image.to(device=AI_DEVICE)
        clean = self.vae_encoder.encode(image[None, :, None].to(GET_DTYPE()))
        clean = clean[0] if clean.ndim == 5 else clean
        image_condition = self.encode_image_condition(image, frames)
        image_condition["clean_first_frame"] = clean
        self.prompt_embeddings = {}
        self.prompts = self.prompt_schedule()
        original_prompt = self.input_info.prompt
        for text in dict.fromkeys(self.prompts):
            self.input_info.prompt = text
            self.prompt_embeddings[text] = self.run_text_encoder(self.input_info)
        self.input_info.prompt = original_prompt
        return {"image_encoder_output": image_condition, "text_encoder_output": self.prompt_embeddings[self.prompts[0]]}

    def _prepare_text_inputs(self, start):
        if len(self.prompts) == 1:
            self.inputs["text_encoder_output"] = self.prompt_embeddings[self.prompts[0]]
            return
        groups = []
        for text in self.prompts:
            if groups and groups[-1][0] == text:
                groups[-1][1] += 1
            else:
                groups.append([text, 1])
        self.inputs["text_segments"] = [count for _, count in groups]
        self.inputs["text_encoder_output"] = {
            key: torch.cat([self.prompt_embeddings[text][key] for text, _ in groups], dim=1) for key, value in self.prompt_embeddings[self.prompts[0]].items() if value is not None
        }

    def _prepare_chunk_inputs(self, start, frames):
        pixels = self.get_proxy_chunk(start, frames).permute(1, 0, 2, 3).to(AI_DEVICE)
        self.inputs["proxy_features"] = self.proxy_encoder.infer(pixels).to(GET_DTYPE())
        self._prepare_text_inputs(start)

    def get_proxy_chunk(self, start, frames):
        return self.proxy_pixels[:, start : start + frames]

    def _run_steps(self, start, frames, total, step_indices=None):
        for step in range(self.scheduler.infer_steps) if step_indices is None else step_indices:
            self.check_stop()
            logger.info(f"==> step_index: {step + 1} / {self.scheduler.infer_steps}")
            with ProfilingContext4DebugL1("step_pre"):
                self.scheduler.step_pre(step, chunk_start=start)
            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)
            with ProfilingContext4DebugL1("step_post"):
                self.scheduler.step_post()
            if self.progress_callback:
                self.progress_callback(100 * (start * self.scheduler.infer_steps + (step + 1) * frames) / (total * self.scheduler.infer_steps), 100)

    def _run_diffusion(self, step_indices=None):
        total = self.scheduler.latents.shape[1]
        self._prepare_chunk_inputs(0, total)
        self._run_steps(0, total, total, step_indices)
        return self.run_vae_decoder(self.scheduler.latents)

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        self.gen_video = self._run_diffusion()
        self.gen_video_final = self.gen_video
        result = self.process_images_after_vae_decoder()
        self.end_run()
        return result

    @torch.no_grad()
    def _run_warmup(self):
        lazy_load = self.config.get("lazy_load", False)
        if lazy_load:
            self.vae_encoder, self.vae_decoder = self.load_vae()
            self.image_encoder = self.load_image_encoder()
        latent_frames = self.config.get("ar_config", {}).get("local_attn_size", 3) + self.config.get("ar_config", {}).get("num_frame_per_chunk", 3) + 1
        num_frames = self.config.get("num_frames")
        if num_frames is None:
            num_frames = (latent_frames - 1) * self.config["vae_stride"][0] + 1
        try:
            for height, width in self.get_warmup_resolutions():
                logger.info("EdenLive warmup: {}x{}, {} frames", height, width, num_frames)
                self.scheduler.generator = None
                stride_t, stride_h, stride_w = self.config["vae_stride"]
                self.input_info = EdenLiveInputInfo(prompt="warmup", negative_prompt="warmup", seed=0, num_frames=num_frames)
                self.input_info.latent_shape = [self.config["out_dim"], (num_frames - 1) // stride_t + 1, height // stride_h, width // stride_w]
                self.proxy_pixels = torch.zeros(self.input_info.latent_shape[1], height, width, 3, dtype=torch.uint8).permute(3, 0, 1, 2)
                try:
                    self.inputs = self._encode_inputs(torch.zeros(3, height, width, device=AI_DEVICE), num_frames)
                    self.scheduler.prepare(0, self.input_info.latent_shape, self.inputs["image_encoder_output"])
                    self._run_diffusion(step_indices=(0,))
                finally:
                    getattr(torch, AI_DEVICE).synchronize()
                    if self.config["cpu_offload"] and self.config.get("offload_granularity", "block") == "model":
                        self.model.to_cpu()
                    self.clear_warmup_state()
        finally:
            self.input_info = None
            self.proxy_pixels = None
            self.prompt_embeddings = {}
            self.prompts = []
            if hasattr(self, "inputs"):
                del self.inputs

    def clear_warmup_state(self):
        self.scheduler.clear()
        self.reset_model_request_state()

    def clean_lazy_load_warmup(self):
        super().clean_lazy_load_warmup()
        del self.proxy_encoder

    def end_run(self):
        self.reset_model_request_state()
        self.proxy_pixels = None
        self.prompt_embeddings = {}
        self.prompts = []
        super().end_run()


class EdenLiveARRunner(WanStreamingVAERunnerMixin, AutoregressiveRunnerMixin, EdenLiveRunner):
    _WARMUP_RESOLUTIONS = ()
    model_class = EdenLiveARModel
    scheduler_class = EdenLiveARScheduler

    def load_recording(self, height, width, fps):
        self.recording_reader = RecordingReader(self.input_info.video_path, height, width, fps, self.input_info.num_frames, self.config["ar_config"]["num_frame_per_chunk"])
        return None, prepare_cover(self.input_info.image_path, height, width), self.recording_reader.num_frames

    def get_proxy_chunk(self, start, frames):
        if self.recording_reader is None:
            return super().get_proxy_chunk(start, frames)
        return self.recording_reader.next_chunk()

    def set_chunk_callback(self, callback):
        """Receive decoded (B,C,T,H,W) chunks in [-1,1] as callback(pixels, index, is_last).

        Async decoding delivers callbacks in order from a worker thread after
        the chunk is ready on GPU. Keep the callback short to avoid queuing output.
        """
        self.chunk_callback = callback

    def on_decoded_chunk(self, pixels, index, *, num_chunks):
        self.chunk_callback(pixels, index, index == num_chunks - 1)

    def prompt_schedule(self):
        return super().prompt_schedule(self.config["ar_config"]["num_frame_per_chunk"])

    def _prepare_text_inputs(self, start):
        index = start // self.config["ar_config"]["num_frame_per_chunk"]
        text = self.prompts[index] if len(self.prompts) > 1 else self.prompts[0]
        self.inputs["text_encoder_output"] = self.prompt_embeddings[text]

    def _run_diffusion(self, step_indices=None):
        total = self.scheduler.latents.shape[1]
        ar = self.config["ar_config"]
        chunk = ar["num_frame_per_chunk"]
        infer = self.model.transformer_infer
        infer.initialize_kv_cache(self.input_info.latent_shape)
        pools = infer.cache_pools
        num_chunks = (total + chunk - 1) // chunk
        on_chunk = partial(self.on_decoded_chunk, num_chunks=num_chunks) if self.chunk_callback is not None else None
        decoder = AsyncVAEChunkDecoder.from_config(self.config, device=AI_DEVICE, vae_decoder=self.vae_decoder, on_chunk=on_chunk, dispatch_in_worker=True)
        try:
            with no_sync_profiling(enabled=decoder.is_async):
                for start in range(0, total, chunk):
                    logger.info(f"start chunk {start // chunk + 1}/{num_chunks}")
                    chunk_profile = ProfilingContext4DebugL1(f"chunk {start // chunk + 1}/{num_chunks} end_to_end") if not decoder.is_async else nullcontext()
                    with chunk_profile:
                        frames = min(chunk, total - start)
                        self._prepare_chunk_inputs(start, frames)
                        self._run_steps(start, frames, total, step_indices)
                        with ProfilingContext4DebugL1("step_pre_in_rerun"):
                            self.scheduler.step_pre(self.scheduler.infer_steps - 1, chunk_start=start, cache_update=True)
                        decode_args = (self.scheduler.latents[:, start : start + frames], start == 0, start + frames == total)
                        with ProfilingContext4DebugL1("🚀 infer_main_in_rerun"):
                            self.model.infer(self.inputs)
                        decoder.submit(self.vae_decoder.cached_decode_withflag, *decode_args)
                decoded_chunks = decoder.finish()
        finally:
            try:
                decoder.finish()
            finally:
                try:
                    if ar.get("kv_offload", False):
                        for pool in pools.values():
                            pool.sync_all()
                finally:
                    self.model.transformer_infer.cache_pools = {}
                    # Drop request aliases; the cache plugin owns retained
                    # graph-bound storage, including after decode failures.
                    pools.clear()
        return torch.cat(decoded_chunks, dim=2)


@RUNNER_REGISTER("wan21_edenlive")
class Wan21EdenLiveRunner(EdenLiveRunner, WanRunner):
    _SUPPORTS_GENERIC_WARMUP = True


@RUNNER_REGISTER("wan22_edenlive")
class Wan22EdenLiveRunner(EdenLiveRunner, Wan22DenseRunner):
    _SUPPORTS_GENERIC_WARMUP = True
    _WARMUP_RESOLUTIONS = ((512, 512), (704, 1280))


@RUNNER_REGISTER("wan21_edenlive_ar")
class Wan21EdenLiveARRunner(EdenLiveARRunner, WanRunner):
    _SUPPORTS_GENERIC_WARMUP = True


@RUNNER_REGISTER("wan22_edenlive_ar")
class Wan22EdenLiveARRunner(EdenLiveARRunner, Wan22DenseRunner):
    _SUPPORTS_GENERIC_WARMUP = True
