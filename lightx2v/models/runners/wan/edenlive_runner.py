import json
from pathlib import Path

import torch
from loguru import logger

from lightx2v.common.kvcache.manager import build_self_attn_kv_cache
from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO
from lightx2v.models.input_encoders.edenlive.recording import frame_prompts, prepare_recording
from lightx2v.models.networks.wan.edenlive_model import EdenLiveARModel, EdenLiveModel, Wan21EdenLiveARModel, Wan21EdenLiveModel
from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner, WanRunner
from lightx2v.models.schedulers.wan.edenlive.scheduler import EdenLiveARScheduler, Wan21EdenLiveScheduler, Wan22EdenLiveScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import EdenLiveInputInfo
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class EdenLiveRunner(WanRunner):
    input_info_cls_by_task = {"i2v": EdenLiveInputInfo}
    supported_request_fields_by_task = {"i2v": WanRunner.supported_request_fields_by_task["i2v"] | {"video_path"}}

    def init_scheduler(self):
        self.scheduler = self.scheduler_class(self.config)

    def load_transformer(self):
        return self.model_class(self.config["model_path"], self.config, self.init_device)

    def load_model(self):
        super().load_model()
        self.proxy_encoder = self.load_proxy_encoder()

    def load_proxy_encoder(self):
        root = Path(self.config["model_path"]) / "vision-encoder"
        return ProxyDINO(str(root / "backbone"), str(root / "projection.safetensors"), self.config["proxy_resize_mode"])

    def encode_image_condition(self, image, frames):
        return {"vae_encoder_out": None, "clip_encoder_out": None}

    def prompt_schedule(self, metadata):
        return [metadata["staticPrompt"]]

    def _run_input_encoder_local_i2v(self):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder, self.vae_decoder = self.load_vae()
            self.image_encoder = self.load_image_encoder()
            self.proxy_encoder = self.load_proxy_encoder()
        height, width = self.get_target_size()
        fps = self.config["fps"]
        proxy, image, frames = prepare_recording(self.input_info.video_path, self.input_info.image_path, height, width, fps, self.input_info.num_frames)
        self.input_info.num_frames = frames
        self.input_info.latent_shape = self.get_latent_shape_with_target_hw()
        self.proxy_pixels = proxy
        image = image.to(device=AI_DEVICE)
        clean = self.vae_encoder.encode(image[None, :, None].to(GET_DTYPE()))
        clean = clean[0] if clean.ndim == 5 else clean
        image_condition = self.encode_image_condition(image, frames)
        image_condition["clean_first_frame"] = clean
        self.prompt_embeddings = {}
        prompt = self.input_info.prompt
        self.prompts = self.prompt_schedule(json.loads(Path(prompt).read_text())) if prompt.endswith(".json") else [prompt]
        original_prompt = self.input_info.prompt
        for text in dict.fromkeys(self.prompts):
            self.input_info.prompt = text
            self.prompt_embeddings[text] = self.run_text_encoder(self.input_info)
        self.input_info.prompt = original_prompt
        return {"image_encoder_output": image_condition, "text_encoder_output": self.prompt_embeddings[self.prompts[0]]}

    def _prepare_text_inputs(self, start):
        self.inputs["text_encoder_output"] = self.prompt_embeddings[self.prompts[0]]

    def _prepare_chunk_inputs(self, start, frames):
        pixels = self.proxy_pixels[:, start : start + frames].permute(1, 0, 2, 3).to(AI_DEVICE)
        self.inputs["proxy_features"] = self.proxy_encoder.infer(pixels.add_(1).mul_(0.5)).to(GET_DTYPE())
        self._prepare_text_inputs(start)

    def _run_steps(self, start, frames, total):
        for step in range(self.scheduler.infer_steps):
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

    def _run_diffusion(self):
        total = self.scheduler.latents.shape[1]
        self._prepare_chunk_inputs(0, total)
        self._run_steps(0, total, total)

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        self.init_run()
        self._run_diffusion()
        self.gen_video = self.run_vae_decoder(self.scheduler.latents)
        self.gen_video_final = self.gen_video
        result = self.process_images_after_vae_decoder()
        self.end_run()
        return result


@RUNNER_REGISTER("wan21_edenlive")
class Wan21EdenLiveRunner(EdenLiveRunner):
    model_class = Wan21EdenLiveModel
    scheduler_class = Wan21EdenLiveScheduler

    def encode_image_condition(self, image, frames):
        height, width = image.shape[-2:]
        pixels = image.new_zeros(1, 3, frames, height, width)
        pixels[0, :, 0] = image
        image_latent = self.vae_encoder.encode(pixels.to(GET_DTYPE()))
        image_latent = image_latent[0] if image_latent.ndim == 5 else image_latent
        mask = image_latent.new_zeros(4, *image_latent.shape[1:])
        mask[:, 0] = 1
        return {"vae_encoder_out": torch.cat((mask, image_latent)), "clip_encoder_out": self.run_image_encoder(image[None])}


@RUNNER_REGISTER("wan22_edenlive")
class Wan22EdenLiveRunner(EdenLiveRunner, Wan22DenseRunner):
    model_class = EdenLiveModel
    scheduler_class = Wan22EdenLiveScheduler

    def prompt_schedule(self, metadata):
        return frame_prompts(metadata, self.input_info.latent_shape[1], self.config["fps"], 1)

    def _prepare_text_inputs(self, start):
        if len(self.prompts) == 1:
            return super()._prepare_text_inputs(start)
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


class EdenLiveARMixin:
    scheduler_class = EdenLiveARScheduler

    def prompt_schedule(self, metadata):
        return frame_prompts(metadata, self.input_info.latent_shape[1], self.config["fps"], self.config["ar_config"]["num_frame_per_chunk"])

    def _prepare_text_inputs(self, start):
        index = start // self.config["ar_config"]["num_frame_per_chunk"]
        text = self.prompts[index] if len(self.prompts) > 1 else self.prompts[0]
        self.inputs["text_encoder_output"] = self.prompt_embeddings[text]

    def _run_diffusion(self):
        total = self.scheduler.latents.shape[1]
        ar = self.config["ar_config"]
        chunk = ar["num_frame_per_chunk"]
        if not 0 <= ar["sink_size"] <= ar["local_attn_size"] - chunk:
            raise ValueError("The attention window must fit sink frames and the current chunk")
        if ar["local_attn_size"] % chunk or ar["sink_size"] % chunk:
            raise ValueError("EdenLive fixed-window RoPE requires whole-chunk window and sink sizes")
        frame_tokens = (self.input_info.latent_shape[2] // 2) * (self.input_info.latent_shape[3] // 2)
        pools = {}
        for branch in [True, False] if self.config["enable_cfg"] else [True]:
            pool = build_self_attn_kv_cache(self.config, ar, ar["local_attn_size"] * frame_tokens, GET_DTYPE(), torch.device(AI_DEVICE))
            pool._init_kv_buffer()
            pools[branch] = pool
        self.model.transformer_infer.cache_pools = pools
        for start in range(0, total, chunk):
            logger.info(f"start chunk {start // chunk + 1}/{(total + chunk - 1) // chunk}")
            frames = min(chunk, total - start)
            self._prepare_chunk_inputs(start, frames)
            self._run_steps(start, frames, total)
            with ProfilingContext4DebugL1("step_pre_in_rerun"):
                self.scheduler.step_pre(self.scheduler.infer_steps - 1, chunk_start=start, cache_update=True)
            with ProfilingContext4DebugL1("🚀 infer_main_in_rerun"):
                self.model.infer(self.inputs)
        if ar.get("kv_offload", False):
            for pool in pools.values():
                pool.sync_all()
        self.model.transformer_infer.cache_pools = {}


@RUNNER_REGISTER("wan21_edenlive_ar")
class Wan21EdenLiveARRunner(EdenLiveARMixin, Wan21EdenLiveRunner):
    model_class = Wan21EdenLiveARModel


@RUNNER_REGISTER("wan22_edenlive_ar")
class Wan22EdenLiveARRunner(EdenLiveARMixin, Wan22EdenLiveRunner):
    model_class = EdenLiveARModel
