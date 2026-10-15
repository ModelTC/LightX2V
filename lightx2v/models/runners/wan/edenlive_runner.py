import json
from pathlib import Path

import torch

from lightx2v.common.kvcache.manager import build_self_attn_kv_cache
from lightx2v.models.input_encoders.edenlive.proxy_dino import ProxyDINO
from lightx2v.models.input_encoders.edenlive.recording import frame_prompts, prepare_recording
from lightx2v.models.networks.wan.edenlive_model import EdenLiveModel
from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner, WanRunner
from lightx2v.models.schedulers.wan.edenlive.scheduler import EdenLiveScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import EdenLiveInputInfo
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("edenlive")
class EdenLiveRunner(WanRunner):
    input_info_cls_by_task = {"i2v": EdenLiveInputInfo}
    supported_request_fields_by_task = {"i2v": WanRunner.supported_request_fields_by_task["i2v"] | {"video_path"}}

    def __init__(self, config):
        super().__init__(config)
        if config["eden_variant"] == "5b":
            from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE

            self.vae_cls = Wan2_2_VAE
            self.vae_name = "Wan2.2_VAE.pth"

    def init_scheduler(self):
        self.scheduler = EdenLiveScheduler(self.config)

    def load_transformer(self):
        return EdenLiveModel(self.config["model_path"], self.config, self.init_device)

    def load_image_encoder(self):
        encoder = super().load_image_encoder(dtype=GET_DTYPE())
        if encoder is not None:
            for module in encoder.model.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.float()
        return encoder

    def load_text_encoder(self):
        from lightx2v.models.input_encoders.hf.wan.t5.model import T5Attention, T5Encoder

        encoders = super().load_text_encoder()
        for encoder in encoders:
            for module in encoder.model.modules():
                if isinstance(module, (T5Attention, T5Encoder)):
                    module.use_sdpa = True
        return encoders

    def run_image_encoder(self, first_frame, last_frame=None):
        from transformers import CLIPImageProcessor

        processor = CLIPImageProcessor.from_pretrained(str(Path(self.config["model_path"]) / "clip/image_processor"), local_files_only=True)
        pixels = ((first_frame.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        pixels = processor(images=list(pixels), return_tensors="pt")["pixel_values"].to(device=AI_DEVICE, dtype=GET_DTYPE())
        if self.image_encoder.cpu_offload:
            self.image_encoder.to_cuda()
        output = self.image_encoder.model.visual(pixels, use_31_block=True).squeeze(0)
        if self.image_encoder.cpu_offload:
            self.image_encoder.to_cpu()
        return output

    def _configure_vae_norm(self, vae):
        from lightx2v.models.video_encoders.hf.wan.vae import RMS_norm as WanRMSNorm
        from lightx2v.models.video_encoders.hf.wan.vae_2_2 import RMS_norm as Wan22RMSNorm

        for module in vae.model.modules():
            if isinstance(module, (WanRMSNorm, Wan22RMSNorm)):
                module.use_fp32_l2_norm = True
        return vae

    def load_vae_encoder(self):
        vae = Wan22DenseRunner.load_vae_encoder(self) if self.config["eden_variant"] == "5b" else super().load_vae_encoder()
        return self._configure_vae_norm(vae)

    def load_vae_decoder(self):
        vae = Wan22DenseRunner.load_vae_decoder(self) if self.config["eden_variant"] == "5b" else super().load_vae_decoder()
        return self._configure_vae_norm(vae)

    _resolve_wan22_vae_paths = Wan22DenseRunner._resolve_wan22_vae_paths
    _build_wan22_vae_config = Wan22DenseRunner._build_wan22_vae_config

    def load_model(self):
        super().load_model()
        self.proxy_encoder = self.load_proxy_encoder()

    def load_proxy_encoder(self):
        root = Path(self.config["model_path"]) / "vision-encoder"
        return ProxyDINO(str(root / "backbone"), str(root / "projection.safetensors"), self.config["proxy_resize_mode"])

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
        image_latent, clip = None, None
        if self.config["eden_variant"] == "14b":
            pixels = image.new_zeros(1, 3, frames, height, width)
            pixels[0, :, 0] = image
            image_latent = self.vae_encoder.encode(pixels.to(GET_DTYPE()))
            image_latent = image_latent[0] if image_latent.ndim == 5 else image_latent
            mask = image_latent.new_zeros(4, *image_latent.shape[1:])
            mask[:, 0] = 1
            image_latent = torch.cat((mask, image_latent))
            clip = self.run_image_encoder(image[None])
        self.prompt_embeddings = {}
        prompt = self.input_info.prompt
        if prompt.endswith(".json"):
            metadata = json.loads(Path(prompt).read_text())
            chunk = self.config["ar_config"]["num_frame_per_chunk"] if self.config["eden_causal"] else 1
            self.prompts = (
                [metadata["staticPrompt"]] if self.config["eden_variant"] == "14b" and not self.config["eden_causal"] else frame_prompts(metadata, self.input_info.latent_shape[1], fps, chunk)
            )
        else:
            self.prompts = [prompt]
        original_prompt = self.input_info.prompt
        for text in dict.fromkeys(self.prompts):
            self.input_info.prompt = text
            self.prompt_embeddings[text] = self.run_text_encoder(self.input_info)
        self.input_info.prompt = original_prompt
        return {"image_encoder_output": {"vae_encoder_out": image_latent, "clip_encoder_out": clip, "clean_first_frame": clean}, "text_encoder_output": self.prompt_embeddings[self.prompts[0]]}

    def _prepare_chunk_inputs(self, start, frames):
        pixels = self.proxy_pixels[:, start : start + frames].permute(1, 0, 2, 3).to(AI_DEVICE)
        self.inputs["proxy_features"] = self.proxy_encoder.infer((pixels + 1) / 2).to(GET_DTYPE())
        if self.config["eden_causal"]:
            index = start // self.config["ar_config"]["num_frame_per_chunk"]
            text = self.prompts[index] if len(self.prompts) > 1 else self.prompts[0]
            self.inputs["text_encoder_output"] = self.prompt_embeddings[text]
        elif len(self.prompts) > 1:
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

    def run_main(self):
        self.init_run()
        total = self.scheduler.latents.shape[1]
        ar = self.config.get("ar_config", {})
        chunk = ar["num_frame_per_chunk"] if self.config["eden_causal"] else total
        if self.config["eden_causal"]:
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
            self._prepare_chunk_inputs(start, min(chunk, total - start))
            for step in range(self.scheduler.infer_steps):
                self.check_stop()
                self.scheduler.step_pre(step, chunk_start=start)
                self.model.infer(self.inputs)
                self.scheduler.step_post()
                if self.progress_callback:
                    self.progress_callback(100 * (start * self.scheduler.infer_steps + (step + 1) * min(chunk, total - start)) / (total * self.scheduler.infer_steps), 100)
            if self.config["eden_causal"]:
                self.scheduler.step_pre(self.scheduler.infer_steps - 1, chunk_start=start, cache_update=True)
                self.model.infer(self.inputs)
        self.gen_video = self.run_vae_decoder(self.scheduler.latents)
        self.gen_video_final = self.gen_video.float()
        result = self.process_images_after_vae_decoder()
        if self.config["eden_causal"] and ar.get("kv_offload", False):
            for pool in self.model.transformer_infer.cache_pools.values():
                pool.sync_all()
        self.model.transformer_infer.cache_pools = {}
        self.end_run()
        return result
