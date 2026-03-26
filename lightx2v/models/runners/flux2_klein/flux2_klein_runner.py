import gc
import os
import torch
from loguru import logger

from lightx2v.models.networks.flux2_klein.model import Flux2KleinTransformerModel
from lightx2v.models.schedulers.flux2_klein.scheduler import Flux2KleinScheduler
from lightx2v.models.input_encoders.hf.flux2_klein.qwen3_model import Flux2Klein_TextEncoder
from lightx2v.models.video_encoders.hf.flux2_klein.vae import Flux2KleinVAE
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v.utils.envs import GET_DTYPE

@RUNNER_REGISTER("flux2_klein")
class Flux2KleinRunner(DefaultRunner):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, config):
        config["vae_scale_factor"] = config.get("vae_scale_factor", 16) # 8 * 2 for patch config
        super().__init__(config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.text_encoders = self.load_text_encoder()
        self.vae = self.load_vae()
        self.model = self.load_transformer()

    def load_transformer(self):
        model_kwargs = {
            "model_path": os.path.join(self.config["model_path"], "transformer"),
            "config": self.config,
            "device": self.init_device,
        }
        return Flux2KleinTransformerModel(**model_kwargs)

    def load_text_encoder(self):
        text_encoder = Flux2Klein_TextEncoder(self.config)
        return [text_encoder]

    def load_vae(self):
        vae = Flux2KleinVAE(self.config)
        return vae
        
    def init_scheduler(self):
        self.scheduler = Flux2KleinScheduler(self.config)

    def init_modules(self):
        logger.info("Initializing Flux2Klein modules...")
        self.load_model()
        self.model.set_scheduler(self.scheduler)
        self.run_dit = self._run_dit_local
        self.run_input_encoder = self._run_input_encoder_local_t2i

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_t2i(self):
        prompt = self.input_info.prompt
        text_encoder_output = self.run_text_encoder(prompt, neg_prompt=self.input_info.negative_prompt)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": None,
        }

    # Copied from diffusers/pipelines/flux2/pipeline_flux2.py _prepare_text_ids
    def _prepare_text_ids(self, x):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t, h, w, l = torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)
        
    # Copied from diffusers/pipelines/flux2/pipeline_flux2.py _prepare_latent_ids
    def _prepare_latent_ids(self, batch_size, height, width):
        t = torch.arange(1)
        # Patched height/width are used here
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    @ProfilingContext4DebugL1("Run Text Encoder")
    def run_text_encoder(self, text, image_list=None, neg_prompt=None):
        prompt_embeds_list, _ = self.text_encoders[0].infer([text])
        prompt_embeds = prompt_embeds_list[0].unsqueeze(0) # (1, seq_len, dim)
        text_ids = self._prepare_text_ids(prompt_embeds).to(AI_DEVICE)
        
        text_encoder_output = {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids
        }
        
        # Flux2Klein uses empty string as negative prompt for CFG
        if self.config.get("sample_guide_scale", 1.0) > 1.0 or self.config.get("enable_cfg", True):
            neg_prompt_embeds_list, _ = self.text_encoders[0].infer([""])
            neg_prompt_embeds = neg_prompt_embeds_list[0].unsqueeze(0)
            neg_text_ids = self._prepare_text_ids(neg_prompt_embeds).to(AI_DEVICE)
            
            text_encoder_output["negative_prompt_embeds"] = neg_prompt_embeds
            text_encoder_output["negative_text_ids"] = neg_text_ids
            
        return text_encoder_output

    @ProfilingContext4DebugL2("Run DiT")
    def _run_dit_local(self, total_steps=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self.model.scheduler.prepare(self.input_info)
        latents, generator = self.run(total_steps)
        return latents, generator

    def run(self, total_steps=None):
        if total_steps is None:
            total_steps = self.model.scheduler.infer_steps
        for step_index in range(total_steps):
            logger.info(f"==> step_index: {step_index + 1} / {total_steps}")

            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)

            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()

            if self.progress_callback:
                self.progress_callback(((step_index + 1) / total_steps) * 100, 100)

        return self.model.scheduler.latents, self.model.scheduler.generator
        
    def set_target_shape(self):
        width = self.config.get("target_width", 1024)
        height = self.config.get("target_height", 1024)

        vae_scale_factor = self.config["vae_scale_factor"]
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        # Strictly follow Flux2KleinPipeline:
        # transformer input channels must match model x_embedder.in_features.
        in_channels = getattr(self.model, "in_channels", None)
        if in_channels is None:
            in_channels = self.config.get("transformer_in_channels", self.config.get("in_channels", 64))

        # input latents are packed as `in_channels` (not pre-4-packed channels)
        # the old hardcode 16*4=64 is incorrect if model expects 128.
        packed_batch = 1
        packed_channels = in_channels
        packed_h = height // 2
        packed_w = width // 2

        self.num_channels_latents = packed_channels // 4 if packed_channels % 4 == 0 else packed_channels
        self.input_info.target_shape = (packed_batch, packed_h * packed_w, packed_channels)
        self.input_info.latent_image_ids = self._prepare_latent_ids(packed_batch, packed_h, packed_w).to(AI_DEVICE)

    def set_img_shapes(self):
        pass

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        B, _, C = latents.shape
        
        # Need original packed height and width. We can recover from latent_image_ids max
        H = int((self.input_info.latent_image_ids[0, :, 1].max() + 1).item())
        W = int((self.input_info.latent_image_ids[0, :, 2].max() + 1).item())
 
        latents = latents.view(B, H, W, C).permute(0, 3, 1, 2)
 
 
        bn_mean = self.vae.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.vae.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.vae.config.batch_norm_eps)
        latents = latents * bn_std + bn_mean
 
        latents = latents.reshape(B, C // 4, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(B, C // 4, H * 2, W * 2)

        images = self.vae.decode(latents, self.input_info)
        return images

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info
        self.set_target_shape()
        self.set_img_shapes()

        self.inputs = self.run_input_encoder()
        logger.info(f"input_info: {self.input_info}")

        latents, generator = self.run_dit()
        images = self.run_vae_decoder(latents)

        if not input_info.return_result_tensor:
            image = images[0]
            image.save(input_info.save_result_path)
            logger.info(f"Image saved: {input_info.save_result_path}")

        torch.cuda.empty_cache()
        gc.collect()

        if input_info.return_result_tensor:
            return {"images": images}
        return {"images": None}
