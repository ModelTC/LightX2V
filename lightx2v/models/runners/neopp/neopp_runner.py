import base64

import numpy as np
import torch
import torchvision.io as io
from PIL import Image

from lightx2v.models.networks.neopp.model import NeoppModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.neopp.scheduler import NeoppMoeScheduler
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import *
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("neopp")
class NeoppRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = self.config.get("patch_size", 16)
        self.merge_size = 2
        self.noise_scale_mode = self.config.get("noise_scale_mode", "resolution")
        self.noise_scale = self.config.get("noise_scale", 1.0)
        self.noise_scale_base_image_seq_len = self.config.get("noise_scale_base_image_seq_len", 64)
        self.noise_scale_max_value = self.config.get("noise_scale_max_value", 8.0)
        llm_config = config["llm_config"]
        head_dim = llm_config["head_dim"]
        self.inv_freq_t = self._build_inv_freq(head_dim // 2, llm_config["rope_theta"])
        self.inv_freq_hw = self._build_inv_freq(head_dim // 4, llm_config["rope_theta_hw"])
        self.past_key_values_cond = None
        self.past_key_values_uncond = None
        self.past_key_values_text_uncond = None
        self.past_key_values_img_uncond = None
        self.num_input_images = config.get("num_input_images", 1)
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

    def init_scheduler(self):
        self.scheduler = NeoppMoeScheduler(self.config)

    def init_modules(self):
        logger.info("Initializing runner modules...")
        self.load_model()
        self.model.set_scheduler(self.scheduler)

    def load_transformer(self):
        """
        MoT: Mixture-of-Transformer-Experts (MoT) architecture
        https://arxiv.org/abs/2505.14683
        """
        model = NeoppModel(self.config["model_path"], self.config, self.init_device)
        return model

    def _build_inv_freq(self, half_head_dim, theta):
        full_dim = half_head_dim * 2
        inv_freq_full = 1.0 / (theta ** (torch.arange(0, full_dim, 2, dtype=torch.float32) / full_dim))
        return inv_freq_full[::2]

    def _compute_rope(self, position_ids, inv_freq):
        inv_freq = inv_freq.cuda()
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=torch.bfloat16), emb.sin().to(dtype=torch.bfloat16)

    def _build_t2i_image_indexes(self, token_h, token_w, text_len, device):
        t_image = torch.full((token_h * token_w,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)

    def run_input_encoder(self):
        with ProfilingContext4DebugL1("run_input_encoder"):
            token_h = self.input_info.target_shape[0] // (self.patch_size * self.merge_size)
            token_w = self.input_info.target_shape[1] // (self.patch_size * self.merge_size)
            self.input_info.latent_shape = self.get_latent_shape_with_target_hw()

            indexes_cond = self._build_t2i_image_indexes(token_h, token_w, self.index_offset_cond, device=self.init_device)
            indexes_uncond = self._build_t2i_image_indexes(token_h, token_w, self.index_offset_uncond, device=self.init_device)

            cos_t_cond, sin_t_cond = self._compute_rope(indexes_cond[0].unsqueeze(0), self.inv_freq_t)
            cos_h_cond, sin_h_cond = self._compute_rope(indexes_cond[1].unsqueeze(0), self.inv_freq_hw)
            cos_w_cond, sin_w_cond = self._compute_rope(indexes_cond[2].unsqueeze(0), self.inv_freq_hw)

            cos_t_uncond, sin_t_uncond = self._compute_rope(indexes_uncond[0].unsqueeze(0), self.inv_freq_t)
            cos_h_uncond, sin_h_uncond = self._compute_rope(indexes_uncond[1].unsqueeze(0), self.inv_freq_hw)
            cos_w_uncond, sin_w_uncond = self._compute_rope(indexes_uncond[2].unsqueeze(0), self.inv_freq_hw)

            if self.seq_p_group is not None:
                world_size = dist.get_world_size(self.seq_p_group)
                cur_rank = dist.get_rank(self.seq_p_group)
                cos_t_cond = torch.chunk(cos_t_cond, world_size, dim=1)[cur_rank]
                sin_t_cond = torch.chunk(sin_t_cond, world_size, dim=1)[cur_rank]
                cos_h_cond = torch.chunk(cos_h_cond, world_size, dim=1)[cur_rank]
                sin_h_cond = torch.chunk(sin_h_cond, world_size, dim=1)[cur_rank]
                cos_w_cond = torch.chunk(cos_w_cond, world_size, dim=1)[cur_rank]
                sin_w_cond = torch.chunk(sin_w_cond, world_size, dim=1)[cur_rank]
                cos_t_uncond = torch.chunk(cos_t_uncond, world_size, dim=1)[cur_rank]
                sin_t_uncond = torch.chunk(sin_t_uncond, world_size, dim=1)[cur_rank]
                cos_h_uncond = torch.chunk(cos_h_uncond, world_size, dim=1)[cur_rank]
                sin_h_uncond = torch.chunk(sin_h_uncond, world_size, dim=1)[cur_rank]
                cos_w_uncond = torch.chunk(cos_w_uncond, world_size, dim=1)[cur_rank]
                sin_w_uncond = torch.chunk(sin_w_uncond, world_size, dim=1)[cur_rank]

            return {
                "past_key_values_cond": self.past_key_values_cond,
                "past_key_values_uncond": self.past_key_values_uncond,
                "cos_sin_cond": (cos_t_cond, sin_t_cond, cos_h_cond, sin_h_cond, cos_w_cond, sin_w_cond),
                "cos_sin_uncond": (cos_t_uncond, sin_t_uncond, cos_h_uncond, sin_h_uncond, cos_w_uncond, sin_w_uncond),
            }

    def get_latent_shape_with_target_hw(self):
        target_height = self.input_info.target_shape[0] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_height"]
        target_width = self.input_info.target_shape[1] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_width"]
        latent_shape = [1, 3, target_height, target_width]
        return latent_shape

    def multi_pipeline_run_debug(self, input_info):
        self.input_info = input_info
        if self.config.get("load_kv_cache_in_pipeline_for_debug", False):
            self.load_kvcache(
                "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor/to_x2v_cond_kv_0_289.pt",
                "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor/to_x2v_uncond_kv_0_9.pt",
            )
            self.set_inference_params(
                index_offset_cond=289,
                index_offset_uncond=9,
                cfg_interval=(-1, 2),
                cfg_scale=4.0,
                cfg_norm="global",
                timestep_shift=3.0,
            )
            self.input_info.save_result_path = self.input_info.save_result_path.replace(".png", "_0.png")

        self.inputs = self.run_input_encoder()
        gen_result = self.run_main()
        self.clear_kvcache()

        self.input_info = self.input_info
        if self.config.get("load_kv_cache_in_pipeline_for_debug", False):
            self.load_kvcache(
                "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor/to_x2v_cond_kv_1_346.pt",
                "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor/to_x2v_uncond_kv_1_12.pt",
            )
            self.set_inference_params(
                index_offset_cond=346,
                index_offset_uncond=12,
                cfg_interval=(-1, 2),
                cfg_scale=4.0,
                cfg_norm="global",
                timestep_shift=3.0,
            )
            self.input_info.save_result_path = self.input_info.save_result_path.replace("_0.png", "_1.png")

        self.inputs = self.run_input_encoder()
        gen_result = self.run_main()
        self.clear_kvcache()
        return gen_result

    def run_pipeline(self, input_info):
        self.input_info = input_info
        if self.config.get("load_kv_cache_in_pipeline_for_debug", False):
            self.load_kvcache(
                "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor/to_x2v_cond_kv_0_289.pt",
                "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor/to_x2v_uncond_kv_0_9.pt",
            )
            self.set_inference_params(
                index_offset_cond=289,
                index_offset_uncond=9,
                cfg_interval=(-1, 2),
                cfg_scale=4.0,
                cfg_norm="global",
                timestep_shift=3.0,
            )

        self.inputs = self.run_input_encoder()
        gen_result = self.run_main()
        self.clear_kvcache()
        return gen_result

    def load_kvcache(self, to_x2v_cond_kv_path, to_x2v_uncond_kv_path):
        self.past_key_values_cond = torch.load(to_x2v_cond_kv_path, map_location="cpu").transpose(2, 3).to(AI_DEVICE)
        self.past_key_values_uncond = torch.load(to_x2v_uncond_kv_path, map_location="cpu").transpose(2, 3).to(AI_DEVICE)
        logger.info(f"Loaded t2i KV cache from {to_x2v_cond_kv_path} and {to_x2v_uncond_kv_path}")
        logger.info(f"KV cache cond shape: {self.past_key_values_cond.shape}")  # [layers, 2, past_seq, num_kv_heads, head_dim]
        logger.info(f"KV cache uncond shape: {self.past_key_values_uncond.shape}")  # [layers, 2, past_seq, num_kv_heads, head_dim]

    # def load_kvcache_i2i(self, to_x2v_cond_path, to_x2v_uncond_kv_path, to_x2v_img_uncond_path):
    #     self.past_key_values_cond = torch.load(to_x2v_cond_path, map_location="cpu").transpose(2, 3).to(AI_DEVICE)
    #     self.past_key_values_uncond = torch.load(to_x2v_uncond_kv_path, map_location="cpu").transpose(2, 3).to(AI_DEVICE)
    #     self.past_key_values_img_uncond = torch.load(to_x2v_img_uncond_path, map_location="cpu").transpose(2, 3).to(AI_DEVICE)
    #     logger.info(f"Loaded i2i KV caches: cond={self.past_key_values_cond.shape}, uncond={self.past_key_values_uncond.shape}")

    def set_inference_params(self, index_offset_cond, index_offset_uncond, cfg_interval=(-1, 2), cfg_scale=4.0, cfg_norm="global", timestep_shift=3.0):
        self.index_offset_cond = index_offset_cond
        self.index_offset_uncond = index_offset_uncond
        self.scheduler.timestep_shift = timestep_shift
        self.model.cfg_interval = cfg_interval
        self.model.cfg_scale = cfg_scale
        self.model.cfg_norm = cfg_norm

    def set_kvcache(self, to_x2v_cond_kv: torch.Tensor, to_x2v_uncond_kv: torch.Tensor):
        self.past_key_values_cond = to_x2v_cond_kv.to(AI_DEVICE)
        self.past_key_values_uncond = to_x2v_uncond_kv.to(AI_DEVICE)
        logger.info(f"KV cache cond shape: {self.past_key_values_cond.shape}")  # [layers, 2, past_seq, num_kv_heads, head_dim]
        logger.info(f"KV cache uncond shape: {self.past_key_values_uncond.shape}")  # [layers, 2, past_seq, num_kv_heads, head_dim]

    # def set_kvcache_i2i(self, to_x2v_cond_kv: torch.Tensor, to_x2v_text_uncond_kv: torch.Tensor, to_x2v_img_uncond_kv: torch.Tensor):
    #     self.past_key_values_cond = to_x2v_cond_kv.to(AI_DEVICE)
    #     self.past_key_values_text_uncond = to_x2v_text_uncond_kv.to(AI_DEVICE)
    #     self.past_key_values_img_uncond = to_x2v_img_uncond_kv.to(AI_DEVICE)
    #     logger.info(f"KV cache cond shape: {self.past_key_values_cond.shape}")  # [layers, 2, past_seq, num_kv_heads, head_dim]
    #     logger.info(f"KV cache text uncond shape: {self.past_key_values_text_uncond.shape}")  # [layers, 2, past_seq, num_kv_heads, head_dim]
    #     logger.info(f"KV cache img uncond shape: {self.past_key_values_img_uncond.shape}")  # [layers, 2, past_seq, num_kv_heads, head_dim]

    def clear_kvcache(self):
        self.past_key_values_cond = None
        self.past_key_values_uncond = None
        self.model.transformer_infer.kv_cache.clear()

    def init_run(self):
        self.model.scheduler.prepare(seed=self.input_info.seed, latent_shape=self.input_info.latent_shape)

    def run_main(self):
        self.init_run()
        infer_steps = self.model.scheduler.infer_steps
        for step_index in range(infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")

            with ProfilingContext4DebugL1("step_pre"):
                self.scheduler.step_pre(step_index)

            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)

            with ProfilingContext4DebugL1("step_post"):
                self.scheduler.step_post()

        if self.config.get("save_result_for_debug", True):
            gen_result = self.process_images_after_vae_decoder_for_debug()
        else:
            gen_result = self.process_images_after_vae_decoder()
        return gen_result

    def process_images_after_vae_decoder(self):
        image = self._denorm(self.scheduler.image_prediction.float())
        image = (image.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
        return base64.b64encode(io.encode_jpeg(image[0]).numpy()).decode("utf-8")

    def process_images_after_vae_decoder_for_debug(self):
        image = self._denorm(self.scheduler.image_prediction.float())
        image = (image.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        grid_image = Image.fromarray(image[0])
        grid_image.save(self.input_info.save_result_path)
        logger.info(f"✅ Image saved successfully to: {self.input_info.save_result_path} ✅")
        return grid_image

    def _denorm(self, x: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        """
        x: [B,3,H,W] normalized ((img-mean)/std). returns [0,1] clamped.
        """
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)
