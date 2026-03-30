import torch
from PIL import Image

from lightx2v.models.networks.neopp.model import NeoppMoeModel, NeoppDenseModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.neopp.scheduler import NeoppMoeScheduler
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import *


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
        if self.config.get("version", "moe") == "moe":
            model = NeoppMoeModel(self.config["model_path"], self.config, self.init_device)
        else:
            model = NeoppDenseModel(self.config["model_path"], self.config, self.init_device)
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

    def run_input_encoder(self, to_x2v_cond_kv_path, to_x2v_uncond_kv_path):
        past_key_values_cond = torch.load(to_x2v_cond_kv_path).transpose(2, 3)  # [layers, 2, past_seq, num_kv_heads, head_dim]
        past_key_values_uncond = torch.load(to_x2v_uncond_kv_path).transpose(2, 3)

        with ProfilingContext4DebugL1("run_input_encoder"):
            input_len_cond = past_key_values_cond.shape[-3]
            input_len_uncond = past_key_values_uncond.shape[-3]

            token_h = self.input_info.target_shape[0] // (self.patch_size * self.merge_size)
            token_w = self.input_info.target_shape[1] // (self.patch_size * self.merge_size)

            indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, input_len_cond, device=self.init_device)
            indexes_image_uncondition = self._build_t2i_image_indexes(token_h, token_w, input_len_uncond, device=self.init_device)

            cos_t_cond, sin_t_cond = self._compute_rope(indexes_image_condition[0].unsqueeze(0), self.inv_freq_t)
            cos_h_cond, sin_h_cond = self._compute_rope(indexes_image_condition[1].unsqueeze(0), self.inv_freq_hw)
            cos_w_cond, sin_w_cond = self._compute_rope(indexes_image_condition[2].unsqueeze(0), self.inv_freq_hw)

            cos_t_uncond, sin_t_uncond = self._compute_rope(indexes_image_uncondition[0].unsqueeze(0), self.inv_freq_t)
            cos_h_uncond, sin_h_uncond = self._compute_rope(indexes_image_uncondition[1].unsqueeze(0), self.inv_freq_hw)
            cos_w_uncond, sin_w_uncond = self._compute_rope(indexes_image_uncondition[2].unsqueeze(0), self.inv_freq_hw)

            self.input_info.latent_shape = self.get_latent_shape_with_target_hw()

            return {
                "past_key_values_cond": past_key_values_cond,
                "past_key_values_uncond": past_key_values_uncond,
                "cos_sin_cond": (cos_t_cond, sin_t_cond, cos_h_cond, sin_h_cond, cos_w_cond, sin_w_cond),
                "cos_sin_uncond": (cos_t_uncond, sin_t_uncond, cos_h_uncond, sin_h_uncond, cos_w_uncond, sin_w_uncond),
            }

    def get_latent_shape_with_target_hw(self):
        target_height = self.input_info.target_shape[0] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_height"]
        target_width = self.input_info.target_shape[1] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_width"]
        latent_shape = [1, 3, target_height, target_width]
        return latent_shape

    def run_pipeline(self, input_info):
        self.input_info = input_info
        if self.config.get("version", "moe") == "moe":
            self.inputs = self.run_input_encoder(
                "/data/nvme1/yongyang/FL/neo_test/vlm_tensor/to_x2v_cond_kv.pt",
                "/data/nvme1/yongyang/FL/neo_test/vlm_tensor/to_x2v_uncond_kv.pt",
            )
        else:
            self.inputs = self.run_input_encoder(
                "/data/nvme1/yongyang/FL/neo_test9b/vlm_tensor/to_x2v_cond_kv.pt",
                "/data/nvme1/yongyang/FL/neo_test9b/vlm_tensor/to_x2v_uncond_kv.pt",
            )
        gen_result = self.run_main()
        return gen_result

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

        gen_result = self.process_images_after_vae_decoder()
        return gen_result

    def process_images_after_vae_decoder(self):
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
