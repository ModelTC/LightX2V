import torch
from PIL import Image

from lightx2v.models.networks.neopp.model import NeoppModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.neopp.scheduler import NeoppScheduler
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import *


@RUNNER_REGISTER("neopp")
class NeoppRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = 16
        self.merge_size = 2
        self.noise_scale_mode = self.config.get("noise_scale_mode", "resolution")
        self.noise_scale = self.config.get("noise_scale", 1.0)
        self.noise_scale_base_image_seq_len = self.config.get("noise_scale_base_image_seq_len", 64)
        self.noise_scale_max_value = self.config.get("noise_scale_max_value", 8.0)

    def init_scheduler(self):
        self.scheduler = NeoppScheduler(self.config)

    def init_modules(self):
        logger.info("Initializing runner modules...")
        self.load_model()
        self.model.set_scheduler(self.scheduler)

    def load_transformer(self):
        """
        MoT: Mixture-of-Transformer-Experts (MoT) architecture
        https://arxiv.org/abs/2505.14683
        """
        print("Loading NeoppModel...")
        print("Model path:", self.config["model_path"])
        print("Config:", self.config)
        print("Init device:", self.init_device)
        model = NeoppModel(self.config["model_path"], self.config, self.init_device)
        return model

    def _build_t2i_image_indexes(self, token_h, token_w, text_len, device):
        t_image = torch.full((token_h * token_w,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)

    def run_input_encoder(self, to_x2v_cond_kv_path, to_x2v_uncond_kv_path):
        past_key_values_cond = torch.load(to_x2v_cond_kv_path)
        past_key_values_uncond = torch.load(to_x2v_uncond_kv_path)

        with ProfilingContext4DebugL1("load_input_encoder"):
            input_len_cond = past_key_values_cond.shape[-2]
            input_len_uncond = past_key_values_uncond.shape[-2]

            token_h = self.input_info.target_shape[0] // (self.patch_size * self.merge_size)
            token_w = self.input_info.target_shape[1] // (self.patch_size * self.merge_size)

            indexes_image_condition = self._build_t2i_image_indexes(token_h, token_w, input_len_cond, device=self.init_device)
            indexes_image_uncondition = self._build_t2i_image_indexes(token_h, token_w, input_len_uncond, device=self.init_device)

            self.input_info.latent_shape = self.get_latent_shape_with_target_hw()

            return {
                "past_key_values_cond": past_key_values_cond,
                "past_key_values_uncond": past_key_values_uncond,
                "indexes_image_condition": indexes_image_condition,
                "indexes_image_uncondition": indexes_image_uncondition,
            }

    def get_latent_shape_with_target_hw(self):
        target_height = self.input_info.target_shape[0] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_height"]
        target_width = self.input_info.target_shape[1] if self.input_info.target_shape and len(self.input_info.target_shape) == 2 else self.config["target_width"]
        latent_shape = [1, 3, target_height, target_width]
        return latent_shape

    def run_pipeline(self, input_info):
        self.input_info = input_info
        self.inputs = self.run_input_encoder(
            "/data/nvme1/yongyang/FL/neo_test/vlm_tensor/to_x2v_cond_kv.pt",
            "/data/nvme1/yongyang/FL/neo_test/vlm_tensor/to_x2v_uncond_kv.pt",
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
