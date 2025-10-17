import gc

import torch
from loguru import logger

from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.networks.wan.sf_model import WanSFModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
from lightx2v.models.video_encoders.hf.wan.vae_sf import WanSFVAE
from lightx2v.utils.envs import *
from lightx2v.utils.memory_profiler import peak_memory_decorator
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("wan2.1_sf")
class WanSFRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        self.vae_cls = WanSFVAE

    def load_transformer(self):
        model = WanSFModel(
            self.config,
            self.config,
            self.init_device,
        )
        if self.config.get("lora_configs") and self.config.lora_configs:
            assert not self.config.get("dit_quantized", False)
            lora_wrapper = WanLoraWrapper(model)
            for lora_config in self.config.lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")
        return model

    def init_scheduler(self):
        self.scheduler = WanSFScheduler(self.config)

    def set_target_shape(self):
        self.num_output_frames = 21
        self.config.target_shape = [16, self.num_output_frames, 60, 104]

    def get_video_segment_num(self):
        self.video_segment_num = self.scheduler.num_blocks

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()
        images = self.vae_decoder.decode(latents.to(GET_DTYPE()), use_cache=True)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        return images

    def init_run(self):
        super().init_run()

    @ProfilingContext4DebugL1("End run segment")
    def end_run_segment(self, segment_idx=None):
        with ProfilingContext4DebugL1("step_pre_in_rerun"):
            self.model.scheduler.step_pre(seg_index=segment_idx, step_index=self.model.scheduler.infer_steps - 1, is_rerun=True)
        with ProfilingContext4DebugL1("🚀 infer_main_in_rerun"):
            self.model.infer(self.inputs)
        self.gen_video_final = torch.cat([self.gen_video_final, self.gen_video], dim=0) if self.gen_video_final is not None else self.gen_video

    @peak_memory_decorator
    def run_segment(self, total_steps=None):
        if total_steps is None:
            total_steps = self.model.scheduler.infer_steps
        for step_index in range(total_steps):
            # only for single segment, check stop signal every step
            if self.video_segment_num == 1:
                self.check_stop()
            logger.info(f"==> step_index: {step_index + 1} / {total_steps}")

            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(seg_index=self.segment_idx, step_index=step_index, is_rerun=False)

            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)

            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()

            if self.progress_callback:
                self.progress_callback(((step_index + 1) / total_steps) * 100, 100)

        return self.model.scheduler.stream_output
