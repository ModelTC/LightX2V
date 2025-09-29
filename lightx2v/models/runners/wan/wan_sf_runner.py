import torch
from loguru import logger

from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.networks.wan.sf_model import WanSFModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
from lightx2v.models.video_encoders.hf.wan.vae_sf import WanSFVAE
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER

torch.manual_seed(42)


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
            assert not self.config.get("dit_quantized", False) or self.config.mm_config.get("weight_auto_quant", False)
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

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self, total_steps=None):
        self.init_run()
        if self.config.get("compile", False):
            self.model.select_graph_for_compile()

        total_blocks = self.scheduler.num_blocks
        for seg_index in range(self.video_segment_num):
            logger.info(f"==> segment_index: {seg_index + 1} / {total_blocks}")

            total_steps = len(self.scheduler.denoising_step_list)
            for step_index in range(total_steps):
                logger.info(f"==> step_index: {step_index + 1} / {total_steps}")

                with ProfilingContext4DebugL1("step_pre"):
                    self.model.scheduler.step_pre(seg_index=seg_index, step_index=step_index, is_rerun=False)

                with ProfilingContext4DebugL1("🚀 infer_main"):
                    self.model.infer(self.inputs)

                with ProfilingContext4DebugL1("step_post"):
                    self.model.scheduler.step_post()

            # rerun with timestep zero to update KV cache using clean context
            with ProfilingContext4DebugL1("step_pre_in_rerun"):
                self.model.scheduler.step_pre(seg_index=seg_index, step_index=step_index, is_rerun=True)

            with ProfilingContext4DebugL1("🚀 infer_main_in_rerun"):
                self.model.infer(self.inputs)

        latents = self.model.scheduler.latents
        # la = torch.split(latents, 3, dim=1)
        # print((la[2]-la[3]).sum())
        # breakpoint()

        self.gen_video = self.run_vae_decoder(latents)

        self.end_run()
