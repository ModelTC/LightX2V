import os
import gc
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.schedulers.wan.step_distill.scheduler import WanStepDistillScheduler
from lightx2v.utils.profiler import ProfilingContext4Debug, ProfilingContext
from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.xlm_roberta.model import CLIPModel
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.causvid_model import WanCausVidModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from loguru import logger
import torch.distributed as dist


@RUNNER_REGISTER("wan2.1_causvid")
class WanCausVidRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        self.num_frame_per_block = self.config.num_frame_per_block
        self.num_frames = self.config.num_frames
        self.frame_seq_length = self.config.frame_seq_length
        self.infer_blocks = self.config.num_blocks
        self.num_fragments = self.config.num_fragments

    def load_transformer(self):
        if self.config.get("lora_configs") and self.config.lora_configs:
            model = WanModel(
                self.config.model_path,
                self.config,
                self.init_device,
            )
            lora_wrapper = WanLoraWrapper(model)
            for lora_config in self.config.lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")
        else:
            model = WanCausVidModel(self.config.model_path, self.config, self.init_device)
        return model

    def set_inputs(self, inputs):
        super().set_inputs(inputs)
        self.config["num_fragments"] = inputs.get("num_fragments", 1)
        self.num_fragments = self.config["num_fragments"]

    def init_scheduler(self):
        scheduler = WanStepDistillScheduler(self.config)
        self.model.set_scheduler(scheduler)

    def set_target_shape(self):
        if self.config.task == "i2v":
            self.config.target_shape = (16, self.config.num_frame_per_block, self.config.lat_h, self.config.lat_w)
            # i2v需根据input shape重置frame_seq_length
            frame_seq_length = (self.config.lat_h // 2) * (self.config.lat_w // 2)
            self.model.transformer_infer.frame_seq_length = frame_seq_length
            self.frame_seq_length = frame_seq_length
        elif self.config.task == "t2v":
            self.config.target_shape = (
                16,
                self.config.num_frame_per_block,
                int(self.config.target_height) // self.config.vae_stride[1],
                int(self.config.target_width) // self.config.vae_stride[2],
            )

    def run(self):
        self.model.transformer_infer._init_kv_cache(dtype=torch.bfloat16, device="cuda")
        self.model.transformer_infer._init_crossattn_cache(dtype=torch.bfloat16, device="cuda")

        output_latents = torch.zeros(
            (self.model.config.target_shape[0], self.num_frames + (self.num_fragments - 1) * (self.num_frames - self.num_frame_per_block), *self.model.config.target_shape[2:]),
            device="cuda",
            dtype=torch.bfloat16,
        )

        start_block_idx = 0

        for fragment_idx in range(self.num_fragments):
            logger.info(f"========> fragment_idx: {fragment_idx + 1} / {self.num_fragments}")

            kv_start = 0
            kv_end = kv_start + self.num_frame_per_block * self.frame_seq_length

            if fragment_idx > 0:
                logger.info("recompute the kv_cache ...")
                with ProfilingContext4Debug("step_pre"):
                    self.model.scheduler.latents = self.model.scheduler.last_sample
                    self.model.scheduler.step_pre(step_index=self.model.scheduler.infer_steps - 1)

                with ProfilingContext4Debug("infer"):
                    self.model.infer(self.inputs, kv_start, kv_end)

                kv_start += self.num_frame_per_block * self.frame_seq_length
                kv_end += self.num_frame_per_block * self.frame_seq_length

            infer_blocks = self.infer_blocks - (fragment_idx > 0)

            for block_idx in range(infer_blocks):
                logger.info(f"=====> block_idx: {block_idx + 1} / {infer_blocks}")
                logger.info(f"=====> kv_start: {kv_start}, kv_end: {kv_end}")
                self.model.scheduler.reset()

                for step_index in range(self.model.scheduler.infer_steps):
                    logger.info(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")

                    with ProfilingContext4Debug("step_pre"):
                        self.model.scheduler.step_pre(step_index=step_index)

                    with ProfilingContext4Debug("infer"):
                        self.model.infer(self.inputs, kv_start, kv_end)

                    with ProfilingContext4Debug("step_post"):
                        self.model.scheduler.step_post()

                kv_start += self.num_frame_per_block * self.frame_seq_length
                kv_end += self.num_frame_per_block * self.frame_seq_length

                output_latents[:, start_block_idx * self.num_frame_per_block : (start_block_idx + 1) * self.num_frame_per_block] = self.model.scheduler.latents
                start_block_idx += 1

        return output_latents, self.model.scheduler.generator

    def end_run(self):
        self.model.scheduler.clear()
        del self.inputs, self.model.scheduler, self.model.transformer_infer.kv_cache, self.model.transformer_infer.crossattn_cache
        gc.collect()
        torch.cuda.empty_cache()
