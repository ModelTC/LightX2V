import gc
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from loguru import logger

from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.df.skyreels_v2_df_scheduler import WanSkyreelsV2DFScheduler
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import ProfilingContext, ProfilingContext4Debug
from lightx2v.utils.registry_factory import RUNNER_REGISTER


@RUNNER_REGISTER("wan2.1_skyreels_v2_df")
class WanSkyreelsV2DFRunner(WanRunner):  # Diffustion foring for SkyReelsV2 DF I2V/T2V
    def __init__(self, config):
        super().__init__(config)

    def init_scheduler(self):
        scheduler = WanSkyreelsV2DFScheduler(self.config)
        self.model.set_scheduler(scheduler)

    def run_image_encoder(self, config, image_encoder, vae_model):
        img = Image.open(config.image_path).convert("RGB")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        h, w = img.shape[1:]
        aspect_ratio = h / w
        max_area = config.target_height * config.target_width
        lat_h = round(np.sqrt(max_area * aspect_ratio) // config.vae_stride[1] // config.patch_size[1] * config.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // config.vae_stride[2] // config.patch_size[2] * config.patch_size[2])
        h = lat_h * config.vae_stride[1]
        w = lat_w * config.vae_stride[2]

        config.lat_h = lat_h
        config.lat_w = lat_w

        vae_encoder_out = vae_model.encode([torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1).cuda()])[0]
        vae_encoder_out = vae_encoder_out.to(GET_DTYPE())
        return vae_encoder_out

    def set_target_shape(self):
        if os.path.isfile(self.config.image_path):
            self.config.target_shape = (16, (self.config.target_video_length - 1) // 4 + 1, self.config.lat_h, self.config.lat_w)
        else:
            self.config.target_shape = (
                16,
                (self.config.target_video_length - 1) // 4 + 1,
                int(self.config.target_height) // self.config.vae_stride[1],
                int(self.config.target_width) // self.config.vae_stride[2],
            )

    def run_input_encoder(self):
        image_encoder_output = None
        if os.path.isfile(self.config.image_path):
            with ProfilingContext("Run Img Encoder"):
                image_encoder_output = self.run_image_encoder(self.config, self.image_encoder, self.vae_model)
        with ProfilingContext("Run Text Encoder"):
            text_encoder_output = self.run_text_encoder(self.config["prompt"], self.text_encoders, self.config, image_encoder_output)
        self.set_target_shape()
        self.inputs = {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output}

        gc.collect()
        torch.cuda.empty_cache()

    def run(self):
        num_frames = self.config.num_frames
        overlap_history = self.config.overlap_history
        base_num_frames = self.config.base_num_frames
        addnoise_condition = self.config.addnoise_condition
        causal_block_size = self.config.causal_block_size

        latent_length = (num_frames - 1) // 4 + 1
        base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else latent_length
        overlap_history_frames = (overlap_history - 1) // 4 + 1
        n_iter = 1 + (latent_length - base_num_frames - 1) // (base_num_frames - overlap_history_frames) + 1

        prefix_video = self.inputs["image_encoder_output"]
        predix_video_latent_length = 0
        if prefix_video is not None:
            predix_video_latent_length = prefix_video.size(1)

        output_video = None
        logger.info(f"Diffusion-Forcing n_iter:{n_iter}")
        for i in range(n_iter):
            if output_video is not None:  # i !=0
                prefix_video = output_video[:, :, -overlap_history:].to(self.model.scheduler.device)
                prefix_video = self.vae_model.encode(prefix_video)[0]  # [(b, c, f, h, w)]
                if prefix_video.shape[1] % causal_block_size != 0:
                    truncate_len = prefix_video.shape[1] % causal_block_size
                    # the length of prefix video is truncated for the casual block size alignment.
                    prefix_video = prefix_video[:, : prefix_video.shape[1] - truncate_len]
                predix_video_latent_length = prefix_video.shape[1]
                finished_frame_num = i * (base_num_frames - overlap_history_frames) + overlap_history_frames
                left_frame_num = latent_length - finished_frame_num
                base_num_frames_iter = min(left_frame_num + overlap_history_frames, base_num_frames)
            else:  # i == 0
                base_num_frames_iter = base_num_frames

            if prefix_video is not None:
                input_dtype = self.model.scheduler.latents.dtype
                self.model.scheduler.latents[:, :predix_video_latent_length] = prefix_video.to(input_dtype)

            self.model.scheduler.generate_timestep_matrix(base_num_frames_iter, base_num_frames_iter, addnoise_condition, predix_video_latent_length, causal_block_size)

            for step_index in range(self.model.scheduler.infer_steps):
                logger.info(f"==> step_index: {step_index + 1} / {self.model.scheduler.infer_steps}")
                with ProfilingContext4Debug("step_pre"):
                    self.model.scheduler.step_pre(step_index=step_index)

                with ProfilingContext4Debug("🚀 infer_main"):
                    self.model.infer(self.inputs)

                with ProfilingContext4Debug("step_post"):
                    self.model.scheduler.step_post()

            videos = self.run_vae(self.model.scheduler.latents, self.model.scheduler.generator)
            self.model.scheduler.prepare(self.inputs["image_encoder_output"])  # reset
            if output_video is None:
                output_video = videos.clamp(-1, 1).cpu()  # b, c, f, h, w
            else:
                output_video = torch.cat([output_video, videos[:, :, overlap_history:].clamp(-1, 1).cpu()], 2)
        return output_video

    def run_pipeline(self):
        self.init_scheduler()
        self.run_input_encoder()
        self.model.scheduler.prepare()
        output_video = self.run()
        self.end_run()
        self.save_video(output_video)
