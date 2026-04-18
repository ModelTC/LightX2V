import torch
from loguru import logger

from lightx2v.models.networks.wan.lingbot_fast_model import WanLingbotFastModel
from lightx2v.models.runners.wan.wan_runner import LingbotRunner, WanRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.wan.lingbot_fast.scheduler import LingbotFastScheduler
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import wan_vae_to_comfy
from lightx2v.utils.video_recorder import VideoRecorder

try:
    import torch.distributed as dist
except ImportError:
    dist = None


@RUNNER_REGISTER("lingbot_world_fast")
class LingbotFastRunner(LingbotRunner):
    """Lingbot fast (autoregressive) runner.

    Inherits LingbotRunner for camera pose handling (set_inputs, load_image_encoder,
    get_encoder_output_i2v, and all camera helper methods).
    Adds SF scheduling and segment-based inference.
    """

    def __init__(self, config):
        WanRunner.__init__(self, config)
        self.control_type = config.get("control_type", "cam")
        self.is_live = config.get("is_live", False)
        if self.is_live:
            self.width = self.config["target_width"]
            self.height = self.config["target_height"]
            self.run_main = self.run_main_live

    def load_transformer(self):
        wan_model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = WanLingbotFastModel(**wan_model_kwargs)
        else:
            model = build_wan_model_with_lora(WanLingbotFastModel, self.config, wan_model_kwargs, lora_configs, model_type="wan2.1")
        return model

    # ---- SF scheduling ----

    def init_scheduler(self):
        self.scheduler = LingbotFastScheduler(self.config)

    def set_target_shape(self):
        num_frame_per_block = self.config["sf_config"].get("num_frame_per_block", 3)
        latent_shape = self.input_info.latent_shape
        lat_f = latent_shape[1]
        lat_h, lat_w = latent_shape[2], latent_shape[3]
        num_output_frames = lat_f - (lat_f % num_frame_per_block)
        self.input_info.latent_shape = [latent_shape[0], num_output_frames, lat_h, lat_w]
        self.config.target_shape = [self.config.get("num_channels_latents", 16), num_output_frames, lat_h, lat_w]

        self.scheduler.num_output_frames = num_output_frames
        self.scheduler.num_blocks = num_output_frames // num_frame_per_block

        p = self.config.get("patch_size", [1, 2, 2])
        frame_seq_length = (lat_h // p[1]) * (lat_w // p[2])
        logger.info(
            "[lingbot_fast] lat_f={}, num_output_frames={}, frame_seq_length={} (lat_h={}, lat_w={}, patch={})",
            lat_f,
            num_output_frames,
            frame_seq_length,
            lat_h,
            lat_w,
            p,
        )

        if hasattr(self, "model") and hasattr(self.model, "transformer_infer"):
            self.model.transformer_infer.reinit_caches(
                frame_seq_length,
                num_output_frames,
            )

    def get_video_segment_num(self):
        self.video_segment_num = self.scheduler.num_blocks

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")

            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(seg_index=segment_idx, step_index=step_index, is_rerun=False)

            with ProfilingContext4DebugL1("infer_main"):
                self.model.infer(self.inputs)

            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()

            if self.progress_callback:
                current_step = segment_idx * infer_steps + step_index + 1
                total_all_steps = self.video_segment_num * infer_steps
                self.progress_callback((current_step / total_all_steps) * 100, 100)

        return self.model.scheduler.stream_output

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self, total_steps=None):
        """Collect all segment latents, then decode at once with normal VAE.

        This matches the source code behavior in image2video_fast.py:
            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)
            videos = self.vae.decode([pred_latent_chunks])
        """
        self.set_target_shape()
        self.init_run()
        if self.config.get("compile", False):
            self.model.select_graph_for_compile(self.input_info)

        all_latents = []
        for segment_idx in range(self.video_segment_num):
            logger.info(f"start segment {segment_idx + 1}/{self.video_segment_num}")
            with ProfilingContext4DebugL1(
                f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                metrics_labels=["DefaultRunner"],
            ):
                self.check_stop()
                self.init_run_segment(segment_idx)
                latents = self.run_segment(segment_idx)
                all_latents.append(latents)

                with ProfilingContext4DebugL1("step_pre_in_rerun"):
                    self.model.scheduler.step_pre(
                        seg_index=segment_idx,
                        step_index=self.model.scheduler.infer_steps - 1,
                        is_rerun=True,
                    )
                with ProfilingContext4DebugL1("infer_main_in_rerun"):
                    self.model.infer(self.inputs)

                torch.cuda.empty_cache()

        all_latents = torch.cat(all_latents, dim=1)
        self.gen_video = self.run_vae_decoder(all_latents)
        self.gen_video_final = self.gen_video
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    # ---- Live streaming mode (per-segment decode, kept for future use) ----

    def get_rank_and_world_size(self):
        rank = 0
        world_size = 1
        if dist is not None and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        return rank, world_size

    def init_video_recorder(self):
        output_video_path = self.input_info.save_result_path
        self.video_recorder = None
        if isinstance(output_video_path, dict):
            output_video_path = output_video_path["data"]
        logger.info(f"init video_recorder with output_video_path: {output_video_path}")
        rank, world_size = self.get_rank_and_world_size()
        if output_video_path and rank == world_size - 1:
            record_fps = self.config.get("target_fps", 16)
            if "video_frame_interpolation" in self.config and self.vfi_model is not None:
                record_fps = self.config["video_frame_interpolation"]["target_fps"]
            self.video_recorder = VideoRecorder(
                livestream_url=output_video_path,
                fps=record_fps,
            )

    @ProfilingContext4DebugL1("End run segment")
    def end_run_segment(self, segment_idx=None):
        with ProfilingContext4DebugL1("step_pre_in_rerun"):
            self.model.scheduler.step_pre(seg_index=segment_idx, step_index=self.model.scheduler.infer_steps - 1, is_rerun=True)
        with ProfilingContext4DebugL1("infer_main_in_rerun"):
            self.model.infer(self.inputs)

        self.gen_video_final = torch.cat([self.gen_video_final, self.gen_video], dim=0) if self.gen_video_final is not None else self.gen_video
        if self.is_live:
            if self.video_recorder:
                stream_video = wan_vae_to_comfy(self.gen_video)
                self.video_recorder.pub_video(stream_video)

        torch.cuda.empty_cache()

    @ProfilingContext4DebugL2("Run DiT")
    def run_main_live(self, total_steps=None):
        try:
            self.init_video_recorder()
            logger.info(f"init video_recorder: {self.video_recorder}")
            rank, world_size = self.get_rank_and_world_size()
            if rank == world_size - 1:
                assert self.video_recorder is not None
                self.video_recorder.start(self.width, self.height)
            if world_size > 1 and dist is not None:
                dist.barrier()
            self.set_target_shape()
            self.init_run()
            if self.config.get("compile", False):
                self.model.select_graph_for_compile(self.input_info)

            for segment_idx in range(self.video_segment_num):
                logger.info(f"start segment {segment_idx + 1}/{self.video_segment_num}")
                with ProfilingContext4DebugL1(
                    f"segment end2end {segment_idx + 1}/{self.video_segment_num}",
                    recorder_mode=GET_RECORDER_MODE(),
                    metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                    metrics_labels=["DefaultRunner"],
                ):
                    self.check_stop()
                    self.init_run_segment(segment_idx)
                    latents = self.run_segment(segment_idx)
                    self.gen_video = self.run_vae_decoder(latents)
                    self.end_run_segment(segment_idx)
        finally:
            if hasattr(self.model, "inputs"):
                self.end_run()
            if self.video_recorder:
                self.video_recorder.stop()
                self.video_recorder = None
