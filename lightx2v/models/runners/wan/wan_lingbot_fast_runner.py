import gc

import torch
from loguru import logger

from lightx2v.common.kvcache import KVCacheManager
from lightx2v.models.networks.wan.lingbot_fast_model import WanLingbotFastModel
from lightx2v.models.runners.wan.wan_runner import LingbotRunner, WanRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.async_vae import AsyncVAEChunkDecoder
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import get_rank_and_world_size
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

    def init_scheduler(self):
        self.scheduler = WanSFScheduler(self.config)

    @ProfilingContext4DebugL1("init kv cache manager")
    def init_kv_cache_manager(self):
        kv_mgr = getattr(self.model, "kv_cache_manager", None)
        if kv_mgr is None:
            kv_mgr = KVCacheManager(config=self.config, device=torch.device("cuda"), sp_group=self.model.seq_p_group)
            self.model.kv_cache_manager = kv_mgr
        kv_mgr.ar_config = dict(self.config.get("ar_config", {}))
        kv_mgr._create_kv_caches(self.input_info.latent_shape)
        self.model.transformer_infer.kv_cache_manager = kv_mgr
        self.input_info.latent_shape = [self.input_info.latent_shape[0], kv_mgr.num_output_frames, self.input_info.latent_shape[2], self.input_info.latent_shape[3]]
        self.scheduler.num_output_frames = kv_mgr.num_output_frames
        self.scheduler.num_chunks = kv_mgr.num_output_frames // self.config.get("ar_config", {}).get("num_frame_per_chunk", 3)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sp_group = getattr(self.model, "seq_p_group", None)
            if sp_group is not None or torch.distributed.get_world_size() > 1:
                torch.distributed.barrier(group=sp_group)

    def get_video_segment_num(self):
        self.video_segment_num = self.scheduler.num_chunks

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")
            self.model.kv_cache_manager.current_step = step_index

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

    def decode_segment_latents(self, segment_idx: int, segment_latents: torch.Tensor) -> torch.Tensor:
        is_first = segment_idx == 0
        is_last = segment_idx == self.video_segment_num - 1
        return self.vae_decoder.cached_decode_withflag(segment_latents.to(GET_DTYPE()), is_first, is_last)

    def init_run(self):
        self.init_kv_cache_manager()
        super().init_run()

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self, total_steps=None):
        self.init_run()
        if self.config.get("compile", False):
            self.model.select_graph_for_compile(self.input_info)

        lazy_vae = self.config.get("lazy_load", False) or self.config.get("unload_modules", False)
        if lazy_vae:
            self.vae_decoder = self.load_vae_decoder()
        vae_decoder = AsyncVAEChunkDecoder.from_config(self.config, device=torch.device("cuda"), vae_decoder=self.vae_decoder)

        with no_sync_profiling(enabled=vae_decoder.is_async):
            with ProfilingContext4DebugL1(
                f"AR chunk total {self.video_segment_num} chunks",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                metrics_labels=["DefaultRunner"],
            ):
                try:
                    for segment_idx in range(self.video_segment_num):
                        logger.info(f"start chunk {segment_idx + 1}/{self.video_segment_num}")
                        with ProfilingContext4DebugL1(
                            f"chunk end2end {segment_idx + 1}/{self.video_segment_num}",
                            recorder_mode=GET_RECORDER_MODE(),
                            metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                            metrics_labels=["DefaultRunner"],
                        ):
                            self.check_stop()
                            self.init_run_segment(segment_idx)
                            latents = self.run_segment(segment_idx)

                            with ProfilingContext4DebugL1("step_pre_in_rerun"):
                                self.model.scheduler.step_pre(
                                    seg_index=segment_idx,
                                    step_index=self.model.scheduler.infer_steps - 1,
                                    is_rerun=True,
                                )
                            with ProfilingContext4DebugL1("infer_main_in_rerun"):
                                self.model.infer(self.inputs)

                        vae_decoder.submit(self.decode_segment_latents, segment_idx, latents)
                        torch.cuda.empty_cache()
                    decoded_chunks = vae_decoder.finish()
                finally:
                    if "vae_decoder" in locals():
                        vae_decoder.finish()
                    if lazy_vae:
                        del self.vae_decoder
                        torch.cuda.empty_cache()
                        gc.collect()

        self.gen_video = torch.cat(decoded_chunks, dim=2)
        self.gen_video_final = self.gen_video
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final

    def init_video_recorder(self):
        output_video_path = self.input_info.save_result_path
        self.video_recorder = None
        if isinstance(output_video_path, dict):
            output_video_path = output_video_path["data"]
        logger.info(f"init video_recorder with output_video_path: {output_video_path}")
        rank, world_size = get_rank_and_world_size()
        if output_video_path and rank == world_size - 1:
            record_fps = self.config.get("target_fps", 16)
            if "video_frame_interpolation" in self.config and self.vfi_model is not None:
                record_fps = self.config["video_frame_interpolation"]["target_fps"]
            self.video_recorder = VideoRecorder(
                livestream_url=output_video_path,
                fps=record_fps,
            )
