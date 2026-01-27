import math
import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.runners.vsr.vsr_wrapper import compute_scaled_and_target_dims
from lightx2v_platform.base.global_var import AI_DEVICE


class NextControl:
    def __init__(self, action: str, data: any = None):
        # action: blank_to_voice, data: prev_video tensor
        # action: wait, data: None
        # action: fetch, data: None
        # action: switch_image, data: image_path
        # action: perform_action, data: action prompt
        self.action = action
        self.data = data


class VAController:
    def __init__(self, model_runner):
        self.reader = None
        self.recorder = None
        self.rank = 0
        self.world_size = 1
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.target_reader_rank = int(os.getenv("READER_RANK", "0")) % self.world_size
        self.target_recorder_rank = int(os.getenv("RECORDER_RANK", "0")) % self.world_size
        self.init_base(model_runner.config, model_runner.input_info, model_runner.vfi_model is not None, model_runner.vsr_model is not None)
        self.init_recorder()
        self.init_reader(model_runner)

    def init_base(self, config, input_info, has_vfi_model, has_vsr_model):
        if "stream_config" in input_info.__dataclass_fields__:
            self.stream_config = input_info.stream_config
            logger.info(f"VAController init base with stream config: {self.stream_config}")
        self.audio_path = input_info.audio_path
        self.output_video_path = input_info.save_result_path
        if isinstance(self.output_video_path, dict):
            self.output_video_path = self.output_video_path["data"]

        self.audio_sr = config.get("audio_sr", 16000)
        self.target_fps = config.get("target_fps", 16)
        self.max_num_frames = config.get("target_video_length", 81)
        self.prev_frame_length = config.get("prev_frame_length", 5)

        self.record_fps = config.get("target_fps", 16)
        if "video_frame_interpolation" in config and has_vfi_model:
            self.record_fps = config["video_frame_interpolation"]["target_fps"]
        self.record_fps = config.get("record_fps", self.record_fps)

        self.tgt_h = input_info.target_shape[0]
        self.tgt_w = input_info.target_shape[1]
        self.record_h, self.record_w = self.tgt_h, self.tgt_w
        if "video_super_resolution" in config and has_vsr_model:
            _, _, self.record_w, self.record_h = compute_scaled_and_target_dims(
                self.record_w,
                self.record_h,
                scale=config["video_super_resolution"]["scale"],
                multiple=128,
            )

        # how many frames to publish stream as a batch
        self.slice_frame = config.get("slice_frame", self.prev_frame_length)
        # estimate the max infer seconds, for immediate switch with local omni
        slice_interval = self.slice_frame / self.record_fps

        est_max_infer_secs = config.get("est_max_infer_secs", 0.6)
        est_max_switch_image_secs = config.get("est_max_switch_image_secs", 0)
        est_max_switch_action_secs = config.get("est_max_switch_action_secs", 0)

        self.est_infer_end_idx = math.ceil(est_max_infer_secs / slice_interval)
        self.est_switch_image_end_idx = math.ceil(est_max_switch_image_secs / slice_interval)
        self.est_switch_action_end_idx = math.ceil(est_max_switch_action_secs / slice_interval)

        max_end_idx = max(self.est_infer_end_idx, self.est_switch_image_end_idx, self.est_switch_action_end_idx)
        self.min_stay_queue_num = max_end_idx * 2 + 1

    def init_recorder(self):
        if not self.output_video_path or self.rank != self.target_recorder_rank:
            return
        logger.info(f"Rank {self.rank} init recorder with: {self.output_video_path}")
        whip_shared_path = os.getenv("WHIP_SHARED_LIB", None)
        if whip_shared_path and self.output_video_path.startswith("http"):
            from lightx2v.deploy.common.va_recorder_x264 import X264VARecorder

            self.recorder = X264VARecorder(
                whip_shared_path=whip_shared_path,
                livestream_url=self.output_video_path,
                fps=self.record_fps,
                sample_rate=self.audio_sr,
                slice_frame=self.slice_frame,
                prev_frame=self.prev_frame_length,
            )
        else:
            from lightx2v.deploy.common.va_recorder import VARecorder

            self.recorder = VARecorder(
                livestream_url=self.output_video_path,
                fps=self.record_fps,
                sample_rate=self.audio_sr,
                slice_frame=self.slice_frame,
                prev_frame=self.prev_frame_length,
                stream_config=self.stream_config,
            )

    def init_reader(self, model_runner=None):
        if not isinstance(self.audio_path, dict):
            return
        assert self.audio_path["type"] == "stream", f"unexcept audio_path: {self.audio_path}"
        segment_duration = self.max_num_frames / self.target_fps
        prev_duration = self.prev_frame_length / self.target_fps
        omni_work_dir = os.getenv("OMNI_WORK_DIR", None)
        if omni_work_dir:
            from lightx2v.deploy.common.va_reader_omni import OmniVAReader

            self.reader = OmniVAReader(
                rank=self.rank,
                world_size=self.world_size,
                stream_url=self.audio_path["data"],
                sample_rate=self.audio_sr,
                segment_duration=segment_duration,
                prev_duration=prev_duration,
                target_rank=self.target_reader_rank,
                model_runner=model_runner,
                huoshan_tts_voice_type=self.audio_path.get("huoshan_tts_voice_type", None),
                stream_config=self.stream_config,
                va_recorder=self.recorder,
            )
        else:
            from lightx2v.deploy.common.va_reader import VAReader

            self.reader = VAReader(
                rank=self.rank,
                world_size=self.world_size,
                stream_url=self.audio_path["data"],
                sample_rate=self.audio_sr,
                segment_duration=segment_duration,
                prev_duration=prev_duration,
                target_rank=self.target_reader_rank,
            )

    def start(self):
        self.reader.start()
        if self.rank == self.target_recorder_rank:
            assert self.recorder is not None, f"recorder is required for stream audio input for rank {self.rank}"
            self.recorder.start(self.record_w, self.record_h)
        if self.world_size > 1:
            dist.barrier()

    def next_control(self):
        from lightx2v.deploy.common.va_reader_omni import OmniVAReader

        if isinstance(self.reader, OmniVAReader):
            action_control = self.omni_reader_action_control()
            if action_control is not None:
                return action_control
            image_control = self.omni_reader_image_control()
            if image_control is not None:
                return image_control
            return self.omni_reader_next_control()
        return NextControl(action="fetch")

    def before_control(self):
        from lightx2v.deploy.common.va_reader_omni import OmniVAReader

        if isinstance(self.reader, OmniVAReader):
            self.len_tensor = torch.tensor([0], dtype=torch.int32, device=AI_DEVICE)
            self.flag_tensor = torch.tensor([0], dtype=torch.int32, device=AI_DEVICE)
            self.prev_tensor = torch.zeros((1, 3, self.prev_frame_length, self.tgt_h, self.tgt_w), dtype=torch.float, device=AI_DEVICE)

    def omni_reader_next_control(self):
        immediate_switch = self.reader.get_immediate_switch()
        if immediate_switch == 1:
            # truncate the stream buffer to keep the max infer time length
            # and broadcast the prev video tensor to all ranks
            if self.rank == self.target_recorder_rank:
                logger.warning(f"runner recv immediate switch, truncate stream buffer")
                video_tensor = self.recorder.truncate_stream_buffer(self.est_infer_end_idx)
                if video_tensor is not None:
                    self.flag_tensor.fill_(1)
                    self.prev_tensor.copy_(video_tensor)
                else:
                    self.flag_tensor.fill_(0)
            dist.broadcast(self.flag_tensor, src=self.target_recorder_rank)
            if self.flag_tensor.item() == 1:
                dist.broadcast(self.prev_tensor, src=self.target_recorder_rank)
                return NextControl(action="blank_to_voice", data=self.prev_tensor)
        else:
            # get the length of stream buffer, broadcast to all ranks
            if self.rank == self.target_recorder_rank:
                stream_buffer_length = self.recorder.get_buffer_stream_size()
                self.len_tensor.copy_(stream_buffer_length)
            dist.broadcast(self.len_tensor, src=self.target_recorder_rank)
            buffer_length = self.len_tensor.item()
            # stream buffer is enough, skip infer
            if buffer_length >= self.min_stay_queue_num:
                return NextControl(action="wait")
        return NextControl(action="fetch")

    def omni_reader_image_control(self):
        image_switch = self.reader.get_image_switch()
        if not isinstance(image_switch, str) or len(image_switch) == 0:
            return None
        if not os.path.exists(image_switch):
            logger.warning(f"Switch image path {image_switch} does not exist")
            return None
        # truncate the stream buffer to keep the max infer time length
        if self.rank == self.target_recorder_rank:
            logger.warning(f"runner recv image switch, truncate stream buffer")
            self.recorder.truncate_stream_buffer(self.est_switch_image_end_idx)
        return NextControl(action="switch_image", data=image_switch)

    def omni_reader_action_control(self):
        action_switch = self.reader.get_action_switch()
        if not isinstance(action_switch, str) or len(action_switch) == 0:
            return None
        # truncate the stream buffer to keep the max infer time length
        if self.rank == self.target_recorder_rank:
            logger.warning(f"runner recv action switch, truncate stream buffer")
            self.recorder.truncate_stream_buffer(self.est_switch_action_end_idx)
        return NextControl(action="perform_action", data=action_switch)

    def pub_livestream(self, images: torch.Tensor, audios: torch.Tensor, gen_video: torch.Tensor, valid_duration=1e9):
        if self.recorder.realtime:
            self.recorder.buffer_stream(images, audios, gen_video, valid_duration=valid_duration)
        else:
            self.recorder.pub_livestream(images, audios)

    def clear(self):
        self.len_tensor = None
        self.flag_tensor = None
        self.prev_tensor = None
        if self.reader is not None:
            try:
                self.reader.stop()
            except Exception as e:
                logger.warning(f"Error stopping reader: {e}")
            self.reader = None
        if self.recorder is not None:
            try:
                self.recorder.stop()
            except Exception as e:
                logger.warning(f"Error stopping recorder: {e}")
            self.recorder = None

    def __del__(self):
        self.clear()
