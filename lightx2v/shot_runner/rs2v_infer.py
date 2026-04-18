import argparse
import json
import os

import numpy as np
import torch
import torchaudio as ta
from loguru import logger

from lightx2v.shot_runner.shot_base import ShotPipeline, load_clip_configs
from lightx2v.shot_runner.utils import RS2V_SlidingWindowReader, save_audio, save_to_video
from lightx2v.utils.input_info import UNSET, calculate_target_video_length_from_duration, init_input_info_from_args
from lightx2v.utils.profiler import *
from lightx2v.utils.utils import is_main_process, seed_all, vae_to_comfyui_image, vae_to_comfyui_image_inplace
from lightx2v.utils.va_controller import VAController


def get_reference_state_sequence(frames_per_clip=17, target_fps=16):
    duration = frames_per_clip / target_fps
    if duration > 3:
        inner_every = 2
    else:
        inner_every = 6
    return [0] + [1] * (inner_every - 1)


class ShotRS2VPipeline(ShotPipeline):  # type:ignore
    def __init__(self, clip_configs):
        super().__init__(clip_configs)

    @staticmethod
    def _parse_audio_path(audio_path):
        if os.path.isdir(audio_path):
            audio_config_path = os.path.join(audio_path, "config.json")
            assert os.path.exists(audio_config_path), "config.json not found in audio_path"
            with open(audio_config_path, "r") as f:
                audio_config = json.load(f)
            audio_files = [os.path.join(audio_path, obj["audio"]) for obj in audio_config["talk_objects"]]
            mask_files = [os.path.join(audio_path, obj["mask"]) for obj in audio_config["talk_objects"]]
        else:
            audio_files = [audio_path]
            mask_files = None
        return audio_files, mask_files

    @staticmethod
    def _load_single_audio(audio_path, target_sr):
        arr, ori_sr = ta.load(audio_path)
        arr = arr.mean(0)
        if ori_sr != target_sr:
            arr = ta.functional.resample(arr, ori_sr, target_sr)
        return arr

    @classmethod
    def _load_audio_array(cls, audio_files, audio_sr, video_duration):
        if len(audio_files) == 1:
            audio_array = cls._load_single_audio(audio_files[0], audio_sr).unsqueeze(0)
        else:
            arrays = [cls._load_single_audio(f, audio_sr) for f in audio_files]
            max_len = max(a.numel() for a in arrays)
            audio_array = torch.zeros(len(arrays), max_len, dtype=torch.float32)
            for i, arr in enumerate(arrays):
                audio_array[i, : arr.numel()] = arr

        if video_duration is not None and video_duration > 0:
            max_samples = int(video_duration * audio_sr)
            if audio_array.shape[1] > max_samples:
                audio_array = audio_array[:, :max_samples]

        return audio_array

    @staticmethod
    def _load_mask_latents(rs2v, mask_files):
        if mask_files is None:
            return None
        latents = [rs2v.process_single_mask(f) for f in mask_files]
        return torch.cat(latents, dim=0)

    @staticmethod
    def _calc_total_clips(total_samples, audio_per_frame, target_video_length):
        total_frames = int(np.ceil(total_samples / audio_per_frame))
        if total_frames <= target_video_length:
            return 1
        remaining = total_frames - target_video_length
        return 1 + int(np.ceil(remaining / (target_video_length + 3)))

    @staticmethod
    def _update_latent_shape(clip_input_info, target_len, vae_stride):
        if hasattr(clip_input_info, "latent_shape") and clip_input_info.latent_shape is not None:
            s = clip_input_info.latent_shape
            new_t = (target_len - 1) // vae_stride + 1
            clip_input_info.latent_shape = [s[0], new_t, s[2], s[3]]

    def _compute_segment_params(self, idx, audio_clip, pad_len, target_video_length, target_fps, audio_per_frame, vae_stride, clip_input_info):
        """Compute per-segment parameters (target_video_length, latent_shape, trimmed audio_clip).

        Returns:
            (is_first, is_last, segment_actual_video_frames, audio_clip)
        """
        is_first = idx == 0
        is_last = pad_len > 0
        segment_actual_video_frames = None

        if is_last:
            actual_audio_samples = audio_clip.shape[1] - pad_len
            actual_video_frames = int(np.ceil(actual_audio_samples / audio_per_frame))
            segment_actual_video_frames = actual_video_frames

            seg_target_len = calculate_target_video_length_from_duration(actual_video_frames / target_fps, target_fps)
            clip_input_info.target_video_length = seg_target_len
            self._update_latent_shape(clip_input_info, seg_target_len, vae_stride)

            logger.info(
                f"Segment {idx}: Last segment with pad_len={pad_len}, "
                f"actual_video_frames={actual_video_frames}, "
                f"calculated target_video_length={seg_target_len}, "
                f"latent_shape={clip_input_info.latent_shape}"
            )
            audio_clip = audio_clip[:, : clip_input_info.target_video_length * audio_per_frame]
        else:
            cur_clip_len = target_video_length if is_first else (target_video_length + 3)
            clip_input_info.target_video_length = cur_clip_len
            if not is_first:
                self._update_latent_shape(clip_input_info, cur_clip_len, vae_stride)

        return is_first, is_last, segment_actual_video_frames, audio_clip

    @staticmethod
    def _trim_segment(gen_clip_video, audio_clip, segment_actual_video_frames, audio_per_frame):
        if segment_actual_video_frames is not None:
            video_seg = gen_clip_video[:, :, :segment_actual_video_frames]
            audio_seg = audio_clip[:, : segment_actual_video_frames * audio_per_frame].sum(dim=0)
        else:
            video_seg = gen_clip_video
            audio_seg = audio_clip.sum(dim=0)
        return video_seg, audio_seg

    @staticmethod
    def _merge_and_save(gen_video_list, cut_audio_list, target_fps, save_result_path):
        gen_lvideo = torch.cat(gen_video_list, dim=2).float()
        gen_lvideo = torch.clamp(gen_lvideo, -1, 1)
        merge_audio = torch.cat(cut_audio_list, dim=0).numpy().astype(np.float32)

        if is_main_process() and save_result_path:
            out_path = os.path.join("./", "video_merge.mp4")
            audio_file = os.path.join("./", "audio_merge.wav")
            save_to_video(gen_lvideo, out_path, target_fps)
            save_audio(merge_audio, audio_file, out_path, output_path=save_result_path)
            os.remove(out_path)
            os.remove(audio_file)

        return gen_lvideo, merge_audio

    @torch.no_grad()
    def generate(self, args):
        rs2v = self.clip_generators["rs2v_clip"]

        target_fps = rs2v.config.get("target_fps", 16)
        audio_sr = rs2v.config.get("audio_sr", 16000)
        audio_per_frame = audio_sr // target_fps
        vae_stride = rs2v.config["vae_stride"][0]

        clip_input_info = init_input_info_from_args(rs2v.config["task"], args)
        clip_input_info = self.check_input_info(clip_input_info, rs2v.config)

        if clip_input_info.target_video_length is None or clip_input_info.target_video_length == UNSET:
            if clip_input_info.video_duration is not None and clip_input_info.video_duration != UNSET:
                segment_duration = min(clip_input_info.video_duration, 5.0)
                clip_input_info.target_video_length = calculate_target_video_length_from_duration(segment_duration, target_fps)
                logger.info(f"Auto-calculated target_video_length={clip_input_info.target_video_length} from video_duration={clip_input_info.video_duration}s (segment={segment_duration}s)")
            else:
                clip_input_info.target_video_length = rs2v.config.get("target_video_length", 81)

        target_video_length = clip_input_info.target_video_length
        base_seed = clip_input_info.seed

        audio_files, mask_files = self._parse_audio_path(clip_input_info.audio_path)
        clip_input_info.audio_num = len(audio_files)

        audio_array = self._load_audio_array(audio_files, audio_sr, clip_input_info.video_duration)
        person_mask_latens = self._load_mask_latents(rs2v, mask_files)

        audio_reader = RS2V_SlidingWindowReader(audio_array, first_clip_len=target_video_length, clip_len=target_video_length + 3, sr=audio_sr, fps=target_fps)
        total_clips = self._calc_total_clips(audio_array.shape[1], audio_per_frame, target_video_length)
        ref_state_seq = get_reference_state_sequence(target_video_length - 3, target_fps)

        rs2v.input_info = clip_input_info
        rs2v.inputs_static = rs2v._run_input_encoder_local_rs2v_static()

        self.va_controller = None
        if clip_input_info.stream_save_video:
            self.va_controller = VAController(rs2v)
            logger.info(f"init va_recorder: {self.va_controller.recorder} and va_reader: {self.va_controller.reader}")

        gen_video_list = []
        cut_audio_list = []

        for idx, (audio_clip, pad_len) in enumerate(iter(audio_reader.next_frame, (None, 0))):
            if audio_clip is None:
                break
            rs2v.check_stop()

            is_first, is_last, segment_actual_frames, audio_clip = self._compute_segment_params(idx, audio_clip, pad_len, target_video_length, target_fps, audio_per_frame, vae_stride, clip_input_info)

            clip_input_info.is_first = is_first
            clip_input_info.is_last = is_last
            clip_input_info.ref_state = ref_state_seq[idx % len(ref_state_seq)]
            clip_input_info.seed = base_seed + idx
            clip_input_info.audio_clip = audio_clip
            clip_input_info.person_mask_latens = person_mask_latens

            if self.progress_callback:
                self.progress_callback(idx + 1, total_clips)

            rs2v.input_info = clip_input_info
            rs2v.inputs = rs2v._run_input_encoder_local_rs2v_dynamic()
            gen_clip_video, audio_clip, gen_latents = rs2v.run_clip_main()

            logger.info(f"Generated rs2v clip {idx + 1}, pad_len={pad_len}, gen_clip_video={gen_clip_video.shape}, audio_clip={audio_clip.shape}, gen_latents={gen_latents.shape}")

            video_seg, audio_seg = self._trim_segment(gen_clip_video, audio_clip, segment_actual_frames, audio_per_frame)
            clip_input_info.overlap_latent = gen_latents[:, -1:]

            if clip_input_info.return_result_tensor or not clip_input_info.stream_save_video:
                gen_video_list.append(video_seg.clone().cpu())
                cut_audio_list.append(audio_seg.cpu())
            elif self.va_controller.recorder is not None:
                video_seg = torch.clamp(video_seg, -1, 1).to(torch.float).cpu()
                video_seg = vae_to_comfyui_image_inplace(video_seg)
                self.va_controller.pub_livestream(video_seg, audio_seg, None)

        if not clip_input_info.return_result_tensor and clip_input_info.stream_save_video:
            return None, None, None

        gen_lvideo, merge_audio = self._merge_and_save(gen_video_list, cut_audio_list, target_fps, clip_input_info.save_result_path)
        return gen_lvideo, merge_audio, audio_sr

    def run_pipeline(self, input_info):
        try:
            gen_lvideo, merge_audio, audio_sr = self.generate(input_info)
        finally:
            if self.va_controller is not None:
                self.va_controller.clear()
                self.va_controller = None
        if isinstance(input_info, dict):
            return_result_tensor = input_info.get("return_result_tensor", False)
        else:
            return_result_tensor = getattr(input_info, "return_result_tensor", False)
        if return_result_tensor:
            video = vae_to_comfyui_image(gen_lvideo)
            audio_tensor = torch.from_numpy(merge_audio).float()
            audio_waveform = audio_tensor.unsqueeze(0).unsqueeze(0)
            return {"video": video, "audio": {"waveform": audio_waveform, "sample_rate": audio_sr}}
        return {"video": None, "audio": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="The seed for random generator")
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default="", help="The path to input image file for image-to-video (i2v) task")
    parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file or directory for audio-to-video (s2v) task")
    parser.add_argument("--save_result_path", type=str, default=None, help="The path to save video path/file")
    parser.add_argument("--return_result_tensor", action="store_true", help="Whether to return result tensor. (Useful for comfyui)")
    parser.add_argument("--target_shape", nargs="+", default=[], help="Set return video or image shape")
    parser.add_argument("--infer_steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--target_video_length", type=int, default=81, help="The target video length for each generated clip")
    parser.add_argument("--video_duration", type=float, default=20, help="Video duration in seconds")
    parser.add_argument("--stream_save_video", action="store_true", help="Whether to save video by stream")

    args = parser.parse_args()

    seed_all(args.seed)
    clip_configs = load_clip_configs(args.config_json)

    with ProfilingContext4DebugL1("Init Pipeline Cost Time"):
        shot_rs2v_pipe = ShotRS2VPipeline(clip_configs)

    with ProfilingContext4DebugL1("Generate Cost Time"):
        shot_rs2v_pipe.generate(args)

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
