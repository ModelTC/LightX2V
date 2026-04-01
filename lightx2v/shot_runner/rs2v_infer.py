import argparse
import json
import os

import numpy as np
import torch
import torchaudio as ta
from loguru import logger

from lightx2v.shot_runner.shot_base import ShotPipeline, load_clip_configs
from lightx2v.shot_runner.utils import RS2V_SlidingWindowReader
from lightx2v.utils.input_info import UNSET, calculate_target_video_length_from_duration, init_input_info_from_args
from lightx2v.utils.profiler import *
from lightx2v.utils.utils import seed_all, vae_to_comfyui_image, vae_to_comfyui_image_inplace
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

    @torch.no_grad()
    def generate(self, args):
        rs2v = self.clip_generators["rs2v_clip"]
        # 获取此clip模型的配置信息
        target_fps = rs2v.config.get("target_fps", 16)
        audio_sr = rs2v.config.get("audio_sr", 16000)
        audio_per_frame = audio_sr // target_fps

        # 获取用户输入信息
        clip_input_info = init_input_info_from_args(rs2v.config["task"], args)
        # 从默认配置中补全输入信息
        clip_input_info = self.check_input_info(clip_input_info, rs2v.config)

        # Auto-calculate target_video_length from video_duration if not explicitly provided
        if clip_input_info.target_video_length is None or clip_input_info.target_video_length == UNSET:
            if clip_input_info.video_duration is not None and clip_input_info.video_duration != UNSET:
                # Calculate for the first segment (max 5s)
                segment_duration = min(clip_input_info.video_duration, 5.0)
                clip_input_info.target_video_length = calculate_target_video_length_from_duration(segment_duration, target_fps)
                logger.info(f"Auto-calculated target_video_length={clip_input_info.target_video_length} from video_duration={clip_input_info.video_duration}s (segment={segment_duration}s)")
            else:
                # Fallback to config default
                clip_input_info.target_video_length = rs2v.config.get("target_video_length", 81)

        target_video_length = clip_input_info.target_video_length

        gen_video_list = []
        cut_audio_list = []
        video_duration = clip_input_info.video_duration

        def get_audio_files_from_audio_path(audio_path):
            if os.path.isdir(audio_path):
                audio_files = []
                mask_files = []
                audio_config_path = os.path.join(audio_path, "config.json")
                assert os.path.exists(audio_config_path), "config.json not found in audio_path"
                with open(audio_config_path, "r") as f:
                    audio_config = json.load(f)
                for talk_object in audio_config["talk_objects"]:
                    audio_files.append(os.path.join(audio_path, talk_object["audio"]))
                    mask_files.append(os.path.join(audio_path, talk_object["mask"]))
            else:
                audio_files = [audio_path]
                mask_files = None
            return audio_files, mask_files

        def load_audio(audio_path, target_sr):
            arr, ori_sr = ta.load(audio_path)
            arr = arr.mean(0)
            if ori_sr != target_sr:
                arr = ta.functional.resample(arr, ori_sr, target_sr)
            return arr

        audio_files, mask_files = get_audio_files_from_audio_path(clip_input_info.audio_path)
        clip_input_info.audio_num = len(audio_files)

        if len(audio_files) == 1:
            audio_array = load_audio(audio_files[0], audio_sr)
            audio_array = audio_array.unsqueeze(0)
        else:
            audio_arrays = []
            max_len = 0
            for a_file in audio_files:
                arr = load_audio(a_file, audio_sr)
                audio_arrays.append(arr)
                max_len = max(max_len, arr.numel())
            num_files = len(audio_arrays)
            audio_array = torch.zeros(num_files, max_len, dtype=torch.float32)
            for i, arr in enumerate(audio_arrays):
                length = arr.numel()
                audio_array[i, :length] = arr

        if video_duration is not None and video_duration > 0:
            max_samples = int(video_duration * audio_sr)
            if audio_array.shape[1] > max_samples:
                audio_array = audio_array[:, :max_samples]

        if mask_files is not None:
            mask_latents = [rs2v.process_single_mask(mask_file) for mask_file in mask_files]
            person_mask_latens = torch.cat(mask_latents, dim=0)
        else:
            person_mask_latens = None

        audio_reader = RS2V_SlidingWindowReader(audio_array, first_clip_len=target_video_length, clip_len=target_video_length + 3, sr=audio_sr, fps=target_fps)

        total_frames = int(np.ceil(audio_array.shape[1] / audio_per_frame))
        if total_frames <= target_video_length:
            total_clips = 1
        else:
            remaining = total_frames - target_video_length
            total_clips = 1 + int(np.ceil(remaining / (target_video_length + 3)))

        ref_state_sq = get_reference_state_sequence(target_video_length - 3, target_fps)

        # 预先运行输入编码的静态部分 (处理ref image的vae编码和文本编码)
        rs2v.input_info = clip_input_info
        rs2v.inputs_static = rs2v._run_input_encoder_local_rs2v_static()

        self.va_controller = VAController(rs2v)
        logger.info(f"init va_recorder: {self.va_controller.recorder} and va_reader: {self.va_controller.reader}")

        idx = 0
        while True:
            audio_clip, pad_len = audio_reader.next_frame()
            if audio_clip is None:
                break

            is_first = True if idx == 0 else False
            is_last = True if pad_len > 0 else False

            pipe = rs2v
            pipe.check_stop()

            # Calculate actual target_video_length for this segment based on audio length
            if is_last and pad_len > 0:
                # For the last segment with padding, calculate actual video frames needed
                actual_audio_samples = audio_clip.shape[1] - pad_len
                actual_video_frames = int(np.ceil(actual_audio_samples / audio_per_frame))
                # Apply the formula to ensure VAE stride constraint
                segment_target_video_length = calculate_target_video_length_from_duration(actual_video_frames / target_fps, target_fps)
                clip_input_info.target_video_length = segment_target_video_length

                # Recalculate latent_shape for this segment
                # latent_shape = [C, T, H, W] where T = (target_video_length - 1) // vae_stride[0] + 1
                if hasattr(clip_input_info, "latent_shape") and clip_input_info.latent_shape is not None:
                    original_latent_shape = clip_input_info.latent_shape
                    # Update only the temporal dimension (index 1)
                    new_latent_t = (segment_target_video_length - 1) // rs2v.config["vae_stride"][0] + 1
                    clip_input_info.latent_shape = [
                        original_latent_shape[0],  # C
                        new_latent_t,  # T
                        original_latent_shape[2],  # H
                        original_latent_shape[3],  # W
                    ]

                logger.info(
                    f"Segment {idx}: Last segment with pad_len={pad_len}, "
                    f"actual_video_frames={actual_video_frames}, "
                    f"calculated target_video_length={segment_target_video_length}, "
                    f"latent_shape={clip_input_info.latent_shape}"
                )
            else:
                # For first/middle segments, use the original target_video_length
                cur_clip_len = target_video_length if is_first else (target_video_length + 3)
                clip_input_info.target_video_length = cur_clip_len

                # Recalculate latent_shape for non-first segments
                if not is_first and hasattr(clip_input_info, "latent_shape") and clip_input_info.latent_shape is not None:
                    original_latent_shape = clip_input_info.latent_shape
                    new_latent_t = (cur_clip_len - 1) // rs2v.config["vae_stride"][0] + 1
                    clip_input_info.latent_shape = [
                        original_latent_shape[0],
                        new_latent_t,
                        original_latent_shape[2],
                        original_latent_shape[3],
                    ]

            clip_input_info.is_first = is_first
            clip_input_info.is_last = is_last
            clip_input_info.ref_state = ref_state_sq[idx % len(ref_state_sq)]
            clip_input_info.seed = clip_input_info.seed + idx
            clip_input_info.audio_clip = audio_clip
            idx = idx + 1
            if self.progress_callback:
                self.progress_callback(idx, total_clips)

            rs2v.input_info = clip_input_info
            clip_input_info.person_mask_latens = person_mask_latens

            # 使用动态输入获取当前 clip 控制参数
            rs2v.inputs = rs2v._run_input_encoder_local_rs2v_dynamic()

            gen_clip_video, audio_clip, gen_latents = rs2v.run_clip_main()
            logger.info(f"Generated rs2v clip {idx}, pad_len {pad_len}, gen_clip_video shape: {gen_clip_video.shape}, audio_clip shape: {audio_clip.shape} gen_latents shape: {gen_latents.shape}")

            video_pad_len = pad_len // audio_per_frame
            audio_pad_len = video_pad_len * audio_per_frame
            video_seg = gen_clip_video[:, :, : gen_clip_video.shape[2] - video_pad_len]
            # Since audio_clip is now multidimensional (N, T), slice on dim 1 and sum on dim 0 to merge tracks
            audio_seg = audio_clip[:, : audio_clip.shape[1] - audio_pad_len].sum(dim=0)
            clip_input_info.overlap_latent = gen_latents[:, -1:]

            if clip_input_info.return_result_tensor:
                gen_video_list.append(video_seg.clone().cpu().float())
                cut_audio_list.append(audio_seg.cpu())
            elif self.va_controller.recorder is not None:
                video_seg = torch.clamp(video_seg, -1, 1).to(torch.float).cpu()
                video_seg = vae_to_comfyui_image_inplace(video_seg)
                self.va_controller.pub_livestream(video_seg, audio_seg, None)

        if not clip_input_info.return_result_tensor:
            return None, None, None

        gen_lvideo = torch.cat(gen_video_list, dim=2).float()
        gen_lvideo = torch.clamp(gen_lvideo, -1, 1)
        merge_audio = torch.cat(cut_audio_list, dim=0).numpy().astype(np.float32)

        return gen_lvideo, merge_audio, audio_sr

    def run_pipeline(self, input_info):
        # input_info = self.update_input_info(input_info)
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
