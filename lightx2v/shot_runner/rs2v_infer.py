import argparse
import os

import numpy as np
import torch
import torchaudio as ta
from loguru import logger

from lightx2v.shot_runner.shot_base import ShotPipeline, load_clip_configs
from lightx2v.shot_runner.utils import RS2V_SlidingWindowReader, save_audio, save_to_video
from lightx2v.utils.input_info import init_input_info_from_args
from lightx2v.utils.profiler import *
from lightx2v.utils.utils import is_main_process, seed_all, vae_to_comfyui_image


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
        target_video_length = rs2v.config.get("target_video_length", 81)
        target_fps = rs2v.config.get("target_fps", 16)
        audio_sr = rs2v.config.get("audio_sr", 16000)
        audio_per_frame = audio_sr // target_fps

        # 获取用户输入信息
        clip_input_info = init_input_info_from_args(rs2v.config["task"], args, infer_steps=3, video_duration=20)
        # 从默认配置中补全输入信息
        clip_input_info = self.check_input_info(clip_input_info, rs2v.config)

        gen_video_list = []
        cut_audio_list = []
        video_duration = clip_input_info.video_duration
        audio_array, ori_sr = ta.load(clip_input_info.audio_path)
        audio_array = audio_array.mean(0)
        if ori_sr != audio_sr:
            audio_array = ta.functional.resample(audio_array, ori_sr, audio_sr)
        if video_duration is not None and video_duration > 0:
            max_samples = int(video_duration * audio_sr)
            if audio_array.numel() > max_samples:
                audio_array = audio_array[:max_samples]
        audio_reader = RS2V_SlidingWindowReader(audio_array, first_clip_len=target_video_length, clip_len=target_video_length + 3, sr=audio_sr, fps=target_fps)

        total_frames = int(np.ceil(audio_array.numel() / audio_per_frame))
        if total_frames <= target_video_length:
            total_clips = 1
        else:
            remaining = total_frames - target_video_length
            total_clips = 1 + int(np.ceil(remaining / (target_video_length + 3)))

        ref_state_sq = get_reference_state_sequence(target_video_length - 3, target_fps)

        idx = 0
        while True:
            audio_clip, pad_len = audio_reader.next_frame()
            if audio_clip is None:
                break

            is_first = True if idx == 0 else False
            is_last = True if pad_len > 0 else False

            pipe = rs2v

            clip_input_info.is_first = is_first
            clip_input_info.is_last = is_last
            clip_input_info.ref_state = ref_state_sq[idx % len(ref_state_sq)]
            clip_input_info.seed = clip_input_info.seed + idx
            clip_input_info.audio_clip = audio_clip
            idx = idx + 1
            if self.progress_callback:
                self.progress_callback(idx, total_clips)

            gen_clip_video, audio_clip, gen_latents = pipe.run_clip_pipeline(clip_input_info)
            logger.info(f"Generated rs2v clip {idx}, pad_len {pad_len}, gen_clip_video shape: {gen_clip_video.shape}, audio_clip shape: {audio_clip.shape} gen_latents shape: {gen_latents.shape}")

            video_pad_len = pad_len // audio_per_frame
            gen_video_list.append(gen_clip_video[:, :, : gen_clip_video.shape[2] - video_pad_len].clone())
            cut_audio_list.append(audio_clip[: audio_clip.shape[0] - pad_len])
            clip_input_info.overlap_latent = gen_latents[:, -1:]

        gen_lvideo = torch.cat(gen_video_list, dim=2).float()
        gen_lvideo = torch.clamp(gen_lvideo, -1, 1)
        merge_audio = np.concatenate(cut_audio_list, axis=0).astype(np.float32)

        if is_main_process() and clip_input_info.save_result_path:
            out_path = os.path.join("./", "video_merge.mp4")
            audio_file = os.path.join("./", "audio_merge.wav")

            save_to_video(gen_lvideo, out_path, 16)
            save_audio(merge_audio, audio_file, out_path, output_path=clip_input_info.save_result_path)
            os.remove(out_path)
            os.remove(audio_file)

        return gen_lvideo, merge_audio, audio_sr

    def run_pipeline(self, input_info):
        # input_info = self.update_input_info(input_info)
        gen_lvideo, merge_audio, audio_sr = self.generate(input_info)
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
