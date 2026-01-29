import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torchaudio as ta
from loguru import logger

from lightx2v.shot_runner.shot_base import ShotConfig, ShotPipeline, load_clip_configs
from lightx2v.shot_runner.utils import RS2V_SlidingWindowReader, save_audio, save_to_video
from lightx2v.utils.profiler import *
from lightx2v.utils.utils import seed_all


def get_reference_state_sequence(frames_per_clip=17, target_fps=16):
    duration = frames_per_clip / target_fps
    if duration > 3:
        inner_every = 2
    else:
        inner_every = 6
    return [0] + [1] * (inner_every - 1)


class ShotRS2VPipeline(ShotPipeline):  # type:ignore
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def generate(self):
        rs2v = self.clip_generators["rs2v_clip"]
        target_video_length = rs2v.config.get("target_video_length", 81)
        target_fps = rs2v.config.get("target_fps", 16)
        audio_sr = rs2v.config.get("audio_sr", 16000)
        audio_per_frame = audio_sr // target_fps

        # 根据 pipe 最长 overlap_len 初始化 tail buffer

        gen_video_list = []
        cut_audio_list = []

        audio_array, ori_sr = ta.load(self.shot_cfg.audio_path)
        audio_array = audio_array.mean(0)
        if ori_sr != audio_sr:
            audio_array = ta.functional.resample(audio_array, ori_sr, audio_sr)
        audio_reader = RS2V_SlidingWindowReader(audio_array, first_clip_len=target_video_length, clip_len=target_video_length + 3, sr=audio_sr, fps=target_fps)

        ref_state_sq = get_reference_state_sequence(target_video_length - 3, target_fps)

        idx = 0
        while True:
            audio_clip, pad_len = audio_reader.next_frame()
            if audio_clip is None:
                break

            is_first = True if idx == 0 else False
            is_last = True if pad_len > 0 else False

            pipe = rs2v
            inputs = self.clip_inputs["rs2v_clip"]
            # inputs.prompt = "A man speaks to the camera with a slightly furrowed brow and focused gaze. He raises both hands upward in powerful, emphatic gestures. "  # 添加动作提示

            inputs.is_first = is_first
            inputs.is_last = is_last
            inputs.ref_state = ref_state_sq[idx % len(ref_state_sq)]
            inputs.seed = inputs.seed + idx  # 不同 clip 使用不同随机种子
            inputs.audio_clip = audio_clip
            idx = idx + 1

            gen_clip_video, audio_clip, gen_latents = pipe.run_clip_pipeline(inputs)
            logger.info(f"Generated rs2v clip {idx}, pad_len {pad_len}, gen_clip_video shape: {gen_clip_video.shape}, audio_clip shape: {audio_clip.shape} gen_latents shape: {gen_latents.shape}")

            video_pad_len = pad_len // audio_per_frame
            gen_video_list.append(gen_clip_video[:, :, : gen_clip_video.shape[2] - video_pad_len])
            cut_audio_list.append(audio_clip[: audio_clip.shape[0] - pad_len])
            inputs.overlap_latent = gen_latents[:, -1:]

        gen_lvideo = torch.cat(gen_video_list, dim=2).float()
        gen_lvideo = torch.clamp(gen_lvideo, -1, 1)
        merge_audio = np.concatenate(cut_audio_list, axis=0).astype(np.float32)
        out_path = os.path.join("./", "video_merge.mp4")
        audio_file = os.path.join("./", "audio_merge.wav")

        save_to_video(gen_lvideo, out_path, 16)
        save_audio(merge_audio, audio_file, out_path, output_path=self.shot_cfg.save_result_path)
        os.remove(out_path)
        os.remove(audio_file)


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

    shot_cfg = ShotConfig(
        seed=args.seed,
        image_path=args.image_path,
        audio_path=args.audio_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        save_result_path=args.save_result_path,
        clip_configs=clip_configs,
        target_shape=args.target_shape,
    )

    with ProfilingContext4DebugL1("Total Cost"):
        shot_stream_pipe = ShotRS2VPipeline(shot_cfg)
        shot_stream_pipe.generate()

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
