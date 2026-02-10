import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torchaudio as ta
from loguru import logger

from lightx2v.shot_runner.shot_base import ShotPipeline, load_clip_configs
from lightx2v.shot_runner.utils import SlidingWindowReader, save_audio, save_to_video
from lightx2v.utils.input_info import init_input_info_from_args
from lightx2v.utils.profiler import *
from lightx2v.utils.utils import seed_all


class ShotStreamPipeline(ShotPipeline):  # type:ignore
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def generate(self, args):
        s2v = self.clip_generators["s2v_clip"]  # s2v一致性强，动态相应差
        f2v = self.clip_generators["f2v_clip"]  # f2v一致性差，动态响应强
        # 根据 pipe 最长 overlap_len 初始化 tail buffer
        self.max_tail_len = max(s2v.config.get("prev_frame_length", None), f2v.config.get("prev_frame_length", None))
        model_fps = s2v.config.get("target_fps", 16)
        model_sr = s2v.config.get("audio_sr", 16000)

        # 获取用户输入信息
        s2v_input_info = init_input_info_from_args(s2v.config["task"], args, infer_steps=3)
        f2v_input_info = init_input_info_from_args(f2v.config["task"], args)
        # 从默认配置中补全输入信息
        s2v_input_info = self.check_input_info(s2v_input_info, s2v.config)
        f2v_input_info = self.check_input_info(f2v_input_info, f2v.config)

        assert s2v_input_info.audio_path == f2v_input_info.audio_path, "s2v and f2v must use the same audio input"

        self.global_tail_video = None

        gen_video_list = []
        cut_audio_list = []

        audio_array, ori_sr = ta.load(args.audio_path)
        audio_array = audio_array.mean(0)
        if ori_sr != model_sr:
            audio_array = ta.functional.resample(audio_array, ori_sr, model_sr)
        audio_reader = SlidingWindowReader(audio_array, frame_len=33)

        # Demo 交替生成 clip
        i = 0
        overlap = 0
        while True:
            audio_clip = audio_reader.next_frame(overlap=overlap)
            if audio_clip is None:
                break

            if i % 2 == 0:
                pipe = s2v
                inputs = s2v_input_info
            else:
                pipe = f2v
                inputs = f2v_input_info
                inputs.prompt = "A man speaks to the camera with a slightly furrowed brow and focused gaze. He raises both hands upward in powerful, emphatic gestures. "  # 添加动作提示

            inputs.seed = inputs.seed + i  # 不同 clip 使用不同随机种子
            inputs.audio_clip = audio_clip
            i = i + 1

            # if i % 4 == 0:
            #    inputs.infer_steps = 2#s2v 一半时间用2步推理

            if self.global_tail_video is not None:  # 根据当前 pipe 需要多少 overlap_len 来裁剪 tail
                inputs.overlap_frame = self.global_tail_video[:, :, -pipe.prev_frame_length :]
            gen_clip_video, audio_clip, _ = pipe.run_clip_pipeline(inputs)
            aligned_len = gen_clip_video.shape[2] - overlap
            gen_video_list.append(gen_clip_video[:, :, :aligned_len])
            cut_audio_list.append(audio_clip[: aligned_len * audio_reader.audio_per_frame])

            overlap = pipe.prev_frame_length
            self.global_tail_video = gen_clip_video[:, :, -self.max_tail_len :]

        gen_lvideo = torch.cat(gen_video_list, dim=2).float()
        gen_lvideo = torch.clamp(gen_lvideo, -1, 1)
        merge_audio = np.concatenate(cut_audio_list, axis=0).astype(np.float32)
        out_path = os.path.join("./", "video_merge.mp4")
        audio_file = os.path.join("./", "audio_merge.wav")

        save_to_video(gen_lvideo, out_path, model_fps)
        save_audio(merge_audio, audio_file, out_path, output_path=args.save_result_path)
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

    with ProfilingContext4DebugL1("Init Pipeline Cost Time"):
        shot_stream_pipe = ShotStreamPipeline(clip_configs)

    with ProfilingContext4DebugL1("Generate Cost Time"):
        shot_stream_pipe.generate(args)

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
