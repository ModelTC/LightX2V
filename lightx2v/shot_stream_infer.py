import argparse
import json
import os
import subprocess
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchaudio as ta
from einops import rearrange
from loguru import logger

from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config
from lightx2v.utils.utils import seed_all


class SlidingWindowReader:
    def __init__(self, samples: torch.Tensor, frame_len: int, sr=16000, fps=16):
        assert isinstance(samples, torch.Tensor)
        assert samples.dim() == 1, "samples 必须是 1D Tensor"

        self.samples = samples
        self.frame_len = frame_len  # 单位：视频帧
        self.audio_per_frame = sr // fps  # samples / frame
        self.pos = 0  # 单位：视频帧

    def next_frame(self, overlap: int):
        assert 0 <= overlap < self.frame_len

        hop_frames = self.frame_len - overlap

        start_sample = self.pos * self.audio_per_frame
        end_sample = start_sample + self.frame_len * self.audio_per_frame

        if end_sample > self.samples.numel():
            return None

        frame = self.samples[start_sample:end_sample]

        self.pos += hop_frames
        return frame.float()


def array_to_video(
    image_array: np.ndarray,
    output_path: str,
    fps: Union[int, float] = 30,
    resolution: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
    disable_log: bool = False,
    lossless: bool = True,
) -> None:
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input should be np.ndarray.")
    assert image_array.ndim == 4
    assert image_array.shape[-1] == 3
    if resolution:
        height, width = resolution
        width += width % 2
        height += height % 2
    else:
        image_array = pad_for_libx264(image_array)
        height, width = image_array.shape[1], image_array.shape[2]
    if lossless:
        command = [
            "ffmpeg",
            "-y",  # (optional) overwrite output file if it exists
            "-f",
            "rawvideo",
            "-s",
            f"{int(width)}x{int(height)}",  # size of one frame
            "-pix_fmt",
            "bgr24",
            "-r",
            f"{fps}",  # frames per second
            "-loglevel",
            "error",
            "-threads",
            "4",
            "-i",
            "-",  # The input comes from a pipe
            "-vcodec",
            "libx264rgb",
            "-crf",
            "0",
            "-an",  # Tells FFMPEG not to expect any audio
            output_path,
        ]
    else:
        command = [
            "ffmpeg",
            "-y",  # (optional) overwrite output file if it exists
            "-f",
            "rawvideo",
            "-s",
            f"{int(width)}x{int(height)}",  # size of one frame
            "-pix_fmt",
            "bgr24",
            "-r",
            f"{fps}",  # frames per second
            "-loglevel",
            "error",
            "-threads",
            "4",
            "-i",
            "-",  # The input comes from a pipe
            "-vcodec",
            "libx264",
            "-an",  # Tells FFMPEG not to expect any audio
            output_path,
        ]

    if not disable_log:
        print(f'Running "{" ".join(command)}"')
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        raise BrokenPipeError("No buffer received.")
    index = 0
    while True:
        if index >= image_array.shape[0]:
            break
        process.stdin.write(image_array[index].tobytes())
        index += 1
    process.stdin.close()
    process.stderr.close()
    process.wait()


def pad_for_libx264(image_array):
    if image_array.ndim == 2 or (image_array.ndim == 3 and image_array.shape[2] == 3):
        hei_index = 0
        wid_index = 1
    elif image_array.ndim == 4 or (image_array.ndim == 3 and image_array.shape[2] != 3):
        hei_index = 1
        wid_index = 2
    else:
        return image_array
    hei_pad = image_array.shape[hei_index] % 2
    wid_pad = image_array.shape[wid_index] % 2
    if hei_pad + wid_pad > 0:
        pad_width = []
        for dim_index in range(image_array.ndim):
            if dim_index == hei_index:
                pad_width.append((0, hei_pad))
            elif dim_index == wid_index:
                pad_width.append((0, wid_pad))
            else:
                pad_width.append((0, 0))
        values = 0
        image_array = np.pad(image_array, pad_width, mode="constant", constant_values=values)
    return image_array


def generate_unique_path(path):
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    index = 1
    new_path = f"{root}-{index}{ext}"
    while os.path.exists(new_path):
        index += 1
        new_path = f"{root}-{index}{ext}"
    return new_path


def save_to_video(gen_lvideo, out_path, target_fps):
    gen_lvideo = rearrange(gen_lvideo, "B C T H W -> B T H W C")
    gen_lvideo = (gen_lvideo[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    gen_lvideo = gen_lvideo[..., ::-1].copy()
    generate_unique_path(out_path)
    array_to_video(gen_lvideo, output_path=out_path, fps=target_fps, lossless=False)


def save_audio(
    audio_array,
    audio_name: str,
    video_name: str,
    sr: int = 16000,
    output_path: Optional[str] = None,
):
    logger.info(f"Saving audio to {audio_name} type: {type(audio_array)}")

    ta.save(
        audio_name,
        torch.tensor(audio_array[None]),
        sample_rate=sr,
    )

    if output_path is None:
        out_video = f"{video_name[:-4]}_with_audio.mp4"
    else:
        out_video = output_path

    parent_dir = os.path.dirname(out_video)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(out_video):
        os.remove(out_video)

    subprocess.call(["ffmpeg", "-y", "-i", video_name, "-i", audio_name, out_video])

    return out_video


def load_clip_configs(main_json_path: str):
    with open(main_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lightx2v_path = cfg["lightx2v_path"]
    clip_configs_raw = cfg["clip_configs"]

    clip_configs = [ClipConfig(name=item["name"], config_json=str(Path(lightx2v_path) / item["path"])) for item in clip_configs_raw]

    return clip_configs


@dataclass
class ClipConfig:
    name: str
    config_json: str


@dataclass
class ShotConfig:
    seed: int
    image_path: str
    audio_path: str
    prompt: str
    negative_prompt: str
    save_result_path: str
    clip_configs: list[ClipConfig]
    target_shape: list[int]


class ShotStreamPipeline:
    def __init__(self, shot_cfg: ShotConfig):
        self.shot_cfg = shot_cfg
        self.clip_generators = {}
        self.clip_inputs = {}
        self.overlap_frame = None
        self.overlap_latent = None

        for clip_config in shot_cfg.clip_configs:
            name = clip_config.name
            self.clip_generators[name] = self.create_clip_generator(clip_config)

            args = Namespace(
                seed=self.shot_cfg.seed,
                prompt=self.shot_cfg.prompt,
                negative_prompt=self.shot_cfg.negative_prompt,
                image_path=self.shot_cfg.image_path,
                audio_path=self.shot_cfg.audio_path,
                save_result_path=self.shot_cfg.save_result_path,
                task=self.clip_generators[name].task,
                return_result_tensor=True,
                overlap_frame=self.overlap_frame,
                overlap_latent=self.overlap_latent,
                target_shape=self.shot_cfg.target_shape,
            )
            self.clip_inputs[name] = set_input_info(args)

    def _init_runner(self, config):
        torch.set_grad_enabled(False)
        runner = RUNNER_REGISTER[config["model_cls"]](config)
        runner.init_modules()
        return runner

    def create_clip_generator(self, clip_config: ClipConfig):
        config = self.get_config_json(clip_config.config_json)
        config = set_config(Namespace(**config))
        print_config(config)

        runner = self._init_runner(config)
        logger.info(f"Clip {clip_config.name} initialized successfully!")
        return runner

    def get_config_json(self, config_json):
        logger.info(f"Loading infer config from {config_json}")
        with open(config_json, "r") as f:
            config = json.load(f)
        return config

    @torch.no_grad()
    def generate(self):
        s2v = self.clip_generators["s2v_clip"]  # s2v一致性强，动态相应差
        f2v = self.clip_generators["f2v_clip"]  # f2v一致性差，动态响应强
        # 根据 pipe 最长 overlap_len 初始化 tail buffer
        self.max_tail_len = max(s2v.prev_frame_length, f2v.prev_frame_length)
        self.global_tail_video = None

        gen_video_list = []
        cut_audio_list = []

        audio_array, ori_sr = ta.load(self.shot_cfg.audio_path)
        audio_array = audio_array.mean(0)
        if ori_sr != 16000:
            audio_array = ta.functional.resample(audio_array, ori_sr, 16000)
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
                inputs = self.clip_inputs["s2v_clip"]
            else:
                pipe = f2v
                inputs = self.clip_inputs["f2v_clip"]
                inputs.prompt = "A man speaks to the camera with a slightly furrowed brow and focused gaze. He raises both hands upward in powerful, emphatic gestures. "  # 添加动作提示

            inputs.seed = self.shot_cfg.seed + i  # 不同 clip 使用不同随机种子
            inputs.audio_clip = audio_clip
            i = i + 1

            if self.global_tail_video is not None:  # 根据当前 pipe 需要多少 overlap_len 来裁剪 tail
                inputs.overlap_frame = self.global_tail_video[:, :, -pipe.prev_frame_length :]

            gen_clip_video, audio_clip = pipe.run_clip_pipeline(inputs)

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

        save_to_video(gen_lvideo, out_path, 16)
        save_audio(merge_audio, audio_file, out_path, output_path="./output_lightx2v_seko_talk.mp4")
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
        shot_stream_pipe = ShotStreamPipeline(shot_cfg)
        shot_stream_pipe.generate()

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
