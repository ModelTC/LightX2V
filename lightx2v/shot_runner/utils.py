import os
import subprocess
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torchaudio as ta
from einops import rearrange
from loguru import logger


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


class RS2V_SlidingWindowReader:
    def __init__(
        self,
        samples: torch.Tensor,
        first_clip_len: int = 81,
        clip_len: int = 84,
        sr: int = 16000,
        fps: int = 16,
    ):
        assert isinstance(samples, torch.Tensor)
        assert samples.dim() == 1, "samples 必须是 1D Tensor"

        self.samples = samples
        self.first_clip_len = first_clip_len
        self.clip_len = clip_len

        self.audio_per_frame = sr // fps  # samples per video frame

        self.pos_frame = 0        # 当前起始帧位置
        self.chunk_idx = 0        # 第几个 chunk（用于区分第一段）

    def next_frame(self):
        cur_clip_len = (
            self.first_clip_len if self.chunk_idx == 0 else self.clip_len
        )

        start_sample = self.pos_frame * self.audio_per_frame
        end_sample = start_sample + cur_clip_len * self.audio_per_frame

        if end_sample > self.samples.numel():
            return None

        frame = self.samples[start_sample:end_sample]

        self.pos_frame += cur_clip_len
        self.chunk_idx += 1

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
