import queue
import random
import os
import ctypes
import signal
import socket
import subprocess
import threading
import time
import traceback

import numpy as np
import torch
import torchaudio as ta
from loguru import logger
from scipy.signal import resample


class VAX64Recorder:
    def __init__(
        self,
        whip_shared_path: str,
        livestream_url: str,
        fps: float = 16.0,
        sample_rate: int = 16000,
    ):
        assert livestream_url.startswith("http"), "VAX64Recorder only support whip http livestream"
        self.livestream_url = livestream_url
        self.fps = fps
        self.sample_rate = sample_rate

        self.width = None
        self.height = None
        self.stoppable_t = None

        # only enable whip shared api for whip http livestream
        self.whip_shared_path = whip_shared_path
        self.whip_shared_lib = None
        self.whip_shared_handle = None

        # queue for send data to whip shared api
        self.queue = queue.Queue()
        self.worker_thread = None

    def worker(self):
        try:
            fail_time, max_fail_time = 0, 10
            packet_secs = 1.0 / self.fps
            audio_chunk = round(48000 * 2 / self.fps)
            audio_samples = round(48000 / self.fps)
            while True:
                try:
                    if self.queue is None:
                        break
                    data = self.queue.get()
                    if data is None:
                        logger.info("Worker thread received stop signal")
                        break
                    audios, images = data

                    for i in range(images.shape[0]):
                        t0 = time.time()
                        cur_audio = audios[i * audio_chunk: (i + 1) * audio_chunk].flatten()
                        audio_ptr = cur_audio.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
                        self.whip_shared_lib.pushRawAudioFrame(self.whip_shared_handle, audio_ptr, audio_samples)

                        cur_video = images[i].flatten()
                        video_ptr = cur_video.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                        self.whip_shared_lib.pushRawVideoFrame(self.whip_shared_handle, video_ptr, self.width, self.height)
                        time.sleep(max(0, packet_secs - (time.time() - t0)))

                    fail_time = 0
                except:  # noqa
                    logger.error(f"Send audio data error: {traceback.format_exc()}")
                    fail_time += 1
                    if fail_time > max_fail_time:
                        logger.error(f"Audio push worker thread failed {fail_time} times, stopping...")
                        break
        except:  # noqa
            logger.error(f"Audio push worker thread error: {traceback.format_exc()}")
        finally:
            logger.info("Audio push worker thread stopped")

    def start_libx264_whip_shared_api(self):
        self.whip_shared_lib = ctypes.CDLL(self.whip_shared_path)

        # define function argtypes and restype
        self.whip_shared_lib.initStream.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.whip_shared_lib.initStream.restype = ctypes.c_void_p

        self.whip_shared_lib.pushRawAudioFrame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]
        self.whip_shared_lib.pushRawVideoFrame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]

        self.whip_shared_lib.destroyStream.argtypes = [ctypes.c_void_p]

        whip_url = ctypes.c_char_p(self.livestream_url.encode("utf-8"))
        self.whip_shared_handle = ctypes.c_void_p(self.whip_shared_lib.initStream(whip_url, 1, 1, 0))
        logger.info(f"WHIP shared API initialized with handle: {self.whip_shared_handle}")

    def convert_data(self, audios, images):
        # Convert audio data to 16-bit integer format
        audio_datas = np.clip(np.round(audios * 32767), -32768, 32767).astype(np.int16)
        # Convert to numpy and scale to [0, 255], convert RGB to BGR for OpenCV/FFmpeg
        image_datas = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        logger.info(f"image_datas: {image_datas.shape} {image_datas.dtype} {image_datas.min()} {image_datas.max()}")
        reample_audios = resample(audio_datas, int(len(audio_datas) * 48000 / self.sample_rate))
        stereo_audios = np.stack([reample_audios, reample_audios], axis=-1).astype(np.int16).reshape(-1)
        return stereo_audios, image_datas

    def start(self, width: int, height: int):
        self.set_video_size(width, height)

    def set_video_size(self, width: int, height: int):
        if self.width is not None and self.height is not None:
            assert self.width == width and self.height == height, "Video size already set"
            return
        self.width = width
        self.height = height
        self.start_libx264_whip_shared_api()
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.start()

    # Publish ComfyUI Image tensor and audio tensor to livestream
    def pub_livestream(self, images: torch.Tensor, audios: np.ndarray):
        N, height, width, C = images.shape
        M = audios.reshape(-1).shape[0]
        assert C == 3, "Input must be [N, H, W, C] with C=3"

        logger.info(f"Publishing video [{N}x{width}x{height}], audio: [{M}]")
        audio_frames = round(M * self.fps / self.sample_rate)
        if audio_frames != N:
            logger.warning(f"Video and audio frames mismatch, {N} vs {audio_frames}")

        self.set_video_size(width, height)

        audio_datas, image_datas = self.convert_data(audios, images)
        self.queue.put((audio_datas, image_datas))
        logger.info(f"Published {N} frames and {M} audio samples")
        self.stoppable_t = time.time() + M / self.sample_rate + 3

    def stop(self, wait=True):
        if wait and self.stoppable_t:
            t = self.stoppable_t - time.time()
            if t > 0:
                logger.warning(f"Waiting for {t} seconds to stop ...")
                time.sleep(t)
            self.stoppable_t = None

        # Send stop signals to queues
        if self.queue:
            self.queue.put(None)

        # Wait for threads to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not stop gracefully")

        # Destroy WHIP shared API
        if self.whip_shared_lib and self.whip_shared_handle:
            self.whip_shared_lib.destroyStream(self.whip_shared_handle)
            self.whip_shared_handle = None
            self.whip_shared_lib = None
            logger.warning("WHIP shared API destroyed")

    def __del__(self):
        self.stop()


def create_simple_video(frames=10, height=480, width=640):
    video_data = []
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.float32)
        stripe_height = height // 8
        colors = [
            [1.0, 0.0, 0.0],  # 红色
            [0.0, 1.0, 0.0],  # 绿色
            [0.0, 0.0, 1.0],  # 蓝色
            [1.0, 1.0, 0.0],  # 黄色
            [1.0, 0.0, 1.0],  # 洋红
            [0.0, 1.0, 1.0],  # 青色
            [1.0, 1.0, 1.0],  # 白色
            [0.5, 0.5, 0.5],  # 灰色
        ]
        for j, color in enumerate(colors):
            start_y = j * stripe_height
            end_y = min((j + 1) * stripe_height, height)
            frame[start_y:end_y, :] = color
        offset = int((i / frames) * width)
        frame = np.roll(frame, offset, axis=1)
        frame = torch.tensor(frame, dtype=torch.float32)
        video_data.append(frame)
    return torch.stack(video_data, dim=0)


if __name__ == "__main__":
    sample_rate = 16000
    fps = 16
    width = 352
    height = 352

    recorder = VAX64Recorder(
        whip_shared_path="/data/nvme0/liuliang1/lightx2v/test_deploy/test_whip_so/src/libagora_go_whip.so",
        livestream_url="https://reverse.st-oc-01.chielo.org/10.5.64.49:8000/rtc/v1/whip/?app=subscribe&stream=ll1&eip=10.120.114.82:8000",
        fps=fps,
        sample_rate=sample_rate,
    )
    recorder.start(width, height)

    time.sleep(5)
    audio_path = "/data/nvme0/liuliang1/lightx2v/test_deploy/media_test/mangzhong.wav"
    audio_array, ori_sr = ta.load(audio_path)
    audio_array = ta.functional.resample(audio_array.mean(0), orig_freq=ori_sr, new_freq=16000)
    audio_array = audio_array.numpy().reshape(-1)
    secs = audio_array.shape[0] // sample_rate
    interval = 1
    space = 10

    i = 0
    while i < space:
        t0 = time.time()
        logger.info(f"space {i} / {space} s")
        cur_audio_array = np.zeros(int(interval * sample_rate), dtype=np.float32)
        num_frames = int(interval * fps)
        images = create_simple_video(num_frames, height, width)
        recorder.pub_livestream(images, cur_audio_array)
        i += interval
        time.sleep(interval - (time.time() - t0))

    started = True

    i = 0
    while i < secs:
        t0 = time.time()
        start = int(i * sample_rate)
        end = int((i + interval) * sample_rate)
        cur_audio_array = audio_array[start:end]
        num_frames = int(interval * fps)
        images = create_simple_video(num_frames, height, width)
        logger.info(f"{i} / {secs} s")
        if started:
            logger.warning(f"start pub_livestream !!!!!!!!!!!!!!!!!!!!!!!")
            started = False
        recorder.pub_livestream(images, cur_audio_array)
        i += interval
        time.sleep(interval - (time.time() - t0))

    recorder.stop()
