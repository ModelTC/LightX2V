import ctypes
import queue
import threading
import time
import traceback

import numpy as np
import torch
import torchaudio as ta
from loguru import logger
from scipy.signal import resample


class X264VARecorder:
    def __init__(
        self,
        whip_shared_path: str,
        livestream_url: str,
        fps: float = 16.0,
        sample_rate: int = 16000,
        slice_frame: int = 1,
        prev_frame: int = 1,
    ):
        assert livestream_url.startswith("http"), "X264VARecorder only support whip http livestream"
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

        assert livestream_url.startswith("http"), "X264VARecorder only support whip http livestream"
        self.realtime = True

        # queue for send data to whip shared api
        self.queue = queue.Queue()
        self.worker_thread = None

        # buffer for stream data
        self.target_sample_rate = 48000
        self.target_samples_per_frame = round(self.target_sample_rate / self.fps)
        self.target_chunks_per_frame = self.target_samples_per_frame * 2
        self.stream_buffer = []
        self.stream_buffer_lock = threading.Lock()
        self.stop_schedule = False
        self.schedule_thread = None
        self.slice_frame = slice_frame
        self.prev_frame = prev_frame
        assert self.slice_frame >= self.prev_frame, "Slice frame must be greater than previous frame"

    def worker(self):
        try:
            fail_time, max_fail_time = 0, 10
            packet_secs = 1.0 / self.fps
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
                        cur_audio = audios[i * self.target_chunks_per_frame : (i + 1) * self.target_chunks_per_frame].flatten()
                        audio_ptr = cur_audio.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
                        self.whip_shared_lib.pushWhipRawAudioFrame(self.whip_shared_handle, audio_ptr, self.target_samples_per_frame)

                        cur_video = images[i].flatten()
                        video_ptr = cur_video.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                        self.whip_shared_lib.pushWhipRawVideoFrame(self.whip_shared_handle, video_ptr, self.width, self.height)

                        if self.realtime and i < images.shape[0] - 1:
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

    def start_libx264_whip_shared_api(self, width: int, height: int):
        self.whip_shared_lib = ctypes.CDLL(self.whip_shared_path)

        # define function argtypes and restype
        self.whip_shared_lib.initWhipStream.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.whip_shared_lib.initWhipStream.restype = ctypes.c_void_p

        self.whip_shared_lib.pushWhipRawAudioFrame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]
        self.whip_shared_lib.pushWhipRawVideoFrame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]

        self.whip_shared_lib.destroyWhipStream.argtypes = [ctypes.c_void_p]

        whip_url = ctypes.c_char_p(self.livestream_url.encode("utf-8"))
        self.whip_shared_handle = ctypes.c_void_p(self.whip_shared_lib.initWhipStream(whip_url, 1, 1, 0, width, height))
        logger.info(f"WHIP shared API initialized with handle: {self.whip_shared_handle}")

    def convert_data(self, audios, images):
        # Convert audio data to 16-bit integer format
        audio_datas = torch.clamp(torch.round(audios * 32767), -32768, 32767).to(torch.int16).cpu().numpy().reshape(-1)
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
        self.start_libx264_whip_shared_api(width, height)
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.start()
        if self.realtime:
            self.schedule_thread = threading.Thread(target=self.schedule_stream_buffer)
            self.schedule_thread.start()

    def buffer_stream(self, images: torch.Tensor, audios: torch.Tensor, gen_video: torch.Tensor):
        N, height, width, C = images.shape
        M = audios.reshape(-1).shape[0]
        assert N % self.slice_frame == 0, "Video frames must be divisible by slice_frame"
        assert C == 3, "Input must be [N, H, W, C] with C=3"

        audio_frames = round(M * self.fps / self.sample_rate)
        if audio_frames != N:
            logger.warning(f"Video and audio frames mismatch, {N} vs {audio_frames}")
        self.set_video_size(width, height)
        audio_datas, image_datas = self.convert_data(audios, images)

        # logger.info(f"Buffer stream images {images.shape} {audios.shape} {gen_video.shape}")
        rets = []
        for i in range(0, N, self.slice_frame):
            end_frame = i + self.slice_frame
            img = image_datas[i:end_frame]
            aud = audio_datas[i * self.target_chunks_per_frame : end_frame * self.target_chunks_per_frame]
            gen = gen_video[:, :, (end_frame - self.prev_frame) : end_frame]
            rets.append((img, aud, gen))

        with self.stream_buffer_lock:
            origin_size = len(self.stream_buffer)
            self.stream_buffer.extend(rets)
            logger.info(f"Buffered {origin_size} + {len(rets)} = {len(self.stream_buffer)} stream segments")

    def get_buffer_stream_size(self):
        return len(self.stream_buffer)

    def truncate_stream_buffer(self, size: int):
        with self.stream_buffer_lock:
            self.stream_buffer = self.stream_buffer[:size]
            logger.info(f"Truncated stream buffer to {len(self.stream_buffer)} segments")
            if len(self.stream_buffer) > 0:
                return self.stream_buffer[-1][2]  # return the last video tensor
            else:
                return None

    def schedule_stream_buffer(self):
        schedule_interval = self.slice_frame / self.fps
        logger.info(f"Schedule stream buffer with interval: {schedule_interval} seconds")
        t = None
        while True:
            try:
                if self.stop_schedule:
                    break
                img, aud, gen = None, None, None
                with self.stream_buffer_lock:
                    if len(self.stream_buffer) > 0:
                        img, aud, gen = self.stream_buffer.pop(0)

                if t is not None:
                    wait_secs = schedule_interval - (time.time() - t)
                    if wait_secs > 0:
                        time.sleep(wait_secs)
                t = time.time()

                if img is not None and aud is not None:
                    self.queue.put((aud, img))
                    # logger.info(f"Scheduled {img.shape[0]} frames and {aud.shape[0]} audio samples to publish")
                    del gen
                    self.stoppable_t = time.time() + img.shape[0] / self.fps + 3
                else:
                    logger.warning(f"No stream buffer to schedule")
            except Exception:
                logger.error(f"Schedule stream buffer error: {traceback.format_exc()}")
                break
        logger.info("Schedule stream buffer thread stopped")

    def stop(self, wait=True):
        if wait and self.stoppable_t:
            t = self.stoppable_t - time.time()
            if t > 0:
                logger.warning(f"Waiting for {t} seconds to stop ...")
                time.sleep(t)
            self.stoppable_t = None

        if self.schedule_thread:
            self.stop_schedule = True
            self.schedule_thread.join(timeout=5)
            if self.schedule_thread and self.schedule_thread.is_alive():
                logger.error(f"Schedule thread did not stop after 5s")

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
            self.whip_shared_lib.destroyWhipStream(self.whip_shared_handle)
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
    width = 452
    height = 352

    recorder = X264VARecorder(
        whip_shared_path="/data/nvme0/liuliang1/lightx2v/test_deploy/test_whip_so/0.1.1/go_whxp.so",
        livestream_url="https://reverse.st-oc-01.chielo.org/10.5.64.49:8000/rtc/v1/whip/?app=subscribe&stream=ll2&eip=10.120.114.82:8000",
        fps=fps,
        sample_rate=sample_rate,
    )
    recorder.start(width, height)

    # time.sleep(5)
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
        recorder.buffer_stream(images, torch.tensor(cur_audio_array, dtype=torch.float32), images)
        i += interval
        time.sleep(interval - (time.time() - t0))

    started = True

    i = 0
    while i < secs:
        t0 = time.time()
        start = int(i * sample_rate)
        end = int((i + interval) * sample_rate)
        cur_audio_array = torch.tensor(audio_array[start:end], dtype=torch.float32)
        num_frames = int(interval * fps)
        images = create_simple_video(num_frames, height, width)
        logger.info(f"{i} / {secs} s")
        if started:
            logger.warning(f"start pub_livestream !!!!!!!!!!!!!!!!!!!!!!!")
            started = False
        recorder.buffer_stream(images, cur_audio_array, images)
        i += interval
        time.sleep(interval - (time.time() - t0))

    recorder.stop()
