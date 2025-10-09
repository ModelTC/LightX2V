import queue
import random
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


class VARecorder:
    def __init__(
        self,
        livestream_url: str,
        fps: float = 16.0,
        sample_rate: int = 16000,
    ):
        self.livestream_url = livestream_url
        self.fps = fps
        self.sample_rate = sample_rate
        self.audio_port = random.choice(range(32000, 40000))
        self.video_port = self.audio_port + 1
        logger.info(f"VARecorder audio port: {self.audio_port}, video port: {self.video_port}")

        self.width = None
        self.height = None
        self.stoppable_t = None
        self.realtime = True

        # ffmpeg process for mix video and audio data and push to livestream
        self.ffmpeg_process = None

        # TCP connection objects
        self.audio_socket = None
        self.video_socket = None
        self.audio_conn = None
        self.video_conn = None
        self.audio_thread = None
        self.video_thread = None

        # queue for send data to ffmpeg process
        self.audio_queue = queue.Queue()
        self.video_queue = queue.Queue()

    def init_sockets(self):
        # TCP socket for send and recv video and audio data
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_socket.bind(("127.0.0.1", self.video_port))
        self.video_socket.listen(1)

        self.audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.audio_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.audio_socket.bind(("127.0.0.1", self.audio_port))
        self.audio_socket.listen(1)

    def audio_worker(self):
        try:
            logger.info("Waiting for ffmpeg to connect to audio socket...")
            self.audio_conn, _ = self.audio_socket.accept()
            logger.info(f"Audio connection established from {self.audio_conn.getpeername()}")
            fail_time, max_fail_time = 0, 10
            while True:
                try:
                    if self.audio_queue is None:
                        break
                    data = self.audio_queue.get()
                    if data is None:
                        logger.info("Audio thread received stop signal")
                        break
                    # Convert audio data to 16-bit integer format
                    audios = torch.clamp(torch.round(data * 32767), -32768, 32767).to(torch.int16)
                    self.audio_conn.send(audios[None].cpu().numpy().tobytes())
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

    def video_worker(self):
        try:
            logger.info("Waiting for ffmpeg to connect to video socket...")
            self.video_conn, _ = self.video_socket.accept()
            logger.info(f"Video connection established from {self.video_conn.getpeername()}")
            fail_time, max_fail_time = 0, 10
            packet_secs = 1.0 / self.fps
            while True:
                try:
                    if self.video_queue is None:
                        break
                    data = self.video_queue.get()
                    if data is None:
                        logger.info("Video thread received stop signal")
                        break

                    # Convert to numpy and scale to [0, 255], convert RGB to BGR for OpenCV/FFmpeg
                    if not self.realtime:
                        frames = (data * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                        self.video_conn.send(frames.tobytes())
                    else:
                        for i in range(data.shape[0]):
                            t0 = time.time()
                            frame = (data[i] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                            self.video_conn.send(frame.tobytes())
                            time.sleep(max(0, packet_secs - (time.time() - t0)))

                    fail_time = 0
                except:  # noqa
                    logger.error(f"Send video data error: {traceback.format_exc()}")
                    fail_time += 1
                    if fail_time > max_fail_time:
                        logger.error(f"Video push worker thread failed {fail_time} times, stopping...")
                        break
        except:  # noqa
            logger.error(f"Video push worker thread error: {traceback.format_exc()}")
        finally:
            logger.info("Video push worker thread stopped")

    def start_ffmpeg_process_local(self):
        """Start ffmpeg process that connects to our TCP sockets"""
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            f"tcp://127.0.0.1:{self.audio_port}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-s",
            f"{self.width}x{self.height}",
            "-i",
            f"tcp://127.0.0.1:{self.video_port}",
            "-ar",
            "44100",
            "-b:v",
            "4M",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            f"{self.fps}",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mp4",
            self.livestream_url,
            "-y",
            "-loglevel",
            "info",
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd)
            logger.info(f"FFmpeg streaming started with PID: {self.ffmpeg_process.pid}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")

    def start_ffmpeg_process_rtmp(self):
        """Start ffmpeg process that connects to our TCP sockets"""
        ffmpeg_cmd = [
            "ffmpeg",
            "-re",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            f"tcp://127.0.0.1:{self.audio_port}",
            "-f",
            "rawvideo",
            "-re",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-s",
            f"{self.width}x{self.height}",
            "-i",
            f"tcp://127.0.0.1:{self.video_port}",
            "-ar",
            "44100",
            "-b:v",
            "2M",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            f"{self.fps}",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "flv",
            self.livestream_url,
            "-y",
            "-loglevel",
            "info",
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd)
            logger.info(f"FFmpeg streaming started with PID: {self.ffmpeg_process.pid}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")

    def start_ffmpeg_process_whip(self):
        """Start ffmpeg process that connects to our TCP sockets"""
        ffmpeg_cmd = [
            "ffmpeg",
            "-re",
            "-fflags",
            "nobuffer",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
            "-flush_packets",
            "1",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-ch_layout",
            "mono",
            "-i",
            f"tcp://127.0.0.1:{self.audio_port}",
            "-f",
            "rawvideo",
            "-re",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-s",
            f"{self.width}x{self.height}",
            "-i",
            f"tcp://127.0.0.1:{self.video_port}",
            "-ar",
            "48000",
            "-c:a",
            "libopus",
            "-ac",
            "2",
            "-b:v",
            "2M",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            f"{self.fps}",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-bf",
            "0",
            "-f",
            "whip",
            self.livestream_url,
            "-y",
            "-loglevel",
            "info",
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd)
            logger.info(f"FFmpeg streaming started with PID: {self.ffmpeg_process.pid}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")

    def start(self, width: int, height: int):
        self.set_video_size(width, height)
        duration = 1.0
        self.pub_livestream(
            torch.zeros((int(self.fps * duration), height, width, 3), dtype=torch.float16),
            torch.zeros(int(self.sample_rate * duration), dtype=torch.float16)
        )
        time.sleep(duration)

    def set_video_size(self, width: int, height: int):
        if self.width is not None and self.height is not None:
            assert self.width == width and self.height == height, "Video size already set"
            return
        self.width = width
        self.height = height
        self.init_sockets()
        if self.livestream_url.startswith("rtmp://"):
            self.start_ffmpeg_process_rtmp()
        elif self.livestream_url.startswith("http"):
            self.start_ffmpeg_process_whip()
        else:
            self.start_ffmpeg_process_local()
            self.realtime = False
        self.audio_thread = threading.Thread(target=self.audio_worker)
        self.video_thread = threading.Thread(target=self.video_worker)
        self.audio_thread.start()
        self.video_thread.start()

    # Publish ComfyUI Image tensor and audio tensor to livestream
    def pub_livestream(self, images: torch.Tensor, audios: torch.Tensor):
        N, height, width, C = images.shape
        M = audios.reshape(-1).shape[0]
        assert C == 3, "Input must be [N, H, W, C] with C=3"

        logger.info(f"Publishing video [{N}x{width}x{height}], audio: [{M}]")
        audio_frames = round(M * self.fps / self.sample_rate)
        if audio_frames != N:
            logger.warning(f"Video and audio frames mismatch, {N} vs {audio_frames}")

        self.set_video_size(width, height)
        self.audio_queue.put(audios)
        self.video_queue.put(images)
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
        if self.audio_queue:
            self.audio_queue.put(None)
        if self.video_queue:
            self.video_queue.put(None)

        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=5)
            if self.audio_thread.is_alive():
                logger.warning("Audio push thread did not stop gracefully")
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=5)
            if self.video_thread.is_alive():
                logger.warning("Video push thread did not stop gracefully")

        # Close TCP connections, sockets
        if self.audio_conn:
            self.audio_conn.close()
        if self.video_conn:
            self.video_conn.close()
        if self.audio_socket:
            self.audio_socket.close()
        if self.video_socket:
            self.video_socket.close()

        while self.audio_queue and self.audio_queue.qsize() > 0:
            self.audio_queue.get_nowait()
        while self.video_queue and self.video_queue.qsize() > 0:
            self.video_queue.get_nowait()
        self.audio_queue = None
        self.video_queue = None
        logger.warning("Cleaned audio and video queues")

        # Stop ffmpeg process
        if self.ffmpeg_process:
            self.ffmpeg_process.send_signal(signal.SIGINT)
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            logger.warning("FFmpeg recorder process stopped")

    def __del__(self):
        self.stop(wait=False)


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
    width = 640
    height = 480

    recorder = VARecorder(
        # livestream_url="rtmp://localhost/live/test",
        # livestream_url="https://reverse.st-oc-01.chielo.org/10.5.64.49:8000/rtc/v1/whip/?app=live&stream=ll_test_video&eip=127.0.0.1:8000",
        livestream_url="/path/to/output_video.mp4",
        fps=fps,
        sample_rate=sample_rate,
    )

    audio_path = "/path/to/test_b_2min.wav"
    audio_array, ori_sr = ta.load(audio_path)
    audio_array = ta.functional.resample(audio_array.mean(0), orig_freq=ori_sr, new_freq=16000)
    audio_array = audio_array.reshape(-1)
    secs = audio_array.shape[0] // sample_rate
    interval = 1

    for i in range(0, secs, interval):
        logger.info(f"{i} / {secs} s")
        start = i * sample_rate
        end = (i + interval) * sample_rate
        cur_audio_array = audio_array[start:end]
        logger.info(f"audio: {cur_audio_array.shape} {cur_audio_array.dtype} {cur_audio_array.min()} {cur_audio_array.max()}")

        num_frames = int(interval * fps)
        images = create_simple_video(num_frames, height, width)
        logger.info(f"images: {images.shape} {images.dtype} {images.min()} {images.max()}")

        recorder.pub_livestream(images, cur_audio_array)
        time.sleep(interval)
    recorder.stop()
