import os
import queue
import socket
import subprocess
import threading
import time
import traceback

import numpy as np
import torch
import torchaudio as ta
from loguru import logger


def pseudo_random(a, b):
    x = str(time.time()).split(".")[1]
    y = int(float("0." + x) * 1000000)
    return a + (y % (b - a + 1))


class VARecorder:
    def __init__(
        self,
        livestream_url: str,
        fps: float = 16.0,
        sample_rate: int = 16000,
        slice_frame: int = 1,
        prev_frame: int = 1,
    ):
        self.livestream_url = livestream_url
        self.fps = fps
        self.sample_rate = sample_rate
        self.audio_port = pseudo_random(32000, 40000)
        self.video_port = self.audio_port + 1
        self.ffmpeg_log_level = os.getenv("FFMPEG_LOG_LEVEL", "error")
        logger.info(f"VARecorder audio port: {self.audio_port}, video port: {self.video_port}, ffmpeg_log_level: {self.ffmpeg_log_level}")

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

        # buffer for stream data
        self.audio_samples_per_frame = round(self.sample_rate / self.fps)
        self.stream_buffer = []
        self.stream_buffer_lock = threading.Lock()
        self.stop_schedule = False
        self.schedule_thread = None
        self.slice_frame = slice_frame
        self.prev_frame = prev_frame
        assert self.slice_frame >= self.prev_frame, "Slice frame must be greater than previous frame"

    def init_sockets(self):
        # TCP socket for send and recv video and audio data
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.video_socket.bind(("127.0.0.1", self.video_port))
        self.video_socket.listen(1)

        self.audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.audio_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.audio_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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
                    try:
                        self.audio_conn.send(audios[None].cpu().numpy().tobytes())
                    except (BrokenPipeError, OSError, ConnectionResetError) as e:
                        logger.info(f"Audio connection closed, stopping worker: {type(e).__name__}")
                        return
                    fail_time = 0
                except (BrokenPipeError, OSError, ConnectionResetError):
                    logger.info("Audio connection closed during queue processing")
                    break
                except Exception:
                    logger.error(f"Send audio data error: {traceback.format_exc()}")
                    fail_time += 1
                    if fail_time > max_fail_time:
                        logger.error(f"Audio push worker thread failed {fail_time} times, stopping...")
                        break
        except Exception:
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
                    for i in range(data.shape[0]):
                        t0 = time.time()
                        frame = (data[i] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                        try:
                            self.video_conn.send(frame.tobytes())
                        except (BrokenPipeError, OSError, ConnectionResetError) as e:
                            logger.info(f"Video connection closed, stopping worker: {type(e).__name__}")
                            return
                        if self.realtime and i < data.shape[0] - 1:
                            time.sleep(max(0, packet_secs - (time.time() - t0)))

                    fail_time = 0
                except (BrokenPipeError, OSError, ConnectionResetError):
                    logger.info("Video connection closed during queue processing")
                    break
                except Exception:
                    logger.error(f"Send video data error: {traceback.format_exc()}")
                    fail_time += 1
                    if fail_time > max_fail_time:
                        logger.error(f"Video push worker thread failed {fail_time} times, stopping...")
                        break
        except Exception:
            logger.error(f"Video push worker thread error: {traceback.format_exc()}")
        finally:
            logger.info("Video push worker thread stopped")

    def start_ffmpeg_process_local(self):
        """Start ffmpeg process that connects to our TCP sockets"""
        ffmpeg_cmd = [
            "ffmpeg",
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
            self.ffmpeg_log_level,
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
            self.ffmpeg_log_level,
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
            self.ffmpeg_log_level,
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
        self.pub_livestream(torch.zeros((int(self.fps * duration), height, width, 3), dtype=torch.float16), torch.zeros(int(self.sample_rate * duration), dtype=torch.float16))
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
        self.schedule_thread = threading.Thread(target=self.schedule_stream_buffer)
        self.audio_thread.start()
        self.video_thread.start()
        self.schedule_thread.start()

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

    def buffer_stream(self, images: torch.Tensor, audios: torch.Tensor, gen_video: torch.Tensor):
        N, height, width, C = images.shape
        M = audios.reshape(-1).shape[0]
        assert N % self.slice_frame == 0, "Video frames must be divisible by slice_frame"
        assert C == 3, "Input must be [N, H, W, C] with C=3"

        audio_frames = round(M * self.fps / self.sample_rate)
        if audio_frames != N:
            logger.warning(f"Video and audio frames mismatch, {N} vs {audio_frames}")
        self.set_video_size(width, height)

        # logger.info(f"Buffer stream images {images.shape} {audios.shape} {gen_video.shape}")
        rets = []
        for i in range(0, N, self.slice_frame):
            end_frame = i + self.slice_frame
            img = images[i : end_frame]
            aud = audios[i * self.audio_samples_per_frame : end_frame * self.audio_samples_per_frame]
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
                return self.stream_buffer[-1][2] # return the last video tensor
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
                    self.audio_queue.put(aud)
                    self.video_queue.put(img)
                    # logger.info(f"Scheduled {img.shape[0]} frames and {aud.shape[0]} audio samples to publish")
                    del gen
                    self.stoppable_t = time.time() + aud.shape[0] / self.sample_rate + 3
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

        self.stop_schedule = True
        self.schedule_thread.join(timeout=5)
        if self.schedule_thread and self.schedule_thread.is_alive():
            logger.error(f"Schedule thread did not stop after 5s")

        # Send stop signals to queues
        if self.audio_queue:
            self.audio_queue.put(None)
        if self.video_queue:
            self.video_queue.put(None)

        # Wait for threads to finish processing queued data (increased timeout)
        queue_timeout = 30  # Increased from 5s to 30s to allow sufficient time for large video frames
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=queue_timeout)
            if self.audio_thread.is_alive():
                logger.error(f"Audio push thread did not stop after {queue_timeout}s")
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=queue_timeout)
            if self.video_thread.is_alive():
                logger.error(f"Video push thread did not stop after {queue_timeout}s")

        # Shutdown connections to signal EOF to FFmpeg
        # shutdown(SHUT_WR) will wait for send buffer to flush, no explicit sleep needed
        if self.audio_conn:
            try:
                self.audio_conn.getpeername()
                self.audio_conn.shutdown(socket.SHUT_WR)
                logger.info("Audio connection shutdown initiated")
            except OSError:
                # Connection already closed, skip shutdown
                pass

        if self.video_conn:
            try:
                self.video_conn.getpeername()
                self.video_conn.shutdown(socket.SHUT_WR)
                logger.info("Video connection shutdown initiated")
            except OSError:
                # Connection already closed, skip shutdown
                pass

        if self.ffmpeg_process:
            is_local_file = not self.livestream_url.startswith(("rtmp://", "http"))
            # Local MP4 files need time to write moov atom and finalize the container
            timeout_seconds = 30 if is_local_file else 10
            logger.info(f"Waiting for FFmpeg to finalize file (timeout={timeout_seconds}s, local_file={is_local_file})")
            logger.info(f"FFmpeg output: {self.livestream_url}")

            try:
                returncode = self.ffmpeg_process.wait(timeout=timeout_seconds)
                if returncode == 0:
                    logger.info(f"FFmpeg process exited successfully (exit code: {returncode})")
                else:
                    logger.warning(f"FFmpeg process exited with non-zero code: {returncode}")
            except subprocess.TimeoutExpired:
                logger.warning(f"FFmpeg process did not exit within {timeout_seconds}s, sending SIGTERM...")
                try:
                    self.ffmpeg_process.terminate()  # SIGTERM
                    returncode = self.ffmpeg_process.wait(timeout=5)
                    logger.warning(f"FFmpeg process terminated with SIGTERM (exit code: {returncode})")
                except subprocess.TimeoutExpired:
                    logger.error("FFmpeg process still running after SIGTERM, killing with SIGKILL...")
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait()  # Wait for kill to complete
                    logger.error("FFmpeg process killed with SIGKILL")
            finally:
                self.ffmpeg_process = None

        if self.audio_conn:
            try:
                self.audio_conn.close()
            except Exception as e:
                logger.debug(f"Error closing audio connection: {e}")
            finally:
                self.audio_conn = None

        if self.video_conn:
            try:
                self.video_conn.close()
            except Exception as e:
                logger.debug(f"Error closing video connection: {e}")
            finally:
                self.video_conn = None

        if self.audio_socket:
            try:
                self.audio_socket.close()
            except Exception as e:
                logger.debug(f"Error closing audio socket: {e}")
            finally:
                self.audio_socket = None

        if self.video_socket:
            try:
                self.video_socket.close()
            except Exception as e:
                logger.debug(f"Error closing video socket: {e}")
            finally:
                self.video_socket = None

        if self.audio_queue:
            while self.audio_queue.qsize() > 0:
                try:
                    self.audio_queue.get_nowait()
                except:  # noqa
                    break
        if self.video_queue:
            while self.video_queue.qsize() > 0:
                try:
                    self.video_queue.get_nowait()
                except:  # noqa
                    break
        self.audio_queue = None
        self.video_queue = None
        logger.info("VARecorder stopped and resources cleaned up")

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
