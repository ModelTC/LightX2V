import os
import queue
import socket
import subprocess
import threading
import time
import traceback

import numpy as np
import torch
from loguru import logger


def pseudo_random(a, b):
    x = str(time.time()).split(".")[1]
    y = int(float("0." + x) * 1000000)
    return a + (y % (b - a + 1))


class VideoRecorder:
    def __init__(
        self,
        livestream_url: str,
        fps: float = 16.0,
    ):
        self.livestream_url = livestream_url
        self.fps = fps
        self.video_port = pseudo_random(32000, 40000)
        self.ffmpeg_log_level = os.getenv("FFMPEG_LOG_LEVEL", "error")
        logger.info(f"VideoRecorder video port: {self.video_port}, ffmpeg_log_level: {self.ffmpeg_log_level}")

        self.width = None
        self.height = None
        self.stoppable_t = None
        self.realtime = True

        # ffmpeg process for video data and push to livestream
        self.ffmpeg_process = None

        # TCP connection objects
        self.video_socket = None
        self.video_conn = None
        self.video_thread = None

        # queue for send data to ffmpeg process
        self.video_queue = queue.Queue()

    def init_sockets(self):
        # TCP socket for send and recv video data
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.video_socket.bind(("127.0.0.1", self.video_port))
        self.video_socket.listen(1)

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
                        if self.realtime:
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
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-color_range",
            "pc",
            "-colorspace",
            "rgb",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "iec61966-2-1",
            "-r",
            str(self.fps),
            "-s",
            f"{self.width}x{self.height}",
            "-i",
            f"tcp://127.0.0.1:{self.video_port}",
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
        self.pub_video(torch.zeros((int(self.fps * duration), height, width, 3), dtype=torch.float16))
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
        self.video_thread = threading.Thread(target=self.video_worker)
        self.video_thread.start()

    # Publish ComfyUI Image tensor to livestream
    def pub_video(self, images: torch.Tensor):
        N, height, width, C = images.shape
        assert C == 3, "Input must be [N, H, W, C] with C=3"

        logger.info(f"Publishing video [{N}x{width}x{height}]")

        self.set_video_size(width, height)
        self.video_queue.put(images)
        logger.info(f"Published {N} frames")

        self.stoppable_t = time.time() + N / self.fps + 3

    def stop(self, wait=True):
        if wait and self.stoppable_t:
            t = self.stoppable_t - time.time()
            if t > 0:
                logger.warning(f"Waiting for {t} seconds to stop ...")
                time.sleep(t)
            self.stoppable_t = None

        # Send stop signals to queues
        if self.video_queue:
            self.video_queue.put(None)

        # Wait for threads to finish processing queued data (increased timeout)
        queue_timeout = 30  # Increased from 5s to 30s to allow sufficient time for large video frames
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=queue_timeout)
            if self.video_thread.is_alive():
                logger.error(f"Video push thread did not stop after {queue_timeout}s")

        # Shutdown connections to signal EOF to FFmpeg
        # shutdown(SHUT_WR) will wait for send buffer to flush, no explicit sleep needed
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

        if self.video_conn:
            try:
                self.video_conn.close()
            except Exception as e:
                logger.debug(f"Error closing video connection: {e}")
            finally:
                self.video_conn = None

        if self.video_socket:
            try:
                self.video_socket.close()
            except Exception as e:
                logger.debug(f"Error closing video socket: {e}")
            finally:
                self.video_socket = None

        if self.video_queue:
            while self.video_queue.qsize() > 0:
                try:
                    self.video_queue.get_nowait()
                except:  # noqa
                    break
        self.video_queue = None
        logger.info("VideoRecorder stopped and resources cleaned up")

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
    fps = 16
    width = 640
    height = 480

    recorder = VideoRecorder(
        # livestream_url="rtmp://localhost/live/test",
        # livestream_url="https://reverse.st-oc-01.chielo.org/10.5.64.49:8000/rtc/v1/whip/?app=live&stream=ll_test_video&eip=127.0.0.1:8000",
        livestream_url="/path/to/output_video.mp4",
        fps=fps,
    )

    secs = 10  # 10秒视频
    interval = 1

    for i in range(0, secs, interval):
        logger.info(f"{i} / {secs} s")

        num_frames = int(interval * fps)
        images = create_simple_video(num_frames, height, width)
        logger.info(f"images: {images.shape} {images.dtype} {images.min()} {images.max()}")

        recorder.pub_video(images)
        time.sleep(interval)
    recorder.stop()
