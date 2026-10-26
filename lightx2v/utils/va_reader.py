import os
import queue
import signal
import subprocess
import threading
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger


class VAReader:
    def __init__(
        self,
        rank: int,
        world_size: int,
        stream_url: str,
        segment_duration: float = 5.0,
        sample_rate: int = 16000,
        audio_channels: int = 1,
        buffer_size: int = 1,
        prev_duration: float = 0.3125,
        target_rank: int = 0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.stream_url = stream_url
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        self.prev_duration = prev_duration
        # int16 = 2 bytes
        self.chunk_size = int(self.segment_duration * self.sample_rate) * 2
        self.prev_size = int(self.prev_duration * self.sample_rate) * 2
        self.prev_chunk = None
        self.buffer_size = buffer_size

        self.audio_queue = queue.Queue(maxsize=self.buffer_size)
        self.audio_thread = None
        self.ffmpeg_process = None
        self.bytes_buffer = bytearray()

        self.target_rank = target_rank % self.world_size

        self.flag_tensor = torch.tensor([0], dtype=torch.int32).to(device="cuda")
        self.audio_tensor = torch.zeros(self.chunk_size, dtype=torch.uint8, device="cuda")

        logger.info(f"VAReader initialized for stream: {stream_url} target_rank: {self.target_rank}")
        logger.info(f"Audio duration per chunk: {segment_duration}s, sample rate: {sample_rate}Hz")

    def start(self):
        if self.rank == self.target_rank:
            if self.stream_url.startswith("rtmp://"):
                self.start_ffmpeg_process_rtmp()
            elif self.stream_url.startswith("http"):
                self.start_ffmpeg_process_whep()
            else:
                raise Exception(f"Unsupported stream URL: {self.stream_url}")
            self.audio_thread = threading.Thread(target=self.audio_worker, daemon=True)
            self.audio_thread.start()
            logger.info(f"VAReader {self.rank}/{self.world_size} started successfully")
        else:
            logger.info(f"VAReader {self.rank}/{self.world_size} wait only")
        if self.world_size > 1:
            logger.info(f"VAReader {self.rank}/{self.world_size} wait barrier")
            dist.barrier()
            logger.info(f"VAReader {self.rank}/{self.world_size} end barrier")

    def start_ffmpeg_process_rtmp(self):
        """Start ffmpeg process read audio from stream"""
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            self.stream_url,
            "-vn",
            # "-acodec",
            # "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.audio_channels),
            "-f",
            "s16le",
            "-",
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            logger.info(f"FFmpeg audio pull process started with PID: {self.ffmpeg_process.pid}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            raise

    def start_ffmpeg_process_whep(self):
        """Start gstream process read audio from stream"""
        ffmpeg_cmd = [
            "gst-launch-1.0",
            "-q",
            "whepsrc",
            f"whep-endpoint={self.stream_url}",
            "video-caps=none",
            "!rtpopusdepay",
            "!opusdec",
            "plc=false",
            "!audioconvert",
            "!audioresample",
            f"!audio/x-raw,format=S16LE,channels={self.audio_channels},rate={self.sample_rate}",
            "!fdsink",
            "fd=1",
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                bufsize=0,
            )
            logger.info(f"FFmpeg audio pull process started with PID: {self.ffmpeg_process.pid}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            raise

    def audio_worker(self):
        logger.info("Audio pull worker thread started")
        try:
            while True:
                if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg process exited, audio worker thread stopped")
                    break
                self.fetch_audio_data()
                time.sleep(0.01)
        except:  # noqa
            logger.error(f"Audio pull worker error: {traceback.format_exc()}")
        finally:
            logger.warning("Audio pull worker thread stopped")

    def fetch_audio_data(self):
        """Fetch audio data from ffmpeg process"""
        try:
            audio_bytes = self.ffmpeg_process.stdout.read(self.chunk_size)
            if not audio_bytes:
                return
            self.bytes_buffer.extend(audio_bytes)
            # logger.info(f"Fetch audio data: {len(audio_bytes)} bytes, bytes_buffer: {len(self.bytes_buffer)} bytes")

            if len(self.bytes_buffer) >= self.chunk_size:
                audio_data = self.bytes_buffer[: self.chunk_size]
                self.bytes_buffer = self.bytes_buffer[self.chunk_size :]

                # first chunk, read original 81 frames
                # for other chunks, read 81 - 5 = 76 frames, concat with previous 5 frames
                if self.prev_chunk is None:
                    logger.info(f"change chunk_size: from {self.chunk_size} to {self.chunk_size - self.prev_size}")
                    self.chunk_size -= self.prev_size
                else:
                    audio_data = self.prev_chunk + audio_data
                self.prev_chunk = audio_data[-self.prev_size :]

                try:
                    self.audio_queue.put_nowait(audio_data)
                except queue.Full:
                    logger.warning(f"Audio queue full:{self.audio_queue.qsize()}, discarded oldest chunk")
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_data)
                logger.info(f"Put audio data: {len(audio_data)} bytes, audio_queue: {self.audio_queue.qsize()}, chunk_size:{self.chunk_size}")

        except:  # noqa
            logger.error(f"Fetch audio data error: {traceback.format_exc()}")

    def braodcast_audio_data(self, audio_data):
        if self.rank == self.target_rank:
            if audio_data is None:
                self.flag_tensor.fill_(0)
            else:
                self.flag_tensor.fill_(1)
                self.audio_tensor.copy_(torch.frombuffer(bytearray(audio_data), dtype=torch.uint8))
                logger.info(f"rank {self.rank} send audio_tensor: {self.audio_tensor.shape}")

        dist.broadcast(self.flag_tensor, src=self.target_rank)
        if self.flag_tensor.item() == 0:
            return None

        dist.broadcast(self.audio_tensor, src=self.target_rank)
        if self.rank != self.target_rank:
            logger.info(f"rank {self.rank} recv audio_tensor: {self.audio_tensor.shape}")
            audio_data = self.audio_tensor.cpu().numpy().tobytes()
        return audio_data

    def bytes_to_ndarray(self, audio_data):
        if audio_data is None:
            return None
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        logger.info(f"Got segment audio rank={self.rank}: {audio_data.shape} {audio_data.dtype} {audio_data.min()} {audio_data.max()}")
        return audio_data

    def get_audio_segment(self, timeout: float = 1.0, fetch_duration: float = None, prev_duration: float = None):
        if fetch_duration is not None and self.segment_duration != fetch_duration:
            logger.warning(f"ignore fetch_duration, {fetch_duration} != {self.segment_duration}")
        if prev_duration is not None and self.prev_duration != prev_duration:
            raise ValueError(f"prev_duration {prev_duration} != {self.prev_duration}")
        audio_data = None
        if self.rank == self.target_rank:
            try:
                audio_data = self.audio_queue.get(timeout=timeout)
            except:  # noqa
                logger.warning(f"Failed to get audio segment: {traceback.format_exc()}")
        if self.world_size > 1:
            audio_data = self.braodcast_audio_data(audio_data)
        audio_data = self.bytes_to_ndarray(audio_data)
        return audio_data, self.segment_duration

    def stop(self):
        # Stop ffmpeg process
        if self.ffmpeg_process:
            self.ffmpeg_process.send_signal(signal.SIGINT)
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            logger.warning("FFmpeg reader process stopped")

        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=5)
            if self.audio_thread.is_alive():
                logger.error("Audio pull thread did not stop gracefully")

        while self.audio_queue and self.audio_queue.qsize() > 0:
            self.audio_queue.get_nowait()
        self.audio_queue = None
        logger.warning("Audio pull queue cleaned")

    def __del__(self):
        self.stop()


class VideoOnlyReader:
    def __init__(
        self,
        rank: int,
        world_size: int,
        stream_url: str,
        fps: float,
        num_frame: int,
        height: int,
        width: int,
        target_rank: int = 0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.stream_url = stream_url
        self.fps = fps
        self.num_frame = num_frame
        self.height = height
        self.width = width
        self.target_rank = target_rank % self.world_size
        if self.num_frame <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError(f"num_frame, height and width must be positive, got {num_frame}, {height}, {width}")

        # Packed RGB, one segment is num_frame frames with no overlap.
        self.frame_bytes = self.width * self.height * 3
        self.chunk_bytes = self.frame_bytes * self.num_frame
        fps_rate = float(self.fps)
        self.fps_num, self.fps_den = (int(fps_rate), 1) if fps_rate == int(fps_rate) else (round(fps_rate * 1000), 1000)

        self.video_queue = queue.Queue(maxsize=1)
        self.video_thread = None
        self.ffmpeg_process = None
        self.bytes_buffer = bytearray()

        self.flag_tensor = torch.tensor([0], dtype=torch.int32).to(device="cuda")
        self.video_tensor = torch.zeros(self.chunk_bytes, dtype=torch.uint8, device="cuda")

        logger.info(f"VideoOnlyReader initialized for stream: {stream_url} target_rank: {self.target_rank}")
        logger.info(f"Video segment: {num_frame} frames, {fps} fps, {width}x{height}")

    def start(self):
        if self.rank == self.target_rank:
            if self.stream_url.startswith("rtmp://"):
                self.start_ffmpeg_process_rtmp()
            elif self.stream_url.startswith("http"):
                self.start_ffmpeg_process_whep()
            else:
                raise Exception(f"Unsupported stream URL: {self.stream_url}")
            self.video_thread = threading.Thread(target=self.video_worker, daemon=True)
            self.video_thread.start()
            logger.info(f"VideoOnlyReader {self.rank}/{self.world_size} started successfully")
        else:
            logger.info(f"VideoOnlyReader {self.rank}/{self.world_size} wait only")
        if self.world_size > 1:
            logger.info(f"VideoOnlyReader {self.rank}/{self.world_size} wait barrier")
            dist.barrier()
            logger.info(f"VideoOnlyReader {self.rank}/{self.world_size} end barrier")

    def start_ffmpeg_process_rtmp(self):
        """Start ffmpeg process read video from stream"""
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            self.stream_url,
            "-an",
            "-vf",
            f"fps={self.fps},scale={self.width}:{self.height}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=0)
            logger.info(f"FFmpeg video pull process started with PID: {self.ffmpeg_process.pid}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            raise

    def start_ffmpeg_process_whep(self):
        """Start gstream process read video from stream"""
        ffmpeg_cmd = [
            "gst-launch-1.0",
            "-q",
            "whepsrc",
            f"whep-endpoint={self.stream_url}",
            "audio-caps=none",
            "!rtph264depay",
            "!h264parse",
            "!avdec_h264",
            "!videoconvert",
            "!videoscale",
            "!videorate",
            f"!video/x-raw,format=RGB,width={self.width},height={self.height},framerate={self.fps_num}/{self.fps_den}",
            "!fdsink",
            "fd=1",
            "sync=false",
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=0)
            logger.info(f"FFmpeg video pull process started with PID: {self.ffmpeg_process.pid}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            raise

    def video_worker(self):
        logger.info("Video pull worker thread started")
        try:
            while True:
                if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg process exited, video worker thread stopped")
                    break
                self.fetch_video_data()
                time.sleep(0.01)
        except:  # noqa
            logger.error(f"Video pull worker error: {traceback.format_exc()}")
        finally:
            logger.warning("Video pull worker thread stopped")

    def fetch_video_data(self):
        """Fetch one segment of packed RGB frames from the pull process"""
        try:
            video_bytes = self.ffmpeg_process.stdout.read(self.chunk_bytes)
            if not video_bytes:
                return
            self.bytes_buffer.extend(video_bytes)

            if len(self.bytes_buffer) >= self.chunk_bytes:
                video_data = bytes(self.bytes_buffer[: self.chunk_bytes])
                del self.bytes_buffer[: self.chunk_bytes]
                try:
                    self.video_queue.put_nowait(video_data)
                except queue.Full:
                    logger.warning(f"Video queue full:{self.video_queue.qsize()}, discarded oldest segment")
                    self.video_queue.get_nowait()
                    self.video_queue.put_nowait(video_data)
                logger.info(f"Put video data: {len(video_data)} bytes, video_queue: {self.video_queue.qsize()}, frames:{self.num_frame}")

        except:  # noqa
            logger.error(f"Fetch video data error: {traceback.format_exc()}")

    def broadcast_video_data(self, video_data):
        if self.rank == self.target_rank:
            if video_data is None:
                self.flag_tensor.fill_(0)
            else:
                self.flag_tensor.fill_(1)
                self.video_tensor.copy_(torch.frombuffer(bytearray(video_data), dtype=torch.uint8))
                logger.info(f"rank {self.rank} send video_tensor: {self.video_tensor.shape}")

        dist.broadcast(self.flag_tensor, src=self.target_rank)
        if self.flag_tensor.item() == 0:
            return None

        dist.broadcast(self.video_tensor, src=self.target_rank)
        if self.rank != self.target_rank:
            logger.info(f"rank {self.rank} recv video_tensor: {self.video_tensor.shape}")
            video_data = self.video_tensor.cpu().numpy().tobytes()
        return video_data

    def bytes_to_ndarray(self, video_data):
        if video_data is None:
            return None
        frames = np.frombuffer(video_data, dtype=np.uint8).reshape(self.num_frame, self.height, self.width, 3).copy()
        logger.info(f"Got segment video rank={self.rank}: {frames.shape} {frames.dtype}")
        return frames

    def get_video_segment(self, timeout: float = 1.0):
        video_data = None
        if self.rank == self.target_rank:
            try:
                video_data = self.video_queue.get(timeout=timeout)
            except:  # noqa
                logger.warning(f"Failed to get video segment: {traceback.format_exc()}")
        if self.world_size > 1:
            video_data = self.broadcast_video_data(video_data)
        return self.bytes_to_ndarray(video_data)

    def stop(self):
        if self.ffmpeg_process:
            self.ffmpeg_process.send_signal(signal.SIGINT)
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            logger.warning("FFmpeg video reader process stopped")

        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=5)
            if self.video_thread.is_alive():
                logger.error("Video pull thread did not stop gracefully")

        while self.video_queue and self.video_queue.qsize() > 0:
            self.video_queue.get_nowait()
        self.video_queue = None
        logger.warning("Video pull queue cleaned")

    def __del__(self):
        self.stop()


def _init_dist():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        logger.info(f"Distributed initialized: rank={rank}, world_size={world_size}")
    return rank, world_size


def run_va_reader():
    rank, world_size = _init_dist()
    reader = VAReader(
        rank,
        world_size,
        "rtmp://localhost/live/test_audio",
        segment_duration=1.0,
        sample_rate=16000,
        audio_channels=1,
        prev_duration=1 / 16,
    )
    reader.start()
    fail_count = 0
    max_fail_count = 2

    try:
        while True:
            audio_data = reader.get_audio_segment(timeout=2)
            if audio_data is not None:
                # logger.info(f"Got audio chunk, shape: {audio_data.shape}, range: [{audio_data.min()}, {audio_data.max()}]")
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > max_fail_count:
                    logger.warning("Failed to get audio chunk, stop reader")
                    reader.stop()
                    break
            time.sleep(0.95)
    finally:
        reader.stop()


def save_video_segment(frames, path, fps):
    """Write one uint8 RGB segment, shaped (T, H, W, 3), to an mp4 file."""
    import av

    height, width = frames.shape[1] // 2 * 2, frames.shape[2] // 2 * 2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    container = av.open(path, "w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=int(fps), options={"crf": "18", "preset": "veryfast"})
        stream.height = height
        stream.width = width
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame[:height, :width], format="rgb24").reformat(format="yuv420p")
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def run_video_only_reader():
    rank, world_size = _init_dist()
    fps = 16
    num_frame = 32
    reader = VideoOnlyReader(
        rank,
        world_size,
        "rtmp://localhost/live/test_video",
        fps=fps,
        num_frame=num_frame,
        height=480,
        width=640,
    )
    reader.start()
    fail_count = 0
    max_fail_count = 2
    segment_index = 0
    debug_dir = os.path.abspath("debug_video_segments")

    try:
        while True:
            video_data = reader.get_video_segment(timeout=2)
            if video_data is not None:
                segment_path = os.path.join(debug_dir, f"segment_{segment_index:04d}.mp4")
                save_video_segment(video_data, segment_path, fps)
                logger.info(f"Got video segment, shape: {video_data.shape}, dtype: {video_data.dtype}, saved: {segment_path}")
                segment_index += 1
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > max_fail_count:
                    logger.warning("Failed to get video segment, stop reader")
                    reader.stop()
                    break
            time.sleep(num_frame / fps * 0.95)
    finally:
        reader.stop()


if __name__ == "__main__":
    run_video_only_reader()
