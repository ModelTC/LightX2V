import datetime
import json
import os
import random
import subprocess
import threading
import time
import traceback
from collections import deque
from copy import deepcopy

import jsonschema
import numpy as np
import torch
import torch.distributed as dist
import zmq
from loguru import logger

try:
    from bson import BSON
except ImportError:
    BSON = None
    logger.warning("BSON is not installed")
from scipy.signal import resample


class AudioInfo:
    def __init__(self, info: dict):
        self.sample_count = info["sample_count"]
        self.sample_rate = info["sample_rate"]
        self.channel_count = info["channel_count"]
        self.sample_fmt = info["sample_fmt"]
        self.pts = info["pts"]

    def is_spec_equal(self, other: "AudioInfo") -> bool:
        return self.sample_fmt == other.sample_fmt and self.sample_rate == other.sample_rate and self.channel_count == other.channel_count

    def duration(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.sample_count / self.sample_rate)

    def __str__(self):
        return "AudioInfo(sample_count={}, sample_rate={}, channel_count={}, sample_fmt={}, pts={})".format(self.sample_count, self.sample_rate, self.channel_count, self.sample_fmt, self.pts)


class ByteBuffer:
    def __init__(self):
        self.buffer = deque()
        self.current_size = 0
        # is the audio belonging to current turn finished
        self.audio_finished = False

    def add(self, byte_data: bytes):
        self.buffer.append(byte_data)
        self.current_size += len(byte_data)

    def get(self, size=1024):
        data = bytearray()

        while size > 0 and len(self.buffer) > 0:
            chunk = self.buffer.popleft()
            if len(chunk) <= size:
                # 如果当前数据小于size，则将当前数据全部添加到data中
                data.extend(chunk)
                self.current_size -= len(chunk)
                size -= len(chunk)
            else:
                # 如果当前数据大于size，则将当前数据的一部分添加到data中，剩余部分留在缓冲区
                data.extend(chunk[:size])
                self.buffer.appendleft(chunk[size:])  # 剩余部分留在缓冲区
                self.current_size -= size
                size = 0

        return bytes(data)

    def mark_finished(self):
        self.audio_finished = True

    def has_more_voice(self):
        return not self.audio_finished

    def __len__(self):
        return self.current_size


class ChatAdapter:
    def __init__(
        self,
        omni_work_dir: str,
        whep_url: str,
        session_id: str,
        account: str,
        config_files: list[str],
        config_schema_path: str,
        seg_duration: float,
        model_runner,
        huoshan_tts_voice_type,
        stream_config: dict,
    ):
        assert os.path.exists(omni_work_dir), f"OMNI work directory {omni_work_dir} does not exist"
        self.omni_work_dir = omni_work_dir
        self.stream_config = stream_config
        self.context = zmq.Context()
        self.w2f_socket = self.context.socket(zmq.PULL)
        self.w2f_url = ChatAdapter.select_and_bind(self.w2f_socket)
        self.f2w_socket = self.context.socket(zmq.PUSH)
        self.f2w_url = ChatAdapter.select_and_bind(self.f2w_socket)
        self.recv_thread = None
        self.audio_buffer = ByteBuffer()
        self.audio_info = None
        self.chat_server_cmd = [
            os.path.join(self.omni_work_dir, "bin", "seko-chatter"),
            "--session-id",
            session_id,
            "--account",
            account,
            "--whep-server-url",
            whep_url,
            "--w2f-endpoint",
            self.w2f_url,
            "--f2w-endpoint",
            self.f2w_url,
            "--config-files",
            *config_files,
        ]
        override_config = {}
        if huoshan_tts_voice_type is not None:
            logger.info(f"Use Huoshan TTS voice type: {huoshan_tts_voice_type}")
            override_config["TTS"] = {
                "default_voice_info": {
                    "voice_type": huoshan_tts_voice_type,
                    "provider": "huoshan_stream_tts",
                }
            }
        system_prompt = stream_config.get("system_prompt", "")
        if system_prompt:
            override_config["model"] = {"system_prompt": system_prompt}
            logger.info(f"Omni use custom system prompt: {system_prompt}")
        with open(config_schema_path, "r") as f:
            schema = json.load(f)
        jsonschema.validate(instance=override_config, schema=schema)
        if override_config is not None:
            self.chat_server_cmd.extend(["--override-config", json.dumps(override_config)])
        self.chatter_proc = None

        self.seg_duration = seg_duration
        self.reset_prev = False
        self.status = "blank"
        self.immediate_switch = 0
        self.image_switch = ""
        self.action_switch = ""
        self.model_runner = model_runner

    def launch_chat_server(self):
        env = {
            "RUST_LOG": "info,duplex_server=debug,backend_5o=debug",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.path.join(self.omni_work_dir, "lib/"),
            "PATH": os.environ["PATH"] + ":" + os.path.join(self.omni_work_dir, "bin/"),
        }
        self.chatter_proc = subprocess.Popen(self.chat_server_cmd, env=env, cwd=self.omni_work_dir)

    @staticmethod
    def select_and_bind(socket: zmq.Socket) -> str:
        # randomly select a port between 1024 and 6553
        retry_count = 20
        err = None
        while retry_count > 0:
            try:
                port = random.randint(1024, 65535)
                # port = 5555
                url = f"tcp://localhost:{port}"
                socket.bind(url)
                return url
            except zmq.error.ZMQError as e:
                retry_count -= 1
                err = e
        raise err

    # immediate switch to status, discard prev_bytes, set immediate_switch to 1
    def immediate_switch_to(self, status):
        logger.warning(f"VA reader immediate switch to {status}")
        self.reset_prev = True
        self.status = status
        self.immediate_switch = 1
        # only no action switch can be paused immediately
        if self.model_runner is not None and self.model_runner.can_pause:
            self.model_runner.pause_signal = True
            logger.warning(f"Model runner pause signal set to True")

    def set_image_switch(self, image_path):
        logger.warning(f"Setting image switch: {image_path}")
        self.image_switch = image_path
        # only blank status and no action switch can be paused immediately
        if self.model_runner is not None and self.model_runner.can_pause:
            self.model_runner.pause_signal = True
            logger.warning(f"Model runner set pause signal for image switch & blank status")

    def set_action_switch(self, prompt):
        logger.warning(f"Setting action switch: {prompt}")
        self.action_switch = prompt
        # only blank status can be paused immediately
        if self.model_runner is not None and self.model_runner.can_pause:
            self.model_runner.pause_signal = True
            logger.warning(f"Model runner set pause signal for action switch & blank status")

    def recv_loop(self):
        while True:
            try:
                message = self.w2f_socket.recv()
            except Exception:
                logger.error(f"Error receiving message: {traceback.format_exc()}")
                break
            try:
                message = BSON.decode(message)
                msg_type = message["type"]
                logger.debug("Received message type: {}".format(msg_type))
                if msg_type == "AgentAudio":
                    audio = message["audio"]
                    if audio["type"] != "Pcm":
                        logger.error("Unsupported audio type: {}".format(audio["type"]))
                        continue
                    pcm_data = audio["data"]
                    audio_info = AudioInfo(audio["info"])
                    logger.debug("Received audio with duration: {}".format(audio_info.duration()))
                    if self.audio_info is None:
                        self.audio_info = audio_info
                    else:
                        # check if the audio info is the same
                        if not self.audio_info.is_spec_equal(audio_info):
                            raise ValueError("Audio info mismatch")
                    self.audio_buffer.add(pcm_data)
                    # if status is blank and has voice, set immediate switch to 1
                    if self.status == "blank" and self.has_voice(self.seg_duration):
                        self.immediate_switch_to("voice")
                elif msg_type == "AgentStartPlay":
                    logger.debug("Received AgentStartPlay, create new audio buffer")
                    self.audio_buffer = ByteBuffer()
                elif msg_type == "AgentEndPlay":
                    logger.debug("Received AgentEndPlay, mark audio finished")
                    self.audio_buffer.mark_finished()
                elif msg_type == "ClearAgentAudio":
                    logger.warning("Received ClearAgentAudio, clear audio buffer")
                    self.audio_buffer = None
                    self.audio_info = None
                    if self.status == "voice":
                        self.status = "blank"
                        # self.immediate_switch_to("blank")
            except Exception as e:
                logger.error("Error decoding message: {}, continue".format(e))
                continue
        logger.warning("recv loop interrupted")

    def start(self):
        self.launch_chat_server()
        self.recv_thread = threading.Thread(target=self.recv_loop)
        self.recv_thread.start()

    def has_voice(self, duration) -> bool:
        if self.audio_info is None or self.audio_buffer.current_size == 0:
            return False
        bytes_count = round(duration * self.audio_info.sample_rate) * self.audio_info.channel_count * 2  # S16LE assumed
        # if not has enough bytes and maybe has more voice, return False
        if self.audio_buffer.current_size < bytes_count and self.audio_buffer.has_more_voice():
            logger.warning(f"Not enough bytes and maybe has more voice, content_size: {self.audio_buffer.current_size}, bytes_count: {bytes_count}")
            return False
        return bytes_count

    def get_audio(self, fetch_duration) -> (bytes, AudioInfo):
        bytes_count = self.has_voice(fetch_duration)
        if bytes_count is False or self.audio_info is None:
            return None
        pcm_data = self.audio_buffer.get(bytes_count)

        # the actual sample count fetched
        sample_count = len(pcm_data) // (self.audio_info.channel_count * 2)
        logger.debug("Fetched {} bytes audio".format(sample_count))
        logger.debug("After fetch, there are {} bytes left".format(self.audio_buffer.current_size))
        audio_info = deepcopy(self.audio_info)
        audio_info.sample_count = sample_count
        return (pcm_data, audio_info)

    def stop(self):
        self.model_runner = None
        if self.chatter_proc is not None:
            self.chatter_proc.terminate()
            self.chatter_proc.wait()
            self.chatter_proc = None
        self.w2f_socket.close()
        self.f2w_socket.close()

    def __del__(self):
        self.stop()


class OmniVAReader:
    def __init__(
        self,
        rank: int,
        world_size: int,
        stream_url: str,
        segment_duration: float = 5.0625,
        sample_rate: int = 16000,
        audio_channels: int = 1,
        buffer_size: int = 1,
        prev_duration: float = 0.3125,
        target_rank: int = 0,
        model_runner=None,
        huoshan_tts_voice_type=None,
        stream_config: dict = {},
        **kwargs,
    ):
        self.rank = rank
        self.world_size = world_size
        self.stream_url = stream_url
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate

        self.audio_channels = audio_channels
        self.prev_duration = prev_duration
        self.all_seg_sample_count = int(self.segment_duration * self.sample_rate)
        self.prev_seg_sample_count = int(self.prev_duration * self.sample_rate)
        self.prev_seg_chunk = None

        self.target_rank = target_rank % self.world_size
        self.flag_tensor = torch.tensor([0], dtype=torch.int32).to(device="cuda")
        self.valid_duration_tensor = torch.tensor([0], dtype=torch.float32).to(device="cuda")
        self.immediate_switch_tensor = torch.tensor([0], dtype=torch.int32).to(device="cuda")
        chunk_size = int(self.segment_duration * self.sample_rate) * 2
        self.audio_tensor = torch.zeros(chunk_size, dtype=torch.uint8, device="cuda")
        self.chat_adapter = None
        self.model_runner = model_runner
        self.huoshan_tts_voice_type = huoshan_tts_voice_type
        self.stream_config = stream_config

        assert self.audio_channels == 1, "Only mono audio is supported for OmniVAReader"
        logger.info(f"VAReader initialized for stream: {stream_url} target_rank: {self.target_rank}")
        logger.info(f"Audio duration per chunk: {segment_duration}s, sample rate: {sample_rate}Hz")

    def init_omni_env(self):
        self.omni_work_dir = os.getenv("OMNI_WORK_DIR", "/path/of/seko_chatter/")
        self.session_id = os.getenv("OMNI_SESSION_ID", "")
        self.account = os.getenv("OMNI_ACCOUNT", "")
        self.config_files = os.getenv("OMNI_CONFIG_FILES", "").split(",")
        self.config_schema_path = os.getenv("OMNI_CONFIG_SCHEMA_PATH", None)
        assert os.path.exists(self.omni_work_dir), f"OMNI work directory {self.omni_work_dir} does not exist"
        assert self.session_id and self.account, "OMNI_SESSION_ID and OMNI_ACCOUNT are required"
        logger.info(
            f"OMNI work directory: {self.omni_work_dir}, session_id: {self.session_id}, account: {self.account}, config_files: {self.config_files}, config_schema_path: {self.config_schema_path}"
        )

    def start(self):
        if self.rank == self.target_rank:
            self.init_omni_env()
            assert self.stream_url.startswith("http"), "Only HTTP stream is supported for OmniVAReader"
            self.chat_adapter = ChatAdapter(
                omni_work_dir=self.omni_work_dir,
                whep_url=self.stream_url,
                session_id=self.session_id,
                account=self.account,
                config_files=self.config_files,
                config_schema_path=self.config_schema_path,
                seg_duration=self.segment_duration,
                model_runner=self.model_runner,
                huoshan_tts_voice_type=self.huoshan_tts_voice_type,
                stream_config=self.stream_config,
            )
            self.chat_adapter.start()
            logger.info(f"OmniVAReader {self.rank}/{self.world_size} started successfully")
        else:
            logger.info(f"OmniVAReader {self.rank}/{self.world_size} wait only")
        if self.world_size > 1:
            logger.info(f"OmniVAReader {self.rank}/{self.world_size} wait barrier")
            dist.barrier()
            logger.info(f"OmniVAReader {self.rank}/{self.world_size} end barrier")

    def braodcast_audio_data(self, audio_data):
        if self.rank == self.target_rank:
            if audio_data is None:
                self.flag_tensor.fill_(0)
            else:
                self.flag_tensor.fill_(1)
                self.audio_tensor.copy_(torch.frombuffer(bytearray(audio_data), dtype=torch.uint8))
                # logger.info(f"rank {self.rank} send audio_tensor: {self.audio_tensor.shape}")

        dist.broadcast(self.flag_tensor, src=self.target_rank)
        if self.flag_tensor.item() == 0:
            return None

        dist.broadcast(self.audio_tensor, src=self.target_rank)
        if self.rank != self.target_rank:
            # logger.info(f"rank {self.rank} recv audio_tensor: {self.audio_tensor.shape}")
            audio_data = self.audio_tensor.cpu().numpy().tobytes()
        return audio_data

    def braodcast_valid_duration(self, valid_duration):
        if self.rank == self.target_rank:
            self.valid_duration_tensor.fill_(valid_duration)
        dist.broadcast(self.valid_duration_tensor, src=self.target_rank)
        return self.valid_duration_tensor.item()

    def bytes_to_ndarray(self, audio_data):
        if audio_data is None:
            return None
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        # logger.info(f"Got segment audio rank={self.rank}: {audio_data.shape} {audio_data.dtype} {audio_data.min()} {audio_data.max()}")
        return audio_data

    def convert_pcm_s16le_to_mono_resampled(self, audio_data, audio_info):
        audio = np.frombuffer(audio_data, dtype=np.int16)
        sample_count = audio_info.sample_count
        assert len(audio) == sample_count * audio_info.channel_count, f"audio length {len(audio)} != sample_count * channel_count {sample_count * audio_info.channel_count}"
        # convert to mono
        if audio_info.channel_count > 1:
            audio = audio.reshape(-1, audio_info.channel_count).mean(axis=1)

        # logger.info(f"audio: {audio.shape} {audio.dtype} {audio.min()} {audio.max()}")
        if audio_info.sample_rate != self.sample_rate:
            sample_count = int(len(audio) * self.sample_rate / audio_info.sample_rate)
            audio = resample(audio, sample_count).astype(np.int16)
            # logger.info(f"resampled audio: {audio.shape} {audio.dtype} {audio.min()} {audio.max()} {sample_count}")
        logger.warning(f"valid audio: {audio.shape} {audio.dtype} {audio.min()} {audio.max()} {sample_count}")
        return audio, sample_count

    def prepare_audio_data(self, chat_audio_result):
        sample_count = 0
        audio = np.array([], dtype=np.int16)

        # convert chat audio result to mono and target sample rate
        if chat_audio_result is not None:
            audio_data, audio_info = chat_audio_result
            audio, sample_count = self.convert_pcm_s16le_to_mono_resampled(audio_data, audio_info)
        valid_duration = sample_count / self.sample_rate

        # if is not the first segment, concat with previous segment
        if self.prev_seg_chunk is not None:
            audio = np.concatenate([self.prev_seg_chunk, audio])
            sample_count = len(audio)
        assert sample_count <= self.all_seg_sample_count, f"audio length {sample_count} > all_seg_sample_count {self.all_seg_sample_count}"

        # pad 0 to the audio to make it the same length as all_seg_sample_count
        if sample_count < self.all_seg_sample_count:
            pad_count = self.all_seg_sample_count - sample_count
            # logger.info(f"pad {pad_count} samples to audio")
            audio = np.pad(audio, (0, pad_count), mode="constant", constant_values=0)
            sample_count = len(audio)

        # update prev seg chunk
        self.prev_seg_chunk = audio[-self.prev_seg_sample_count :]
        # logger.info(f"audio: {audio.shape} {audio.dtype} {audio.min()} {audio.max()} {sample_count}, prev seg chunk: {self.prev_seg_chunk.shape}")
        return audio.tobytes(), valid_duration

    def get_fetch_duration(self):
        fetch_duration = self.segment_duration
        # after immediate switch, reset prev seg chunk
        if self.chat_adapter is not None and self.chat_adapter.reset_prev:
            self.prev_seg_chunk = None
            self.chat_adapter.reset_prev = False
            logger.warning(f"Reset prev seg chunk")
        # first segment, fetch segment_duration, else fetch segment_duration - prev_duration
        if self.prev_seg_chunk is not None:
            fetch_duration -= self.prev_duration
        return fetch_duration

    def change_segment_duration(self, segment_duration):
        if segment_duration is None or self.segment_duration == segment_duration:
            return
        if self.rank == self.target_rank:
            logger.warning(f"segment duration changed: {self.segment_duration} -> {segment_duration}")
        self.segment_duration = segment_duration
        self.all_seg_sample_count = int(self.segment_duration * self.sample_rate)
        chunk_size = int(self.segment_duration * self.sample_rate) * 2
        self.audio_tensor = torch.zeros(chunk_size, dtype=torch.uint8, device="cuda")
        if self.chat_adapter is not None:
            self.chat_adapter.seg_duration = segment_duration

    def get_audio_segment(self, fetch_duration: float = None, prev_duration: float = None):
        audio_data = None
        valid_duration = 0
        if prev_duration is not None and self.prev_duration != prev_duration:
            raise ValueError(f"prev_duration {prev_duration} != {self.prev_duration}")
        self.change_segment_duration(fetch_duration)

        if self.rank == self.target_rank:
            try:
                fetch_duration = self.get_fetch_duration()
                # logger.info(f"Get segment, fetch_duration: {fetch_duration}")
                if self.chat_adapter.status == "voice":
                    audio_result = self.chat_adapter.get_audio(fetch_duration)
                    audio_data, valid_duration = self.prepare_audio_data(audio_result)
                    # think all voice segments inferred, naturally switch to blank
                    if audio_result is None:
                        logger.info(f"Think all voice segments inferred, naturally switch to blank")
                        self.chat_adapter.status = "blank"
                else:
                    audio_data, valid_duration = self.prepare_audio_data(None)
            except Exception as e:
                logger.warning(f"Failed to get voice segment: {e}")
                return None, 0
        if self.world_size > 1:
            audio_data = self.braodcast_audio_data(audio_data)
            valid_duration = self.braodcast_valid_duration(valid_duration)
        audio_data = self.bytes_to_ndarray(audio_data)
        return audio_data, valid_duration

    def get_immediate_switch(self):
        if self.rank == self.target_rank:
            if self.chat_adapter is not None and self.chat_adapter.immediate_switch == 1:
                self.immediate_switch_tensor.fill_(1)
                # reset immediate switch
                self.chat_adapter.immediate_switch = 0
            else:
                self.immediate_switch_tensor.fill_(0)
        if self.world_size > 1:
            dist.broadcast(self.immediate_switch_tensor, src=self.target_rank)
        return self.immediate_switch_tensor.item()

    def get_image_switch(self):
        data = "" if self.chat_adapter is None else self.chat_adapter.image_switch
        image_switch = self.broadcast_data(data)
        # reset image switch
        if self.chat_adapter is not None:
            self.chat_adapter.image_switch = ""
        return image_switch

    def get_action_switch(self):
        data = "" if self.chat_adapter is None else self.chat_adapter.action_switch
        action_switch = self.broadcast_data(data)
        # reset action switch
        if self.chat_adapter is not None:
            self.chat_adapter.action_switch = ""
        return action_switch

    def broadcast_data(self, data):
        if self.world_size <= 1:
            return data
        if self.rank == self.target_rank:
            val = json.dumps(data, ensure_ascii=False).encode("utf-8")
            T = torch.frombuffer(bytearray(val), dtype=torch.uint8).to(device="cuda")
            S = torch.tensor([T.shape[0]], dtype=torch.int32).to(device="cuda")
        else:
            S = torch.zeros(1, dtype=torch.int32, device="cuda")
        dist.broadcast(S, src=self.target_rank)
        if self.rank != self.target_rank:
            T = torch.zeros(S.item(), dtype=torch.uint8, device="cuda")
        dist.broadcast(T, src=self.target_rank)
        if self.rank != self.target_rank:
            val = T.cpu().numpy().tobytes()
            data = json.loads(val.decode("utf-8"))
        return data

    def stop(self):
        self.model_runner = None
        if self.chat_adapter is not None:
            self.chat_adapter.stop()
            self.chat_adapter = None
            logger.warning("OmniVAReader stopped")

    def __del__(self):
        self.stop()


class SekoAROmniVAReader(OmniVAReader):
    def __init__(
        self,
        rank: int,
        world_size: int,
        stream_url: str,
        latent_per_chunk: int = 3,
        audio_window_secs: float = 1.0,
        look_ahead_secs: float = 0.0,
        video_fps: float = 16,
        sample_rate: int = 16000,
        audio_channels: int = 1,
        target_rank: int = 0,
        model_runner=None,
        huoshan_tts_voice_type=None,
        stream_config: dict = {},
        **kwargs,
    ):
        self.rank = rank
        self.world_size = world_size
        self.stream_url = stream_url
        self.sample_rate = sample_rate

        self.audio_channels = audio_channels
        self.latent_per_chunk = latent_per_chunk
        self.audio_window_secs = audio_window_secs
        self.look_ahead_secs = look_ahead_secs
        self.video_fps = video_fps

        self.target_rank = target_rank % self.world_size
        self.origin_flag_tensor = torch.tensor([0], dtype=torch.int32).to(device="cuda")
        self.latent_flag_tensor = torch.tensor([0], dtype=torch.int32).to(device="cuda")
        self.valid_duration_tensor = torch.tensor([0], dtype=torch.float32).to(device="cuda")
        self.immediate_switch_tensor = torch.tensor([0], dtype=torch.int32).to(device="cuda")

        self.chat_adapter = None
        self.model_runner = model_runner
        self.huoshan_tts_voice_type = huoshan_tts_voice_type
        self.stream_config = stream_config

        assert self.audio_channels == 1, "Only mono audio is supported for OmniVAReader"
        self.initAR()
        logger.info(f"VAReader initialized for stream: {stream_url} target_rank: {self.target_rank}")

    def initAR(self):
        # ref: https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/input_encoders/hf/seko_audio/audio_adapter.py#L267-L280
        # 假定 frame_per_latent = 4（每个latent包含4个frame）
        # 左边界帧索引：[0, 1, 5,  9, 13, 17, ...]
        # 右边界帧索引：[1, 5, 9, 13, 17, 21, ...]
        # 中心时间戳（秒）：[(0+1)/2/fps, (1+5)/2/fps, (5+9)/2/fps, ...]，除第一个外，其余为(latent_idx*4-1)/fps
        # 上边界时间戳（秒）:[1/fps, 5/fps, 9/fps, ...]+look_ahead_secs，即(latent_idx*4+1)/fps + look_ahead_secs
        # 每一个 latent 对应的音频时间范围（假定每个chunk包含3个latent，即每3个latent取一次音频）：
        #    [0.5/fps - audio_window_secs/2, 0.5/fps + audio_window_secs/2] 上界应小于 1/fps + look_ahead_secs
        #    [3/fps - audio_window_secs/2, 3/fps + audio_window_secs/2]     上界应小于 5/fps + look_ahead_secs
        # 取 [7/fps - audio_window_secs/2, 7/fps + audio_window_secs/2]     上界应小于 9/fps + look_ahead_secs
        #    [11/fps - audio_window_secs/2, 11/fps + audio_window_secs/2]   上界应小于 13/fps + look_ahead_secs
        #    [15/fps - audio_window_secs/2, 15/fps + audio_window_secs/2]   上界应小于 17/fps + look_ahead_secs
        # 取 [19/fps - audio_window_secs/2, 19/fps + audio_window_secs/2]   上界应小于 21/fps + look_ahead_secs
        #    [23/fps - audio_window_secs/2, 23/fps + audio_window_secs/2]   上界应小于 25/fps + look_ahead_secs
        #    ...
        # 每次取音频应当取 chunk 对应所有 latents 的音频块范围上界的最大值，每次按chunk取的音频段为：
        #    [0, 9/fps + look_ahead_secs]
        #    [9/fps + look_ahead_secs, 21/fps + look_ahead_secs]
        #    [21/fps + look_ahead_secs, 33/fps + look_ahead_secs]
        #    ...
        # 需要根据这个恢复每个chunk内各个latent对应的音频段，用于audio encode 计算
        # 这些音频对应的输出音频段应当按照中心坐标对齐：
        #    [-3/fps, 9/fps],
        #    [9/fps, 21/fps],
        #    [21/fps, 33/fps],
        #    ...
        #  [后面-12/fps, (latent_per_chunk - 1 ) * 4 + 1  + chunk_idx * letent_per_chunk * 4]

        # 第一次取 9/fps + look_ahead_secs 秒的音频
        self.segment_duration = ((self.latent_per_chunk - 1) * 4 + 1) / self.video_fps + self.look_ahead_secs
        # 后续每次取 12/fps 秒的音频
        self.other_fetch_duration = self.latent_per_chunk * 4 / self.video_fps
        self.chunk_idx = 0
        self.last_chunk_audios = None

        # 当前chunk的原音频，每个chunk合成的视频长度固定，对应音频长度为: 12/fps
        self.origin_audio_tensor = torch.zeros(int(self.sample_rate * self.other_fetch_duration) * 2, dtype=torch.uint8, device="cuda")
        # 当前chunk内各个latent对应的音频段，用于audio encode 计算
        self.latent_audio_tensor = torch.zeros(self.latent_per_chunk * int(self.sample_rate * self.audio_window_secs) * 2, dtype=torch.uint8, device="cuda")

    # 获取当前latent对应的边界索引帧序号
    def get_latent_idxs(self, i, base_idx):
        latent_idx = self.chunk_idx * self.latent_per_chunk + i
        if latent_idx == 0:
            center_secs = 0.5 / self.video_fps
        else:
            center_secs = (latent_idx * 4 - 1) / self.video_fps
        # 当前 latent 音频窗口上界索引
        right_idx = int((center_secs + self.audio_window_secs / 2) * self.sample_rate)
        # 当前 latent 音频窗口下界索引
        left_idx = right_idx - int(self.sample_rate * self.audio_window_secs)
        # 当前 latent 音频段上界索引
        look_ahead_secs = (latent_idx * 4 + 1) / self.video_fps + self.look_ahead_secs
        look_ahead_idx = int(look_ahead_secs * self.sample_rate)
        logger.debug(f"latent_idx: {latent_idx} center_secs: {center_secs} right_idx: {right_idx} left_idx: {left_idx} look_ahead_secs: {look_ahead_secs} look_ahead_idx: {look_ahead_idx}")

        # 对齐当前缓存的所有音频的起始位置（只缓存了最多两个chunk的音频）
        left_idx -= base_idx
        right_idx -= base_idx
        look_ahead_idx -= base_idx
        logger.debug(f"adjusted idxs: {left_idx} {right_idx} {look_ahead_idx}")
        return left_idx, right_idx, look_ahead_idx

    # 获取当前chunk对应的原始音频的开始和结束索引帧
    def get_chunk_origin_idxs(self, base_idx):
        # end secs: 9/fps + chunk_idx * 12/fps
        end_secs = ((self.latent_per_chunk - 1) * 4 + 1 + self.chunk_idx * self.latent_per_chunk * 4) / self.video_fps
        start_secs = end_secs - (self.latent_per_chunk * 4) / self.video_fps
        start_idx = int(start_secs * self.sample_rate)
        end_idx = int(end_secs * self.sample_rate)
        logger.debug(f"chunk_idx: {self.chunk_idx} start_idx: {start_idx} end_idx: {end_idx}")
        # 对齐当前缓存的所有音频的起始位置（只缓存了最多两个chunk的音频）
        start_idx -= base_idx
        end_idx -= base_idx
        logger.debug(f"adjusted idxs: {start_idx} {end_idx}")
        return start_idx, end_idx

    def prepare_ar_audios(self, merged_audio, origin_audios, latent_audios):
        # 当前chunk音频的上一个chunk音频的开始时间，用于对齐音频初始位置
        last_chunk_start_secs = 0
        if self.chunk_idx > 1:
            last_chunk_start_secs = self.segment_duration + (self.chunk_idx - 2) * self.other_fetch_duration
        base_idx = int(last_chunk_start_secs * self.sample_rate)
        audio_length = len(merged_audio)
        logger.debug(
            "chunk idx: ", self.chunk_idx, "start_secs: ", last_chunk_start_secs, "end_secs: ", last_chunk_start_secs + self.segment_duration, "base_idx: ", base_idx, "audio_length: ", audio_length
        )

        # 构造原始音频，用于合成最终视频
        start_idx, end_idx = self.get_chunk_origin_idxs(base_idx)
        pad_idx = max(-start_idx, 0)
        real_start = max(start_idx, 0)
        real_len = min(end_idx, audio_length) - real_start
        logger.debug(f"----- center audios pad_idx: {pad_idx} real_start: {real_start} real_len: {real_len} -----")
        origin_audios[pad_idx : pad_idx + real_len] = merged_audio[real_start : real_start + real_len]

        # 构造 latent 音频，用于 audio encode
        for i in range(self.latent_per_chunk):
            left_idx, right_idx, look_ahead_idx = self.get_latent_idxs(i, base_idx)
            pad_idx = max(-left_idx, 0)
            real_start = max(left_idx, 0)
            real_len = min(look_ahead_idx, right_idx, audio_length) - real_start
            logger.debug(f"----- {i} pad_idx: {pad_idx} real_start: {real_start} real_len: {real_len} -----")
            latent_audios[i, pad_idx : pad_idx + real_len] = merged_audio[real_start : real_start + real_len]

    def prepare_audio_data(self, chat_audio_result):
        sample_count = 0
        audio = np.array([], dtype=np.int16)
        origin_audios = np.zeros(int(self.sample_rate * self.other_fetch_duration), dtype=np.int16)
        latent_audios = np.zeros((self.latent_per_chunk, int(self.sample_rate * self.audio_window_secs)), dtype=np.int16)

        # convert chat audio result to mono and target sample rate
        if chat_audio_result is not None:
            audio_data, audio_info = chat_audio_result
            audio, sample_count = self.convert_pcm_s16le_to_mono_resampled(audio_data, audio_info)
        valid_duration = sample_count / self.sample_rate
        if sample_count <= 0:
            return origin_audios.tobytes(), latent_audios.tobytes(), valid_duration

        if self.chunk_idx == 0:
            expect_count = int(self.segment_duration * self.sample_rate)
        else:
            expect_count = int(self.other_fetch_duration * self.sample_rate)
        assert sample_count <= expect_count, f"audio length {sample_count} > expect_count {expect_count}"

        # pad 0 to the audio to make it the same length as expect_count
        if sample_count < expect_count:
            pad_count = expect_count - sample_count
            logger.debug(f"pad {pad_count} samples to audio")
            audio = np.pad(audio, (0, pad_count), mode="constant", constant_values=0)

        # if is not the first segment, concat with previous segment
        if self.last_chunk_audios is not None:
            logger.debug(f"concat {self.last_chunk_audios.shape} with {audio.shape}")
            merged_audio = np.concatenate([self.last_chunk_audios, audio])
        else:
            merged_audio = audio
        self.prepare_ar_audios(merged_audio, origin_audios, latent_audios)
        logger.debug(
            f"chunk[{self.chunk_idx}] origin_audios: {origin_audios.shape} {origin_audios.dtype} {origin_audios.min()} {origin_audios.max()} latent_audios: {latent_audios.shape} {latent_audios.dtype} {latent_audios.min()} {latent_audios.max()} {sample_count}, valid_duration: {valid_duration}"
        )

        # update prev seg chunk
        self.last_chunk_audios = audio
        self.chunk_idx += 1
        return origin_audios.tobytes(), latent_audios.tobytes(), valid_duration

    def get_fetch_duration(self):
        fetch_duration = self.segment_duration
        # after immediate switch, reset prev seg chunk
        if self.chat_adapter is not None and self.chat_adapter.reset_prev:
            self.chat_adapter.reset_prev = False
            self.chunk_idx = 0
            self.last_chunk_audios = None
            logger.warning(f"Reset prev seg chunk")
        # first segment, fetch segment_duration, else fetch self.other_fetch_duration
        if self.chunk_idx > 0 and self.last_chunk_audios is not None:
            fetch_duration = self.other_fetch_duration
        return fetch_duration

    def get_audio_segment(self):
        origin_audio = None
        latent_audio = None
        valid_duration = 0
        try:
            if self.rank == self.target_rank:
                fetch_duration = self.get_fetch_duration()
                # logger.info(f"Get segment, fetch_duration: {fetch_duration}")
                if self.chat_adapter.status == "voice":
                    audio_result = self.chat_adapter.get_audio(fetch_duration)
                    origin_audio, latent_audio, valid_duration = self.prepare_audio_data(audio_result)
                    # think all voice segments inferred, naturally switch to blank
                    if audio_result is None:
                        logger.info(f"Think all voice segments inferred, naturally switch to blank")
                        self.chat_adapter.status = "blank"
                        self.chunk_idx = 0
                        self.last_chunk_audios = None
                else:
                    origin_audio, latent_audio, valid_duration = self.prepare_audio_data(None)

            if self.world_size > 1:
                origin_audio = self.braodcast_audio_data(origin_audio, self.origin_audio_tensor, self.origin_flag_tensor)
                latent_audio = self.braodcast_audio_data(latent_audio, self.latent_audio_tensor, self.latent_flag_tensor)
                valid_duration = self.braodcast_valid_duration(valid_duration)

            origin_audio = self.bytes_to_ndarray(origin_audio)
            latent_audio = self.bytes_to_ndarray(latent_audio)
            if latent_audio is not None:
                # latent_audio: (latent_per_chunk, audio_window * sr)
                latent_audio = latent_audio.reshape(self.latent_per_chunk, -1)
            return origin_audio, latent_audio, valid_duration

        except Exception:
            logger.warning(f"Failed to get voice segment: {traceback.format_exc()}")
            return None, None, 0

    def braodcast_audio_data(self, audio_data, audio_tensor, flag_tensor):
        if self.rank == self.target_rank:
            if audio_data is None:
                flag_tensor.fill_(0)
            else:
                flag_tensor.fill_(1)
                audio_tensor.copy_(torch.frombuffer(bytearray(audio_data), dtype=torch.uint8))
                # logger.info(f"rank {self.rank} send audio_tensor: {self.audio_tensor.shape}")

        dist.broadcast(flag_tensor, src=self.target_rank)
        if flag_tensor.item() == 0:
            return None

        dist.broadcast(audio_tensor, src=self.target_rank)
        if self.rank != self.target_rank:
            # logger.info(f"rank {self.rank} recv audio_tensor: {self.audio_tensor.shape}")
            audio_data = audio_tensor.cpu().numpy().tobytes()
        return audio_data


if __name__ == "__main__":
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 0))
    if WORLD_SIZE > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        logger.info(f"Distributed initialized: rank={RANK}, world_size={WORLD_SIZE}")

    reader = OmniVAReader(
        RANK,
        WORLD_SIZE,
        "https://reverse.st-oc-01.chielo.org/10.5.64.49:8000/rtc/v1/whep/?app=publish&stream=test_stream_ll&eip=10.120.114.82:8000",
        segment_duration=17 / 16,
        sample_rate=16000,
        audio_channels=1,
        prev_duration=1 / 16,
    )
    reader.start()
    fail_count = 0
    max_fail_count = 100000000

    try:
        while True:
            audio_data = reader.get_audio_segment(timeout=1)
            if audio_data is not None:
                logger.info(f"Got audio chunk, shape: {audio_data.shape}, range: [{audio_data.min()}, {audio_data.max()}]")
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
