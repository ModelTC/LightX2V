import threading
import traceback
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.minimax_h3.packing import AUDIO_LATENTS_PER_SECOND, causal_chunk_audio_range
from lightx2v.utils.va_reader_omni import AudioInfo, ByteBuffer, SekoAROmniVAReader


def parse_pcm_audio_format(audio_format: str) -> AudioInfo:
    """Parse strings like pcm_s16le/16000/1 into AudioInfo."""
    fmt = (audio_format or "pcm_s16le/16000/1").strip()
    parts = fmt.split("/")
    sample_fmt = parts[0] if parts else "pcm_s16le"
    sample_rate = int(parts[1]) if len(parts) > 1 and parts[1] else 16000
    channel_count = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    if sample_fmt != "pcm_s16le":
        raise ValueError(f"unsupported audio_format {audio_format}, expected pcm_s16le/<sr>/<ch>")
    if channel_count != 1:
        raise ValueError(f"only mono audio is supported, got channels={channel_count}")
    return AudioInfo(
        {
            "sample_count": 0,
            "sample_rate": sample_rate,
            "channel_count": channel_count,
            "sample_fmt": sample_fmt,
            "pts": 0,
        }
    )


class WsAudioSource:
    """Duck-typed ChatAdapter used by SekoAROmniVAReader, fed by websocket messages.

    Message mapping:
      audio_start (Empty) -> AgentStartPlay
      AudioChunk          -> AgentAudio
      audio_end (Empty)   -> AgentEndPlay
      Interrupt           -> ClearAgentAudio
    """

    def __init__(self, seg_duration: float, sample_rate: int = 16000, audio_format: str = "pcm_s16le/16000/1", model_runner=None):
        self.seg_duration = seg_duration
        self.model_runner = model_runner
        self.audio_buffer = ByteBuffer()
        self.audio_info = parse_pcm_audio_format(audio_format)
        if self.audio_info.sample_rate != sample_rate:
            logger.warning(f"WsAudioSource sample_rate {self.audio_info.sample_rate} != reader {sample_rate}")
        self.status = "blank"
        self.reset_prev = False
        self.immediate_switch = 0
        self.image_switch = ""
        self.action_switch = ""
        self.skip_immediate_switch = False
        self.started_event = threading.Event()
        self._lock = threading.Lock()

    def mark_started(self):
        self.started_event.set()

    def immediate_switch_to(self, status):
        logger.warning(f"VA reader immediate switch to {status}")
        self.status = status
        if self.skip_immediate_switch:
            logger.warning(f"VA reader skip immediate switch to {status}")
            return
        self.immediate_switch = 1
        self.reset_prev = True
        if self.model_runner is not None and self.model_runner.can_pause:
            self.model_runner.pause_signal = True
            logger.warning("Model runner pause signal set to True")

    def on_audio_start(self, note: str = ""):
        logger.debug(f"WsAudioSource AudioStart note={note}")
        with self._lock:
            self.audio_buffer = ByteBuffer()

    def on_audio_chunk(self, pcm_data: bytes, sequence: int = 0):
        if not pcm_data:
            return
        with self._lock:
            if self.audio_buffer is None:
                self.audio_buffer = ByteBuffer()
            self.audio_buffer.add(pcm_data)
            if self.status == "blank" and self.has_voice(self.seg_duration):
                self.immediate_switch_to("voice")

    def on_audio_end(self, note: str = ""):
        logger.debug(f"WsAudioSource AudioEnd note={note}")
        with self._lock:
            if self.audio_buffer is not None:
                self.audio_buffer.mark_finished()

    def on_clear_audio(self, note: str = ""):
        logger.warning(f"WsAudioSource Interrupt/ClearAudio note={note}")
        with self._lock:
            self.audio_buffer = None
            if self.status == "voice":
                self.status = "blank"

    def on_action_switch(self, prompt: str, duration: float = 0):
        prompt = str(prompt).strip()
        if not prompt:
            return
        if duration != -1 and duration <= 0:
            logger.warning(f"WsAudioSource ignore action prompt, duration must be -1 or > 0, got {duration}")
            return
        logger.debug(f"WsAudioSource action prompt change: {prompt} duration={duration}")
        with self._lock:
            self.action_switch = {"prompt": prompt, "duration": duration}

    def has_voice(self, duration) -> bool:
        if self.audio_info is None or self.audio_buffer is None or self.audio_buffer.current_size == 0:
            return False
        bytes_count = round(duration * self.audio_info.sample_rate) * self.audio_info.channel_count * 2
        if self.audio_buffer.current_size < bytes_count and self.audio_buffer.has_more_voice():
            logger.warning(f"Not enough bytes and maybe has more voice, content_size: {self.audio_buffer.current_size}, bytes_count: {bytes_count}")
            return False
        return bytes_count

    def get_audio(self, fetch_duration):
        with self._lock:
            bytes_count = self.has_voice(fetch_duration)
            if bytes_count is False or self.audio_info is None or self.audio_buffer is None:
                return None
            pcm_data = self.audio_buffer.get(bytes_count)
            sample_count = len(pcm_data) // (self.audio_info.channel_count * 2)
            audio_info = deepcopy(self.audio_info)
            audio_info.sample_count = sample_count
            return (pcm_data, audio_info)

    def stop(self):
        self.model_runner = None
        with self._lock:
            self.audio_buffer = None


class SekoARWSReader(SekoAROmniVAReader):
    """SekoAROmniVAReader that uses WsAudioSource instead of seko-chatter."""

    def __init__(self, *args, audio_source=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._preset_audio_source = audio_source

    def start(self):
        if self.rank == self.target_rank:
            if self._preset_audio_source is not None:
                self.chat_adapter = self._preset_audio_source
            else:
                self.chat_adapter = WsAudioSource(
                    seg_duration=self.segment_duration,
                    sample_rate=self.sample_rate,
                    model_runner=self.model_runner,
                )
            self.chat_adapter.model_runner = self.model_runner
            self.chat_adapter.seg_duration = self.segment_duration
            self.chat_adapter.mark_started()
            logger.info(f"SekoARWSReader {self.rank}/{self.world_size} started without omni chatter")
        else:
            logger.info(f"SekoARWSReader {self.rank}/{self.world_size} wait only")
        if self.world_size > 1:
            logger.info(f"SekoARWSReader {self.rank}/{self.world_size} wait barrier")
            dist.barrier()
            logger.info(f"SekoARWSReader {self.rank}/{self.world_size} end barrier")

    def stop(self):
        self.model_runner = None
        if self.chat_adapter is not None:
            try:
                self.chat_adapter.stop()
            except Exception as e:
                logger.warning(f"Error stopping WsAudioSource: {e}")
            self.chat_adapter = None
            logger.warning("SekoARWSReader stopped")


class MinimaxH3CausalWSReader(SekoARWSReader):
    """Websocket audio reader that returns one MiniMax-H3 waveform per call"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"MinimaxH3CausalWSReader sample_rate={self.sample_rate} segment_duration={self.segment_duration}s samples={self.all_seg_sample_count}")

    def start(self):
        if self.rank == self.target_rank:
            if self._preset_audio_source is not None:
                self.chat_adapter = self._preset_audio_source
            else:
                self.chat_adapter = WsAudioSource(
                    seg_duration=self.segment_duration,
                    sample_rate=self.sample_rate,
                    model_runner=self.model_runner,
                )
            self.chat_adapter.model_runner = self.model_runner
            self.chat_adapter.seg_duration = self.segment_duration
            # 不提前推理，每推完一段，等待一段时间后再拿音频推下一段
            self.chat_adapter.skip_immediate_switch = True
            self.chat_adapter.mark_started()
            logger.info(f"MinimaxH3CausalWSReader {self.rank}/{self.world_size} started without omni chatter")
        else:
            logger.info(f"MinimaxH3CausalWSReader {self.rank}/{self.world_size} wait only")
        if self.world_size > 1:
            logger.info(f"MinimaxH3CausalWSReader {self.rank}/{self.world_size} wait barrier")
            dist.barrier()
            logger.info(f"MinimaxH3CausalWSReader {self.rank}/{self.world_size} end barrier")

    def initAR(self):
        audio_start, audio_end = causal_chunk_audio_range(0)
        self.segment_duration = (audio_end - audio_start) / AUDIO_LATENTS_PER_SECOND
        samples = int(round(self.segment_duration * self.sample_rate))
        self.all_seg_sample_count = samples
        self.origin_audio_tensor = torch.zeros(samples * 2, dtype=torch.uint8, device="cuda")

    def prepare_audio_data(self, chat_audio_result):
        """Resample one fetch and pad or crop it to ``all_seg_sample_count``."""
        audio = np.array([], dtype=np.int16)
        sample_count = 0
        if chat_audio_result is not None:
            audio_data, audio_info = chat_audio_result
            audio, sample_count = self.convert_pcm_s16le_to_mono_resampled(audio_data, audio_info)
        if sample_count < self.all_seg_sample_count:
            audio = np.pad(audio, (0, self.all_seg_sample_count - sample_count))
        elif sample_count > self.all_seg_sample_count:
            audio = audio[: self.all_seg_sample_count]
            sample_count = self.all_seg_sample_count
        valid_duration = sample_count / self.sample_rate
        logger.debug(f"origin_audios: {audio.shape} {audio.dtype} {audio.min()} {audio.max()} {sample_count}, valid_duration: {valid_duration}")
        return audio.tobytes(), valid_duration

    def get_audio_segment(self, segment_duration):
        """Pull ``segment_duration`` seconds and return mono float32 audio."""
        samples = int(round(float(segment_duration) * self.sample_rate))
        if samples <= 0:
            raise ValueError(f"samples must be positive, got {samples}")
        if samples != self.all_seg_sample_count:
            if self.rank == self.target_rank:
                logger.debug(f"MinimaxH3CausalWSReader reset audio samples {self.all_seg_sample_count} -> {samples}")
            self.origin_audio_tensor = torch.zeros(samples * 2, dtype=torch.uint8, device="cuda")
            self.all_seg_sample_count = samples

        origin_audio = None
        valid_duration = 0
        try:
            if self.rank == self.target_rank and self.chat_adapter is not None:
                if self.chat_adapter.reset_prev:
                    self.chat_adapter.reset_prev = False

                if self.chat_adapter.status == "voice":
                    audio_result = self.chat_adapter.get_audio(segment_duration)
                    origin_audio, valid_duration = self.prepare_audio_data(audio_result)
                    # think all voice segments inferred, naturally switch to blank
                    if audio_result is None:
                        logger.info(f"Think all voice segments inferred, naturally switch to blank")
                        self.chat_adapter.status = "blank"
                else:
                    origin_audio, valid_duration = self.prepare_audio_data(None)

            if self.world_size > 1:
                origin_audio = self.braodcast_audio_data(origin_audio, self.origin_audio_tensor, self.origin_flag_tensor)
                valid_duration = self.braodcast_valid_duration(valid_duration)

            origin_audio = self.bytes_to_ndarray(origin_audio)
            return origin_audio, valid_duration

        except Exception:
            logger.warning(f"Failed to get voice segment: {traceback.format_exc()}")
            return None, 0
