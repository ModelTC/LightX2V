# -*- coding: utf-8 -*-
"""
Audio Source Separation Module
Separates different voice tracks in audio, supports multi-person audio separation
"""

import base64
import io
import os
import tempfile
import traceback
from collections import defaultdict
from typing import Dict, Optional, Union

import torch
import torch.serialization
import torchaudio
from loguru import logger

# Import pyannote.audio for speaker diarization
from pyannote.audio import Audio, Pipeline

# Fix for PyTorch 2.6 compatibility: allow pyannote.audio classes in torch.load
# PyTorch 2.6 changed torch.load default to weights_only=True for security
try:
    # Add safe globals for pyannote.audio classes
    # This allows torch.load to work with pyannote.audio model files
    from pyannote.audio.core.task import Specifications

    torch.serialization.add_safe_globals([Specifications])
except (ImportError, AttributeError) as e:
    # If pyannote.audio is not installed or class doesn't exist, log warning
    # The actual error will be handled when Pipeline.from_pretrained is called
    logger.debug(f"Could not add pyannote.audio safe globals (may need to use weights_only=False): {e}")

_origin_torch_load = torch.load


def our_torch_load(checkpoint_file, *args, **kwargs):
    kwargs["weights_only"] = False
    return _origin_torch_load(checkpoint_file, *args, **kwargs)


class AudioSeparator:
    """
    Audio separator for separating different voice tracks in audio using pyannote.audio
    Supports multi-person conversation separation, maintains duration (other speakers' tracks are empty)
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        sample_rate: int = 16000,
    ):
        """
        Initialize audio separator

        Args:
            model_path: Model path (if using custom model), default uses pyannote/speaker-diarization-community-1
            device: Device ('cpu', 'cuda', etc.), None for auto selection
            sample_rate: Target sample rate, default 16000
        """
        self.sample_rate = sample_rate
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_pyannote(model_path)

    def _init_pyannote(self, model_path: str = None):
        """Initialize pyannote.audio pipeline"""
        try:
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
            model_name = model_path or "pyannote/speaker-diarization-community-1"

            # Fix for PyTorch 2.6: use safe_globals context manager to allow pyannote.audio classes
            # PyTorch 2.6 changed torch.load default to weights_only=True
            try:
                from pyannote.audio.core.task import Specifications

                safe_globals_context = torch.serialization.safe_globals([Specifications])
            except (ImportError, AttributeError):
                # If Specifications class is not available, use empty context
                safe_globals_context = torch.serialization.safe_globals([])

            try:
                torch.load = our_torch_load
                # Try loading with token if available
                if huggingface_token:
                    self.pipeline = Pipeline.from_pretrained(model_name, token=huggingface_token)
                else:
                    # Try without token (may work for public models)
                    self.pipeline = Pipeline.from_pretrained(model_name)
            except Exception as e:
                if "gated" in str(e).lower() or "token" in str(e).lower():
                    raise RuntimeError(f"Model requires authentication. Set HUGGINGFACE_TOKEN or HF_TOKEN environment variable: {e}")
                # If safe_globals didn't work, try with weights_only=False as fallback
                if "weights_only" in str(e).lower() or "Unsupported global" in str(e):
                    logger.warning(f"PyTorch 2.6 compatibility issue detected, attempting fallback: {e}")
                    # Note: We can't directly control weights_only in Pipeline.from_pretrained,
                    # but the safe_globals should have worked. If not, the error will be raised.
                raise RuntimeError(f"Failed to load pyannote model: {e}")
            finally:
                torch.load = _origin_torch_load

            # Move pipeline to specified device
            if self.device:
                self.pipeline.to(torch.device(self.device))

            # Initialize Audio helper for waveform loading
            self.pyannote_audio = Audio()

            logger.info("Initialized pyannote.audio speaker diarization pipeline")
        except Exception as e:
            logger.error(f"Failed to initialize pyannote: {e}")
            raise RuntimeError(f"Failed to initialize pyannote.audio pipeline: {e}")

    def separate_speakers(
        self,
        audio_path: Union[str, bytes],
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 5,
    ) -> Dict:
        """
        Separate different speakers in audio

        Args:
            audio_path: Audio file path or bytes data
            num_speakers: Specified number of speakers, None for auto detection
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            Dict containing:
                - speakers: List of speaker audio segments, each containing:
                    - speaker_id: Speaker ID (0, 1, 2, ...)
                    - audio: torch.Tensor audio data [channels, samples]
                    - segments: List of (start_time, end_time) tuples
                    - sample_rate: Sample rate
        """
        try:
            # Load audio
            if isinstance(audio_path, bytes):
                # 尝试从字节数据推断音频格式
                # 检查是否是 WAV 格式（RIFF 头）
                is_wav = audio_path[:4] == b"RIFF" and audio_path[8:12] == b"WAVE"
                # 检查是否是 MP3 格式（ID3 或 MPEG 头）
                is_mp3 = audio_path[:3] == b"ID3" or audio_path[:2] == b"\xff\xfb" or audio_path[:2] == b"\xff\xf3"

                # 根据格式选择后缀
                if is_wav:
                    suffix = ".wav"
                elif is_mp3:
                    suffix = ".mp3"
                else:
                    # 默认尝试 WAV，如果失败会抛出错误
                    suffix = ".wav"

                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                    tmp_file.write(audio_path)
                    tmp_audio_path = tmp_file.name
                try:
                    result = self._separate_speakers_internal(tmp_audio_path, num_speakers, min_speakers, max_speakers)
                finally:
                    # 确保临时文件被删除
                    try:
                        os.unlink(tmp_audio_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {tmp_audio_path}: {e}")
                return result
            else:
                return self._separate_speakers_internal(audio_path, num_speakers, min_speakers, max_speakers)

        except Exception as e:
            logger.error(f"Speaker separation failed: {traceback.format_exc()}")
            raise RuntimeError(f"Audio separation error: {e}")

    def _separate_speakers_internal(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 5,
    ) -> Dict:
        """Internal method: execute speaker separation"""

        # Load audio
        waveform, original_sr = torchaudio.load(audio_path)
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(original_sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Ensure waveform is float32 and normalized (pyannote expects this format)
        if waveform.dtype != torch.float32:
            waveform = waveform.float()

        # Ensure waveform is in range [-1, 1] (normalize if needed)
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()

        if self.pipeline is None:
            raise RuntimeError("Pyannote pipeline not initialized")

        return self._separate_with_pyannote(audio_path, waveform, num_speakers, min_speakers, max_speakers)

    def _separate_with_pyannote(
        self,
        audio_path: str,
        waveform: torch.Tensor,
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int,
    ) -> Dict:
        """Use pyannote.audio for speaker diarization"""
        try:
            # Use waveform dict to avoid AudioDecoder dependency issues
            # Pipeline can accept either file path or waveform dict
            # Using waveform dict is more reliable when torchcodec is not properly installed
            audio_input = {
                "waveform": waveform,
                "sample_rate": self.sample_rate,
            }

            # Run speaker diarization
            output = self.pipeline(
                audio_input,
                min_speakers=min_speakers if num_speakers is None else num_speakers,
                max_speakers=max_speakers if num_speakers is None else num_speakers,
            )

            # Extract audio segments for each speaker
            speakers_dict = defaultdict(list)
            for turn, speaker in output.speaker_diarization:
                print(f"Speaker: {speaker}, Start time: {turn.start}, End time: {turn.end}")
                start_time = turn.start
                end_time = turn.end
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)

                # Extract audio segment for this time period
                segment_audio = waveform[:, start_sample:end_sample]
                speakers_dict[speaker].append((start_time, end_time, segment_audio))

            # Generate complete audio for each speaker (other speakers' segments are empty)
            speakers = []
            audio_duration = waveform.shape[1] / self.sample_rate
            num_samples = waveform.shape[1]

            for speaker_id, segments in speakers_dict.items():
                # Create zero-filled audio
                speaker_audio = torch.zeros_like(waveform)

                # Fill in this speaker's segments
                for start_time, end_time, segment_audio in segments:
                    start_sample = int(start_time * self.sample_rate)
                    end_sample = int(end_time * self.sample_rate)
                    # Ensure no out-of-bounds
                    end_sample = min(end_sample, num_samples)
                    segment_len = end_sample - start_sample
                    if segment_len > 0 and segment_audio.shape[1] > 0:
                        actual_len = min(segment_len, segment_audio.shape[1])
                        speaker_audio[:, start_sample : start_sample + actual_len] = segment_audio[:, :actual_len]

                speakers.append(
                    {
                        "speaker_id": speaker_id,
                        "audio": speaker_audio,
                        "segments": [(s[0], s[1]) for s in segments],
                        "sample_rate": self.sample_rate,
                    }
                )

            logger.info(f"Separated audio into {len(speakers)} speakers using pyannote")
            return {"speakers": speakers, "method": "pyannote"}

        except Exception as e:
            logger.error(f"Pyannote separation failed: {e}")
            raise RuntimeError(f"Audio separation failed: {e}")

    def save_speaker_audio(self, speaker_audio: torch.Tensor, output_path: str, sample_rate: int = None):
        """
        Save speaker audio to file

        Args:
            speaker_audio: Audio tensor [channels, samples]
            output_path: Output path
            sample_rate: Sample rate, if None uses self.sample_rate
        """
        sr = sample_rate if sample_rate else self.sample_rate
        torchaudio.save(output_path, speaker_audio, sr)
        logger.info(f"Saved speaker audio to {output_path}")

    def speaker_audio_to_base64(self, speaker_audio: torch.Tensor, sample_rate: int = None, format: str = "wav") -> str:
        """
        Convert speaker audio tensor to base64 encoded string without saving to file

        Args:
            speaker_audio: Audio tensor [channels, samples]
            sample_rate: Sample rate, if None uses self.sample_rate
            format: Audio format (default: "wav")

        Returns:
            Base64 encoded audio string
        """
        sr = sample_rate if sample_rate else self.sample_rate

        # Use BytesIO to save audio to memory
        buffer = io.BytesIO()
        torchaudio.save(buffer, speaker_audio, sr, format=format)

        # Get the audio bytes
        audio_bytes = buffer.getvalue()

        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        logger.debug(f"Converted speaker audio to base64, size: {len(audio_bytes)} bytes")
        return audio_base64

    def separate_and_save(
        self,
        audio_path: Union[str, bytes],
        output_dir: str,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 5,
    ) -> Dict:
        """
        Separate audio and save to files

        Args:
            audio_path: Input audio path or bytes data
            output_dir: Output directory
            num_speakers: Specified number of speakers
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            Separation result dictionary, containing output file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        result = self.separate_speakers(audio_path, num_speakers, min_speakers, max_speakers)

        output_paths = []
        for speaker in result["speakers"]:
            speaker_id = speaker["speaker_id"]
            output_path = os.path.join(output_dir, f"{speaker_id}.wav")
            self.save_speaker_audio(speaker["audio"], output_path, speaker["sample_rate"])
            output_paths.append(output_path)
            speaker["output_path"] = output_path

        result["output_paths"] = output_paths
        return result


def separate_audio_tracks(
    audio_path: str,
    output_dir: str = None,
    num_speakers: int = None,
    model_path: str = None,
) -> Dict:
    """
    Convenience function: separate different audio tracks

    Args:
        audio_path: Audio file path
        output_dir: Output directory, if None does not save files
        num_speakers: Number of speakers
        model_path: Model path (optional)

    Returns:
        Separation result dictionary
    """
    separator = AudioSeparator(model_path=model_path)

    if output_dir:
        return separator.separate_and_save(audio_path, output_dir, num_speakers=num_speakers)
    else:
        return separator.separate_speakers(audio_path, num_speakers=num_speakers)


if __name__ == "__main__":
    # Test code
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_separator.py <audio_path> [output_dir] [num_speakers]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./separated_audio"
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    separator = AudioSeparator()
    result = separator.separate_and_save(audio_path, output_dir, num_speakers=num_speakers)

    print(f"Separated audio into {len(result['speakers'])} speakers:")
    for speaker in result["speakers"]:
        print(f"  Speaker {speaker['speaker_id']}: {len(speaker['segments'])} segments")
        if "output_path" in speaker:
            print(f"    Saved to: {speaker['output_path']}")
