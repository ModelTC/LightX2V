"""Native HuBERT encoder and TBDub's audio/video time-window contract."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from lightx2v_platform.base.global_var import AI_DEVICE

from .infer import HubertInfer
from .weights import load_hubert_weights


def pad_with_fadeout_silence(audio, target_length, sample_rate=16000, fade_ms=10.0):
    """Append silence with the upstream fade; never change real audio samples."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) >= target_length:
        return audio
    padding = np.zeros(target_length - len(audio), dtype=np.float32)
    fade_samples = min(len(padding), max(0, round(fade_ms / 1000 * sample_rate)))
    if fade_samples and len(audio):
        padding[:fade_samples] = np.linspace(float(audio[-1]), 0.0, fade_samples + 1, dtype=np.float32)[1:]
    return np.concatenate((audio, padding))


def load_audio(path, sample_rate=16000):
    """Match TBDub's mono PCM16 ffmpeg decoding, including .npy inputs."""
    path = os.fspath(path)
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False).astype(np.float32)
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-threads", "0", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Failed to load audio: {result.stderr.decode(errors='replace')}")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def normalize_audio(audio):
    """Single unpadded Wav2Vec2FeatureExtractor input, without HF runtime."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio):
        raise ValueError("HuBERT expects a nonempty 16-kHz mono waveform.")
    if not np.isfinite(audio).all():
        raise ValueError("HuBERT audio must contain only finite samples.")
    return (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)


class HubertAudioEncoder:
    """Extract CPU FP32 [audio_frame, layer, channel] conditioning features.

    Only the prefix ending at max(layer_indices) is loaded and executed. The
    four TBDub states (9, 10, 11, 12) therefore do not need layers 13--24.
    """

    def __init__(
        self,
        model_path,
        device=None,
        cpu_offload=False,
        num_frames=77,
        audio_feat_window_size=0,
        vid_fps=25,
        layer_indices=(9, 10, 11, 12),
    ):
        self.model_path = Path(model_path)
        with (self.model_path / "config.json").open() as handle:
            self.config = json.load(handle)
        with (self.model_path / "preprocessor_config.json").open() as handle:
            processor_config = json.load(handle)
        self._validate_config(processor_config, layer_indices)
        if vid_fps != 25 or num_frames < 5 or (num_frames - 1) % 4:
            raise ValueError("TBDub audio windows require 25 fps and a video length of 4n+1 (at least 5).")
        if audio_feat_window_size < 0:
            raise ValueError("audio_feat_window_size must be nonnegative.")
        self.sample_rate = 16000
        self.aud_feature_fps = 50
        self.vid_fps = vid_fps
        self.num_frames = num_frames
        self.token_frames = (num_frames - 1) // 4 + 1
        self.audio_feat_window_size = audio_feat_window_size
        self.layer_indices = tuple(layer_indices)
        self.embedding_num_layers = len(layer_indices)
        self.computation_device = torch.device(device if device is not None else AI_DEVICE)
        self.cpu_offload = cpu_offload
        self.device = torch.device("cpu") if cpu_offload else self.computation_device
        self.weights = load_hubert_weights(self.model_path, self.config, max(layer_indices), self.device)
        self.infer = HubertInfer(self.config, self.layer_indices)

    def _validate_config(self, processor_config, layer_indices):
        if self.config.get("model_type") != "hubert" or not self.config.get("do_stable_layer_norm") or self.config.get("feat_extract_norm") != "layer":
            raise ValueError("Native TBDub HuBERT requires stable layer norm and layer-normalized feature convolutions.")
        if self.config.get("feat_extract_activation") != "gelu" or self.config.get("hidden_act") != "gelu":
            raise ValueError("Native TBDub HuBERT currently supports GELU activations only.")
        if self.config.get("adapter_attn_dim") is not None or self.config.get("conv_pos_batch_norm", False):
            raise ValueError("HuBERT attention adapters and positional batch normalization are not supported.")
        if not layer_indices or min(layer_indices) < 0 or max(layer_indices) > self.config["num_hidden_layers"]:
            raise ValueError("layer_indices must select existing HuBERT hidden states.")
        if processor_config.get("sampling_rate") != 16000 or not processor_config.get("do_normalize"):
            raise ValueError("TBDub requires normalized HuBERT audio sampled at 16 kHz.")
        if int(np.prod(self.config["conv_stride"])) != 320:
            raise ValueError("TBDub's audio windows require a HuBERT feature stride of 320 samples (50 fps).")

    def to(self, device):
        self.device = torch.device(device)
        self.weights.to_device(self.device)
        return self

    def to_cpu(self):
        return self.to("cpu")

    def to_cuda(self):
        return self.to(self.computation_device)

    @torch.no_grad()
    def audio2feat(self, audio_path_or_array, pad_end=None):
        """A file gets 77/25 s trailing silence; arrays match upstream no-pad.

        Set pad_end=True when passing a decoded waveform to obtain the same
        features as the file-path call. Keep the returned tensor across chunks.
        """
        is_path = isinstance(audio_path_or_array, (str, os.PathLike))
        audio = load_audio(audio_path_or_array, self.sample_rate) if is_path else np.asarray(audio_path_or_array, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("HuBERT expects a 16-kHz mono waveform.")
        if pad_end is None:
            pad_end = is_path
        if pad_end:
            audio = pad_with_fadeout_silence(audio, len(audio) + round(self.num_frames / self.vid_fps * self.sample_rate), self.sample_rate)
        values = normalize_audio(audio)
        feature_length = len(values)
        for kernel, stride in zip(self.config["conv_kernel"], self.config["conv_stride"]):
            feature_length = (feature_length - kernel) // stride + 1
        if feature_length < 1:
            raise ValueError("Audio is too short for HuBERT's convolutional feature extractor.")
        try:
            if self.cpu_offload:
                self.to_cuda()
            inputs = torch.from_numpy(values).to(self.device)
            return self.infer.infer(inputs, self.weights).detach().cpu().float()
        finally:
            if self.cpu_offload:
                self.to_cpu()

    def crop_overlap_audio_window(self, features, start_index):
        """Return [latent_frames, window_audio_frames, layers * channels]."""
        if features.ndim != 3 or features.shape[0] == 0 or features.shape[1] != self.embedding_num_layers:
            raise ValueError("Expected nonempty HuBERT features shaped [frames, selected_layers, channels].")
        # Upstream einops 'f n c -> f (c n)': channel-major, layers interleaved.
        flattened = features.transpose(1, 2).flatten(1)
        # Token zero covers [start-3,start+1), all later tokens advance by four.
        centers = torch.arange(self.token_frames, device=features.device) * 4 - 3 + int(start_index)
        window = self.audio_feat_window_size
        offsets = torch.arange(-window * 2, (4 + window) * 2, device=features.device)
        indices = (centers[:, None] * 2 + offsets).clamp(0, len(features) - 1)
        return flattened[indices]


__all__ = ["HubertAudioEncoder"]
