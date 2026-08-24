"""Raw T2AV sample preparation for MiniMax-H3 cache construction."""

from pathlib import Path

import torch
import torch.nn.functional as F

from lightx2v_train.model_zoo.native.minimax_h3 import audio_latent_num_frames, video_latent_num_frames
from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER


class MiniMaxH3T2AVProcessor:
    """Prepare synchronized RGB video and stereo audio for the H3 VAEs."""

    unconditional_prompt = " "
    requires_audio = True

    def __init__(self, config):
        model_config = config["model"]
        self.video_fps = int(model_config.get("video_fps", 24))
        self.audio_sample_rate = int(model_config.get("audio_sampling_rate", 32000))
        self.audio_latents_per_second = int(model_config.get("audio_latents_per_second", 40))
        if self.video_fps <= 0 or self.audio_sample_rate <= 0 or self.audio_latents_per_second <= 0:
            raise ValueError("MiniMax-H3 video/audio rates must be positive.")
        if self.video_fps != 24 or self.audio_latents_per_second != 40:
            raise ValueError("MiniMax-H3 T2AV requires video_fps=24 and audio_latents_per_second=40.")
        if self.audio_sample_rate % self.audio_latents_per_second:
            raise ValueError("MiniMax-H3 audio_sampling_rate must be divisible by audio_latents_per_second.")
        self.audio_hop_length = self.audio_sample_rate // self.audio_latents_per_second

    def __call__(self, sample):
        video = sample.get("inputs", {}).get("video")
        if not torch.is_tensor(video) or video.ndim != 4 or video.shape[0] != 3:
            shape = tuple(video.shape) if torch.is_tensor(video) else type(video).__name__
            raise ValueError(f"MiniMax-H3 expects decoded video [3,F,H,W], got {shape}.")
        num_frames = int(video.shape[1])
        video_latent_num_frames(num_frames)

        audio_path = sample.get("meta", {}).get("audio_path")
        if not audio_path:
            raise ValueError("MiniMax-H3 T2AV cache construction requires an audio path for every video.")
        waveform = self._load_audio(audio_path)
        audio_frames = audio_latent_num_frames(num_frames)
        target_samples = audio_frames * self.audio_hop_length
        video_start_time = float(sample["meta"].get("video_start_time", 0.0))
        if video_start_time < 0:
            raise ValueError(f"MiniMax-H3 video_start_time must be non-negative, got {video_start_time}.")
        audio_start_sample = int(round(video_start_time * self.audio_sample_rate))
        waveform = self._slice_audio(waveform, audio_start_sample, target_samples)

        # VideoDataset normalizes pixels to [-1, 1], while H3's ImageNet VAE
        # normalization consumes [0, 1].
        sample["inputs"]["video"] = ((video + 1.0) * 0.5).clamp_(0.0, 1.0)
        sample["inputs"]["audio"] = waveform
        sample["meta"].update(
            {
                "num_frames": num_frames,
                "target_height": int(video.shape[-2]),
                "target_width": int(video.shape[-1]),
                "audio_sample_rate": self.audio_sample_rate,
                "audio_start_sample": audio_start_sample,
            }
        )
        return sample

    def _load_audio(self, audio_path):
        try:
            import torchaudio
        except ImportError as error:
            raise ImportError("MiniMax-H3 cache construction requires torchaudio.") from error

        waveform, sample_rate = torchaudio.load(Path(audio_path))
        waveform = waveform.float()
        if sample_rate != self.audio_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.audio_sample_rate,
            )
        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        if waveform.shape[0] != 2:
            raise ValueError(f"MiniMax-H3 audio must be mono or stereo, got {waveform.shape[0]} channels.")
        return waveform.contiguous()

    @staticmethod
    def _slice_audio(waveform, start_sample, target_samples):
        waveform = waveform[..., start_sample : start_sample + target_samples]
        if waveform.shape[-1] < target_samples:
            waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
        return waveform.contiguous()


@SAMPLE_PROCESSOR_REGISTER("minimax_h3_t2av")
def build_minimax_h3_t2av_processor(config):
    return MiniMaxH3T2AVProcessor(config)
