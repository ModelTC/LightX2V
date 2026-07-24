import torch
import torchaudio
from torch import nn


class RealMelSpectrogram(nn.Module):
    """Mel spectrogram frontend that avoids complex-valued magnitude ops."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer(
            "mel_filter",
            torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=0.0,
                f_max=sample_rate / 2.0,
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm="slaney",
                mel_scale="slaney",
            ),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert a waveform of shape (..., samples) to (..., mel, frames)."""
        input_shape = waveform.shape
        waveform = waveform.reshape(-1, input_shape[-1])

        complex_spec = torch.stft(
            input=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # torch.view_as_real is a zero-copy view. Magnitude is then computed
        # entirely with real-valued ops, avoiding complex abs on NPU.
        real_spec = torch.view_as_real(complex_spec)
        real = real_spec[..., 0]
        imag = real_spec[..., 1]
        magnitude = torch.sqrt(real.square() + imag.square())

        magnitude = magnitude.reshape(input_shape[:-1] + magnitude.shape[-2:])
        mel_filter = self.mel_filter.to(dtype=magnitude.dtype)
        return torch.matmul(magnitude.transpose(-1, -2), mel_filter).transpose(-1, -2)
