"""BigVGAN component: mel in, mono waveform out; reference implementation reused."""

import json
from pathlib import Path

import torch


class SeedVCVocoder:
    def __init__(self, path, device, sample_rate, mel_channels, hop_length):
        from .bigvgan.bigvgan import BigVGAN
        from .bigvgan.env import AttrDict

        path = Path(path)
        metadata = json.loads((path / "config.json").read_text())
        expected = {"sampling_rate": sample_rate, "num_mels": mel_channels, "hop_size": hop_length}
        for key, value in expected.items():
            if metadata[key] != value:
                raise ValueError(f"Vocoder {key}={metadata[key]} does not match the acoustic model ({value})")
        self.model = BigVGAN(AttrDict(metadata))
        state = torch.load(path / "bigvgan_generator.pt", map_location="cpu", weights_only=True)["generator"]
        has_weight_norm = any(key.endswith(".weight_g") for key in state)
        if not has_weight_norm:
            self.model.remove_weight_norm()
        self.model.load_state_dict(state, strict=True)
        if has_weight_norm:
            self.model.remove_weight_norm()
        self.model = self.model.eval().requires_grad_(False).float().to(device)

    @torch.inference_mode()
    def decode(self, mel):
        return self.model(mel.float()).reshape(-1).float().cpu()

    def close(self):
        self.model = None
