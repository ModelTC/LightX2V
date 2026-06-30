import json
import math
import os

import torch
import torch.nn as nn
from safetensors import safe_open
from torch.nn.utils import weight_norm


class Snake1d(nn.Module):
    def __init__(self, hidden_dim, logscale=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, hidden_states):
        shape = hidden_states.shape
        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)
        beta = self.beta if not self.logscale else torch.exp(self.beta)
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class Cosmos3AudioResidualUnit(nn.Module):
    def __init__(self, dimension, dilation=1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_state):
        output_tensor = self.conv1(self.snake1(hidden_state))
        output_tensor = self.conv2(self.snake2(output_tensor))
        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        return hidden_state + output_tensor


class Cosmos3AudioDecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, output_padding=0):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=output_padding,
            )
        )
        self.res_unit1 = Cosmos3AudioResidualUnit(output_dim, dilation=1)
        self.res_unit2 = Cosmos3AudioResidualUnit(output_dim, dilation=3)
        self.res_unit3 = Cosmos3AudioResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state):
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.res_unit3(hidden_state)
        return hidden_state


class Cosmos3AudioDecoder(nn.Module):
    def __init__(self, channels, input_channels, audio_channels, upsampling_ratios, channel_multiples):
        super().__init__()
        channel_multiples = [1] + list(channel_multiples)
        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))
        block = []
        for stride_index, stride in enumerate(upsampling_ratios):
            block.append(
                Cosmos3AudioDecoderBlock(
                    input_dim=channels * channel_multiples[len(upsampling_ratios) - stride_index],
                    output_dim=channels * channel_multiples[len(upsampling_ratios) - stride_index - 1],
                    stride=stride,
                    output_padding=stride % 2,
                )
            )
        self.block = nn.ModuleList(block)
        self.snake1 = Snake1d(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


class Cosmos3SoundTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sampling_rate = int(config.get("sampling_rate", 48000))
        self.hop_size = int(config.get("hop_size") or math.prod(config.get("dec_strides", [2, 4, 5, 6, 8])))
        self.decoder = Cosmos3AudioDecoder(
            channels=int(config.get("dec_dim", 320)),
            input_channels=int(config.get("vocoder_input_dim", 64)),
            audio_channels=int(config.get("dec_out_channels", 2)),
            upsampling_ratios=list(reversed(config.get("dec_strides", [2, 4, 5, 6, 8]))),
            channel_multiples=list(config.get("dec_c_mults", [1, 2, 4, 8, 16])),
        )

    @classmethod
    def from_pretrained(cls, model_path, device, dtype):
        tokenizer_path = os.path.join(model_path, "sound_tokenizer")
        with open(os.path.join(tokenizer_path, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(config)
        weight_path = os.path.join(tokenizer_path, "diffusion_pytorch_model.safetensors")
        state_dict = {}
        with safe_open(weight_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("decoder."):
                    state_dict[key] = f.get_tensor(key)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in unexpected if key.startswith("decoder.")]
        missing = [key for key in missing if key.startswith("decoder.")]
        if missing or unexpected:
            raise RuntimeError(f"Failed to load Cosmos3 sound decoder, missing={missing}, unexpected={unexpected}")
        return model.to(device=device, dtype=dtype).eval()

    @torch.no_grad()
    def decode(self, latents):
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        audio = self.decoder(latents).clamp(-1.0, 1.0)
        return audio.squeeze(0) if squeeze else audio
