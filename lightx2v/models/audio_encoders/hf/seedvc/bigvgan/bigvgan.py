# Copyright (c) 2024 NVIDIA CORPORATION. Licensed under the MIT license.
# Adapted from NVIDIA/BigVGAN via Plachtaa/seed-vc at 51383efd921027683c89e5348211d93ff12ac2a8.
"""BigVGAN generator with the upstream PyTorch anti-aliasing path."""

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from . import activations
from .alias_free_activation.native_ops.act import Activation1d as TorchActivation1d
from .env import AttrDict
from .utils import get_padding, init_weights


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(self, h: AttrDict, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5), activation: str = None):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))) for d in dilation])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))) for _ in range(len(dilation))])
        self.convs2.apply(init_weights)
        self.num_layers = len(self.convs1) + len(self.convs2)
        Activation1d = TorchActivation1d
        if activation == "snake":
            self.activations = nn.ModuleList([Activation1d(activation=activations.Snake(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        elif activation == "snakebeta":
            self.activations = nn.ModuleList([Activation1d(activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale)) for _ in range(self.num_layers)])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = (self.activations[::2], self.activations[1::2])
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


class BigVGAN(torch.nn.Module):
    def __init__(self, h: AttrDict):
        super().__init__()
        self.h = h
        Activation1d = TorchActivation1d
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        if h.resblock != "1":
            raise ValueError("Seed-VC requires BigVGAN AMPBlock1 weights")
        resblock_class = AMPBlock1
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([weight_norm(ConvTranspose1d(h.upsample_initial_channel // 2**i, h.upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2))]))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock_class(h, ch, k, d, activation=h.activation))
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale) if h.activation == "snake" else activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale) if h.activation == "snakebeta" else None
        )
        if activation_post is None:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")
        self.activation_post = Activation1d(activation=activation_post)
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final))
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)
        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for stage in self.ups:
                for conv in stage:
                    remove_weight_norm(conv)
            for block in self.resblocks:
                block.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass
