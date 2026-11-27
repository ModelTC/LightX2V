# Copyright (c) 2024 NVIDIA CORPORATION. Licensed under the MIT license.
# Adapted from NVIDIA/BigVGAN via Plachtaa/seed-vc at 51383efd921027683c89e5348211d93ff12ac2a8.
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
