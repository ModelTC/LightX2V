"""One-dimensional convolution with the shared convolution weight lifecycle."""

import torch.nn.functional as F

from .conv2d import Conv2dWeight


class Conv1dWeight(Conv2dWeight):
    def apply(self, input_tensor):
        return F.conv1d(input_tensor, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
