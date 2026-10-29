"""Spatial geometry of a native RelayVSR checkpoint, independent of output resizing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RelayVSRGeometry:
    scale: int

    def __post_init__(self):
        if self.scale not in (2, 4):
            raise ValueError("RelayVSR native scale must be 2 or 4")

    @property
    def latent_stride(self):
        return 16 // self.scale

    @property
    def wan_stride(self):
        return 2 * self.latent_stride

    @property
    def encoder_strides(self):
        return (2, 2, self.latent_stride // 4)

    def latent_grid(self, height, width):
        return height // self.latent_stride, width // self.latent_stride

    def wan_grid(self, height, width):
        return height // self.wan_stride, width // self.wan_stride

    def padded_size(self, height, width):
        alignment = self.wan_stride
        return height + (-height) % alignment, width + (-width) % alignment

    def output_size(self, request, height, width):
        target = (height * self.scale, width * self.scale)
        if request.size:
            if len(request.size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in request.size):
                raise ValueError("RelayVSR size must be [height, width] with integer dimensions")
            target = tuple(request.size)
        if self.scale == 2:
            if request.sr_ratio != 2 or target != (height * 2, width * 2):
                raise ValueError("relayvsr_x2 supports only sr_ratio=2 and an exact native 2x output size")
        elif not request.size:
            if request.sr_ratio not in (2, 4):
                raise ValueError("relayvsr_x4 sr_ratio must be 2 or 4; 2x resizes the native 4x result")
            target = (int(height * request.sr_ratio), int(width * request.sr_ratio))
        if min(target) <= 0 or target[0] > height * self.scale or target[1] > width * self.scale:
            raise ValueError(f"RelayVSR output dimensions must be positive and no larger than the native {self.scale}x result")
        return target
