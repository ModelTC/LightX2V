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

    def validate_sr_ratio(self, sr_ratio):
        if not 1 <= sr_ratio <= self.scale:
            raise ValueError(f"relayvsr_x{self.scale} sr_ratio must be between 1 and {self.scale}, got {sr_ratio}")

    def output_size(self, request, height, width):
        size = getattr(request, "size", None)
        if size:
            # An explicit [height, width] resizes the native result to exactly that size,
            # scaling each axis independently; sr_ratio is ignored.
            if self.scale == 2:
                raise ValueError("relayvsr_x2 has an exact 2x output and does not accept size")
            if len(size) != 2:
                raise ValueError(f"RelayVSR size must be [height, width], got {list(size)}")
            target = (int(size[0]), int(size[1]))
        else:
            self.validate_sr_ratio(request.sr_ratio)
            target = (int(height * request.sr_ratio), int(width * request.sr_ratio))
        if min(target) <= 0:
            raise ValueError("RelayVSR output dimensions must be positive")
        return target
