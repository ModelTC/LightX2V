import torch

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v_platform.base.global_var import AI_DEVICE


class WanDancerVAE(WanVAE):
    """Wan VAE with the exact overlap/weighting used by Wan-Dancer."""

    @staticmethod
    def _mask_1d(length, left_bound, right_bound, border_width):
        value = torch.ones(length)
        if not left_bound:
            value[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            value[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return value

    @classmethod
    def _mask(cls, data, bounds, border):
        height = cls._mask_1d(data.shape[-2], bounds[0], bounds[1], border[0])
        width = cls._mask_1d(data.shape[-1], bounds[2], bounds[3], border[1])
        return torch.minimum(height[:, None], width[None, :])[None, None, None]

    @staticmethod
    def _tasks(height, width, tile, stride):
        tasks = []
        for top in range(0, height, stride[0]):
            if top - stride[0] >= 0 and top - stride[0] + tile[0] >= height:
                continue
            for left in range(0, width, stride[1]):
                if left - stride[1] >= 0 and left - stride[1] + tile[1] >= width:
                    continue
                tasks.append((top, top + tile[0], left, left + tile[1]))
        return tasks

    def _encode_tiled(self, video):
        _, _, frames, height, width = video.shape
        tile, stride = (240, 416), (120, 208)
        out_frames = (frames + 3) // 4
        values = torch.zeros((1, 16, out_frames, height // 8, width // 8), dtype=video.dtype, device="cpu")
        weight = torch.zeros((1, 1, out_frames, height // 8, width // 8), dtype=video.dtype, device="cpu")
        for top, bottom, left, right in self._tasks(height, width, tile, stride):
            encoded = self.model.encode(video[..., top:bottom, left:right].to(AI_DEVICE), self.scale).cpu()
            mask = self._mask(
                encoded,
                (top == 0, bottom >= height, left == 0, right >= width),
                ((tile[0] - stride[0]) // 8, (tile[1] - stride[1]) // 8),
            ).to(encoded.dtype)
            target_top, target_left = top // 8, left // 8
            values[..., target_top : target_top + encoded.shape[-2], target_left : target_left + encoded.shape[-1]].add_(encoded * mask)
            weight[..., target_top : target_top + encoded.shape[-2], target_left : target_left + encoded.shape[-1]].add_(mask)
        return values / weight

    def _decode_tiled(self, latents):
        _, _, frames, height, width = latents.shape
        tile, stride = (30, 52), (15, 26)
        out_frames = frames * 4 - 3
        values = torch.zeros((1, 3, out_frames, height * 8, width * 8), dtype=latents.dtype, device="cpu")
        weight = torch.zeros((1, 1, out_frames, height * 8, width * 8), dtype=latents.dtype, device="cpu")
        for top, bottom, left, right in self._tasks(height, width, tile, stride):
            decoded = self.model.decode(latents[..., top:bottom, left:right].to(AI_DEVICE), self.scale).cpu()
            mask = self._mask(
                decoded,
                (top == 0, bottom >= height, left == 0, right >= width),
                ((tile[0] - stride[0]) * 8, (tile[1] - stride[1]) * 8),
            ).to(decoded.dtype)
            target_top, target_left = top * 8, left * 8
            values[..., target_top : target_top + decoded.shape[-2], target_left : target_left + decoded.shape[-1]].add_(decoded * mask)
            weight[..., target_top : target_top + decoded.shape[-2], target_left : target_left + decoded.shape[-1]].add_(mask)
        return (values / weight).clamp_(-1, 1)

    def encode(self, video, world_size_h=None, world_size_w=None):
        if self.cpu_offload:
            self.to_cuda()
        result = self._encode_tiled(video).squeeze(0)
        if self.cpu_offload:
            self.to_cpu()
        return result

    def decode(self, latents):
        if self.cpu_offload:
            self.to_cuda()
        result = self._decode_tiled(latents.unsqueeze(0))
        if self.cpu_offload:
            self.to_cpu()
        return result

    def decode_framewise(self, latents):
        if self.cpu_offload:
            self.to_cuda()
        frames = [self._decode_tiled(latents[:, index : index + 1].unsqueeze(0)) for index in range(latents.shape[1])]
        result = torch.cat(frames, dim=2)
        if self.cpu_offload:
            self.to_cpu()
        return result
