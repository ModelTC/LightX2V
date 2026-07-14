import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class WanDancerPreInfer(WanPreInfer):
    def __init__(self, config):
        super().__init__(config)
        # Dancer uses Wan-I2V's CLIP conditioning path even though its public task is s2v.
        self.task = "i2v"
        self.music_heads = 4
        self.music_head_dim = 64
        self.freqs = torch.cat(
            [
                self._rope_params(2794, self.head_size - 4 * (self.head_size // 6)),
                self._rope_params(2794, 2 * (self.head_size // 6)),
                self._rope_params(2794, 2 * (self.head_size // 6)),
            ],
            dim=1,
        ).to(AI_DEVICE)

    @staticmethod
    def _rope_params(max_seq_len, dim, theta=10000.0):
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        phase = torch.outer(torch.arange(max_seq_len, dtype=torch.float64), inv)
        return torch.polar(torch.ones_like(phase), phase)

    @staticmethod
    def _interpolate_temporal_freqs(freqs, frame_num, fps):
        interval = 30.0 / (fps + 1e-6)
        total = int(interval * frame_num + 0.5)
        source = freqs[:total]
        result = torch.zeros((frame_num, source.shape[1]), device=source.device, dtype=source.dtype)
        result[0] = source[0]
        result[-1] = source[total - 1]
        for index in range(1, frame_num - 1):
            position = index * interval
            low = int(position)
            high = min(low + 1, total - 1)
            result[index] = source[low] * (1.0 - position + low) + source[high] * (position - low)
        return result

    def prepare_cos_sin(self, grid_sizes, freqs):
        c = self.head_size // 2
        temporal, height, width = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        frames, grid_h, grid_w = grid_sizes
        if self._use_global:
            temporal = self._interpolate_temporal_freqs(temporal, frames, self._input_fps)
        result = torch.cat(
            [
                temporal[:frames].view(frames, 1, 1, -1).expand(frames, grid_h, grid_w, -1),
                height[:grid_h].view(1, grid_h, 1, -1).expand(frames, grid_h, grid_w, -1),
                width[:grid_w].view(1, 1, grid_w, -1).expand(frames, grid_h, grid_w, -1),
            ],
            dim=-1,
        ).reshape(frames * grid_h * grid_w, 1, -1)
        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            rank = dist.get_rank(self.seq_p_group)
            padding = (-result.shape[0]) % world_size
            if padding:
                result = F.pad(result, (0, 0, 0, 0, 0, padding), value=1.0)
            result = torch.chunk(result, world_size, dim=0)[rank]
        return result

    @staticmethod
    def _rotate_music(x):
        length = x.shape[0]
        inv = 1.0 / (10000 ** (torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32) / x.shape[-1]))
        phase = torch.outer(torch.arange(length, device=x.device, dtype=torch.float32), inv)
        phase = torch.repeat_interleave(phase, 2, dim=-1).to(x)
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return x * phase.cos() + rotated * phase.sin()

    def _music_layer(self, layer, x):
        normalized = layer.norm1.apply(x)
        qk = self._rotate_music(normalized)
        weight = layer.in_proj_weight.tensor
        bias = layer.in_proj_bias.tensor
        size = x.shape[-1]
        attended = F.multi_head_attention_forward(
            qk.unsqueeze(1),
            qk.unsqueeze(1),
            normalized.unsqueeze(1),
            size,
            self.music_heads,
            weight,
            bias,
            None,
            None,
            False,
            0.1,
            layer.out_proj._get_actual_weight().t(),
            layer.out_proj._get_actual_bias(),
            training=False,
            need_weights=False,
        )[0].squeeze(1)
        x = x + attended
        hidden = layer.linear1.apply(layer.norm2.apply(x))
        return x + layer.linear2.apply(F.gelu(hidden))

    def _encode_music(self, weights, music_feature, frame_num):
        music_feature = music_feature.to(device=AI_DEVICE, dtype=GET_DTYPE())
        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            rank = dist.get_rank(self.seq_p_group)
            length = (music_feature.shape[0] // world_size) * world_size
            music_feature = torch.chunk(music_feature[:length], world_size, dim=0)[rank]
        x = weights.music_projection.apply(music_feature)
        for layer in weights.music_layers:
            x = self._music_layer(layer, x)
        if self.seq_p_group is not None:
            gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(self.seq_p_group))]
            dist.all_gather(gathered, x, group=self.seq_p_group)
            x = torch.cat(gathered, dim=0)
        x = F.interpolate(x[None, None], size=(149, 4800), mode="bilinear").squeeze(1)
        final_length = frame_num * 8
        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            final_length = (final_length // world_size) * world_size
        x = F.interpolate(x.unsqueeze(1), size=(final_length, self.dim), mode="bilinear").squeeze(1)
        if self.seq_p_group is not None:
            x = torch.chunk(x, dist.get_world_size(self.seq_p_group), dim=1)[dist.get_rank(self.seq_p_group)]
        return x.squeeze(0).contiguous()

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        image_output = inputs["image_encoder_output"]
        self._input_fps = float(image_output["input_fps"])
        self._use_global = int(self._input_fps + 0.5) != 30
        original_patch = weights.patch_embedding
        if self._use_global:
            weights.patch_embedding = weights.patch_embedding_global
        # Force a rebuild between stages/segments with distinct fps or shapes.
        if getattr(self, "_last_fps", None) != self._input_fps:
            self.grid_sizes = (0, 0, 0)
            self.cos_sin = None
        try:
            output = super().infer(weights, inputs, kv_start=kv_start, kv_end=kv_end)
        finally:
            weights.patch_embedding = original_patch
        self._last_fps = self._input_fps

        ref = image_output["ref_clip_encoder_out"]
        ref = weights.ref_proj_0.apply(ref)
        ref = weights.ref_proj_1.apply(ref)
        ref = F.gelu(ref, approximate="none")
        ref = weights.ref_proj_3.apply(ref)
        ref = weights.ref_proj_4.apply(ref)
        output.context = torch.cat([ref, output.context], dim=0)

        cache_key = "_dancer_music_context"
        if cache_key not in inputs:
            inputs[cache_key] = self._encode_music(weights, image_output["music_feature"], output.grid_sizes.tuple[0])
        output.adapter_args.update(
            music_context=inputs[cache_key],
            use_global=self._use_global,
            enable_skip_layer=bool(image_output.get("enable_skip_layer", True)),
        )
        return output
