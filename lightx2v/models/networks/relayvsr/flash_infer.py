import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

from .attention import get_local_attention_ops
from .compile import RelayCompileMixin, mark_sequence_dynamic
from .rope import apply_rope


class FlashPreInfer:
    def frame(self, weights, lr):
        x = lr
        for i in (0, 2, 4):
            x = F.silu(getattr(weights, f"conv{i}").apply(x))
        x = weights.conv6.apply(x).flatten(2).transpose(1, 2)[0]
        return weights.lr_norm.apply(x) + weights.video_type.tensor[0]

    def condition(self, weights, latent):
        x = weights.condition.apply(latent).flatten(2).transpose(1, 2)[0]
        return weights.condition_norm.apply(x) + weights.condition_type.tensor[0]


class FlashTransformerInfer(RelayCompileMixin, BaseTransformerInfer):
    def __init__(self, config, parallel, deployment):
        self.config, self.parallel = config, parallel
        self.init_compile(deployment)
        self.local_attention = get_local_attention_ops(deployment)[0]

    def run_block(self, block_idx, block, x, condition, past_key, past_value, q_rope, k_rope, windows, height, width):
        if self.use_compile:
            mark_sequence_dynamic(x, *q_rope, *k_rope)
            if not condition:
                mark_sequence_dynamic(past_key, past_value)
        return super().run_block(block_idx, block, x, condition, past_key, past_value, q_rope, k_rope, windows, height, width)

    def infer_block(self, block, x, condition, past_key, past_value, q_rope, k_rope, windows, height, width):
        tokens = q_rope[0].shape[0]
        normalized = block.attention_norm.apply(x)
        query = block.query.apply(normalized).view(-1, 8, 64)
        key_op = block.condition_key if condition else block.key
        value_op = block.condition_value if condition else block.value
        key = block.key_norm.apply(key_op.apply(normalized).view(-1, 2, 64))
        value = block.value_norm.apply(value_op.apply(normalized).view(-1, 2, 64))
        query = self.parallel.to_heads(query, tokens)
        key = self.parallel.to_heads(key, tokens)
        value = self.parallel.to_heads(value, tokens)
        all_key = key if condition else torch.cat((past_key, key), dim=0)
        all_value = value if condition else torch.cat((past_value, value), dim=0)
        query = apply_rope(query, q_rope)
        all_key = apply_rope(all_key, k_rope)
        query, all_key, all_value = query.transpose(0, 1)[None], all_key.transpose(0, 1)[None], all_value.transpose(0, 1)[None]
        if torch.compiler.is_compiling():
            spatial = self.config["spatial_attention"]
            names = ("condition_self_window_size",) if condition else ("start_condition_window_size", "end_condition_window_size", "video_window_size")
            # The attention leaf repeats the video window for the dynamic KV
            # length; Python history length does not specialize the block.
            windows = tuple(tuple(spatial[name]) for name in names)
        attended = self.local_attention(query, all_key, all_value, height, width, windows)
        attended = attended[0].transpose(0, 1)
        x = x + block.output.apply(self.parallel.to_sequence(attended))
        gate, up = block.gate_up.apply(block.feed_forward_norm.apply(x)).chunk(2, -1)
        x = x + block.down.apply(F.silu(gate) * up)
        return x, key, value

    def prepare_condition(self, blocks, x, rope, height, width):
        if self.use_compile and self.parallel.size > 1:
            x = x.clone()  # Normalize the first SP shard's storage offset.
        frequencies = rope.for_positions(height, width, (0,))
        window = tuple(self.config["spatial_attention"]["condition_self_window_size"])
        caches = []
        for index, block in enumerate(blocks):
            x, key, value = self.run_block(index, block, x, True, None, None, frequencies, frequencies, (window,), height, width)
            caches.append((key, value))
        return caches

    def infer(self, blocks, x, conditions, history, rope, frame, height, width):
        if self.use_compile and self.parallel.size > 1:
            x = x.clone()
        spatial = self.config["spatial_attention"]
        previous = min(frame, self.config["window_size"] - 1)
        q_rope = rope.for_positions(height, width, (0,))
        k_rope = rope.for_positions(height, width, (0, 0, *range(-previous, 1)))
        windows = (tuple(spatial["start_condition_window_size"]), tuple(spatial["end_condition_window_size"])) + (tuple(spatial["video_window_size"]),) * (previous + 1)
        for index, block in enumerate(blocks):
            ck, cv = conditions[index]
            if previous:
                past_key = torch.cat((ck, history.k_cache(index)), 0)
                past_value = torch.cat((cv, history.v_cache(index)), 0)
            else:
                past_key, past_value = ck, cv
            x, key, value = self.run_block(index, block, x, False, past_key, past_value, q_rope, k_rope, windows, height, width)
            history.store_kv(key, value, index)
        return x


class FlashPostInfer:
    def infer(self, weights, hidden, lr):
        height, width = lr.shape[-2] // 4, lr.shape[-1] // 4
        pixels = weights.projection.apply(F.silu(weights.head.apply(hidden)))
        residual = F.pixel_shuffle(pixels.t().reshape(1, 768, height, width), 16)
        return F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False) + residual
