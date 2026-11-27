import math
from functools import cache, partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.wan.infer.infinitetalk.audio_windows import pack_later_audio_windows
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, ROPE_REGISTER

# RoPE position assigned by the release to the shared silent bystander stream.
SILENT_BACKGROUND_POSITION = 100


def modulate(tokens, shift, scale):
    normalized = F.layer_norm(tokens.float(), (tokens.shape[-1],), eps=1e-6)
    return (normalized * (1 + scale[:, None, :]) + shift[:, None, :]).to(tokens.dtype)


def gather_spatial_shards(tile, group, spatial_split):
    """Reassemble [..., height, width, channels] from row-major spatial tiles."""
    rows, cols = spatial_split
    shards = [torch.empty_like(tile) for _ in range(rows * cols)]
    dist.all_gather(shards, tile, group=group)
    strips = [torch.cat(shards[start : start + cols], dim=-2) for start in range(0, len(shards), cols)]
    return strips[0] if rows == 1 else torch.cat(strips, dim=-3)


class AvatarPreInfer:
    def __init__(self, config, weights):
        self.weights = weights
        self.patch_size = tuple(config["patch_size"])
        half_dim = config["frequency_embedding_size"] // 2
        self.time_frequencies = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim).to(weights.patch.weight.device)
        self.text_length = config["model_max_length"]
        self.vae_scale = config["vae_scale"]
        self.audio_window = config["audio_window"]
        self.context_tokens = config["context_tokens"]
        self.head_dim = config["hidden_size"] // config["num_heads"]
        self.class_interval = config["class_interval"]
        self.class_range = config["class_range"]

    def _rope(self, grid, top, left, device, continuation):
        spatial_dim = 2 * (self.head_dim // 6)
        dims = (self.head_dim - 2 * spatial_dim, spatial_dim, spatial_dim)
        # The continuation anchor uses temporal position 10. The segment starts at 0.
        time = torch.cat((torch.tensor([10]), torch.arange(grid[0] - 1))) if continuation else torch.arange(grid[0])
        t, h, w = grid
        axes = (time, torch.arange(top, top + h), torch.arange(left, left + w))
        phase = []
        for axis, dim in zip(axes, dims):
            frequency = 1 / 10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            phase.append(torch.outer(axis.float(), frequency))
        angles = torch.cat((phase[0][:, None, None].expand(t, h, w, -1), phase[1][None, :, None].expand(t, h, w, -1), phase[2][None, None, :].expand(t, h, w, -1)), dim=-1).to(device)
        return angles.cos(), angles.sin()

    def project_text(self, text):
        weights = self.weights
        text = weights.text_out.apply(F.gelu(weights.text_in.apply(text), approximate="tanh"))
        # The released DiT attends to projected text plus zero-valued padding.
        return F.pad(text, (0, 0, 0, self.text_length - text.shape[0]))

    def prepare(self, projected_text, audio, speaker_masks, latent_shape, num_cond_latents, spatial_split, rank, rope=None):
        weights = self.weights
        speakers = audio.shape[0]
        later = pack_later_audio_windows(audio, self.vae_scale, self.audio_window)
        projected = weights.audio_later.apply(later.flatten(2).flatten(0, 1)).unflatten(0, (speakers, -1))
        if num_cond_latents == 0 or speaker_masks is not None:
            first = weights.audio_first.apply(audio[:, 0].flatten(1))
            projected = torch.cat((first[:, None], projected), dim=1)
        audio = weights.audio_hidden.apply(projected.flatten(0, 1).relu()).relu()
        audio = weights.audio_out.apply(audio).unflatten(-1, (self.context_tokens, -1))
        audio = weights.audio_norm.apply(audio).unflatten(0, (speakers, -1)).transpose(0, 1).flatten(1, 2)

        rows, cols = spatial_split
        row, col = divmod(rank, cols)
        pt, ph, pw = self.patch_size
        frames, height, width = latent_shape[-3:]
        grid = (frames // pt, height // ph // rows, width // pw // cols)
        top, left = row * grid[1], col * grid[2]
        audio = audio[-(grid[0] - num_cond_latents) :]
        offsets = torch.arange(audio.shape[0] + 1, device=audio.device, dtype=torch.int32)
        speaker_rope = None
        if speaker_masks is not None:
            frequencies = (1 / 10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)).to(audio.device)
            centers = [self.class_interval / 2, self.class_range - self.class_interval / 2]
            if speakers == 3:
                centers.append(SILENT_BACKGROUND_POSITION)
            centers = frequencies.new_tensor(centers)
            angles = torch.outer(centers.repeat_interleave(self.context_tokens), frequencies).unsqueeze(0)
            speaker_rope = (frequencies, (angles.cos(), angles.sin()))
        return {
            "speaker_masks": speaker_masks,
            "speaker_rope": speaker_rope,
            "text": projected_text,
            "audio": audio,
            "audio_cu_seqlens": (offsets * (grid[1] * grid[2]), offsets * audio.shape[1]),
            "num_cond_latents": num_cond_latents,
            "spatial_slices": (slice(top * ph, (top + grid[1]) * ph), slice(left * pw, (left + grid[2]) * pw)),
            "rope": rope if rope is not None else self._rope(grid, top, left, audio.device, num_cond_latents > 1),
        }

    def embed_timesteps(self, timesteps):
        phase = timesteps.float()[:, None] * self.time_frequencies
        frequencies = torch.cat((phase.cos(), phase.sin()), dim=-1)
        hidden = F.silu(self.weights.time_in.apply(frequencies))
        return F.silu(self.weights.time_out.apply(hidden))

    def infer(self, latents, spatial_slices):
        height, width = spatial_slices
        patch = self.weights.patch
        tokens = patch.apply(latents[..., height, width].to(patch.weight.dtype))[0]
        return rearrange(tokens, "c t h w -> t (h w) c").contiguous()


class AvatarSelfAttention:
    def __init__(self, attn_type, num_shards):
        self.attention = ATTN_WEIGHT_REGISTER[attn_type]()
        self.num_shards = num_shards

    def apply_ulysses(self, ulysses, group, q, k, v, **kwargs):
        output, _ = ulysses.apply(q, k, v, attention_module=self, seq_p_group=group, attention_kwargs=kwargs)
        return output

    def apply(self, q, k, v, frame_tokens, num_cond_latents, layer_cache=None, **kwargs):
        cond_end = num_cond_latents * frame_tokens
        if layer_cache is not None:
            # Store K/V after the head exchange. Keep each rank's prefix before its
            # noisy rows when reconstructing the sequence for attention.
            keys, values = (part.unflatten(0, (self.num_shards, -1)) for part in (k, v))
            if num_cond_latents:
                layer_cache["k"] = keys[:, :cond_end].clone(memory_format=torch.contiguous_format)
                layer_cache["v"] = values[:, :cond_end].clone(memory_format=torch.contiguous_format)
            else:
                k = torch.cat((layer_cache["k"], keys), dim=1).flatten(0, 1)
                v = torch.cat((layer_cache["v"], values), dim=1).flatten(0, 1)
        if not num_cond_latents:
            return self.attention.apply(q, k, v)

        # Ulysses concatenates spatial shards in rank order, each with all its frames.
        def rows(tensor, start, end=None):
            return tensor.unflatten(0, (self.num_shards, -1))[:, start:end].flatten(0, 1)

        ends = (frame_tokens, cond_end) if num_cond_latents > 1 else (frame_tokens,)
        groups = []
        start = 0
        for end in ends:
            # The anchor and the overlap frames each attend only within their group.
            groups.append(self.attention.apply(rows(q, start, end), rows(k, start, end), rows(v, start, end)))
            start = end
        groups.append(self.attention.apply(rows(q, cond_end), k, v))
        return torch.cat([group.unflatten(0, (self.num_shards, -1)) for group in groups], dim=1).flatten(0, 1)


@cache
def get_self_attention(attn_type, num_shards):
    """Share the stateless frame-group attention with the NVLink custom op."""
    return AvatarSelfAttention(attn_type, num_shards)


class AvatarTransformerInfer(BaseTransformerInfer):
    def __init__(self, config, attention_group, spatial_split):
        self.init_compile(config)
        self.num_heads = config["num_heads"]
        self.class_interval = config["class_interval"]
        self.class_range = config["class_range"]
        attn_type = config["attn_type"]
        num_shards = math.prod(spatial_split)
        grouped = get_self_attention(attn_type, num_shards)
        self.attention = grouped.attention
        if attention_group is None:
            self.self_attention = grouped.apply
        elif config["parallel"].get("seq_p_nvlink", False):
            from .nvlink import nvlink_self_attention

            self.self_attention = partial(nvlink_self_attention, group_name=attention_group.group_name, size=num_shards, attn_type=attn_type)
        else:
            self.self_attention = partial(grouped.apply_ulysses, ATTN_WEIGHT_REGISTER["ulysses"](), attention_group)
        self.attention_group = attention_group
        self.spatial_split = spatial_split
        self.rope = ROPE_REGISTER["torch_real_rope"](layout="interleaved", compute_dtype=torch.float32)

    def _heads(self, projections):
        return projections.unflatten(-1, (self.num_heads, -1))

    def _speaker_scores(self, q, reference_k, masks):
        head_dim = q.shape[-1]
        q = q * head_dim**-0.5
        scores = []
        # Each V channel collects the attention mass of one region, shared across heads.
        for regions in masks.split(head_dim):
            num_regions = regions.shape[0]
            values = F.pad(regions.T, (0, head_dim - num_regions)).contiguous()
            values = values[:, None, :].expand_as(reference_k)
            mass = self.attention.apply(q, reference_k, values, softmax_scale=1.0)
            mass = mass.unflatten(-1, (2, self.num_heads // 2, head_dim))[..., :num_regions]
            group_scores = (mass / regions.sum(dim=1)).mean(dim=2)
            scores.append((group_scores[:, 0] + group_scores[:, 1]) / 2)
        return scores[0] if len(scores) == 1 else torch.cat(scores, dim=1)

    def _speaker_positions(self, q, reference_k, masks):
        scores = self._speaker_scores(q, reference_k, masks)
        people = scores[:, :2]
        limits = torch.stack((people.amax(dim=0), -people.amin(dim=0)))
        if self.attention_group is not None:
            dist.all_reduce(limits, op=dist.ReduceOp.MAX, group=self.attention_group)
        maximum, minimum = limits[0], -limits[1]
        positions = (people - minimum) / (maximum - minimum + 1e-8) * self.class_interval
        positions[:, 1] += self.class_range - self.class_interval
        background_pos = SILENT_BACKGROUND_POSITION if masks.shape[0] > 3 else self.class_range // 2
        positions = F.pad(positions, (0, 1), value=background_pos)
        return positions.gather(1, scores.argmax(dim=1, keepdim=True).clamp_max(2))

    def _self_attn(self, tokens, condition, frame_tokens, weights, strengths, layer_cache):
        num_cond_latents = condition["num_cond_latents"]
        q, k, v = weights.qkv.apply(tokens, strengths).unflatten(-1, (3, self.num_heads, -1)).unbind(1)
        rope = condition["rope"]
        q, k = self.rope.apply(weights.q_norm.apply(q), weights.k_norm.apply(k), tuple(part.flatten(0, 2) for part in rope))
        cond_end = num_cond_latents * frame_tokens
        attended = self.self_attention(q, k, v, frame_tokens=frame_tokens, num_cond_latents=num_cond_latents, layer_cache=layer_cache)
        positions = None
        if condition["speaker_masks"] is not None:
            if layer_cache is not None and not num_cond_latents:
                reference_k = layer_cache["reference_k"]
            else:
                reference_k = k[:frame_tokens]
                if self.attention_group is not None:
                    tile = reference_k.flatten(1).unflatten(0, (rope[0].shape[1], -1)).contiguous()
                    reference_k = self._heads(gather_spatial_shards(tile, self.attention_group, self.spatial_split).flatten(0, 1))
                if layer_cache is not None:
                    layer_cache["reference_k"] = reference_k if self.attention_group is not None else layer_cache["k"][0, :frame_tokens]
            positions = self._speaker_positions(q[cond_end:], reference_k, condition["speaker_masks"])
        return weights.out.apply(attended, strengths), positions

    def _text_attn(self, tokens, text, weights, strengths):
        q = weights.q_norm.apply(self._heads(weights.q.apply(tokens, strengths)))
        k, v = (self._heads(part) for part in weights.kv.apply(text, strengths).chunk(2, dim=-1))
        k = weights.k_norm.apply(k)
        return weights.out.apply(self.attention.apply(q, k, v))

    def _audio_attn(self, tokens, condition, positions, weights):
        audio = condition["audio"]
        frames, spatial_tokens, channels = tokens.shape
        q = weights.q_norm.apply(self._heads(weights.q.apply(tokens.flatten(0, 1)))).unflatten(0, (frames, spatial_tokens))
        k, v = weights.kv.apply(audio.flatten(0, 1)).unflatten(-1, (2, self.num_heads, -1)).unbind(1)
        k = weights.k_norm.apply(k).unflatten(0, audio.shape[:2])
        v = v.unflatten(0, audio.shape[:2])
        if positions is not None:
            frequencies, key_rope = condition["speaker_rope"]
            angles = (positions.float() * frequencies).unflatten(0, (frames, spatial_tokens))
            q = self.rope.apply_single(q, (angles.cos(), angles.sin()))
            k = self.rope.apply_single(k, key_rope)
        attended = self.attention.apply(q, k, v, *condition["audio_cu_seqlens"], spatial_tokens, audio.shape[1])
        return weights.out.apply(attended).view(frames, spatial_tokens, channels)

    def infer(self, tokens, modulations, condition, blocks, strengths, layer_caches):
        if self.use_compile:
            # Frame and spatial extents vary by segment and request. The head and channel sizes are fixed.
            for tensor in (condition["audio"], *condition["audio_cu_seqlens"]):
                torch._dynamo.maybe_mark_dynamic(tensor, 0)
            for part in condition["rope"]:
                torch._dynamo.maybe_mark_dynamic(part, (0, 1, 2))
            if condition["speaker_masks"] is not None:
                torch._dynamo.maybe_mark_dynamic(condition["speaker_masks"], 1)
        for block_idx, block in enumerate(blocks):
            layer_cache = layer_caches[block_idx] if layer_caches is not None else None
            if self.use_compile:
                torch._dynamo.maybe_mark_dynamic(tokens, (0, 1))
            tokens = self.run_block(block_idx, block, tokens, modulations[block_idx], condition, strengths, layer_cache)
        if self.use_compile and layer_caches is not None and condition["num_cond_latents"]:
            # Newly cached tensors end in [tokens, heads, head_dim].
            for cache in layer_caches:
                for tensor in cache.values():
                    torch._dynamo.maybe_mark_dynamic(tensor, tensor.ndim - 3)
        return tokens

    def infer_block(self, weights, tokens, modulation, condition, strengths, layer_cache):
        num_cond_latents = condition["num_cond_latents"]
        video, audio = modulation
        if num_cond_latents:
            video = torch.cat((video[:1].expand(num_cond_latents, -1), video[-1:].expand(tokens.shape[0] - num_cond_latents, -1)))
        shift, scale, gate, ffn_shift, ffn_scale, ffn_gate = video.chunk(6, dim=-1)
        normalized = modulate(tokens, shift, scale).flatten(0, 1)
        attended, positions = self._self_attn(normalized, condition, tokens.shape[1], weights.self_attn, strengths, layer_cache)
        tokens = (tokens + gate[:, None, :] * attended.view_as(tokens)).to(tokens.dtype)

        noisy = tokens[num_cond_latents:]
        text_input = weights.text_norm.apply(noisy.flatten(0, 1))
        noisy.add_(self._text_attn(text_input, condition["text"], weights.text_attn, strengths).view_as(noisy))
        audio_input = weights.audio_norm.apply(noisy)
        attended = self._audio_attn(audio_input, condition, positions, weights.audio_attn)
        shift, scale, gate = audio.chunk(3, dim=-1)
        tokens[num_cond_latents:] = (noisy + gate[:, None, :] * modulate(attended, shift, scale)).to(tokens.dtype)

        normalized = modulate(tokens, ffn_shift, ffn_scale).flatten(0, 1)
        hidden = F.silu(weights.ffn_gate.apply(normalized, strengths)) * weights.ffn_up.apply(normalized, strengths)
        hidden = weights.ffn_out.apply(hidden, strengths).view_as(tokens)
        return (tokens + ffn_gate[:, None, :] * hidden).to(tokens.dtype)


class AvatarPostInfer:
    def __init__(self, config, weights, attention_group, spatial_split):
        self.weights = weights
        self.patch_size = tuple(config["patch_size"])
        self.attention_group = attention_group
        self.spatial_split = spatial_split

    def infer(self, tokens, modulation, grid):
        shift, scale = modulation.chunk(2, dim=-1)
        tokens = modulate(tokens, shift, scale)
        patches = self.weights.final_out.apply(tokens.flatten(0, 1).float())
        patches = patches.view(*grid, patches.shape[-1])
        if self.attention_group is not None:
            patches = gather_spatial_shards(patches, self.attention_group, self.spatial_split)
        pt, ph, pw = self.patch_size
        return rearrange(patches, "t h w (pt ph pw c) -> 1 c (t pt) (h ph) (w pw)", pt=pt, ph=ph, pw=pw).float()
