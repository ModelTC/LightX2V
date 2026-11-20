import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from lightx2v.models.networks.wan.infer.infinitetalk.audio_windows import pack_later_audio_windows
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, ROPE_REGISTER

# RoPE position assigned by the release to the shared silent bystander stream.
SILENT_BACKGROUND_POSITION = 100


def modulate(tokens, shift, scale):
    normalized = F.layer_norm(tokens.float(), (tokens.shape[-1],), eps=1e-6)
    return (normalized * (1 + scale[:, None, :]) + shift[:, None, :]).to(tokens.dtype)


def gather_width_shards(tile, group):
    """Gather contiguous [..., width, channels] tiles in rank order."""
    shards = [torch.empty_like(tile) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, tile, group=group)
    return torch.cat(shards, dim=-2)


def _speaker_region_scores(q, k, masks):
    # Keep the reference reduction layout and the fused softmax. Early BF16
    # rounding can change which speaker wins a near tie.
    q = (q * q.shape[-1] ** -0.5).transpose(1, 2)
    k = k.transpose(1, 2)
    probabilities = (q @ k.transpose(-2, -1)).softmax(-1)
    regions = [((probabilities * mask[None, None, None]).sum(-1) / mask[None, None, None].sum()).permute(0, 2, 1).mean(-1) for mask in masks]
    return torch.cat(regions)


# Upstream specializes the first segment, then compiles a variable query length
# for continuation. Select each kernel explicitly so HTTP request order is immaterial.
_first_region_scores = torch.compile(_speaker_region_scores, dynamic=False)
_continuation_region_scores = torch.compile(_speaker_region_scores)


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

    def _rope(self, grid, width_start, width_end, device, continuation):
        spatial_dim = 2 * (self.head_dim // 6)
        dims = (self.head_dim - 2 * spatial_dim, spatial_dim, spatial_dim)
        # The continuation anchor uses temporal position 10. The segment starts at 0.
        time = torch.cat((torch.tensor([10]), torch.arange(grid[0] - 1))) if continuation else torch.arange(grid[0])
        axes = (time, torch.arange(grid[1]), torch.arange(width_start, width_end))
        phase = []
        for axis, dim in zip(axes, dims):
            frequency = 1 / 10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            phase.append(torch.outer(axis.float(), frequency))
        t, h, w = grid[0], grid[1], width_end - width_start
        angles = (
            torch.cat(
                (phase[0][:, None, None].expand(t, h, w, -1), phase[1][None, :, None].expand(t, h, w, -1), phase[2][None, None, :].expand(t, h, w, -1)),
                dim=-1,
            )
            .flatten(0, 2)
            .to(device)
        )
        return angles.cos(), angles.sin()

    def project_text(self, text):
        weights = self.weights
        text = weights.text_out.apply(F.gelu(weights.text_in.apply(text), approximate="tanh"))
        # The released DiT attends to projected text plus zero-valued padding.
        return F.pad(text, (0, 0, 0, self.text_length - text.shape[0]))

    def prepare(self, projected_text, audio, speaker_masks, latent_shape, num_cond_latents, width_shards, rank, rope=None):
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

        grid = tuple(latent_size // patch for latent_size, patch in zip(latent_shape[-3:], self.patch_size))
        width = grid[2] // width_shards
        start, end = rank * width, (rank + 1) * width
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
            "audio_cu_seqlens": (offsets * (grid[1] * width), offsets * audio.shape[1]),
            "grid": grid,
            "num_cond_latents": num_cond_latents,
            "width_slice": slice(start * self.patch_size[2], end * self.patch_size[2]),
            "rope": rope if rope is not None else self._rope(grid, start, end, audio.device, num_cond_latents > 1),
        }

    def infer(self, latents, timesteps, width_slice):
        weights = self.weights
        tokens = weights.patch.apply(latents[..., width_slice].to(weights.patch.weight.dtype))[0]
        tokens = rearrange(tokens, "c t h w -> t (h w) c").contiguous()
        phase = timesteps.float()[:, None] * self.time_frequencies
        frequencies = torch.cat((phase.cos(), phase.sin()), dim=-1)
        time = weights.time_out.apply(F.silu(weights.time_in.apply(frequencies)))
        return tokens, F.silu(time)


class AvatarTransformerInfer:
    def __init__(self, config, attention_group):
        self.num_heads = config["num_heads"]
        self.class_interval = config["class_interval"]
        self.class_range = config["class_range"]
        self.attention = ATTN_WEIGHT_REGISTER[config["attn_type"]]()
        self.ulysses = ATTN_WEIGHT_REGISTER[config["parallel"].get("seq_p_attn_type", "ulysses")]() if attention_group is not None else None
        self.attention_group = attention_group
        self.rope = ROPE_REGISTER["torch_real_rope"](layout="interleaved", compute_dtype=torch.float32)

    def _heads(self, projections):
        return projections.unflatten(-1, (self.num_heads, -1))

    def _attend(self, q, k, v):
        if self.ulysses is None:
            return self.attention.apply(q, k, v)
        attended, _ = self.ulysses.apply(q, k, v, attention_module=self.attention, seq_p_group=self.attention_group)
        return attended

    def _speaker_positions(self, q, reference_k, masks, grid_height, continuation):
        if self.attention_group is not None:
            tile = reference_k.transpose(1, 2).unflatten(2, (grid_height, -1)).contiguous()
            reference_k = gather_width_shards(tile, self.attention_group).flatten(2, 3).transpose(1, 2)

        # The release splits heads in two and reduces all noisy queries at their
        # native shape. Changing that shape can flip a near-tied speaker assignment.
        groups = []
        for qh, kh in zip(q.chunk(2, dim=2), reference_k.chunk(2, dim=2)):
            if continuation:
                torch._dynamo.mark_dynamic(qh, 1)
                groups.append(_continuation_region_scores(qh, kh, masks))
            else:
                groups.append(_first_region_scores(qh, kh, masks))
        scores = ((groups[0] + groups[1]) / 2).T
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

    def _self_attn(self, tokens, condition, frame_tokens, weights, strengths):
        num_cond_latents = condition["num_cond_latents"]
        q, k, v = weights.qkv.apply(tokens, strengths).unflatten(-1, (3, self.num_heads, -1)).unbind(1)
        q, k = self.rope.apply(weights.q_norm.apply(q), weights.k_norm.apply(k), condition["rope"])
        cond_end = num_cond_latents * frame_tokens
        # Ulysses exchanges equal-sized Q and KV shards.
        if self.ulysses is None:
            noisy = self.attention.apply(q[cond_end:], k, v)
        else:
            noisy = self._attend(q, k, v)[cond_end:]
        if num_cond_latents:
            groups = [self._attend(q[:frame_tokens], k[:frame_tokens], v[:frame_tokens])]
            if num_cond_latents > 1:
                # Overlap frames attend to each other, independently of the anchor.
                history = slice(frame_tokens, cond_end)
                groups.append(self._attend(q[history], k[history], v[history]))
            noisy = torch.cat((*groups, noisy))
        positions = None
        if condition["speaker_masks"] is not None:
            positions = self._speaker_positions(q[None, cond_end:], k[None, :frame_tokens], condition["speaker_masks"], condition["grid"][1], num_cond_latents > 1)
        return weights.out.apply(noisy, strengths), positions

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

    def infer(self, tokens, time, condition, weights, strengths):
        num_cond_latents = condition["num_cond_latents"]
        shift, scale, gate, ffn_shift, ffn_scale, ffn_gate = weights.modulation.apply(time).chunk(6, dim=-1)
        normalized = modulate(tokens, shift, scale).flatten(0, 1)
        attended, positions = self._self_attn(normalized, condition, tokens.shape[1], weights.self_attn, strengths)
        tokens = (tokens + gate[:, None, :] * attended.view_as(tokens)).to(tokens.dtype)

        noisy = tokens[num_cond_latents:]
        text_input = weights.text_norm.apply(noisy.flatten(0, 1))
        noisy.add_(self._text_attn(text_input, condition["text"], weights.text_attn, strengths).view_as(noisy))
        audio_input = weights.audio_norm.apply(noisy)
        attended = self._audio_attn(audio_input, condition, positions, weights.audio_attn)
        shift, scale, gate = weights.audio_modulation.apply(time[num_cond_latents:]).chunk(3, dim=-1)
        tokens[num_cond_latents:] = (noisy + gate[:, None, :] * modulate(attended, shift, scale)).to(tokens.dtype)

        normalized = modulate(tokens, ffn_shift, ffn_scale).flatten(0, 1)
        hidden = F.silu(weights.ffn_gate.apply(normalized, strengths)) * weights.ffn_up.apply(normalized, strengths)
        hidden = weights.ffn_out.apply(hidden, strengths).view_as(tokens)
        return (tokens + ffn_gate[:, None, :] * hidden).to(tokens.dtype)


class AvatarPostInfer:
    def __init__(self, config, weights, attention_group):
        self.weights = weights
        self.patch_size = tuple(config["patch_size"])
        self.attention_group = attention_group

    def infer(self, tokens, time, grid):
        shift, scale = self.weights.final_modulation.apply(time).chunk(2, dim=-1)
        tokens = modulate(tokens, shift, scale)
        patches = self.weights.final_out.apply(tokens.flatten(0, 1).float())
        patches = patches.view(grid[0], grid[1], -1, patches.shape[-1])
        if self.attention_group is not None:
            patches = gather_width_shards(patches, self.attention_group)
        pt, ph, pw = self.patch_size
        return rearrange(patches, "t h w (pt ph pw c) -> 1 c (t pt) (h ph) (w pw)", pt=pt, ph=ph, pw=pw).float()
