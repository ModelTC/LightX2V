import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer


class TBDubPreInfer(WanPreInfer):
    def rope_params(self, max_seq_len, dim, theta=10000):
        # TBDub forms both frequencies and complex rotations in float64.
        frequencies = torch.outer(torch.arange(max_seq_len, dtype=torch.float64), 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)))
        return torch.polar(torch.ones_like(frequencies), frequencies)

    def prepare_cos_sin(self, grid_sizes, freqs):
        frames, height, width = grid_sizes
        half = self.head_size // 2
        temporal, vertical, horizontal = freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
        sequences = []
        for row_offset, col_offset in ((height, width), (0, 0)):
            sequences.append(
                torch.cat(
                    [
                        temporal[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                        vertical[row_offset : row_offset + height].view(1, height, 1, -1).expand(frames, height, width, -1),
                        horizontal[col_offset : col_offset + width].view(1, 1, width, -1).expand(frames, height, width, -1),
                    ],
                    dim=-1,
                ).reshape(frames * height * width, 1, -1)
            )
        result = torch.cat(sequences)
        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            rank = dist.get_rank(self.seq_p_group)
            padding = (-result.shape[0]) % world_size
            if padding:
                result = F.pad(result, (0, 0, 0, 0, 0, padding), value=1)
            result = result.chunk(world_size, dim=0)[rank]
        return result

    def infer_audio(self, weights, audio_feats):
        frames, window, channels = audio_feats.shape
        x = weights.audio_proj_in.apply(audio_feats.reshape(-1, channels).to(self.sensitive_layer_dtype))
        dim = x.shape[-1]
        heads = self.config.get("audio_projection_heads", 24)
        head_dim = dim // heads
        cu = torch.arange(frames + 1, device=x.device, dtype=torch.int32) * window
        for layer in weights.audio_layers:
            h = layer.norm1.apply(x).reshape(frames, window, dim)
            h = (h + layer.position.tensor[:, :window]).reshape(-1, dim).to(self.sensitive_layer_dtype)
            q = layer.q.apply(h).reshape(frames, window, heads, head_dim)
            k = layer.k.apply(h).reshape(frames, window, heads, head_dim)
            v = layer.v.apply(h).reshape(frames, window, heads, head_dim)
            h = layer.attn.apply(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=window, max_seqlen_kv=window)
            x = x + layer.out.apply(h.reshape(-1, dim))
            h = layer.ff_in.apply(layer.norm2.apply(x).to(self.sensitive_layer_dtype))
            value, gate = h.chunk(2, dim=-1)
            x = x + layer.ff_out.apply(value * F.gelu(gate))
        return weights.audio_proj_out.apply(x).reshape(frames, window, -1).to(self.infer_dtype)

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        latents = self.scheduler.latents
        reference = inputs["ref_latents"]
        if reference.shape != latents.shape or latents.ndim != 4:
            raise ValueError("TBDub expects matching unbatched [C,F,H,W] reference and target latents")
        x = weights.patch_embedding.apply(torch.cat((reference, latents), dim=1).unsqueeze(0))
        total_frames, height, width = x.shape[2:]
        frames = total_frames // 2
        target_tokens = frames * height * width
        x = x.flatten(2).transpose(1, 2).squeeze(0).contiguous()

        timestep = self.scheduler.timestep_input.to(device=x.device, dtype=latents.dtype).reshape(1)
        positions = timestep.expand(frames, height * width).clone()
        motion = inputs.get("motion_latents")
        if motion is not None:
            positions[: motion.shape[1]] = 0
        positions = torch.cat((torch.zeros_like(positions), positions)).flatten()
        frequencies = 10000 ** (-torch.arange(self.freq_dim // 2, device=x.device, dtype=torch.float64) / (self.freq_dim // 2))
        sinusoid = torch.outer(positions.double(), frequencies)
        embed = torch.cat((sinusoid.cos(), sinusoid.sin()), dim=-1).to(self.sensitive_layer_dtype)
        embed = weights.time_embedding_2.apply(F.silu(weights.time_embedding_0.apply(embed)))
        embed0 = weights.time_projection_1.apply(F.silu(embed)).unflatten(1, (6, self.dim))
        context = inputs["text_encoder_output"]["context"].reshape(-1, self.config["text_dim"]).to(self.sensitive_layer_dtype)
        context = weights.text_embedding_2.apply(F.gelu(weights.text_embedding_0.apply(context), approximate="tanh"))
        audio = self.infer_audio(weights, inputs["audio_feats"])
        if audio.shape[0] != frames:
            raise ValueError(f"TBDub audio frame count {audio.shape[0]} does not match latent frames {frames}")

        grid = (frames, height, width)
        if self.cos_sin is None or self.grid_sizes != grid:
            self.grid_sizes = grid
            self.cos_sin = self.prepare_rope_cache(self.prepare_cos_sin(grid, self.freqs))
        return WanPreInferModuleOutput(
            embed=embed,
            embed0=embed0,
            x=x,
            context=context,
            grid_sizes=GridOutput(tensor=torch.tensor([grid], device=x.device, dtype=torch.int32), tuple=grid),
            cos_sin=self.cos_sin,
            rope_positions=self.rope_positions,
            valid_token_len=2 * target_tokens,
            adapter_args={"audio_embedding": audio, "target_tokens": target_tokens},
        )
