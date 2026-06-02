from copy import deepcopy

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.module_io import GridOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.s2v.casual_audio import apply_casual_audio_encoder, mm_weight_fp32
from lightx2v.models.networks.wan.infer.s2v.framepack import inject_motion_tokens
from lightx2v.models.networks.wan.infer.s2v.module_io import WanS2VPreInferModuleOutput
from lightx2v.models.networks.wan.infer.s2v.rope import rope_precompute
from lightx2v_platform.base.global_var import AI_DEVICE


class WanS2VPreInfer(WanPreInfer):
    def __init__(self, config):
        super().__init__(config)
        self.text_len = config["text_len"]
        self.add_last_motion = config.get("add_last_motion", True)
        self.zero_timestep = config.get("zero_timestep", True)
        # Wan model.py builds RoPE table in float64; base WanPreInfer uses float32.
        head_size = config["dim"] // config["num_heads"]
        self.freqs = torch.cat(
            [
                self.rope_params(1024, head_size - 4 * (head_size // 6)),
                self.rope_params(1024, 2 * (head_size // 6)),
                self.rope_params(1024, 2 * (head_size // 6)),
            ],
            dim=1,
        ).to(device=AI_DEVICE)

    @staticmethod
    def rope_params(max_seq_len, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(
            torch.arange(max_seq_len),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float64).div(dim)),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    @staticmethod
    def sinusoidal_embedding_1d_wan(dim, position):
        assert dim % 2 == 0
        half = dim // 2
        position = position.type(torch.float64)
        sinusoid = torch.outer(
            position,
            torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float64).div(half)),
        )
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).float()

    def _compute_time_embed(self, weights, t):
        with torch.amp.autocast("cuda", dtype=torch.float32):
            sin = self.sinusoidal_embedding_1d_wan(self.freq_dim, t)
            embed = mm_weight_fp32(weights.time_embedding_0, sin)
            embed = F.silu(embed)
            embed = mm_weight_fp32(weights.time_embedding_2, embed)
            embed0 = mm_weight_fp32(weights.time_projection_1, F.silu(embed)).unflatten(1, (6, self.dim))
        return embed, embed0

    @torch.no_grad()
    def infer(self, weights, inputs):
        s2v = inputs["s2v"]
        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        x = self.scheduler.latents
        t = self.scheduler.timestep_input
        ref_latents = s2v["ref_latents"]
        motion_latents = s2v["motion_latents"]
        cond_states = s2v["cond_latents"]
        audio_input = s2v["audio_input"]

        if not self.scheduler.infer_condition:
            audio_input = 0.0 * audio_input
        motion_frames = s2v["motion_frames"]
        drop_motion_frames = s2v["drop_motion_frames"]
        add_last_motion = self.add_last_motion * s2v.get("add_last_motion", True)

        mf0, mf1 = motion_frames
        audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, mf0), audio_input], dim=-1)

        audio_emb_res = apply_casual_audio_encoder(
            weights.casual_audio_encoder,
            audio_input,
            self.config.get("num_audio_token", 4),
            self.config.get("enable_adain", False),
            audio_dim=self.config.get("audio_dim", 1024),
            out_dim=self.config["dim"],
        )

        if self.config.get("enable_adain", False):
            audio_emb_global, audio_emb = audio_emb_res
            audio_emb_global = audio_emb_global[:, mf1:].clone()
        else:
            audio_emb = audio_emb_res
            audio_emb_global = None
        merged_audio_emb = audio_emb[:, mf1:, :]

        x = weights.patch_embedding.apply(x.unsqueeze(0))
        cond_input = cond_states if cond_states.dim() == 5 else cond_states.unsqueeze(0)
        cond = weights.cond_encoder.apply(cond_input)
        x = x + cond
        grid_sizes = torch.stack([torch.tensor(x.shape[2:], dtype=torch.long, device=x.device)])
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([x.size(1)], dtype=torch.long, device=x.device)
        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes_list = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

        ref_input = ref_latents if ref_latents.dim() == 5 else ref_latents.unsqueeze(0)
        ref = weights.patch_embedding.apply(ref_input)
        batch_size = ref.size(0)
        height, width = ref.shape[3], ref.shape[4]
        ref_grid_sizes = [
            [
                torch.tensor([30, 0, 0], device=x.device).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([31, height, width], device=x.device).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([1, height, width], device=x.device).unsqueeze(0).repeat(batch_size, 1),
            ]
        ]
        ref = ref.flatten(2).transpose(1, 2)
        original_seq_len = seq_lens[0].item()
        seq_lens = seq_lens + torch.tensor([ref.size(1)], dtype=torch.long, device=x.device)
        grid_sizes_list = grid_sizes_list + ref_grid_sizes
        x = torch.cat([x, ref], dim=1)

        mask_input = [torch.zeros([1, x.shape[1]], dtype=torch.long, device=AI_DEVICE)]
        mask_input[0][:, original_seq_len:] = 1

        b, s, n, d = x.size(0), x.size(1), self.config["num_heads"], self.dim // self.config["num_heads"]
        pre_compute_freqs = rope_precompute(x.detach().view(b, s, n, d), grid_sizes_list, self.freqs, start=None)
        x_list = [x]
        pre_compute_freqs = [pre_compute_freqs]

        frame_packer = getattr(weights, "frame_packer", None)

        x_list, seq_lens, pre_compute_freqs, mask_input = inject_motion_tokens(
            x_list,
            seq_lens,
            pre_compute_freqs,
            mask_input,
            motion_latents,
            frame_packer,
            self.freqs,
            self.config,
            drop_motion_frames,
            add_last_motion,
        )

        x = torch.cat(x_list, dim=0)
        pre_compute_freqs = torch.cat(pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)
        x = x + weights.trainable_cond_mask.apply(mask_input).to(x.dtype)

        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
        embed, embed0 = self._compute_time_embed(weights, t)

        if self.zero_timestep:
            embed = embed[:-1]
            zero_e0 = embed0[-1:]
            embed0 = embed0[:-1]
            embed0 = torch.cat([embed0.unsqueeze(2), zero_e0.unsqueeze(2).repeat(embed0.size(0), 1, 1, 1)], dim=2)
            embed0 = [embed0, original_seq_len]
        else:
            embed0 = embed0.unsqueeze(2).repeat(1, 1, 2, 1)
            embed0 = [embed0, 0]

        u = context.squeeze(0)
        if self.sensitive_layer_dtype != self.infer_dtype:
            u = u.to(self.infer_dtype)
        # Wan text_embedding runs Linear0 on the full padded [text_len, text_dim] tensor.
        u_padded = torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        out = weights.text_embedding_0.apply(u_padded)
        out = F.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out).unsqueeze(0)

        return WanS2VPreInferModuleOutput(
            embed=embed,
            embed0=embed0,
            x=x,
            context=context,
            grid_sizes=GridOutput(tensor=original_grid_sizes, tuple=tuple(original_grid_sizes[0].tolist())),
            freqs=pre_compute_freqs,
            seq_lens=seq_lens,
            original_seq_len=original_seq_len,
            merged_audio_emb=merged_audio_emb,
            audio_emb_global=audio_emb_global,
            s2v_extra={"grid_sizes": grid_sizes_list, "context_lens": None},
        )
