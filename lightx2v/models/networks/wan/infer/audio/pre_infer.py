import math

import torch

from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.utils.envs import *

from ..module_io import WanPreInferModuleOutput
from ..utils import rope_params, sinusoidal_embedding_1d


class WanAudioPreInfer(WanPreInfer):
    def __init__(self, config):
        assert (config["dim"] % config["num_heads"]) == 0 and (config["dim"] // config["num_heads"]) % 2 == 0
        d = config["dim"] // config["num_heads"]
        self.config = config
        self.task = config["task"]
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        ).cuda()
        self.freq_dim = config["freq_dim"]
        self.dim = config["dim"]
        self.text_len = config["text_len"]
        self.rope_t_dim = d // 2 - 2 * (d // 6)
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

        if config.parallel:
            self.sp_size = config.parallel.get("seq_p_size", 1)
        else:
            self.sp_size = 1

    def infer(self, weights, inputs):
        prev_latents = inputs["previmg_encoder_output"]["prev_latents"]
        if self.config.model_cls == "wan2.2_audio":
            hidden_states = self.scheduler.latents
        else:
            prev_latents = prev_latents.unsqueeze(0)
            prev_mask = inputs["previmg_encoder_output"]["prev_mask"]
            hidden_states = self.scheduler.latents.unsqueeze(0)
            hidden_states = torch.cat([hidden_states, prev_mask, prev_latents], dim=1)
            hidden_states = hidden_states.squeeze(0)

        x = hidden_states
        t = torch.stack([self.scheduler.timesteps[self.scheduler.step_index]])

        if self.config.model_cls == "wan2.2_audio":
            _, lat_f, lat_h, lat_w = self.scheduler.latents.shape
            F = (lat_f - 1) * self.config.vae_stride[0] + 1
            max_seq_len = ((F - 1) // self.config.vae_stride[0] + 1) * lat_h * lat_w // (self.config.patch_size[1] * self.config.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
            temp_ts = (self.scheduler.mask[0][:, ::2, ::2] * t).flatten()
            temp_ts = torch.cat([temp_ts, temp_ts.new_ones(max_seq_len - temp_ts.size(0)) * t])
            t = temp_ts.unsqueeze(0)

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]
        ref_image_encoder = inputs["image_encoder_output"]["vae_encoder_out"].to(self.scheduler.latents.dtype)

        num_channels, _, height, width = x.shape
        ref_num_channels, ref_num_frames, _, _ = ref_image_encoder.shape

        if ref_num_channels != num_channels:
            zero_padding = torch.zeros(
                (num_channels - ref_num_channels, ref_num_frames, height, width),
                dtype=self.scheduler.latents.dtype,
                device=self.scheduler.latents.device,
            )
            ref_image_encoder = torch.concat([ref_image_encoder, zero_padding], dim=0)
        y = ref_image_encoder

        # embeddings
        x = weights.patch_embedding.apply(x.unsqueeze(0))
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long).unsqueeze(0)
        x = x.flatten(2).transpose(1, 2).contiguous()
        seq_lens = torch.tensor(x.size(1), dtype=torch.long).cuda().unsqueeze(0)

        y = weights.patch_embedding.apply(y.unsqueeze(0))
        y = y.flatten(2).transpose(1, 2).contiguous()
        ref_seq_lens = torch.tensor(y.size(1), dtype=torch.long).cuda().unsqueeze(0)

        x = torch.cat([x, y], dim=1).squeeze(0)

        ####for r2v # zero temporl component corresponding to ref embeddings
        self.freqs[grid_sizes[0][0] :, : self.rope_t_dim] = 0
        grid_sizes[:, 0] += 1

        if self.config.model_cls == "wan2.2_audio":
            ref_seq_len = ref_seq_lens.item()
            t = torch.cat(
                [
                    t,
                    torch.zeros(
                        (1, ref_seq_len),
                        dtype=t.dtype,
                        device=t.device,
                    ),
                ],
                dim=1,
            )

        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        if self.sensitive_layer_dtype != self.infer_dtype:
            embed = weights.time_embedding_0.apply(embed.to(self.sensitive_layer_dtype))
        else:
            embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)

        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)
        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))

        # text embeddings
        stacked = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        if self.sensitive_layer_dtype != self.infer_dtype:
            out = weights.text_embedding_0.apply(stacked.squeeze(0).to(self.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(stacked.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)
        if self.clean_cuda_cache:
            del out, stacked
            torch.cuda.empty_cache()

        if self.task == "i2v" and self.config.get("use_image_encoder", True):
            context_clip = weights.proj_0.apply(clip_fea)
            if self.clean_cuda_cache:
                del clip_fea
                torch.cuda.empty_cache()
            context_clip = weights.proj_1.apply(context_clip)
            context_clip = torch.nn.functional.gelu(context_clip, approximate="none")
            context_clip = weights.proj_3.apply(context_clip)
            context_clip = weights.proj_4.apply(context_clip)
            if self.clean_cuda_cache:
                torch.cuda.empty_cache()
            context = torch.concat([context_clip, context], dim=0)

        if self.clean_cuda_cache:
            if self.config.get("use_image_encoder", True):
                del context_clip
            torch.cuda.empty_cache()

        return WanPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x,
            embed0=embed0.squeeze(0),
            seq_lens=seq_lens,
            freqs=self.freqs,
            context=context,
            adapter_output={"audio_encoder_output": inputs["audio_encoder_output"]},
        )
