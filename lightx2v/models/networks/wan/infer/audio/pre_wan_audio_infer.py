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
            prev_mask = inputs["previmg_encoder_output"]["prev_mask"]
            if prev_latents is not None:
                hidden_states = (1.0 - prev_mask[0]) * prev_latents + prev_mask[0] * hidden_states
        else:
            prev_latents = prev_latents.unsqueeze(0)
            prev_mask = inputs["previmg_encoder_output"]["prev_mask"]
            hidden_states = self.scheduler.latents.unsqueeze(0)
            hidden_states = torch.cat([hidden_states, prev_mask, prev_latents], dim=1)
            hidden_states = hidden_states.squeeze(0)

        x = [hidden_states]
        t = torch.stack([self.scheduler.timesteps[self.scheduler.step_index]])

        if self.config.model_cls == "wan2.2_audio":
            _, lat_f, lat_h, lat_w = self.scheduler.latents.shape
            F = (lat_f - 1) * self.config.vae_stride[0] + 1
            max_seq_len = ((F - 1) // self.config.vae_stride[0] + 1) * lat_h * lat_w // (self.config.patch_size[1] * self.config.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

            temp_ts = (prev_mask[0][0][:, ::2, ::2] * t).flatten()
            temp_ts = torch.cat([temp_ts, temp_ts.new_ones(max_seq_len - temp_ts.size(0)) * t])
            t = temp_ts.unsqueeze(0)

        audio_dit_blocks = []
        audio_encoder_output = inputs["audio_encoder_output"]
        audio_model_input = {
            "audio_input_feat": audio_encoder_output.to(hidden_states.device),
            "latent_shape": hidden_states.shape,
            "timestep": t,
        }
        audio_dit_blocks.append(inputs["audio_adapter_pipe"](**audio_model_input))
        # audio_dit_blocks = None##Debug Drop Audio

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]
        seq_len = self.scheduler.seq_len

        clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]
        ref_image_encoder = inputs["image_encoder_output"]["vae_encoder_out"].to(self.scheduler.latents.dtype)
        batch_size = len(x)
        num_channels, _, height, width = x[0].shape
        _, ref_num_channels, ref_num_frames, _, _ = ref_image_encoder.shape

        # embeddings
        x = [weights.patch_embedding.apply(u.unsqueeze(0)) for u in x]
        x_grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long).cuda()
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
        valid_patch_length = x[0].size(0)

        if ref_num_channels != num_channels:
            zero_padding = torch.zeros(
                (batch_size, num_channels - ref_num_channels, ref_num_frames, height, width),
                dtype=self.scheduler.latents.dtype,
                device=self.scheduler.latents.device,
            )
            ref_image_encoder = torch.concat([ref_image_encoder, zero_padding], dim=1)
        y = list(torch.unbind(ref_image_encoder, dim=0))  # 第一个batch维度变成list

        y = [weights.patch_embedding.apply(u.unsqueeze(0)) for u in y]
        # y_grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in y])
        y = [u.flatten(2).transpose(1, 2).squeeze(0) for u in y]
        ref_seq_lens = torch.tensor([u.size(0) for u in y], dtype=torch.long).cuda()

        x = [torch.cat([a, b], dim=0) for a, b in zip(x, y)]
        x = torch.stack(x, dim=0)
        seq_lens = seq_lens + ref_seq_lens

        ####for r2v # zero temporl component corresponding to ref embeddings
        self.freqs[x_grid_sizes[0][0] :, : x_grid_sizes[0][2]] = 0
        x_grid_sizes[:, 0] += 1

        if self.config.model_cls == "wan2.2_audio":
            bt = t.size(0)
            ref_seq_len = ref_seq_lens[0].item()
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
                del clip_fea
                torch.cuda.empty_cache()
            context = torch.concat([context_clip, context], dim=0)

        if self.clean_cuda_cache:
            if self.config.get("use_image_encoder", True):
                del context_clip
            torch.cuda.empty_cache()

        return WanPreInferModuleOutput(
            embed=embed,
            grid_sizes=x_grid_sizes,
            x=x.squeeze(0),
            embed0=embed0.squeeze(0),
            seq_lens=seq_lens,
            freqs=self.freqs,
            context=context,
            audio_dit_blocks=audio_dit_blocks,
            valid_patch_length=valid_patch_length,
        )
