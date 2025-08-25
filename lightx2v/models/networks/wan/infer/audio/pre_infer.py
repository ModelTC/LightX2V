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
        prev_mask = inputs["previmg_encoder_output"]["prev_mask"]
        hidden_states = torch.cat([self.scheduler.latents, prev_mask, prev_latents], dim=0)

        x = hidden_states
        t = self.scheduler.timestep_input

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]
        # seq_len = self.scheduler.seq_len

        clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]
        ref_image_encoder = inputs["image_encoder_output"]["vae_encoder_out"].to(self.scheduler.latents.dtype)
        # batch_size = len(x)
        num_channels, _, height, width = x.shape
        ref_num_channels, ref_num_frames, _, _ = ref_image_encoder.shape

        if ref_num_channels != num_channels:
            zero_padding = torch.zeros(
                (num_channels - ref_num_channels, ref_num_frames, height, width),
                dtype=self.scheduler.latents.dtype,
                device=self.scheduler.latents.device,
            )
            ref_image_encoder = torch.concat([ref_image_encoder, zero_padding], dim=0)
        y = ref_image_encoder  # 第一个batch维度变成list

        # embeddings
        x = weights.patch_embedding.apply(x.unsqueeze(0))
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long).unsqueeze(0)
        x = x.flatten(2).transpose(1, 2).contiguous()
        seq_lens = torch.tensor(x.size(1), dtype=torch.long).cuda().unsqueeze(0)

        y = weights.patch_embedding.apply(y.unsqueeze(0))
        y = y.flatten(2).transpose(1, 2).contiguous()

        x = torch.cat([x, y], dim=1)

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
            grid_sizes=grid_sizes,
            x=x.squeeze(0),
            embed0=embed0.squeeze(0),
            seq_lens=seq_lens,
            freqs=self.freqs,
            context=context,
            adapter_output={"audio_encoder_output": inputs["audio_encoder_output"]},
        )
