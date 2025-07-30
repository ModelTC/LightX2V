import torch
from diffusers.models.embeddings import TimestepEmbedding
from .utils import rope_params, sinusoidal_embedding_1d, guidance_scale_embedding
from lightx2v.utils.envs import *


class WanPreInfer:
    def __init__(self, config):
        assert (config["dim"] % config["num_heads"]) == 0 and (config["dim"] // config["num_heads"]) % 2 == 0
        self.config = config
        d = config["dim"] // config["num_heads"]
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)
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
        self.enable_dynamic_cfg = config.get("enable_dynamic_cfg", False)
        self.cfg_scale = config.get("cfg_scale", 4.0)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, inputs, positive, kv_start=0, kv_end=0):
        x = self.scheduler.latents

        if self.scheduler.flag_df:
            t = self.scheduler.df_timesteps[self.scheduler.step_index].unsqueeze(0)
            assert t.dim() == 2  # df推理模型timestep是二维
        else:
            t = torch.stack([self.scheduler.timesteps[self.scheduler.step_index]])

        if positive:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        if self.task == "i2v":
            if self.config.get("use_image_encoder", True):
                clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]

            if self.config.get("changing_resolution", False):
                image_encoder = inputs["image_encoder_output"]["vae_encode_out"][self.scheduler.changing_resolution_index]
            else:
                image_encoder = inputs["image_encoder_output"]["vae_encode_out"]

            frame_seq_length = (image_encoder.size(2) // 2) * (image_encoder.size(3) // 2)
            if kv_end - kv_start >= frame_seq_length:  # 如果是CausalVid, image_encoder取片段
                idx_s = kv_start // frame_seq_length
                idx_e = kv_end // frame_seq_length
                image_encoder = image_encoder[:, idx_s:idx_e, :, :]
            y = image_encoder
            x = torch.cat([x, y], dim=0)

        # embeddings
        x = weights.patch_embedding.apply(x.unsqueeze(0))
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long).unsqueeze(0)
        x = x.flatten(2).transpose(1, 2).contiguous()
        seq_lens = torch.tensor(x.size(1), dtype=torch.long).cuda().unsqueeze(0)

        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        if self.enable_dynamic_cfg:
            s = torch.tensor([self.cfg_scale], dtype=torch.float32).to(x.device)
            cfg_embed = guidance_scale_embedding(s, embedding_dim=256, cfg_range=(1.0, 6.0), target_range=1000.0, dtype=torch.float32).type_as(x)
            cfg_embed = weights.cfg_cond_proj_1.apply(cfg_embed)
            cfg_embed = torch.nn.functional.silu(cfg_embed)
            cfg_embed = weights.cfg_cond_proj_2.apply(cfg_embed)
            embed = embed + cfg_embed
        if GET_DTYPE() != "BF16":
            embed = weights.time_embedding_0.apply(embed.float())
        else:
            embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)

        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))

        if self.scheduler.flag_df:
            b, f = t.shape
            assert b == len(x)  # batch_size == 1
            embed = embed.view(b, f, 1, 1, self.dim)
            embed0 = embed0.view(b, f, 1, 1, 6, self.dim)
            embed = embed.repeat(1, 1, grid_sizes[0][1], grid_sizes[0][2], 1).flatten(1, 3)
            embed0 = embed0.repeat(1, 1, grid_sizes[0][1], grid_sizes[0][2], 1, 1).flatten(1, 3)
            embed0 = embed0.transpose(1, 2).contiguous()

        # text embeddings
        stacked = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        if GET_DTYPE() != "BF16":
            out = weights.text_embedding_0.apply(stacked.squeeze(0).float())
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
            if self.clean_cuda_cache:
                torch.cuda.empty_cache()
            context_clip = weights.proj_3.apply(context_clip)
            context_clip = weights.proj_4.apply(context_clip)
            context = torch.concat([context_clip, context], dim=0)
        if self.clean_cuda_cache:
            if self.config.get("use_image_encoder", True):
                del context_clip
            torch.cuda.empty_cache()
        return (
            embed,
            grid_sizes,
            (x.squeeze(0), embed0.squeeze(0), seq_lens, self.freqs, context),
        )
