import torch

from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer

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

    def infer(self, weights, inputs, positive):
        prev_latents = inputs["previmg_encoder_output"]["prev_latents"].unsqueeze(0)
        prev_mask = inputs["previmg_encoder_output"]["prev_mask"]

        hidden_states = self.scheduler.latents.unsqueeze(0)
        hidden_states = torch.cat([hidden_states, prev_mask, prev_latents], dim=1)
        hidden_states = hidden_states.squeeze(0)

        x = [hidden_states]
        t = torch.stack([self.scheduler.timesteps[self.scheduler.step_index]])

        audio_dit_blocks = []
        audio_encoder_output = inputs["audio_encoder_output"]
        audio_model_input = {
            "audio_input_feat": audio_encoder_output.to(hidden_states.device),
            "latent_shape": hidden_states.shape,
            "timestep": t,
        }
        audio_dit_blocks.append(inputs["audio_adapter_pipe"](**audio_model_input))
        ##audio_dit_blocks = None##Debug Drop Audio

        if positive:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]
        seq_len = self.scheduler.seq_len

        clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]
        ref_image_encoder = inputs["image_encoder_output"]["vae_encoder_out"]
        batch_size = len(x)
        num_channels, num_frames, height, width = x[0].shape
        _, ref_num_channels, ref_num_frames, _, _ = ref_image_encoder.shape

        if ref_num_channels != num_channels:
            zero_padding = torch.zeros(
                (batch_size, num_channels - ref_num_channels, ref_num_frames, height, width),
                dtype=self.scheduler.latents.dtype,
                device=self.scheduler.latents.device,
            )
            ref_image_encoder = torch.concat([ref_image_encoder, zero_padding], dim=1)
        y = list(torch.unbind(ref_image_encoder, dim=0))  # 第一个batch维度变成list
        # embeddings
        x = [weights.patch_embedding.apply(u.unsqueeze(0)) for u in x]
        x_grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long).cuda()
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
        valid_patch_length = x[0].size(0)
        y = [weights.patch_embedding.apply(u.unsqueeze(0)) for u in y]
        y_grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in y])
        y = [u.flatten(2).transpose(1, 2).squeeze(0) for u in y]

        x = [torch.cat([a, b], dim=0) for a, b in zip(x, y)]
        x = torch.stack(x, dim=0)

        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)

        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))

        # text embeddings
        stacked = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        out = weights.text_embedding_0.apply(stacked.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)

        if self.task == "i2v" and self.config.get("use_image_encoder", True):
            context_clip = weights.proj_0.apply(clip_fea)
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

        return (embed, x_grid_sizes, (x.squeeze(0), embed0.squeeze(0), seq_lens, self.freqs, context, audio_dit_blocks), valid_patch_length)
