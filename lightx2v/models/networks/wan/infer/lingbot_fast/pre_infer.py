import torch

from lightx2v.models.networks.wan.infer.lingbot.pre_infer import WanLingbotPreInfer
from lightx2v.models.networks.wan.infer.module_io import GridOutput
from lightx2v.models.networks.wan.infer.self_forcing.pre_infer import (
    WanSFPreInferModuleOutput,
    rope_params,
    sinusoidal_embedding_1d,
)
from lightx2v_platform.base.global_var import AI_DEVICE


class WanLingbotFastPreInfer(WanLingbotPreInfer):
    """Fast (autoregressive) pre-infer: inherits lingbot camera handling, adds SF scheduling."""

    def __init__(self, config):
        super().__init__(config)
        d = config["dim"] // config["num_heads"]
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        ).to(AI_DEVICE)

    def time_embedding(self, weights, embed):
        embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        return embed

    def time_projection(self, weights, embed):
        embed0 = torch.nn.functional.silu(embed)
        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))
        return embed0

    def _build_lingbot_conditional_dict(self, weights, inputs, x_tokens: torch.Tensor) -> dict:
        image_encoder_output = inputs.get("image_encoder_output") or {}
        dit_cond_dict = image_encoder_output.get("dit_cond_dict") or {}
        c2ws_plucker_emb = dit_cond_dict.get("c2ws_plucker_emb", None)
        if c2ws_plucker_emb is None:
            return {}
        if isinstance(c2ws_plucker_emb, (list, tuple)):
            if len(c2ws_plucker_emb) == 0:
                return {}
            c2ws_plucker_emb = c2ws_plucker_emb[0]
        if c2ws_plucker_emb.dim() == 4:
            c2ws_plucker_emb = c2ws_plucker_emb.unsqueeze(0)
        if c2ws_plucker_emb.dim() != 5:
            return {}

        seg_start = self.scheduler.seg_index * self.scheduler.num_frame_per_chunk
        seg_end = min(
            (self.scheduler.seg_index + 1) * self.scheduler.num_frame_per_chunk,
            self.scheduler.num_output_frames,
        )
        sliced = c2ws_plucker_emb[:, :, seg_start:seg_end, :, :]

        original = dit_cond_dict["c2ws_plucker_emb"]
        dit_cond_dict["c2ws_plucker_emb"] = sliced
        result = super()._build_lingbot_conditional_dict(weights, inputs, x_tokens)
        dit_cond_dict["c2ws_plucker_emb"] = original
        return result

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        x = self.scheduler.latents_input
        t = self.scheduler.timestep_input

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        image_encoder_output = inputs.get("image_encoder_output") or {}
        vae_encoder_out = image_encoder_output.get("vae_encoder_out", None)

        if vae_encoder_out is not None:
            seg_start = self.scheduler.seg_index * self.scheduler.num_frame_per_chunk
            seg_end = min(
                (self.scheduler.seg_index + 1) * self.scheduler.num_frame_per_chunk,
                self.scheduler.num_output_frames,
            )
            vae_chunk = vae_encoder_out[:, seg_start:seg_end]
            x = torch.cat([x, vae_chunk], dim=0)

        x = weights.patch_embedding.apply(x.unsqueeze(0))
        grid_sizes_t, grid_sizes_h, grid_sizes_w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        seq_lens = torch.tensor(x.size(1), dtype=torch.int32, device=x.device).unsqueeze(0)

        embed_tmp = sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        embed = self.time_embedding(weights, embed_tmp)
        embed0 = self.time_projection(weights, embed)

        if self.sensitive_layer_dtype != self.infer_dtype:
            out = weights.text_embedding_0.apply(context.squeeze(0).to(self.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(context.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)
        if self.clean_cuda_cache:
            del out
            torch.cuda.empty_cache()

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        grid_sizes = GridOutput(
            tensor=torch.tensor(
                [[grid_sizes_t, grid_sizes_h, grid_sizes_w]],
                dtype=torch.int32,
                device=x.device,
            ),
            tuple=(grid_sizes_t, grid_sizes_h, grid_sizes_w),
        )

        if self.cos_sin is None or self.grid_sizes != grid_sizes.tuple:
            freqs = self.freqs.clone()
            self.grid_sizes = grid_sizes.tuple
            self.cos_sin = self.prepare_cos_sin(grid_sizes.tuple, freqs)

        result = WanSFPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x.squeeze(0),
            embed0=embed0.squeeze(0),
            seq_lens=seq_lens,
            freqs=self.freqs,
            context=context,
            cos_sin=self.cos_sin,
        )

        result.conditional_dict = self._build_lingbot_conditional_dict(weights, inputs, result.x)
        # print(result.conditional_dict)
        # exit()

        return result
