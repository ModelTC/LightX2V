import torch

from lightx2v.utils.envs import *

from .module_io import GridOutput, WanPreInferModuleOutput
from .utils import guidance_scale_embedding, rope_params, sinusoidal_embedding_1d


class WanPreInfer:
    def __init__(self, config):
        assert (config["dim"] % config["num_heads"]) == 0 and (config["dim"] // config["num_heads"]) % 2 == 0
        self.config = config
        d = config["dim"] // config["num_heads"]
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)
        self.task = config["task"]
        self.device = torch.device(self.config.get("run_device", "cuda"))
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        ).to(self.device)
        self.freq_dim = config["freq_dim"]
        self.dim = config["dim"]
        self.enable_dynamic_cfg = config.get("enable_dynamic_cfg", False)
        self.cfg_scale = config.get("cfg_scale", 4.0)
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        x = self.scheduler.latents
        t = self.scheduler.timestep_input

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        if self.task in ["i2v", "flf2v", "animate", "s2v"]:
            if self.config.get("use_image_encoder", True):
                clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]

            if self.config.get("changing_resolution", False):
                image_encoder = inputs["image_encoder_output"]["vae_encoder_out"][self.scheduler.changing_resolution_index]
            else:
                image_encoder = inputs["image_encoder_output"]["vae_encoder_out"]

            if image_encoder is not None:
                frame_seq_length = (image_encoder.size(2) // 2) * (image_encoder.size(3) // 2)
                if kv_end - kv_start >= frame_seq_length:  # 如果是CausalVid, image_encoder取片段
                    idx_s = kv_start // frame_seq_length
                    idx_e = kv_end // frame_seq_length
                    image_encoder = image_encoder[:, idx_s:idx_e, :, :]
                y = image_encoder
                x = torch.cat([x, y], dim=0)

        # embeddings
        x = weights.patch_embedding.apply(x.unsqueeze(0))

        if hasattr(self, "after_patch_embedding"):
            x, motion_vec = self.after_patch_embedding(weights, x, inputs["image_encoder_output"]["pose_latents"], inputs["image_encoder_output"]["face_pixel_values"])
        else:
            motion_vec = None

        grid_sizes_t, grid_sizes_h, grid_sizes_w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        seq_lens = torch.tensor(x.size(1), dtype=torch.int32, device=x.device).unsqueeze(0)

        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        if self.enable_dynamic_cfg:
            s = torch.tensor([self.cfg_scale], dtype=torch.float32, device=x.device)
            cfg_embed = guidance_scale_embedding(s, embedding_dim=256, cfg_range=(1.0, 6.0), target_range=1000.0, dtype=torch.float32).type_as(x)
            cfg_embed = weights.cfg_cond_proj_1.apply(cfg_embed)
            cfg_embed = torch.nn.functional.silu(cfg_embed)
            cfg_embed = weights.cfg_cond_proj_2.apply(cfg_embed)
            embed = embed + cfg_embed
        if self.sensitive_layer_dtype != self.infer_dtype:
            embed = weights.time_embedding_0.apply(embed.to(self.sensitive_layer_dtype))
        else:
            embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)

        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))

        # text embeddings
        if self.sensitive_layer_dtype != self.infer_dtype:
            out = weights.text_embedding_0.apply(context.squeeze(0).to(self.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(context.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)
        if self.clean_cuda_cache:
            del out
            torch.cuda.empty_cache()

        if self.task in ["i2v", "flf2v", "animate"] and self.config.get("use_image_encoder", True):
            if self.task == "flf2v":
                _, n, d = clip_fea.shape
                clip_fea = clip_fea.view(2 * n, d)
                clip_fea = clip_fea + weights.emb_pos.tensor.squeeze()
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

        grid_sizes = GridOutput(tensor=torch.tensor([[grid_sizes_t, grid_sizes_h, grid_sizes_w]], dtype=torch.int32, device=x.device), tuple=(grid_sizes_t, grid_sizes_h, grid_sizes_w))
        return WanPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x.squeeze(0),
            embed0=embed0.squeeze(0),
            seq_lens=seq_lens,
            freqs=self.freqs,
            context=context,
            adapter_args={"motion_vec": motion_vec},
        )
