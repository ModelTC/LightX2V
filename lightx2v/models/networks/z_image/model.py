import torch
import torch.distributed as dist
from torch.nn import functional as F

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.z_image.infer.offload.transformer_infer import ZImageOffloadTransformerInfer
from lightx2v.models.networks.z_image.infer.post_infer import ZImagePostInfer
from lightx2v.models.networks.z_image.infer.pre_infer import ZImagePreInfer
from lightx2v.models.networks.z_image.infer.transformer_infer import ZImageTransformerInfer
from lightx2v.models.networks.z_image.weights.post_weights import ZImagePostWeights
from lightx2v.models.networks.z_image.weights.pre_weights import ZImagePreWeights
from lightx2v.models.networks.z_image.weights.transformer_weights import ZImageTransformerWeights
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class ZImageTransformerModel(BaseTransformerModel):
    pre_weight_class = ZImagePreWeights
    transformer_weight_class = ZImageTransformerWeights
    post_weight_class = ZImagePostWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, None, lora_path, lora_strength)
        if self.lazy_load:
            self.remove_keys.extend(["layers."])

        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = ZImageTransformerInfer if not self.cpu_offload else ZImageOffloadTransformerInfer
        else:
            assert NotImplementedError
        self.pre_infer_class = ZImagePreInfer
        self.post_infer_class = ZImagePostInfer

    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    @torch.no_grad()
    def _infer_cond_uncond(self, latents_input, prompt_embeds, infer_condition=True):
        self.scheduler.infer_condition = infer_condition
        pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            hidden_states=latents_input,
            encoder_hidden_states=prompt_embeds,
        )

        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        hidden_states = self.transformer_infer.infer(
            block_weights=self.transformer_weights,
            pre_infer_out=pre_infer_out,
        )

        if self.config["seq_parallel"]:
            hidden_states = self._seq_parallel_post_process(hidden_states, pre_infer_out.x_item_seqlens[0])

        noise_pred = self.post_infer.infer(
            self.post_weight,
            hidden_states,
            pre_infer_out.temb_img_silu,  # Use timestep embedding (t), not text embedding!
            image_tokens_len=pre_infer_out.image_tokens_len,
        )

        return noise_pred

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)

        seqlen = pre_infer_out.hidden_states.shape[0]
        padding_size = (world_size - (seqlen % world_size)) % world_size
        if padding_size > 0:
            pre_infer_out.hidden_states = F.pad(pre_infer_out.hidden_states, (0, 0, 0, padding_size))
            pre_infer_out.x_freqs_cis = F.pad(pre_infer_out.x_freqs_cis, (0, 0, 0, padding_size))

        pre_infer_out.hidden_states = torch.chunk(pre_infer_out.hidden_states, world_size, dim=0)[cur_rank]
        pre_infer_out.x_freqs_cis = torch.chunk(pre_infer_out.x_freqs_cis, world_size, dim=0)[cur_rank]
        pre_infer_out.x_item_seqlens = [pre_infer_out.hidden_states.shape[0]]
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, hidden_states, image_shard_len):
        image_hidden_states = hidden_states[:image_shard_len]
        world_size = dist.get_world_size(self.seq_p_group)
        gathered_hidden_states = [torch.empty_like(image_hidden_states) for _ in range(world_size)]
        dist.all_gather(gathered_hidden_states, image_hidden_states, group=self.seq_p_group)
        return torch.cat(gathered_hidden_states, dim=0)

    @compiled_method()
    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.post_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        latents = self.scheduler.latents
        latents_input = latents

        if self.config["enable_cfg"]:
            if self.config["cfg_parallel"]:
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
                else:
                    noise_pred = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["negative_prompt_embeds"], infer_condition=False)

                # post_infer already extracts image part, so noise_pred is already [B, T_img, out_dim]
                # No need to extract again
                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                noise_pred_cond = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
                noise_pred_uncond = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["negative_prompt_embeds"], infer_condition=False)

                # post_infer already extracts image part, so noise_pred is already [B, T_img, out_dim]
                # Just ensure both have the same sequence length (should be same, but double-check)
                min_seq_len = min(noise_pred_cond.shape[1], noise_pred_uncond.shape[1])
                noise_pred_cond = noise_pred_cond[:, :min_seq_len, :]
                noise_pred_uncond = noise_pred_uncond[:, :min_seq_len, :]

            comb_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
            noise_pred_cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            self.scheduler.noise_pred = comb_pred * (noise_pred_cond_norm / noise_norm)
        else:
            # ==================== No CFG Processing ====================
            noise_pred = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)

            # post_infer already extracts image part, so noise_pred is already [B, T_img, out_dim]
            # No need to extract again

            self.scheduler.noise_pred = noise_pred

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
