import time

import torch
import torch.distributed as dist
from loguru import logger
from torch.nn import functional as F

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.qwen_image.infer.offload.transformer_infer import QwenImageOffloadTransformerInfer
from lightx2v.models.networks.qwen_image.infer.post_infer import QwenImagePostInfer
from lightx2v.models.networks.qwen_image.infer.pre_infer import QwenImagePreInfer
from lightx2v.models.networks.qwen_image.infer.transformer_infer import QwenImageTransformerInfer
from lightx2v.models.networks.qwen_image.weights.post_weights import QwenImagePostWeights
from lightx2v.models.networks.qwen_image.weights.pre_weights import QwenImagePreWeights
from lightx2v.models.networks.qwen_image.weights.transformer_weights import (
    QwenImageTransformerWeights,
    release_weight_module_device_tensors,
)
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class QwenImageTransformerModel(BaseTransformerModel):
    pre_weight_class = QwenImagePreWeights
    transformer_weight_class = QwenImageTransformerWeights
    post_weight_class = QwenImagePostWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, None, lora_path, lora_strength)
        self._offload_weights_active = False
        self.in_channels = self.config["in_channels"]
        self.attention_kwargs = {}
        if self.lazy_load:
            self.remove_keys.extend(["blocks."])
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = QwenImageTransformerInfer if not self.cpu_offload else QwenImageOffloadTransformerInfer
        else:
            assert NotImplementedError
        self.pre_infer_class = QwenImagePreInfer
        self.post_infer_class = QwenImagePostInfer

    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        first_block = self.transformer_weights.blocks[0]
        self.pre_infer.set_rope(
            img_rope=first_block.compute_phases[0].rope,
            txt_rope=first_block.compute_phases[1].rope,
        )
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    def _init_offload_manager(self):
        if hasattr(self.transformer_weights, "offload_block_cuda_buffers"):
            self.transformer_infer.offload_manager.init_cuda_buffer(
                blocks_cuda_buffer=self.transformer_weights.offload_block_cuda_buffers,
            )
        if self.lazy_load and hasattr(self.transformer_weights, "offload_block_cpu_buffers"):
            self.transformer_infer.offload_manager.init_cpu_buffer(
                blocks_cpu_buffer=self.transformer_weights.offload_block_cpu_buffers,
            )

    def prepare_offload_weights(self):
        """Keep the largest configured set of weights resident for the DiT loop."""
        if not self.cpu_offload:
            return
        if self._offload_weights_active:
            if (self.offload_granularity == "block" and self.config.get("offload_persistent_resident_blocks", False)) or self._keep_model_weights_resident_on_this_rank():
                return
            raise RuntimeError("Qwen-Image offload weights are already active")

        self._offload_weights_active = True
        if self.offload_granularity == "model":
            transfer_start = time.perf_counter()
            self.to_cuda()
            if self.config.get("qwen_image_rank_aware_model_offload", False):
                torch_device_module.synchronize()
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                free_bytes, total_bytes = torch_device_module.mem_get_info()
                logger.info(
                    f"[QwenImage] Rank {rank}: full DiT model H2D completed in "
                    f"{time.perf_counter() - transfer_start:.3f}s; device free={free_bytes / 2**30:.2f} GiB, "
                    f"total={total_bytes / 2**30:.2f} GiB"
                )
        else:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()
            self.transformer_weights.resident_blocks_to_cuda()
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            resident_count = len(self.transformer_weights.resident_block_indices)
            if hasattr(torch_device_module, "mem_get_info"):
                free_bytes, total_bytes = torch_device_module.mem_get_info()
                logger.info(
                    f"[QwenImage] Rank {rank}: prepared {resident_count}/{self.config['num_layers']} resident DiT blocks; device free={free_bytes / 2**30:.2f} GiB, total={total_bytes / 2**30:.2f} GiB"
                )

    def finish_offload_weights(self):
        """Finish one DiT pass while optionally retaining immutable weights."""
        if not self.cpu_offload or not self._offload_weights_active:
            return
        if self._keep_model_weights_resident_on_this_rank():
            torch_device_module.synchronize()
            return
        if self.offload_granularity == "block" and self.config.get("offload_persistent_resident_blocks", False):
            torch_device_module.synchronize()
            if hasattr(self.transformer_infer.offload_manager, "reset_slots"):
                self.transformer_infer.offload_manager.reset_slots()
            return
        self.force_cleanup_offload_weights()

    def force_cleanup_offload_weights(self):
        """Release DiT-resident weights without copying immutable weights back to CPU."""
        if not self.cpu_offload or not self._offload_weights_active:
            return

        torch_device_module.synchronize()
        if self.offload_granularity == "model":
            if self.config.get("qwen_image_model_offload_release_only", False):
                release_start = time.perf_counter()
                release_weight_module_device_tensors(self.pre_weight)
                release_weight_module_device_tensors(self.transformer_weights)
                release_weight_module_device_tensors(self.post_weight)
                torch_device_module.empty_cache()
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                logger.info(f"[QwenImage] Rank {rank}: released full DiT device replica without D2H in {time.perf_counter() - release_start:.3f}s")
            else:
                self.to_cpu()
        else:
            if hasattr(self.transformer_infer.offload_manager, "reset_slots"):
                self.transformer_infer.offload_manager.reset_slots()
            release_weight_module_device_tensors(self.pre_weight)
            release_weight_module_device_tensors(self.post_weight)
            self.transformer_weights.release_resident_blocks()
        self._offload_weights_active = False

    def _keep_model_weights_resident_on_this_rank(self):
        return (
            self.offload_granularity == "model"
            and self.config.get("qwen_image_rank_aware_model_offload", False)
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
            and dist.get_rank() != 0
        )

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
        noise_pred = self.post_infer.infer(self.post_weight, hidden_states, pre_infer_out.temb_txt_silu)

        if self.config["seq_parallel"]:
            noise_pred = self._seq_parallel_post_process(noise_pred)
        return noise_pred

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        seqlen = pre_infer_out.hidden_states.shape[0]
        padding_size = (world_size - (seqlen % world_size)) % world_size
        if padding_size > 0:
            pre_infer_out.hidden_states = F.pad(pre_infer_out.hidden_states, (0, 0, 0, padding_size))
        pre_infer_out.hidden_states = torch.chunk(pre_infer_out.hidden_states, world_size, dim=0)[cur_rank]
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, noise_pred):
        world_size = dist.get_world_size(self.seq_p_group)
        gathered_noise_pred = [torch.empty_like(noise_pred) for _ in range(world_size)]
        dist.all_gather(gathered_noise_pred, noise_pred, group=self.seq_p_group)
        noise_pred = torch.cat(gathered_noise_pred, dim=1)
        return noise_pred

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self._offload_weights_active:
                pass
            elif self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.post_weight.to_cuda()

        latents = self.scheduler.latents
        if self.config["task"] == "i2i":
            image_latents = torch.cat([item["image_latents"] for item in inputs["image_encoder_output"]], dim=1)
            latents_input = torch.cat([latents, image_latents], dim=1)
        else:
            latents_input = latents

        if self.config["enable_cfg"]:
            if self.config["cfg_parallel"]:
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
                    if self.config["task"] == "i2i":
                        noise_pred = noise_pred[:, : latents.size(1)]
                else:
                    noise_pred = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["negative_prompt_embeds"], infer_condition=False)
                    if self.config["task"] == "i2i":
                        noise_pred = noise_pred[:, : latents.size(1)]
                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                noise_pred_cond = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
                if self.config["task"] == "i2i":
                    noise_pred_cond = noise_pred_cond[:, : latents.size(1)]

                noise_pred_uncond = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["negative_prompt_embeds"], infer_condition=False)
                if self.config["task"] == "i2i":
                    noise_pred_uncond = noise_pred_uncond[:, : latents.size(1)]

            comb_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
            noise_pred_cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            self.scheduler.noise_pred = comb_pred * (noise_pred_cond_norm / noise_norm)
        else:
            # ==================== No CFG Processing ====================
            noise_pred = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
            if self.config["task"] == "i2i":
                noise_pred = noise_pred[:, : latents.size(1)]
            self.scheduler.noise_pred = noise_pred

        if self.cpu_offload:
            if self._offload_weights_active:
                pass
            elif self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()
