import torch
import torch.distributed as dist

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.longcat_image.infer.post_infer import LongCatImagePostInfer
from lightx2v.models.networks.longcat_image.infer.pre_infer import LongCatImagePreInfer
from lightx2v.models.networks.longcat_image.infer.offload.transformer_infer import LongCatImageOffloadTransformerInfer
from lightx2v.models.networks.longcat_image.infer.transformer_infer import LongCatImageTransformerInfer
from lightx2v.models.networks.longcat_image.weights.post_weights import LongCatImagePostWeights
from lightx2v.models.networks.longcat_image.weights.pre_weights import LongCatImagePreWeights
from lightx2v.models.networks.longcat_image.weights.transformer_weights import LongCatImageTransformerWeights
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v.utils.envs import *


class LongCatImageTransformerModel(BaseTransformerModel):
    """Transformer model for LongCat Image.

    Handles weight loading and inference for the LongCat architecture
    (10 double-stream blocks + 20 single-stream blocks).
    """

    pre_weight_class = LongCatImagePreWeights
    transformer_weight_class = LongCatImageTransformerWeights
    post_weight_class = LongCatImagePostWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)
        # Use transformer_in_channels to avoid conflict with VAE's in_channels
        self.in_channels = self.config.get("transformer_in_channels", self.config.get("in_channels", 64))
        self.attention_kwargs = {}
        if self.config["seq_parallel"]:
            raise NotImplementedError("Sequence parallel is not implemented for LongCatImageTransformerModel")
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        if self.cpu_offload and self.offload_granularity == "block":
            self.transformer_infer_class = LongCatImageOffloadTransformerInfer
        else:
            self.transformer_infer_class = LongCatImageTransformerInfer
        self.pre_infer_class = LongCatImagePreInfer
        self.post_infer_class = LongCatImagePostInfer

    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager_double") and hasattr(self.transformer_infer, "offload_manager_single"):
            self._init_offload_manager()

    def _init_offload_manager(self):
        """Initialize offload managers for double and single block buffers."""
        self.transformer_infer.offload_manager_double.init_cuda_buffer(
            blocks_cuda_buffer=self.transformer_weights.offload_double_block_cuda_buffers
        )
        self.transformer_infer.offload_manager_single.init_cuda_buffer(
            blocks_cuda_buffer=self.transformer_weights.offload_single_block_cuda_buffers
        )

    @torch.no_grad()
    def _infer_cond_uncond(self, latents_input, prompt_embeds, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            hidden_states=latents_input,
            encoder_hidden_states=prompt_embeds,
        )

        hidden_states = self.transformer_infer.infer(
            block_weights=self.transformer_weights,
            pre_infer_out=pre_infer_out,
        )

        noise_pred = self.post_infer.infer(self.post_weight, hidden_states, pre_infer_out.temb)

        return noise_pred

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        raise NotImplementedError("Sequence parallel pre-process is not implemented for LongCatImageTransformerModel")

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        raise NotImplementedError("Sequence parallel post-process is not implemented for LongCatImageTransformerModel")

    @compiled_method()
    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model":
                self.to_cuda()
            elif self.offload_granularity == "block":
                self.pre_weight.to_cuda()
                self.post_weight.to_cuda()

        latents = self.scheduler.latents

        if self.config.get("enable_cfg", True):
            # Check if CFG parallel should be used
            # Note: I2I task may have different sequence lengths for positive/negative prompts,
            # which is not yet supported in CFG parallel mode
            use_cfg_parallel = self.config.get("cfg_parallel", False)
            if use_cfg_parallel and hasattr(self.scheduler, "input_image_latents") and self.scheduler.input_image_latents is not None:
                # I2I task: check if sequence lengths match
                if hasattr(self.scheduler, "image_rotary_emb") and hasattr(self.scheduler, "negative_image_rotary_emb"):
                    pos_len = self.scheduler.image_rotary_emb[0].shape[0]
                    neg_len = self.scheduler.negative_image_rotary_emb[0].shape[0]
                    if pos_len != neg_len:
                        from lightx2v.utils.utils import logger

                        if dist.get_rank() == 0:
                            logger.warning(f"CFG parallel disabled for I2I task due to sequence length mismatch (positive: {pos_len}, negative: {neg_len}). Falling back to sequential CFG.")
                        use_cfg_parallel = False

            if use_cfg_parallel:
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(latents, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
                else:
                    noise_pred = self._infer_cond_uncond(latents, inputs["text_encoder_output"]["negative_prompt_embeds"], infer_condition=False)

                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1

                # Apply CFG with optional renormalization
                noise_pred = self.scheduler.apply_cfg(noise_pred_cond, noise_pred_uncond)
                self.scheduler.noise_pred = noise_pred
            else:
                # ==================== CFG Sequential Processing ====================
                noise_pred_cond = self._infer_cond_uncond(latents, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
                noise_pred_uncond = self._infer_cond_uncond(latents, inputs["text_encoder_output"]["negative_prompt_embeds"], infer_condition=False)

                # Apply CFG with optional renormalization
                noise_pred = self.scheduler.apply_cfg(noise_pred_cond, noise_pred_uncond)
                self.scheduler.noise_pred = noise_pred
        else:
            # ==================== No CFG Processing ====================
            noise_pred = self._infer_cond_uncond(latents, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
            self.scheduler.noise_pred = noise_pred

        if self.cpu_offload:
            if self.offload_granularity == "model":
                self.to_cpu()
            elif self.offload_granularity == "block":
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()
