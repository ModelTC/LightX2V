import torch
import torch.distributed as dist

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.flux2_klein.infer.post_infer import Flux2KleinPostInfer
from lightx2v.models.networks.flux2_klein.infer.pre_infer import Flux2KleinPreInfer
from lightx2v.models.networks.flux2_klein.infer.transformer_infer import Flux2KleinTransformerInfer
from lightx2v.models.networks.flux2_klein.weights.post_weights import Flux2KleinPostWeights
from lightx2v.models.networks.flux2_klein.weights.pre_weights import Flux2KleinPreWeights
from lightx2v.models.networks.flux2_klein.weights.transformer_weights import Flux2KleinTransformerWeights
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v.utils.envs import *


class Flux2KleinTransformerModel(BaseTransformerModel):
    pre_weight_class = Flux2KleinPreWeights
    transformer_weight_class = Flux2KleinTransformerWeights
    post_weight_class = Flux2KleinPostWeights

    def __init__(self, config, model_path, device):
        super().__init__(model_path, config, device)
        # Use transformer_in_channels to avoid conflict with VAE's in_channels
        self.in_channels = self.config.get("transformer_in_channels", self.config.get("in_channels", 64))
        self.attention_kwargs = {}
        if self.config["seq_parallel"]:
            raise NotImplementedError("Sequence parallel is not implemented for Flux2KleinTransformerModel")
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        self.transformer_infer_class = Flux2KleinTransformerInfer
        self.pre_infer_class = Flux2KleinPreInfer
        self.post_infer_class = Flux2KleinPostInfer

    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    # Weight loading is handled by base class _apply_weights()
    # Modulation weights are loaded into transformer_weights by WeightModule.load()

    @torch.no_grad()
    def _infer_cond_uncond(self, latents_input, prompt_embeds, infer_condition=True, txt_ids=None, img_ids=None):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            hidden_states=latents_input,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
        )

        hidden_states = self.transformer_infer.infer(
            block_weights=self.transformer_weights,
            pre_infer_out=pre_infer_out,
        )

        noise_pred = self.post_infer.infer(self.post_weight, hidden_states, pre_infer_out.timestep)

        return noise_pred

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        raise NotImplementedError("Sequence parallel pre-process is not implemented for Flux2KleinTransformerModel")

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        raise NotImplementedError("Sequence parallel post-process is not implemented for Flux2KleinTransformerModel")

    @compiled_method()
    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            self.to_cuda()

        latents = self.scheduler.latents
        do_cfg = self.config.get("enable_cfg", True) and self.config.get("sample_guide_scale", 1.0) > 1.0

        if do_cfg:
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

                text_ids = inputs["text_encoder_output"].get("text_ids", None)
                img_ids = getattr(self.scheduler, "latent_image_ids", None)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(
                        latents,
                        inputs["text_encoder_output"]["prompt_embeds"],
                        infer_condition=True,
                        txt_ids=text_ids,
                        img_ids=img_ids,
                    )
                else:
                    noise_pred = self._infer_cond_uncond(
                        latents,
                        inputs["text_encoder_output"]["negative_prompt_embeds"],
                        infer_condition=False,
                        txt_ids=inputs["text_encoder_output"].get("negative_text_ids", text_ids),
                        img_ids=img_ids,
                    )

                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1

                # Apply CFG
                guidance_scale = self.config.get("sample_guide_scale", 1.0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                self.scheduler.noise_pred = noise_pred
            else:
                # ==================== CFG Sequential Processing ====================
                text_ids = inputs["text_encoder_output"].get("text_ids", None)
                img_ids = getattr(self.scheduler, "latent_image_ids", None)

                noise_pred_cond = self._infer_cond_uncond(
                    latents,
                    inputs["text_encoder_output"]["prompt_embeds"],
                    infer_condition=True,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                )
                noise_pred_uncond = self._infer_cond_uncond(
                    latents,
                    inputs["text_encoder_output"]["negative_prompt_embeds"],
                    infer_condition=False,
                    txt_ids=inputs["text_encoder_output"].get("negative_text_ids", text_ids),
                    img_ids=img_ids,
                )

                # Apply CFG
                guidance_scale = self.config.get("sample_guide_scale", 1.0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                self.scheduler.noise_pred = noise_pred
        else:
            # ==================== No CFG Processing ====================
            text_ids = inputs["text_encoder_output"].get("text_ids", None)
            img_ids = getattr(self.scheduler, "latent_image_ids", None)
            noise_pred = self._infer_cond_uncond(
                latents,
                inputs["text_encoder_output"]["prompt_embeds"],
                infer_condition=True,
                txt_ids=text_ids,
                img_ids=img_ids,
            )
            self.scheduler.noise_pred = noise_pred
