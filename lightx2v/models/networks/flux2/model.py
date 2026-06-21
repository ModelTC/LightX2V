import torch
import torch.distributed as dist
from torch.nn import functional as F

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.flux2.infer.feature_caching.transformer_infer import Flux2TransformerInferAdaCaching
from lightx2v.models.networks.flux2.infer.offload.transformer_infer import Flux2OffloadTransformerInfer
from lightx2v.models.networks.flux2.infer.post_infer import Flux2PostInfer
from lightx2v.models.networks.flux2.infer.pre_infer import Flux2DevPreInfer, Flux2PreInfer
from lightx2v.models.networks.flux2.infer.transformer_infer import Flux2TransformerInfer
from lightx2v.models.networks.flux2.weights.post_weights import Flux2PostWeights
from lightx2v.models.networks.flux2.weights.pre_weights import Flux2DevPreWeights, Flux2PreWeights
from lightx2v.models.networks.flux2.weights.transformer_weights import Flux2TransformerWeights
from lightx2v.utils.custom_compiler import compiled_method


class _Flux2TransformerModelBase(BaseTransformerModel):
    """Shared base for both Klein and Dev transformer models."""

    transformer_weight_class = Flux2TransformerWeights
    post_weight_class = Flux2PostWeights

    def __init__(self, config, model_path, device):
        super().__init__(model_path, config, device)
        self.in_channels = self.config.get("transformer_in_channels", self.config.get("in_channels", 64))
        self.attention_kwargs = {}
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager_double") and hasattr(self.transformer_infer, "offload_manager_single"):
            self._init_offload_manager()

    def _init_offload_manager(self):
        self.transformer_infer.offload_manager_double.init_cuda_buffer(blocks_cuda_buffer=self.transformer_weights.offload_double_block_cuda_buffers)
        self.transformer_infer.offload_manager_single.init_cuda_buffer(blocks_cuda_buffer=self.transformer_weights.offload_single_block_cuda_buffers)

    @torch.no_grad()
    def _infer_cond_uncond(self, latents_input, prompt_embeds, infer_condition=True, txt_ids=None, img_ids=None):
        self.scheduler.infer_condition = infer_condition

        input_image_latents = getattr(self.scheduler, "input_image_latents", None)
        input_image_ids = getattr(self.scheduler, "input_image_ids", None)

        orig_seq_len = latents_input.shape[1]

        if input_image_latents is not None:
            latents_input = torch.cat([latents_input, input_image_latents], dim=1)
            if img_ids is not None and input_image_ids is not None:
                img_ids = torch.cat([img_ids, input_image_ids], dim=1)

        pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            hidden_states=latents_input,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
        )

        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        hidden_states = self.transformer_infer.infer(
            block_weights=self.transformer_weights,
            pre_infer_out=pre_infer_out,
        )

        noise_pred = self.post_infer.infer(self.post_weight, hidden_states, pre_infer_out.timestep)

        if self.config["seq_parallel"]:
            noise_pred = self._seq_parallel_post_process(noise_pred)

        noise_pred = noise_pred[:, :orig_seq_len, :]

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


class Flux2KleinTransformerModel(_Flux2TransformerModelBase):
    """Flux2 Klein transformer: supports CFG (sequential and parallel)."""

    pre_weight_class = Flux2PreWeights

    def _init_infer_class(self):
        feature_caching = self.config.get("feature_caching", "NoCaching")
        if feature_caching in ("NoCaching", "None"):
            if self.cpu_offload and self.offload_granularity == "block":
                self.transformer_infer_class = Flux2OffloadTransformerInfer
            else:
                self.transformer_infer_class = Flux2TransformerInfer
        elif feature_caching == "Ada":
            if self.cpu_offload and self.offload_granularity == "block":
                raise NotImplementedError("Flux2 AdaCache does not support block-level cpu_offload yet")
            self.transformer_infer_class = Flux2TransformerInferAdaCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {feature_caching}")
        self.pre_infer_class = Flux2PreInfer
        self.post_infer_class = Flux2PostInfer

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
        do_cfg = self.config.get("enable_cfg", True) and self.config.get("sample_guide_scale", 1.0) > 1.0

        if do_cfg:
            use_cfg_parallel = self.config.get("cfg_parallel", False)
            if use_cfg_parallel and hasattr(self.scheduler, "input_image_latents") and self.scheduler.input_image_latents is not None:
                if hasattr(self.scheduler, "image_rotary_emb") and hasattr(self.scheduler, "negative_image_rotary_emb"):
                    pos_len = self.scheduler.image_rotary_emb[0].shape[0]
                    neg_len = self.scheduler.negative_image_rotary_emb[0].shape[0]
                    if pos_len != neg_len:
                        from lightx2v.utils.utils import logger

                        if dist.get_rank() == 0:
                            logger.warning(f"CFG parallel disabled for I2I task due to sequence length mismatch (positive: {pos_len}, negative: {neg_len}). Falling back to sequential CFG.")
                        use_cfg_parallel = False

            if use_cfg_parallel:
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
                noise_pred_cond = noise_pred_list[0]
                noise_pred_uncond = noise_pred_list[1]

                guidance_scale = self.config.get("sample_guide_scale", 1.0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                self.scheduler.noise_pred = noise_pred
            else:
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

                guidance_scale = self.config.get("sample_guide_scale", 1.0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                self.scheduler.noise_pred = noise_pred
        else:
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

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()


class Flux2DevTransformerModel(_Flux2TransformerModelBase):
    """Flux2 Dev transformer: single forward pass with embedded guidance (no CFG)."""

    pre_weight_class = Flux2DevPreWeights

    def _init_infer_class(self):
        feature_caching = self.config.get("feature_caching", "NoCaching")
        if feature_caching in ("NoCaching", "None"):
            if self.cpu_offload and self.offload_granularity == "block":
                self.transformer_infer_class = Flux2OffloadTransformerInfer
            else:
                self.transformer_infer_class = Flux2TransformerInfer
        elif feature_caching == "Ada":
            if self.cpu_offload and self.offload_granularity == "block":
                raise NotImplementedError("Flux2 AdaCache does not support block-level cpu_offload yet")
            self.transformer_infer_class = Flux2TransformerInferAdaCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {feature_caching}")
        self.pre_infer_class = Flux2DevPreInfer
        self.post_infer_class = Flux2PostInfer

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
        txt_ids = inputs["text_encoder_output"].get("text_ids", None)
        img_ids = getattr(self.scheduler, "latent_image_ids", None)

        noise_pred = self._infer_cond_uncond(
            latents,
            inputs["text_encoder_output"]["prompt_embeds"],
            infer_condition=True,
            txt_ids=txt_ids,
            img_ids=img_ids,
        )
        self.scheduler.noise_pred = noise_pred

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
