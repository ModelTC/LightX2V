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
from lightx2v_platform.base import global_var


class _Flux2TransformerModelBase(BaseTransformerModel):
    """Shared base for both Klein and Dev transformer models."""

    transformer_weight_class = Flux2TransformerWeights
    post_weight_class = Flux2PostWeights

    def __init__(self, config, model_path, device):
        super().__init__(model_path, config, device)
        self.in_channels = self.config.get("transformer_in_channels", self.config.get("in_channels", 64))
        self.attention_kwargs = {}
        self._combined_img_ids_cache = None
        self._init_tensor_parallel()
        self._init_infer_class()
        self._init_weights()
        # In PipeFusion mode, weights were loaded to CPU to avoid OOM;
        # move only this stage's subset to GPU.
        if self.config.get("pipefusion_parallel", False):
            self.to_cuda()
        self._init_infer()

    def _init_tensor_parallel(self):
        if self.config.get("tensor_parallel", False):
            self.use_tp = True
            self.tp_group = self.config.get("device_mesh").get_group(mesh_dim="tensor_p")
            self.tp_rank = dist.get_rank(self.tp_group)
            self.tp_size = dist.get_world_size(self.tp_group)
        else:
            self.use_tp = False
            self.tp_group = None
            self.tp_rank = 0
            self.tp_size = 1

    def _should_load_weights(self):
        if self.config.get("device_mesh") is None:
            return True
        if dist.is_initialized() and self.use_tp:
            return dist.get_rank() == 0
        return super()._should_load_weights()

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        if not self.use_tp:
            return super()._load_ckpt(unified_dtype, sensitive_layer)
        original_device = self.device
        self.device = torch.device("cpu")
        try:
            return super()._load_ckpt(unified_dtype, sensitive_layer)
        finally:
            self.device = original_device

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        if not self.use_tp:
            return super()._load_quant_ckpt(unified_dtype, sensitive_layer)
        original_device = self.device
        self.device = torch.device("cpu")
        try:
            return super()._load_quant_ckpt(unified_dtype, sensitive_layer)
        finally:
            self.device = original_device

    def _rank_device(self):
        ai_device = global_var.AI_DEVICE
        if ai_device is None:
            return torch.device("cpu")
        if dist.is_initialized():
            return torch.device(f"{ai_device}:{dist.get_rank()}")
        return torch.device(ai_device)

    def _sync_device(self):
        ai_device = global_var.AI_DEVICE
        device_module = getattr(torch, ai_device, None) if ai_device else None
        if device_module is not None and hasattr(device_module, "synchronize"):
            device_module.synchronize()

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        if not self.use_tp:
            return super()._load_weights_from_rank0(weight_dict, is_weight_loader)
        if self.cpu_offload:
            raise NotImplementedError("Flux2 tensor parallel weight loading does not support cpu_offload yet.")

        global_src_rank = 0
        target_device = self._rank_device()

        if is_weight_loader:
            processed_weight_dict = {}
            meta_dict = {}
            processed_bias_keys = set()
            for key, tensor in weight_dict.items():
                split_type = self._get_split_type(key)
                if key.endswith(".weight") and split_type is not None:
                    split_weights = self._split_weight_for_tp(key, tensor, self.tp_size)
                    for rank_idx, split_weight in enumerate(split_weights):
                        rank_key = f"{key}__tp_rank_{rank_idx}"
                        processed_weight_dict[rank_key] = split_weight.contiguous()
                    meta_dict[key] = {"shape": split_weights[0].shape, "dtype": split_weights[0].dtype, "is_tp": True}

                    bias_key = key.replace(".weight", ".bias")
                    if bias_key in weight_dict and split_type in ("col", "ff_fused_col", "single_fused_col"):
                        bias_splits = self._split_bias_for_tp(bias_key, weight_dict[bias_key], split_type, self.tp_size)
                        for rank_idx, split_bias in enumerate(bias_splits):
                            processed_weight_dict[f"{bias_key}__tp_rank_{rank_idx}"] = split_bias.contiguous()
                        meta_dict[bias_key] = {"shape": bias_splits[0].shape, "dtype": bias_splits[0].dtype, "is_tp": True}
                        processed_bias_keys.add(bias_key)
                elif key not in processed_bias_keys:
                    processed_weight_dict[key] = tensor
                    meta_dict[key] = {"shape": tensor.shape, "dtype": tensor.dtype, "is_tp": False}

            obj_list = [meta_dict]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]
            weight_dict = processed_weight_dict
        else:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]

        distributed_weight_dict = {key: torch.empty(meta["shape"], dtype=meta["dtype"], device=target_device) for key, meta in synced_meta_dict.items()}
        dist.barrier()

        for key in sorted(synced_meta_dict.keys()):
            meta = synced_meta_dict[key]
            if meta.get("is_tp", False):
                for rank_idx in range(self.tp_size):
                    if is_weight_loader:
                        src_tensor = weight_dict[f"{key}__tp_rank_{rank_idx}"].to(target_device, non_blocking=True)
                    else:
                        src_tensor = torch.empty(meta["shape"], dtype=meta["dtype"], device=target_device)
                    dist.broadcast(src_tensor, src=global_src_rank)
                    if rank_idx == self.tp_rank:
                        distributed_weight_dict[key].copy_(src_tensor, non_blocking=True)
                    del src_tensor
            else:
                if is_weight_loader:
                    distributed_weight_dict[key].copy_(weight_dict[key].to(target_device, non_blocking=True), non_blocking=True)
                dist.broadcast(distributed_weight_dict[key], src=global_src_rank)

        self._sync_device()
        return distributed_weight_dict

    def _get_split_type(self, key):
        if ".norm_" in key:
            return None
        if key.endswith(".weight") and "single_transformer_blocks." in key and ".attn.to_qkv_mlp_proj." in key:
            return "single_fused_col"
        if key.endswith(".weight") and "single_transformer_blocks." in key and ".attn.to_out." in key:
            return "single_fused_row"
        if key.endswith(".weight") and (".ff.linear_in." in key or ".ff_context.linear_in." in key):
            return "ff_fused_col"
        col_patterns = (
            ".attn.to_q.",
            ".attn.to_k.",
            ".attn.to_v.",
            ".attn.add_q_proj.",
            ".attn.add_k_proj.",
            ".attn.add_v_proj.",
        )
        row_patterns = (
            ".attn.to_out.0.",
            ".attn.to_add_out.",
            ".ff.linear_out.",
            ".ff_context.linear_out.",
        )
        if any(pattern in key for pattern in col_patterns):
            return "col"
        if any(pattern in key for pattern in row_patterns):
            return "row"
        return None

    def _split_bias_for_tp(self, key, bias, split_type, tp_size):
        if split_type == "col":
            assert bias.shape[0] % tp_size == 0, f"bias dimension ({bias.shape[0]}) must be divisible by tp_size ({tp_size}) for {key}"
            return list(torch.chunk(bias, tp_size, dim=0))

        if split_type == "ff_fused_col":
            assert bias.shape[0] % 2 == 0, f"invalid fused SwiGLU bias dim for {key}: {bias.shape[0]}"
            ffn_dim = bias.shape[0] // 2
            assert ffn_dim % tp_size == 0, f"ffn_dim ({ffn_dim}) must be divisible by tp_size ({tp_size}) for {key}"
            gate, up = torch.split(bias, [ffn_dim, ffn_dim], dim=0)
            gate_chunks = torch.chunk(gate, tp_size, dim=0)
            up_chunks = torch.chunk(up, tp_size, dim=0)
            return [torch.cat([gate_chunks[rank_idx], up_chunks[rank_idx]], dim=0) for rank_idx in range(tp_size)]

        if split_type == "single_fused_col":
            inner_dim = self.config["num_attention_heads"] * self.config["attention_head_dim"]
            ffn_dim_twice = bias.shape[0] - 3 * inner_dim
            assert ffn_dim_twice > 0 and ffn_dim_twice % 2 == 0, f"invalid fused qkv/mlp bias dim for {key}: {bias.shape[0]}"
            ffn_dim = ffn_dim_twice // 2
            assert inner_dim % tp_size == 0, f"inner_dim ({inner_dim}) must be divisible by tp_size ({tp_size}) for {key}"
            assert ffn_dim % tp_size == 0, f"ffn_dim ({ffn_dim}) must be divisible by tp_size ({tp_size}) for {key}"
            q, k, v, mlp_1, mlp_2 = torch.split(bias, [inner_dim, inner_dim, inner_dim, ffn_dim, ffn_dim], dim=0)
            chunks = [torch.chunk(part, tp_size, dim=0) for part in (q, k, v, mlp_1, mlp_2)]
            return [torch.cat([part_chunks[rank_idx] for part_chunks in chunks], dim=0) for rank_idx in range(tp_size)]

        raise ValueError(f"Unsupported Flux2 TP bias split type {split_type} for {key}")

    def _split_weight_for_tp(self, key, weight, tp_size):
        split_type = self._get_split_type(key)
        if split_type is None:
            return [weight] * tp_size

        if split_type == "col":
            assert weight.shape[0] % tp_size == 0, f"out_dim ({weight.shape[0]}) must be divisible by tp_size ({tp_size}) for {key}"
            return list(torch.chunk(weight, tp_size, dim=0))

        if split_type == "row":
            assert weight.shape[1] % tp_size == 0, f"in_dim ({weight.shape[1]}) must be divisible by tp_size ({tp_size}) for {key}"
            return list(torch.chunk(weight, tp_size, dim=1))

        if split_type == "ff_fused_col":
            assert weight.shape[0] % 2 == 0, f"invalid fused SwiGLU out_dim for {key}: {weight.shape[0]}"
            ffn_dim = weight.shape[0] // 2
            assert ffn_dim % tp_size == 0, f"ffn_dim ({ffn_dim}) must be divisible by tp_size ({tp_size}) for {key}"
            gate, up = torch.split(weight, [ffn_dim, ffn_dim], dim=0)
            gate_chunks = torch.chunk(gate, tp_size, dim=0)
            up_chunks = torch.chunk(up, tp_size, dim=0)
            return [torch.cat([gate_chunks[rank_idx], up_chunks[rank_idx]], dim=0) for rank_idx in range(tp_size)]

        inner_dim = self.config["num_attention_heads"] * self.config["attention_head_dim"]
        if split_type == "single_fused_col":
            ffn_dim_twice = weight.shape[0] - 3 * inner_dim
            assert ffn_dim_twice > 0 and ffn_dim_twice % 2 == 0, f"invalid fused qkv/mlp out_dim for {key}: {weight.shape[0]}"
            ffn_dim = ffn_dim_twice // 2
            assert inner_dim % tp_size == 0, f"inner_dim ({inner_dim}) must be divisible by tp_size ({tp_size}) for {key}"
            assert ffn_dim % tp_size == 0, f"ffn_dim ({ffn_dim}) must be divisible by tp_size ({tp_size}) for {key}"
            q, k, v, mlp_1, mlp_2 = torch.split(weight, [inner_dim, inner_dim, inner_dim, ffn_dim, ffn_dim], dim=0)
            return [
                torch.cat(
                    [
                        torch.chunk(q, tp_size, dim=0)[rank_idx],
                        torch.chunk(k, tp_size, dim=0)[rank_idx],
                        torch.chunk(v, tp_size, dim=0)[rank_idx],
                        torch.chunk(mlp_1, tp_size, dim=0)[rank_idx],
                        torch.chunk(mlp_2, tp_size, dim=0)[rank_idx],
                    ],
                    dim=0,
                )
                for rank_idx in range(tp_size)
            ]

        if split_type == "single_fused_row":
            ffn_dim = weight.shape[1] - inner_dim
            assert ffn_dim > 0, f"invalid fused output in_dim for {key}: {weight.shape[1]}"
            assert inner_dim % tp_size == 0, f"inner_dim ({inner_dim}) must be divisible by tp_size ({tp_size}) for {key}"
            assert ffn_dim % tp_size == 0, f"ffn_dim ({ffn_dim}) must be divisible by tp_size ({tp_size}) for {key}"
            attn, mlp = torch.split(weight, [inner_dim, ffn_dim], dim=1)
            return [
                torch.cat(
                    [
                        torch.chunk(attn, tp_size, dim=1)[rank_idx],
                        torch.chunk(mlp, tp_size, dim=1)[rank_idx],
                    ],
                    dim=1,
                )
                for rank_idx in range(tp_size)
            ]

        raise ValueError(f"Unsupported Flux2 TP split type {split_type} for {key}")

    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.pre_infer.set_rope(self.transformer_weights.double_blocks[0].rope)
        if hasattr(self.transformer_infer, "offload_manager_double") and hasattr(self.transformer_infer, "offload_manager_single"):
            self._init_offload_manager()

    def _init_offload_manager(self):
        self.transformer_infer.offload_manager_double.init_cuda_buffer(blocks_cuda_buffer=self.transformer_weights.offload_double_block_cuda_buffers)
        self.transformer_infer.offload_manager_single.init_cuda_buffer(blocks_cuda_buffer=self.transformer_weights.offload_single_block_cuda_buffers)

    def _get_combined_img_ids(self, img_ids, input_image_ids):
        cached = self._combined_img_ids_cache
        if cached is not None:
            cached_sources, combined = cached
            if cached_sources[0] is img_ids and cached_sources[1] is input_image_ids:
                return combined

        combined = torch.cat([img_ids, input_image_ids], dim=1)
        self._combined_img_ids_cache = ((img_ids, input_image_ids), combined)
        return combined

    @torch.no_grad()
    def _infer_cond_uncond(self, latents_input, prompt_embeds, infer_condition=True, txt_ids=None, img_ids=None):
        self.scheduler.infer_condition = infer_condition

        input_image_latents = getattr(self.scheduler, "input_image_latents", None)
        input_image_ids = getattr(self.scheduler, "input_image_ids", None)

        orig_seq_len = latents_input.shape[1]

        if input_image_latents is not None:
            latents_input = torch.cat([latents_input, input_image_latents], dim=1)
            if img_ids is not None and input_image_ids is not None:
                img_ids = self._get_combined_img_ids(img_ids, input_image_ids)

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
        if self.config.get("pipefusion_parallel", False):
            from lightx2v.models.networks.flux2.infer.pipefusion.transformer_infer import (
                Flux2PipeFusionTransformerInfer,
            )

            self.transformer_infer_class = Flux2PipeFusionTransformerInfer
        elif feature_caching in ("NoCaching", "None"):
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
