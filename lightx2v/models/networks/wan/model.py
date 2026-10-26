import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.offload.shared_weight_coordinator import coordinate_rank_local_error
from lightx2v.common.ops.mm.fp8_f16_accum import fp8_f16_accum_mm_unavailable_reason, validate_fp8_f16_accum_qmax
from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channelF16Accum, unwrap_tp_weight
from lightx2v.common.ops.mm.optimized_linear import make_gelu_linear, make_projection_group
from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.wan.infer.feature_caching.transformer_infer import (
    WanTransformerInferAdaCaching,
    WanTransformerInferCustomCaching,
    WanTransformerInferDualBlock,
    WanTransformerInferDynamicBlock,
    WanTransformerInferFirstBlock,
    WanTransformerInferMagCaching,
    WanTransformerInferTaylorCaching,
    WanTransformerInferTeaCaching,
)
from lightx2v.models.networks.wan.infer.offload.transformer_infer import (
    WanOffloadTransformerInfer,
    validate_wan_residency_config,
)
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.models.networks.wan.shared_block_weights import (
    WanFp8VllmSharedBlockAdapter,
)
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.envs import *
from lightx2v.utils.fp8_f16_accum_checkpoint import ACTIVATION_QMAX, WAN_QUANTIZATION_PROFILE, validate_checkpoint
from lightx2v.utils.utils import *
from lightx2v_platform.base.global_var import AI_DEVICE


class WanModel(BaseTransformerModel):
    pre_weight_class = WanPreWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device, model_type="wan2.1", lora_path=None, lora_strength=1.0):
        validate_wan_residency_config(config)
        if config.get("dit_quant_scheme") == "fp8-f16-accum":
            activation_qmax = validate_fp8_f16_accum_qmax(config.get("dit_fp8_activation_qmax", ACTIVATION_QMAX))
            if activation_qmax > ACTIVATION_QMAX:
                raise ValueError(f"Wan FP16 accumulation requires activation qmax <= {ACTIVATION_QMAX}")
            self.fp8_activation_qmax = activation_qmax
            reason = fp8_f16_accum_mm_unavailable_reason()
            if reason is not None:
                raise RuntimeError(f"Wan fp8-f16-accum is unavailable: {reason}")
            if not config.get("dummy_model", False):
                validate_checkpoint(config.get("dit_quantized_ckpt", model_path), profile=WAN_QUANTIZATION_PROFILE)
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)
        if self.lazy_load:
            self.remove_keys.extend(["blocks."])
        self.sensitive_layer = {
            "norm",
            "embedding",
            "modulation",
            "time",
            "img_emb.proj.0",
            "img_emb.proj.4",
            "before_proj",  # vace
            "after_proj",  # vace
        }
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _apply_weights(self, weight_dict=None):
        super()._apply_weights(weight_dict)
        if self.config.get("dit_quant_scheme") != "fp8-f16-accum":
            return
        for _, leaf in self.transformer_weights.named_weight_leaves():
            linear = unwrap_tp_weight(leaf)
            if type(linear) is MMWeightWfp8channelAfp8channelF16Accum:
                linear.enable_fp8_f16_accum(self.fp8_activation_qmax)

    def prepare_autoregressive_projection_groups(self):
        """Attach shared projection and GELU paths after model weights are loaded."""
        fuse_gelu = self.config.get("use_compile", False) and self.config.get("compile_backend", "default") == "default" and self.config.get("compile_options", {}).get("emulate_precision_casts", True)
        blocks = self.transformer_weights.blocks
        for index, block in enumerate(blocks):
            self._prepare_projection_phases(block.compute_phases, fuse_gelu=fuse_gelu, cache_kv=index == len(blocks) - 1)

        if self.config.get("dit_quant_scheme") != "fp8-f16-accum" or self.config.get("cpu_offload_block_contiguous", False):
            return
        offload_blocks = getattr(self.transformer_weights, "offload_block_cuda_buffers", None)
        if offload_blocks is not None:
            for block in offload_blocks:
                self._prepare_projection_phases(block.compute_phases, fuse_gelu=fuse_gelu, cache_kv=True, offload_buffers=True)
        offload_phases = getattr(self.transformer_weights, "offload_phase_cuda_buffers", None)
        if offload_phases is not None:
            self._prepare_projection_phases(offload_phases, fuse_gelu=fuse_gelu, cache_kv=True, offload_buffers=True)

    def to_cuda(self):
        super().to_cuda()
        if self.config.get("dit_quant_scheme") == "fp8-f16-accum" and self.cpu_offload and self.offload_granularity == "model":
            # Model offload replaces every tensor, so groups prepared while the
            # weights were on CPU cannot use the newly uploaded CUDA weights.
            self.prepare_autoregressive_projection_groups()

    def to_cpu(self):
        super().to_cpu()
        if self.config.get("dit_quant_scheme") == "fp8-f16-accum" and self.cpu_offload and self.offload_granularity == "model":
            # A packed group owns its CUDA allocation even after the original
            # projection modules have moved to CPU. Release that allocation.
            for block in self.transformer_weights.blocks:
                self._clear_projection_phases(block.compute_phases)

    @staticmethod
    def _clear_projection_phases(phases):
        phases[0].self_attn_projection_group = None
        phases[0].self_attn_cache_kv_projection_group = None
        phases[1].cross_kv_projection_group = None
        phases[1].cross_img_kv_projection_group = None
        phases[2].ffn_gelu = None

    @staticmethod
    def _prepare_projection_phases(phases, *, fuse_gelu, cache_kv, offload_buffers=False):
        self_attn, cross_attn, ffn = phases[:3]
        query, key, value = self_attn.self_attn_q, self_attn.self_attn_k, self_attn.self_attn_v
        self_attn.self_attn_projection_group = make_projection_group((query, key, value), packed=True, offload_buffers=offload_buffers)
        self_attn.self_attn_cache_kv_projection_group = make_projection_group((key, value), offload_buffers=offload_buffers) if cache_kv else None
        cross_attn.cross_kv_projection_group = make_projection_group((cross_attn.cross_attn_k, cross_attn.cross_attn_v), offload_buffers=offload_buffers)
        cross_attn.cross_img_kv_projection_group = (
            make_projection_group((cross_attn.cross_attn_k_img, cross_attn.cross_attn_v_img), offload_buffers=offload_buffers) if hasattr(cross_attn, "cross_attn_k_img") else None
        )
        ffn.ffn_gelu = make_gelu_linear(ffn.ffn_2, enabled=fuse_gelu, offload_buffers=offload_buffers)

    def _load_shared_cpu_weights(self, unified_dtype, sensitive_layer):
        """Load rank-private non-block tensors plus NUMA-shared DiT blocks."""
        local_error = None
        try:
            adapter = WanFp8VllmSharedBlockAdapter(self.config, lora_path=self.lora_path)
            non_block_path = os.path.join(self.config["dit_quantized_ckpt"], "non_block.safetensors")
            if not os.path.isfile(non_block_path):
                raise FileNotFoundError(f"Wan non-block checkpoint not found: {non_block_path}")
            logger.info(f"[SharedCPUWeightsInfo] Loading rank-private Wan non-block weights from {non_block_path}")
            private_weights = self._load_safetensor_to_dict(non_block_path, unified_dtype, sensitive_layer)
        except Exception as error:
            local_error = error

        # Every rank completes local checkpoint I/O before entering the arena
        # collectives, so one bad file cannot strand healthy peers.
        coordinate_rank_local_error("Wan checkpoint preflight", local_error)

        allocation = adapter.materialize()
        try:
            weight_map = adapter.build_weight_map(private_weights)
        except BaseException:
            try:
                allocation.close()
            except Exception as cleanup_error:
                logger.error(f"Failed to close Wan shared CPU weight arena after initialization error: {cleanup_error}")
            raise
        return weight_map

    # ------------------------------------------------------------------ TP --
    def _rank_device(self):
        if dist.is_initialized():
            return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
        return torch.device(AI_DEVICE)

    def _get_split_type(self, key):
        if not key.endswith(".weight"):
            return None
        col_infixes = (
            ".self_attn.q.",
            ".self_attn.k.",
            ".self_attn.v.",
            ".self_attn.norm_q.",
            ".self_attn.norm_k.",
            ".cross_attn.q.",
            ".cross_attn.k.",
            ".cross_attn.v.",
            ".cross_attn.norm_q.",
            ".cross_attn.norm_k.",
            ".cross_attn.norm_k_img.",
            ".cross_attn.k_img.",
            ".cross_attn.v_img.",
        )
        row_infixes = (".self_attn.o.", ".cross_attn.o.", ".ffn.2.")
        if any(s in key for s in col_infixes):
            return "col"
        if any(s in key for s in row_infixes):
            return "row"
        if ".ffn.0." in key:
            return "col"
        return None

    def _split_tensor_for_tp(self, key, tensor):
        prefix, _, attribute = key.rpartition(".")
        split_type = self._get_split_type(f"{prefix}.weight")
        if split_type is None or attribute not in ("weight", "bias", "weight_scale"):
            return None
        if attribute == "weight":
            split_dim = 0 if split_type == "col" else 1
        else:
            # Bias and per-output scales follow output channels; row shards keep both whole.
            if split_type != "col" or (attribute == "weight_scale" and tensor.numel() == 1):
                return None
            split_dim = 0
        if tensor.shape[split_dim] % self.tp_size:
            raise ValueError(f"Cannot split {key} shape {tuple(tensor.shape)} across tensor parallel size {self.tp_size} on dimension {split_dim}")
        return tensor.chunk(self.tp_size, dim=split_dim)

    def _should_load_weights(self):
        if self.use_tp and self._use_local_tp_load():
            return True
        return super()._should_load_weights()

    def _use_local_tp_load(self):
        mode = self.config.get("parallel", {}).get("tp_load_mode", "broadcast")
        if mode not in ("broadcast", "local"):
            raise ValueError(f"Unsupported Wan TP load mode: {mode!r}; expected 'broadcast' or 'local'.")
        return mode == "local"

    def _shard_weights_locally(self, weight_dict):
        local_weights = {}
        for key, tensor in weight_dict.items():
            shards = self._split_tensor_for_tp(key, tensor)
            local_weights[key] = (tensor if shards is None else shards[self.tp_rank].contiguous()).to(self.device)
        return local_weights

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        if not self.use_tp:
            return super()._load_weights_from_rank0(weight_dict, is_weight_loader)

        if self._use_local_tp_load():
            if not is_weight_loader or weight_dict is None:
                raise RuntimeError("Wan TP local loading requires every rank to load the shared checkpoint")
            return self._shard_weights_locally(weight_dict)

        src_rank = 0
        target_device = self._rank_device()

        if is_weight_loader:
            processed, meta = {}, {}
            for key, tensor in weight_dict.items():
                shards = self._split_tensor_for_tp(key, tensor)
                if shards is None:
                    processed[key] = tensor
                    meta[key] = {"shape": tensor.shape, "dtype": tensor.dtype, "is_tp": False}
                else:
                    for rank, shard in enumerate(shards):
                        processed[f"{key}__tp_{rank}"] = shard.contiguous()
                    meta[key] = {"shape": shards[0].shape, "dtype": shards[0].dtype, "is_tp": True}
            obj_list = [meta]
        else:
            obj_list = [None]

        dist.broadcast_object_list(obj_list, src=src_rank)
        synced_meta = obj_list[0]

        distributed = {k: torch.empty(m["shape"], dtype=m["dtype"], device=target_device) for k, m in synced_meta.items()}

        for key in sorted(synced_meta.keys()):
            m = synced_meta[key]
            if m["is_tp"]:
                for r in range(self.tp_size):
                    buf = processed[f"{key}__tp_{r}"].to(target_device) if is_weight_loader else torch.empty(m["shape"], dtype=m["dtype"], device=target_device)
                    dist.broadcast(buf, src=src_rank, group=self.tp_group)
                    if r == self.tp_rank:
                        distributed[key].copy_(buf)
                    del buf
            else:
                if is_weight_loader:
                    distributed[key].copy_(processed[key].to(target_device))
                dist.broadcast(distributed[key], src=src_rank, group=self.tp_group)

        return {name: tensor.to(self.device) for name, tensor in distributed.items()}

    # ------------------------------------------------------------------ TP --

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer

        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WanTransformerInfer if not self.cpu_offload else WanOffloadTransformerInfer
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = WanTransformerInferTeaCaching
        elif self.config["feature_caching"] == "TaylorSeer":
            self.transformer_infer_class = WanTransformerInferTaylorCaching
        elif self.config["feature_caching"] == "Ada":
            self.transformer_infer_class = WanTransformerInferAdaCaching
        elif self.config["feature_caching"] == "Custom":
            self.transformer_infer_class = WanTransformerInferCustomCaching
        elif self.config["feature_caching"] == "FirstBlock":
            self.transformer_infer_class = WanTransformerInferFirstBlock
        elif self.config["feature_caching"] == "DualBlock":
            self.transformer_infer_class = WanTransformerInferDualBlock
        elif self.config["feature_caching"] == "DynamicBlock":
            self.transformer_infer_class = WanTransformerInferDynamicBlock
        elif self.config["feature_caching"] == "Mag":
            self.transformer_infer_class = WanTransformerInferMagCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        if hasattr(self.pre_infer, "set_rope"):
            first_attn = self.transformer_weights.blocks[0].compute_phases[0]
            rope = getattr(first_attn, "dreamzero_rope", None)
            self.pre_infer.set_rope(rope if rope is not None else first_attn.rope)
        if hasattr(self.pre_infer, "set_audio_rope"):
            self.pre_infer.set_audio_rope(self.transformer_weights.blocks[0].compute_phases[2].rope_1d)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    def _should_init_empty_model(self):
        if self.config.get("lora_configs") and self.config["lora_configs"] and not self.config.get("lora_dynamic_apply", False):
            if self.model_type in ["wan2.1"]:
                return True
            if self.model_type in ["wan2.2_moe_high_noise"]:
                for lora_config in self.config["lora_configs"]:
                    if lora_config["name"] == "high_noise_model":
                        return True
            if self.model_type in ["wan2.2_moe_low_noise"]:
                for lora_config in self.config["lora_configs"]:
                    if lora_config["name"] == "low_noise_model":
                        return True
        return False

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)

        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        x = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)
        if x is None:
            return None

        if self.config["seq_parallel"]:
            x = self._seq_parallel_post_process(x)

        noise_pred = self.post_infer.infer(x, pre_infer_out)[0]

        if self.clean_cuda_cache:
            del x, pre_infer_out
            torch.cuda.empty_cache()

        return noise_pred

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        x = pre_infer_out.x
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        padding_size = (world_size - (x.shape[0] % world_size)) % world_size
        if padding_size > 0:
            x = F.pad(x, (0, 0, 0, padding_size))

        pre_infer_out.x = torch.chunk(x, world_size, dim=0)[cur_rank]

        if self.config["model_cls"] == "wan2.2" and self.config["task"] in ["i2v", "s2v", "rs2v"]:
            embed, embed0 = pre_infer_out.embed, pre_infer_out.embed0

            padding_size = (world_size - (embed.shape[0] % world_size)) % world_size
            if padding_size > 0:
                embed = F.pad(embed, (0, 0, 0, padding_size))
                embed0 = F.pad(embed0, (0, 0, 0, 0, 0, padding_size))

            pre_infer_out.embed = torch.chunk(embed, world_size, dim=0)[cur_rank]
            pre_infer_out.embed0 = torch.chunk(embed0, world_size, dim=0)[cur_rank]

        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        world_size = dist.get_world_size(self.seq_p_group)
        gathered_x = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_x, x, group=self.seq_p_group)
        combined_output = torch.cat(gathered_x, dim=0)
        return combined_output

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        if self.config["enable_cfg"]:
            assert self.scheduler.sample_guide_scale is not None and self.scheduler.sample_guide_scale > 1.0, f"CFG requires sample_guide_scale > 1, got {self.scheduler.sample_guide_scale!r}"
            if self.config["cfg_parallel"]:
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
                else:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=False)

                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
                noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)

            noise_pred_guided = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
            self.scheduler.noise_pred_cond = noise_pred_cond
            self.scheduler.noise_pred_uncond = noise_pred_uncond
            self.scheduler.noise_pred_guided = noise_pred_guided
            self.scheduler.noise_pred = noise_pred_guided
        else:
            # ==================== No CFG ====================
            noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
            self.scheduler.noise_pred_cond = noise_pred
            self.scheduler.noise_pred_uncond = None
            self.scheduler.noise_pred_guided = noise_pred
            self.scheduler.noise_pred = noise_pred

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
