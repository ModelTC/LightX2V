"""MiniMax-H3-Image, using the shared H3 transformer blocks."""

from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.minimax_h3.adaln_cache import validate_adaln_cache_config
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer, validate_minimax_h3_residency_config
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.models.networks.minimax_h3.parallel import gather_packed_sequence, shard_packed_sequence, split_tensor_parallel_weight
from lightx2v.models.networks.minimax_h3.weights.transformer_weights import MiniMaxH3TransformerWeights
from lightx2v.models.networks.minimax_h3_image.checkpoint import convert_pdd_heads, convert_transformer_tensor
from lightx2v.models.networks.minimax_h3_image.infer import MiniMaxH3ImagePostInfer, MiniMaxH3ImagePreInfer
from lightx2v.models.networks.minimax_h3_image.weights import MiniMaxH3ImagePostWeights, MiniMaxH3ImagePreWeights
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class MiniMaxH3ImageModel(BaseTransformerModel):
    pre_weight_class = MiniMaxH3ImagePreWeights
    transformer_weight_class = MiniMaxH3TransformerWeights
    post_weight_class = MiniMaxH3ImagePostWeights

    def __init__(self, model_path, config, device):
        validate_minimax_h3_residency_config(config)
        if config.get("enable_cfg", False) or config.get("cfg_parallel", False):
            raise ValueError("MiniMax-H3-Image is guidance-distilled and does not have a CFG/unconditional branch")
        for option in (
            "dit_quantized",
            "lazy_load",
            "dit_disk_streaming",
            "shared_cpu_weights",
        ):
            if config.get(option, False):
                raise NotImplementedError(f"MiniMax-H3-Image does not support {option}")
        if config.get("lora_configs") or config.get("lora_dynamic_apply", False):
            raise NotImplementedError("LoRA loading is not implemented for MiniMax-H3-Image")
        if config.get("dit_quant_scheme", "Default") != "Default":
            raise ValueError("MiniMax-H3-Image requires original BF16/FP32 transformer weights")
        if GET_DTYPE() != torch.bfloat16:
            raise ValueError("MiniMax-H3-Image requires DTYPE=BF16")
        self.use_adaln_cache = bool(config.get("use_adaln_cache", False))
        if self.use_adaln_cache:
            validate_adaln_cache_config(config)
            self.remove_keys = [".adaln_proj.", "time_embedder.", "norm_out.linear."]
        # Match Diffusers' BF16 sine/cosine before applying RoPE to Q/K.
        config.setdefault("rope_type", "torch_real_rope")
        config.setdefault("rope_compute_dtype", "bf16")
        checkpoint_root = Path(model_path).expanduser()
        transformer_path = Path(config.get("dit_original_ckpt") or checkpoint_root / "transformer")
        self.pdd_heads_path = Path(config.get("pdd_heads_path") or checkpoint_root / "pdd_fused_heads.safetensors")
        if config.get("tensor_parallel", False) or config.get("seq_parallel", False):
            config["load_from_rank0"] = True
        super().__init__(str(transformer_path), config, torch.device(device))
        self._validate_parallel_config()
        self.block_offload = self.cpu_offload and self.offload_granularity == "block"
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        self.pre_infer_class = MiniMaxH3ImagePreInfer
        self.transformer_infer_class = MiniMaxH3OffloadTransformerInfer if self.cpu_offload else MiniMaxH3TransformerInfer
        self.post_infer_class = MiniMaxH3ImagePostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if self.block_offload:
            self._init_offload_manager()

    def _validate_parallel_config(self):
        if self.use_tp:
            dimensions = {"num_attention_heads": self.config["num_attention_heads"], "ffn_dim": self.config["ffn_dim"]}
            if not self.use_adaln_cache:
                dimensions["adaln_output_size"] = 18 * self.config["hidden_size"]
            invalid = {name: value for name, value in dimensions.items() if value % self.tp_size}
            if invalid:
                raise ValueError(f"MiniMax-H3-Image TP size {self.tp_size} must divide {invalid}")
        if self.seq_p_group is not None:
            parallel_type = self.config.get("parallel", {}).get("seq_p_attn_type", "ulysses")
            seq_size = dist.get_world_size(self.seq_p_group)
            if parallel_type == "ulysses" and self.config["num_attention_heads"] % (self.tp_size * seq_size):
                raise ValueError("MiniMax-H3-Image Ulysses attention heads must be divisible by tensor_p_size * seq_p_size")

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        self._checkpoint_load_error = None
        try:
            return self._read_checkpoint()
        except Exception as error:
            if not (dist.is_initialized() and dist.get_world_size() > 1 and self.config.get("device_mesh") is not None and self.config.get("load_from_rank0", False)):
                raise
            self._checkpoint_load_error = error
            return {}

    def _read_checkpoint(self):
        distributed_load = self.config.get("device_mesh") is not None and self.config.get("load_from_rank0", False)
        load_device = "cpu" if distributed_load else str(self.device)
        checkpoint_path = Path(self.model_path)
        paths = sorted(checkpoint_path.glob("*.safetensors")) if checkpoint_path.is_dir() else [checkpoint_path]
        if not paths:
            raise FileNotFoundError(f"MiniMax-H3-Image transformer weights not found: {checkpoint_path}")
        with safe_open(str(paths[0]), framework="pt", device="cpu") as handle:
            is_diffusers = any(name.startswith(("proj_in.", "transformer_blocks.", "pdd_proj_out.", "token_refiner.refiner_blocks.")) for name in handle.keys())
        fp32_modules = ("proj_in", "time_embedder", "pdd_proj_out") if is_diffusers else ("video_patch_proj", "time_embedder")
        weight_dict = {}
        for path in paths:
            with safe_open(str(path), framework="pt", device=load_device) as handle:
                for name in handle.keys():
                    if name == "rope.inv_freq" or (not is_diffusers and name.startswith("final_layer.video_out.")) or any(key in name for key in self.remove_keys):
                        continue
                    tensor = handle.get_tensor(name)
                    # Keep the released FP32 projections before converting keys.
                    dtype = torch.float32 if any(module in name.split(".") for module in fp32_modules) else torch.bfloat16
                    tensor = tensor.to(dtype=dtype)
                    converted = {name: tensor} if is_diffusers else convert_transformer_tensor(name, tensor, self.config)
                    weight_dict.update(converted)
        if not is_diffusers:
            with safe_open(str(self.pdd_heads_path), framework="pt", device=load_device) as handle:
                weight_dict.update(convert_pdd_heads({name: handle.get_tensor(name) for name in ("video_out.weight", "video_out.bias")}, self.config))
        self._record_checkpoint_metadata(weight_dict)
        return weight_dict

    def _split_checkpoint_tensor(self, name, tensor, rank):
        return split_tensor_parallel_weight(name, tensor, self.tp_size, rank, text_refiner_tensor_parallel=self.config.get("text_refiner_tensor_parallel", True))

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        # Convert on CPU before splitting the logical Q/K/V and SwiGLU segments.
        # Transfer one tensor shard at a time; no rank holds a full GPU checkpoint.
        rank = dist.get_rank()
        metadata = None
        load_error = getattr(self, "_checkpoint_load_error", None)
        if is_weight_loader and load_error is None:
            try:
                metadata = []
                for name, tensor in weight_dict.items():
                    local = self._split_checkpoint_tensor(name, tensor, 0)
                    metadata.append((name, tuple(local.shape), local.dtype, local.shape != tensor.shape))
            except Exception as error:
                load_error = error
        status = [None if load_error is None else f"{type(load_error).__name__}: {load_error}"]
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(f"MiniMax-H3-Image rank 0 checkpoint initialization failed: {status[0]}") from load_error
        coordinates = [None] * dist.get_world_size()
        dist.all_gather_object(coordinates, self.tp_rank)
        objects = [metadata]
        dist.broadcast_object_list(objects, src=0)
        target_device = torch.device(self.device)
        communication_device = torch.device("cpu") if dist.get_backend() == "gloo" else torch.device(AI_DEVICE)
        local_weights = {}
        for name, shape, dtype, sharded in objects[0]:
            if sharded:
                if rank == 0:
                    tensor = weight_dict.pop(name)
                    local_weights[name] = self._split_checkpoint_tensor(name, tensor, coordinates[0]).to(device=target_device, copy=True)
                    for peer in range(1, len(coordinates)):
                        shard = self._split_checkpoint_tensor(name, tensor, coordinates[peer]).to(device=communication_device)
                        dist.send(shard, dst=peer)
                        del shard
                else:
                    tensor = torch.empty(shape, dtype=dtype, device=communication_device)
                    dist.recv(tensor, src=0)
                    local_weights[name] = tensor.to(device=target_device)
            else:
                tensor = weight_dict.pop(name).to(device=communication_device) if rank == 0 else torch.empty(shape, dtype=dtype, device=communication_device)
                dist.broadcast(tensor, src=0)
                local_weights[name] = tensor.to(device=target_device)
        logger.info("MiniMax-H3-Image rank {} loaded TP shard {}/{} from rank 0", rank, self.tp_rank, self.tp_size)
        return local_weights

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        if not infer_condition:
            raise ValueError("MiniMax-H3-Image has no unconditional branch")
        prompt_embeds = inputs["text_encoder_output"]["prompt_embeds"]
        pre = self.pre_infer.infer(self.pre_weight, prompt_embeds)
        if self.seq_p_group is not None:
            pre = self._seq_parallel_pre_process(pre)
        hidden_states = self.transformer_infer.infer(self.transformer_weights, pre)
        if self.seq_p_group is not None:
            hidden_states = self._seq_parallel_post_process(hidden_states, pre)
        return self.post_infer.infer(self.post_weight, hidden_states, pre)

    def _seq_parallel_pre_process(self, pre_infer_out):
        pre = shard_packed_sequence(pre_infer_out, self.seq_p_group, allow_media_prefix=True)
        if pre.sequence_parallel_state is None and self.scheduler.step_index == 0 and dist.get_rank() == 0:
            logger.info("MiniMax-H3-Image packed sequence has fewer rows than SP ranks; replicating its dense attention while retaining TP")
        return pre

    def _seq_parallel_post_process(self, output, pre_infer_out):
        if pre_infer_out.sequence_parallel_state is None:
            return output
        return gather_packed_sequence(output, pre_infer_out, self.seq_p_group)

    def to_cpu(self):
        if not self.block_offload:
            return super().to_cpu()
        self.transformer_infer.release_resident_blocks()
        self.pre_weight.to_cpu()
        self.post_weight.to_cpu()
        # Keep the two device slots bound to their buffers for the next request.
        self.transformer_infer.offload_manager.need_init_first_buffer = True

    @torch.no_grad()
    def infer(self, inputs):
        if self.block_offload and self.scheduler.step_index == 0:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()
        self.scheduler.video_noise_pred = self._infer_cond_uncond(inputs)
