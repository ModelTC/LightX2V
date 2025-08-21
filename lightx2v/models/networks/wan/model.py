import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open

from lightx2v.common.ops.attn import MaskMap
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
)
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *

try:
    import gguf
except ImportError:
    gguf = None


class WanModel:
    pre_weight_class = WanPreWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        self.model_path = model_path
        self.config = config
        self.cpu_offload = self.config.get("cpu_offload", False)
        self.offload_granularity = self.config.get("offload_granularity", "block")

        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.dit_quantized = self.config.mm_config.get("mm_type", "Default") != "Default"

        if self.dit_quantized:
            dit_quant_scheme = self.config.mm_config.get("mm_type").split("-")[1]
            if self.config.model_cls == "wan2.1_distill":
                dit_quant_scheme = "distill_" + dit_quant_scheme
            if dit_quant_scheme == "gguf":
                self.dit_quantized_ckpt = find_gguf_model_path(config, "dit_quantized_ckpt", subdir=dit_quant_scheme)
                self.config.use_gguf = True
            else:
                self.dit_quantized_ckpt = find_hf_model_path(
                    config,
                    self.model_path,
                    "dit_quantized_ckpt",
                    subdir=dit_quant_scheme,
                )
            quant_config_path = os.path.join(self.dit_quantized_ckpt, "config.json")
            if os.path.exists(quant_config_path):
                with open(quant_config_path, "r") as f:
                    quant_model_config = json.load(f)
                self.config.update(quant_model_config)
        else:
            self.dit_quantized_ckpt = None
            assert not self.config.get("lazy_load", False)

        self.config.dit_quantized_ckpt = self.dit_quantized_ckpt

        self.weight_auto_quant = self.config.mm_config.get("weight_auto_quant", False)
        if self.dit_quantized:
            assert self.weight_auto_quant or self.dit_quantized_ckpt is not None

        self.device = device
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

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

    def _should_load_weights(self):
        """Determine if current rank should load weights from disk."""
        if self.config.get("device_mesh") is None:
            # Single GPU mode
            return True
        elif dist.is_initialized():
            # Multi-GPU mode, only rank 0 loads
            if dist.get_rank() == 0:
                logger.info(f"Loading weights from {self.model_path}")
                return True
        return False

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        with safe_open(file_path, framework="pt") as f:
            return {
                key: (f.get_tensor(key).to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else f.get_tensor(key).to(GET_SENSITIVE_DTYPE())).pin_memory().to(self.device)
                for key in f.keys()
            }

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        safetensors_path = find_hf_model_path(self.config, self.model_path, "dit_original_ckpt", subdir="original")
        safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        weight_dict = {}
        for file_path in safetensors_files:
            file_weights = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
            weight_dict.update(file_weights)
        return weight_dict

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        ckpt_path = self.dit_quantized_ckpt
        logger.info(f"Loading quant dit model from {ckpt_path}")

        index_files = [f for f in os.listdir(ckpt_path) if f.endswith(".index.json")]
        if not index_files:
            raise FileNotFoundError(f"No *.index.json found in {ckpt_path}")

        index_path = os.path.join(ckpt_path, index_files[0])
        logger.info(f" Using safetensors index: {index_path}")

        with open(index_path, "r") as f:
            index_data = json.load(f)

        weight_dict = {}
        for filename in set(index_data["weight_map"].values()):
            safetensor_path = os.path.join(ckpt_path, filename)
            with safe_open(safetensor_path, framework="pt") as f:
                logger.info(f"Loading weights from {safetensor_path}")
                for k in f.keys():
                    if f.get_tensor(k).dtype in [
                        torch.float16,
                        torch.bfloat16,
                        torch.float,
                    ]:
                        if unified_dtype or all(s not in k for s in sensitive_layer):
                            weight_dict[k] = f.get_tensor(k).pin_memory().to(GET_DTYPE()).to(self.device)
                        else:
                            weight_dict[k] = f.get_tensor(k).pin_memory().to(GET_SENSITIVE_DTYPE()).to(self.device)
                    else:
                        weight_dict[k] = f.get_tensor(k).pin_memory().to(self.device)

        return weight_dict

    def _load_quant_split_ckpt(self, unified_dtype, sensitive_layer):
        lazy_load_model_path = self.dit_quantized_ckpt
        logger.info(f"Loading splited quant model from {lazy_load_model_path}")
        pre_post_weight_dict = {}

        safetensor_path = os.path.join(lazy_load_model_path, "non_block.safetensors")
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if f.get_tensor(k).dtype in [
                    torch.float16,
                    torch.bfloat16,
                    torch.float,
                ]:
                    if unified_dtype or all(s not in k for s in sensitive_layer):
                        pre_post_weight_dict[k] = f.get_tensor(k).pin_memory().to(GET_DTYPE()).to(self.device)
                    else:
                        pre_post_weight_dict[k] = f.get_tensor(k).pin_memory().to(GET_SENSITIVE_DTYPE()).to(self.device)
                else:
                    pre_post_weight_dict[k] = f.get_tensor(k).pin_memory().to(self.device)

        return pre_post_weight_dict

    def _load_gguf_ckpt(self):
        gguf_path = self.dit_quantized_ckpt
        logger.info(f"Loading gguf-quant dit model from {gguf_path}")
        reader = gguf.GGUFReader(gguf_path)
        for tensor in reader.tensors:
            # TODO: implement _load_gguf_ckpt
            pass

    def _init_weights(self, weight_dict=None):
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        # Some layers run with float32 to achieve high accuracy
        sensitive_layer = {
            "norm",
            "embedding",
            "modulation",
            "time",
            "img_emb.proj.0",
            "img_emb.proj.4",
            "before_proj",  # vace
            "after_proj",  # vace
        }

        if weight_dict is None:
            is_weight_loader = self._should_load_weights()
            if is_weight_loader:
                if not self.dit_quantized or self.weight_auto_quant:
                    # Load original weights
                    weight_dict = self._load_ckpt(unified_dtype, sensitive_layer)
                else:
                    # Load quantized weights
                    if not self.config.get("lazy_load", False):
                        weight_dict = self._load_quant_ckpt(unified_dtype, sensitive_layer)
                    else:
                        weight_dict = self._load_quant_split_ckpt(unified_dtype, sensitive_layer)

            if self.config.get("device_mesh") is not None:
                weight_dict = self._load_weights_distribute(weight_dict, is_weight_loader)

            self.original_weight_dict = weight_dict
        else:
            self.original_weight_dict = weight_dict

        # Initialize weight containers
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)

        # Load weights into containers
        self.pre_weight.load(self.original_weight_dict)
        self.transformer_weights.load(self.original_weight_dict)

    def _load_weights_distribute(self, weight_dict, is_weight_loader):
        global_src_rank = 0
        target_device = "cpu" if self.cpu_offload else "cuda"

        if is_weight_loader:
            meta_dict = {}
            for key, tensor in weight_dict.items():
                meta_dict[key] = {"shape": tensor.shape, "dtype": tensor.dtype}

            obj_list = [meta_dict]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]
        else:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]

        distributed_weight_dict = {}
        for key, meta in synced_meta_dict.items():
            distributed_weight_dict[key] = torch.empty(meta["shape"], dtype=meta["dtype"], device=target_device)

        if target_device == "cuda":
            dist.barrier(device_ids=[torch.cuda.current_device()])

        for key in sorted(synced_meta_dict.keys()):
            if is_weight_loader:
                distributed_weight_dict[key].copy_(weight_dict[key], non_blocking=True)

            if target_device == "cpu":
                if is_weight_loader:
                    gpu_tensor = distributed_weight_dict[key].cuda()
                    dist.broadcast(gpu_tensor, src=global_src_rank)
                    distributed_weight_dict[key].copy_(gpu_tensor.cpu(), non_blocking=True)
                    del gpu_tensor
                    torch.cuda.empty_cache()
                else:
                    gpu_tensor = torch.empty_like(distributed_weight_dict[key], device="cuda")
                    dist.broadcast(gpu_tensor, src=global_src_rank)
                    distributed_weight_dict[key].copy_(gpu_tensor.cpu(), non_blocking=True)
                    del gpu_tensor
                    torch.cuda.empty_cache()

                if distributed_weight_dict[key].is_pinned():
                    distributed_weight_dict[key].copy_(distributed_weight_dict[key], non_blocking=True)
            else:
                dist.broadcast(distributed_weight_dict[key], src=global_src_rank)

        if target_device == "cuda":
            torch.cuda.synchronize()
        else:
            for tensor in distributed_weight_dict.values():
                if tensor.is_pinned():
                    tensor.copy_(tensor, non_blocking=False)

        logger.info(f"Weights distributed across {dist.get_world_size()} devices on {target_device}")
        return distributed_weight_dict

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)

    def to_cpu(self):
        self.pre_weight.to_cpu()
        self.transformer_weights.to_cpu()

    def to_cuda(self):
        self.pre_weight.to_cuda()
        self.transformer_weights.to_cuda()

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        if self.transformer_infer.mask_map is None:
            _, c, h, w = self.scheduler.latents.shape
            video_token_num = c * (h // 2) * (w // 2)
            self.transformer_infer.mask_map = MaskMap(video_token_num, c)

        if self.config["enable_cfg"]:
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

            self.scheduler.noise_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            # ==================== No CFG ====================
            self.scheduler.noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)

        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        x = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)

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

        if self.config["model_cls"] == "wan2.2" and self.config["task"] == "i2v":
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
