import os
import torch
import torch.distributed as dist
import glob
import json
from lightx2v.common.ops.attn import MaskMap
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.models.networks.wan.infer.feature_caching.transformer_infer import (
    WanTransformerInferTeaCaching,
    WanTransformerInferTaylorCaching,
    WanTransformerInferAdaCaching,
    WanTransformerInferCustomCaching,
    WanTransformerInferFirstBlock,
    WanTransformerInferDualBlock,
    WanTransformerInferDynamicBlock,
)
from lightx2v.models.networks.wan.infer.dist_infer.transformer_infer import WanTransformerDistInfer
from safetensors import safe_open
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *
from loguru import logger


class WanModel:
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        self.model_path = model_path
        self.config = config
        self.cpu_offload = self.config.get("cpu_offload", False)
        self.offload_granularity = self.config.get("offload_granularity", "block")

        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.dit_quantized = self.config.mm_config.get("mm_type", "Default") != "Default"

        if self.dit_quantized:
            dit_quant_scheme = self.config.mm_config.get("mm_type").split("-")[1]
            self.dit_quantized_ckpt = find_hf_model_path(config, "dit_quantized_ckpt", subdir=dit_quant_scheme)
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
        if self.config["seq_parallel"]:
            self.transformer_infer_class = WanTransformerDistInfer
        else:
            if self.config["feature_caching"] == "NoCaching":
                self.transformer_infer_class = WanTransformerInfer
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
            else:
                raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def _load_safetensor_to_dict(self, file_path, use_bf16, skip_bf16):
        with safe_open(file_path, framework="pt") as f:
            return {key: (f.get_tensor(key).to(torch.bfloat16) if use_bf16 or all(s not in key for s in skip_bf16) else f.get_tensor(key)).pin_memory().to(self.device) for key in f.keys()}

    def _load_ckpt(self, use_bf16, skip_bf16):
        safetensors_path = find_hf_model_path(self.config, "dit_original_ckpt", subdir="original")
        safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        weight_dict = {}
        for file_path in safetensors_files:
            file_weights = self._load_safetensor_to_dict(file_path, use_bf16, skip_bf16)
            weight_dict.update(file_weights)
        return weight_dict

    def _load_quant_ckpt(self, use_bf16, skip_bf16):
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
                    if f.get_tensor(k).dtype == torch.float:
                        if use_bf16 or all(s not in k for s in skip_bf16):
                            weight_dict[k] = f.get_tensor(k).pin_memory().to(torch.bfloat16).to(self.device)
                        else:
                            weight_dict[k] = f.get_tensor(k).pin_memory().to(self.device)
                    else:
                        weight_dict[k] = f.get_tensor(k).pin_memory().to(self.device)

        return weight_dict

    def _load_quant_split_ckpt(self, use_bf16, skip_bf16):
        lazy_load_model_path = self.dit_quantized_ckpt
        logger.info(f"Loading splited quant model from {lazy_load_model_path}")
        pre_post_weight_dict = {}

        safetensor_path = os.path.join(lazy_load_model_path, "non_block.safetensors")
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if f.get_tensor(k).dtype == torch.float:
                    if use_bf16 or all(s not in k for s in skip_bf16):
                        pre_post_weight_dict[k] = f.get_tensor(k).pin_memory().to(torch.bfloat16).to(self.device)
                    else:
                        pre_post_weight_dict[k] = f.get_tensor(k).pin_memory().to(self.device)
                else:
                    pre_post_weight_dict[k] = f.get_tensor(k).pin_memory().to(self.device)

        return pre_post_weight_dict

    def _init_weights(self, weight_dict=None):
        use_bf16 = GET_DTYPE() == "BF16"
        # Some layers run with float32 to achieve high accuracy
        skip_bf16 = {
            "norm",
            "embedding",
            "modulation",
            "time",
            "img_emb.proj.0",
            "img_emb.proj.4",
        }
        if weight_dict is None:
            if not self.dit_quantized or self.weight_auto_quant:
                self.original_weight_dict = self._load_ckpt(use_bf16, skip_bf16)
            else:
                if not self.config.get("lazy_load", False):
                    self.original_weight_dict = self._load_quant_ckpt(use_bf16, skip_bf16)
                else:
                    self.original_weight_dict = self._load_quant_split_ckpt(use_bf16, skip_bf16)
        else:
            self.original_weight_dict = weight_dict
        # init weights
        self.pre_weight = self.pre_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        # load weights
        self.pre_weight.load(self.original_weight_dict)
        self.post_weight.load(self.original_weight_dict)
        self.transformer_weights.load(self.original_weight_dict)

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        if self.config["cfg_parallel"]:
            self.infer_func = self.infer_with_cfg_parallel
        else:
            self.infer_func = self.infer_wo_cfg_parallel

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)

    def to_cpu(self):
        self.pre_weight.to_cpu()
        self.post_weight.to_cpu()
        self.transformer_weights.to_cpu()

    def to_cuda(self):
        self.pre_weight.to_cuda()
        self.post_weight.to_cuda()
        self.transformer_weights.to_cuda()

    @torch.no_grad()
    def infer(self, inputs):
        return self.infer_func(inputs)

    @torch.no_grad()
    def infer_wo_cfg_parallel(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.post_weight.to_cuda()

        if self.transformer_infer.mask_map is None:
            _, c, h, w = self.scheduler.latents.shape
            video_token_num = c * (h // 2) * (w // 2)
            self.transformer_infer.mask_map = MaskMap(video_token_num, c)

        embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=True)
        x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
        noise_pred_cond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        self.scheduler.noise_pred = noise_pred_cond

        if self.clean_cuda_cache:
            del x, embed, pre_infer_out, noise_pred_cond, grid_sizes
            torch.cuda.empty_cache()

        if self.config["enable_cfg"]:
            embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=False)
            x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
            noise_pred_uncond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

            self.scheduler.noise_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (self.scheduler.noise_pred - noise_pred_uncond)

            if self.clean_cuda_cache:
                del x, embed, pre_infer_out, noise_pred_uncond, grid_sizes
                torch.cuda.empty_cache()

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()

    @torch.no_grad()
    def infer_with_cfg_parallel(self, inputs):
        assert self.config["enable_cfg"], "enable_cfg must be True"
        cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
        assert dist.get_world_size(cfg_p_group) == 2, f"cfg_p_world_size must be equal to 2"
        cfg_p_rank = dist.get_rank(cfg_p_group)

        if cfg_p_rank == 0:
            embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=True)
            x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
            noise_pred = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]
        else:
            embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=False)
            x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
            noise_pred = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
        dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)

        noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
        noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1
        self.scheduler.noise_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)


class Wan22MoeModel(WanModel):
    def _load_ckpt(self, use_bf16, skip_bf16):
        safetensors_files = glob.glob(os.path.join(self.model_path, "*.safetensors"))
        weight_dict = {}
        for file_path in safetensors_files:
            file_weights = self._load_safetensor_to_dict(file_path, use_bf16, skip_bf16)
            weight_dict.update(file_weights)
        return weight_dict

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload and self.offload_granularity != "model":
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()

        embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=True)
        x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
        noise_pred_cond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        self.scheduler.noise_pred = noise_pred_cond

        if self.config["enable_cfg"]:
            embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=False)
            x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
            noise_pred_uncond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

            self.scheduler.noise_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (self.scheduler.noise_pred - noise_pred_uncond)

        if self.cpu_offload and self.offload_granularity != "model":
            self.pre_weight.to_cpu()
            self.post_weight.to_cpu()
