import os
import torch
import json
from lightx2v.models.networks.hunyuan.weights.pre_weights import HunyuanPreWeights
from lightx2v.models.networks.hunyuan.weights.post_weights import HunyuanPostWeights
from lightx2v.models.networks.hunyuan.weights.transformer_weights import HunyuanTransformerWeights
from lightx2v.models.networks.hunyuan.infer.pre_infer import HunyuanPreInfer
from lightx2v.models.networks.hunyuan.infer.post_infer import HunyuanPostInfer
from lightx2v.models.networks.hunyuan.infer.transformer_infer import HunyuanTransformerInfer
from lightx2v.models.networks.hunyuan.infer.feature_caching.transformer_infer import (
    HunyuanTransformerInferTaylorCaching,
    HunyuanTransformerInferTeaCaching,
    HunyuanTransformerInferAdaCaching,
    HunyuanTransformerInferCustomCaching,
)
import lightx2v.attentions.distributed.ulysses.wrap as ulysses_dist_wrap
import lightx2v.attentions.distributed.ring.wrap as ring_dist_wrap
from lightx2v.utils.envs import *
from loguru import logger
from safetensors import safe_open


class HunyuanModel:
    pre_weight_class = HunyuanPreWeights
    post_weight_class = HunyuanPostWeights
    transformer_weight_class = HunyuanTransformerWeights

    def __init__(self, model_path, config, device, args):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.args = args

        self.dit_quantized = self.config.mm_config.get("mm_type", "Default") != "Default"
        self.dit_quantized_ckpt = self.config.get("dit_quantized_ckpt", None)
        self.weight_auto_quant = self.config.mm_config.get("weight_auto_quant", False)
        if self.dit_quantized:
            assert self.weight_auto_quant or self.dit_quantized_ckpt is not None

        self._init_infer_class()
        self._init_weights()
        self._init_infer()

        if config["parallel_attn_type"]:
            if config["parallel_attn_type"] == "ulysses":
                ulysses_dist_wrap.parallelize_hunyuan(self)
            elif config["parallel_attn_type"] == "ring":
                ring_dist_wrap.parallelize_hunyuan(self)
            else:
                raise Exception(f"Unsuppotred parallel_attn_type")

        if self.config["cpu_offload"]:
            self.to_cpu()

    def _load_ckpt(self):
        if self.args.task == "t2v":
            ckpt_path = os.path.join(self.model_path, "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt")
        else:
            ckpt_path = os.path.join(self.model_path, "hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt")
        weight_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)["module"]
        return weight_dict

    def _load_quant_ckpt(self):
        ckpt_path = self.config.dit_quantized_ckpt
        logger.info(f"Loading quant dit model from {ckpt_path}")

        if ckpt_path.endswith(".pth"):
            logger.info(f"Loading {ckpt_path} as PyTorch model.")
            weight_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        else:
            index_files = [f for f in os.listdir(ckpt_path) if f.endswith(".index.json")]
            if not index_files:
                raise FileNotFoundError(f"No .pth file or *.index.json found in {ckpt_path}")

            index_path = os.path.join(ckpt_path, index_files[0])
            logger.info(f" Using safetensors index: {index_path}")

            with open(index_path, "r") as f:
                index_data = json.load(f)

            weight_dict = {}
            for filename in set(index_data["weight_map"].values()):
                safetensor_path = os.path.join(ckpt_path, filename)
                with safe_open(safetensor_path, framework="pt", device=str(self.device)) as f:
                    logger.info(f"Loading weights from {safetensor_path}")
                    for k in f.keys():
                        weight_dict[k] = f.get_tensor(k)
                        if weight_dict[k].dtype == torch.float:
                            weight_dict[k] = weight_dict[k].to(torch.bfloat16)

        return weight_dict

    def _init_weights(self):
        if not self.dit_quantized or self.weight_auto_quant:
            weight_dict = self._load_ckpt()
        else:
            weight_dict = self._load_quant_ckpt()
        # init weights
        self.pre_weight = self.pre_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        # load weights
        self.pre_weight.load(weight_dict)
        self.post_weight.load(weight_dict)
        self.transformer_weights.load(weight_dict)

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)

    def save_weights(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pre_state_dict = self.pre_weight.state_dict()
        logger.info(pre_state_dict.keys())

        post_state_dict = self.post_weight.state_dict()
        logger.info(post_state_dict.keys())

        transformer_state_dict = self.transformer_weights.state_dict()
        logger.info(transformer_state_dict.keys())

        save_dict = {}
        save_dict.update(pre_state_dict)
        save_dict.update(post_state_dict)
        save_dict.update(transformer_state_dict)

        save_path = os.path.join(save_path, "quant_weights.pth")
        torch.save(save_dict, save_path)
        logger.info(f"Save weights to {save_path}")

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
        if self.config["cpu_offload"]:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()

        inputs = self.pre_infer.infer(self.pre_weight, inputs)
        inputs = self.transformer_infer.infer(self.transformer_weights, *inputs)
        self.scheduler.noise_pred = self.post_infer.infer(self.post_weight, *inputs)

        if self.config["cpu_offload"]:
            self.pre_weight.to_cpu()
            self.post_weight.to_cpu()

    def _init_infer_class(self):
        self.pre_infer_class = HunyuanPreInfer
        self.post_infer_class = HunyuanPostInfer
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = HunyuanTransformerInfer
        elif self.config["feature_caching"] == "TaylorSeer":
            self.transformer_infer_class = HunyuanTransformerInferTaylorCaching
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = HunyuanTransformerInferTeaCaching
        elif self.config["feature_caching"] == "Ada":
            self.transformer_infer_class = HunyuanTransformerInferAdaCaching
        elif self.config["feature_caching"] == "Custom":
            self.transformer_infer_class = HunyuanTransformerInferCustomCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")
