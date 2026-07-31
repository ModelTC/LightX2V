import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.networks.wan.infer.audio.post_infer import WanAudioPostInfer
from lightx2v.models.networks.wan.infer.audio.pre_infer import WanAudioARPreInfer, WanAudioPreInfer
from lightx2v.models.networks.wan.infer.audio.transformer_infer import WanAudioARTransformerInfer, WanAudioTransformerInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.audio.transformer_weights import WanAudioTransformerWeights
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import load_weights
from lightx2v_platform.base.global_var import AI_DEVICE


class WanAudioModel(WanModel):
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanAudioTransformerWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, lora_path=lora_path, lora_strength=lora_strength)

    def _load_adapter_ckpt(self):
        if self.config.get("adapter_model_path", None) is None:
            if self.config.get("adapter_quantized", False):
                if self.config.get("adapter_quant_scheme", None) in ["fp8", "fp8-q8f", "fp8-vllm", "fp8-sgl", "fp8-torchao", "fp8-triton"]:
                    adapter_model_name = "audio_adapter_model_fp8.safetensors"
                elif self.config.get("adapter_quant_scheme", None) in ["int8", "int8-q8f", "int8-vllm", "int8-torchao", "int8-sgl", "int8-triton", "int8-tmo", "int8-npu", "int8-iluvatar"]:
                    adapter_model_name = "audio_adapter_model_int8.safetensors"
                elif self.config.get("adapter_quant_scheme", None) in ["mxfp4"]:
                    adapter_model_name = "audio_adapter_model_mxfp4.safetensors"
                elif self.config.get("adapter_quant_scheme", None) in ["mxfp6", "mxfp6-mxfp8"]:
                    adapter_model_name = "audio_adapter_model_mxfp6.safetensors"
                elif self.config.get("adapter_quant_scheme", None) in ["mxfp8"]:
                    adapter_model_name = "audio_adapter_model_mxfp8.safetensors"
                else:
                    raise ValueError(f"Unsupported quant_scheme: {self.config.get('adapter_quant_scheme', None)}")
            else:
                adapter_model_name = "audio_adapter_model.safetensors"
            self.config["adapter_model_path"] = os.path.join(self.config["model_path"], adapter_model_name)

        if self.config.get("dummy_model", False):
            from lightx2v.models.networks.base_model import SAFETENSORS_DTYPE_MAP, BaseTransformerModel

            dummy_device = str(self.device)
            logger.info(f"[DummyModel] Generating random adapter weights on device={dummy_device}")
            tensors_meta = BaseTransformerModel._read_safetensors_metadata(self.config["adapter_model_path"])
            adapter_weights_dict = {}

            for key, meta in tensors_meta.items():
                if "audio" in key:
                    continue
                shape = meta["shape"]
                dtype = GET_DTYPE()
                original_dtype = SAFETENSORS_DTYPE_MAP.get(meta["dtype"])
                if original_dtype is not None and not original_dtype.is_floating_point:
                    dtype = original_dtype
                adapter_weights_dict[key] = torch.randn(shape, dtype=dtype, device=dummy_device) if dtype.is_floating_point else torch.zeros(shape, dtype=dtype, device=dummy_device)
            return adapter_weights_dict

        adapter_offload = self.config.get("cpu_offload", False)
        load_from_rank0 = self.config.get("load_from_rank0", False)
        adapter_weights_dict = load_weights(self.config["adapter_model_path"], cpu_offload=adapter_offload, remove_key="audio", load_from_rank0=load_from_rank0)
        target_device = torch.device("cpu") if adapter_offload else torch.device(AI_DEVICE)
        target_dtype = GET_DTYPE()
        for key, tensor in adapter_weights_dict.items():
            adapter_weights_dict[key] = tensor.to(device=target_device, dtype=target_dtype) if (tensor.is_floating_point() and tensor.dtype != torch.float8_e4m3fn) else tensor.to(device=target_device)
        return adapter_weights_dict

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = WanAudioPreInfer
        self.post_infer_class = WanAudioPostInfer
        self.transformer_infer_class = WanAudioTransformerInfer

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        x = pre_infer_out.x
        person_mask_latens = pre_infer_out.adapter_args["person_mask_latens"]

        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)

        padding_size = (world_size - (x.shape[0] % world_size)) % world_size
        if padding_size > 0:
            x = F.pad(x, (0, 0, 0, padding_size))
            if person_mask_latens is not None:
                person_mask_latens = F.pad(person_mask_latens, (0, padding_size))

        pre_infer_out.x = torch.chunk(x, world_size, dim=0)[cur_rank]
        if person_mask_latens is not None:
            pre_infer_out.adapter_args["person_mask_latens"] = torch.chunk(person_mask_latens, world_size, dim=1)[cur_rank]

        if self.config["model_cls"] in ["wan2.2", "wan2.2_audio"] and self.config["task"] in ["i2v", "s2v", "rs2v"]:
            embed, embed0 = pre_infer_out.embed, pre_infer_out.embed0
            padding_size = (world_size - (embed.shape[0] % world_size)) % world_size
            if padding_size > 0:
                embed = F.pad(embed, (0, 0, 0, padding_size))
                embed0 = F.pad(embed0, (0, 0, 0, 0, 0, padding_size))
            pre_infer_out.embed = torch.chunk(embed, world_size, dim=0)[cur_rank]
            pre_infer_out.embed0 = torch.chunk(embed0, world_size, dim=0)[cur_rank]
        return pre_infer_out


class WanAudioARModel(WanAudioModel):
    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = WanAudioARPreInfer
        self.post_infer_class = WanAudioPostInfer
        self.transformer_infer_class = WanAudioARTransformerInfer

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)
        if self.config["seq_parallel"] and not inputs.get("_ar_ref_prefill", False):
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        x = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)

        if inputs.get("_ar_ref_prefill", False):
            noise_pred = None
        else:
            if self.config["seq_parallel"]:
                x = self._seq_parallel_post_process(x)
            noise_pred = self.post_infer.infer(x, pre_infer_out)[0]
            self.scheduler.noise_pred = noise_pred

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
        return noise_pred
