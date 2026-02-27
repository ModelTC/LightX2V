import torch

from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.self_forcing.pre_infer import WanSFPreInfer
from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import WanSFTransformerInfer
from lightx2v.models.networks.wan.model import WanModel


class WanSFModel(WanModel):
    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, lora_path=lora_path, lora_strength=lora_strength)
        if config["model_cls"] not in ["wan2.1_sf_mtxg2"] and not config.get("dit_quantized", False):
            self.to_cuda()

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        file_path = self.config["sf_model_path"]
        _weight_dict = torch.load(file_path)
        _weight_dict = _weight_dict.get("generator_ema", _weight_dict)
        weight_dict = {}
        for k, v in _weight_dict.items():
            name = k[6:]
            weight = v.to(torch.bfloat16)
            weight_dict.update({name: weight})
        del _weight_dict
        return weight_dict

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        _weight_dict = super()._load_quant_ckpt(unified_dtype, sensitive_layer)
        weight_dict = {}
        for k, v in _weight_dict.items():
            weight_dict.update({k[6:]: v})
        del _weight_dict
        return weight_dict

    def _init_infer_class(self):
        self.pre_infer_class = WanSFPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanSFTransformerInfer

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        current_start_frame = self.scheduler.seg_index * self.scheduler.num_frame_per_block
        current_end_frame = (self.scheduler.seg_index + 1) * self.scheduler.num_frame_per_block
        noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)

        self.scheduler.noise_pred[:, current_start_frame:current_end_frame] = noise_pred
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
