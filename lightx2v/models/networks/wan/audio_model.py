import os

import torch
import torch.distributed as dist

from lightx2v.models.networks.wan.infer.audio.post_infer import WanAudioPostInfer
from lightx2v.models.networks.wan.infer.audio.pre_infer import WanAudioPreInfer
from lightx2v.models.networks.wan.infer.audio.transformer_infer import WanAudioTransformerInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.audio.transformer_weights import WanAudioTransformerWeights
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.utils import load_weights


class WanAudioModel(WanModel):
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanAudioTransformerWeights

    def __init__(self, model_path, config, device):
        self.config = config
        self._load_adapter_ckpt()
        super().__init__(model_path, config, device)

    def _load_adapter_ckpt(self):
        if self.config.get("adapter_model_path", None) is None:
            if self.config.get("adapter_quantized", False):
                if self.config.get("adapter_quant_scheme", None) in ["fp8", "fp8-q8f"]:
                    adapter_model_name = "audio_adapter_model_fp8.safetensors"
                elif self.config.get("adapter_quant_scheme", None) == "int8":
                    adapter_model_name = "audio_adapter_model_int8.safetensors"
                else:
                    raise ValueError(f"Unsupported quant_scheme: {self.config.get('adapter_quant_scheme', None)}")
            else:
                adapter_model_name = "audio_adapter_model.safetensors"
            self.config.adapter_model_path = os.path.join(self.config.model_path, adapter_model_name)

        adapter_offload = self.config.get("cpu_offload", False)
        self.adapter_weights_dict = load_weights(self.config.adapter_model_path, cpu_offload=adapter_offload, remove_key="audio")
        if not adapter_offload and not dist.is_initialized():
            for key, value in self.adapter_weights_dict.items():
                self.adapter_weights_dict[key] = value.cuda()

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = WanAudioPreInfer
        self.post_infer_class = WanAudioPostInfer
        self.transformer_infer_class = WanAudioTransformerInfer

    def compile(self):
        new_inputs = {}
        new_inputs["text_encoder_output"] = {}
        new_inputs["text_encoder_output"]["context"] = torch.randn(1, 512, 4096, dtype=torch.bfloat16).cuda()
        new_inputs["text_encoder_output"]["context_null"] = torch.randn(1, 512, 4096, dtype=torch.bfloat16).cuda()

        new_inputs["image_encoder_output"] = {}
        new_inputs["image_encoder_output"]["clip_encoder_out"] = torch.randn(257, 1280, dtype=torch.bfloat16).cuda()
        new_inputs["image_encoder_output"]["vae_encoder_out"] = torch.randn(16, 1, 90, 160, dtype=torch.bfloat16).cuda()

        new_inputs["audio_encoder_output"] = torch.randn(1, 21, 128, 1024, dtype=torch.bfloat16).cuda()

        new_inputs["previmg_encoder_output"] = {}
        new_inputs["previmg_encoder_output"]["prev_latents"] = torch.randn(16, 21, 90, 160, dtype=torch.bfloat16).cuda()
        new_inputs["previmg_encoder_output"]["prev_mask"] = torch.randn(4, 21, 90, 160, dtype=torch.bfloat16).cuda()

        self.scheduler.latents = torch.randn(16, 21, 90, 160, dtype=torch.bfloat16).cuda()
        self.scheduler.timestep_input = torch.tensor([600.0], dtype=torch.float32).cuda()
        self.scheduler.audio_adapter_t_emb = torch.randn(1, 3, 5120, dtype=torch.bfloat16).cuda()

        self.enable_compile_mode("_infer_cond_uncond")

        self._infer_cond_uncond(new_inputs, infer_condition=True, graph_name="graph_01")

        self.disable_compile_mode("_infer_cond_uncond")

        self.select_graph("_infer_cond_uncond", "graph_01")
