import os

import torch
import torch.distributed as dist
from loguru import logger

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
        load_from_rank0 = self.config.get("load_from_rank0", False)
        self.adapter_weights_dict = load_weights(self.config.adapter_model_path, cpu_offload=adapter_offload, remove_key="audio", load_from_rank0=load_from_rank0)
        if not dist.is_initialized() and not adapter_offload:
            for key in self.adapter_weights_dict:
                self.adapter_weights_dict[key] = self.adapter_weights_dict[key].cuda()

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = WanAudioPreInfer
        self.post_infer_class = WanAudioPostInfer
        self.transformer_infer_class = WanAudioTransformerInfer

    def get_graph_name(self, shape):
        return f"graph_{shape[0]}x{shape[1]}"

    def start_compile(self, shape):
        graph_name = self.get_graph_name(shape)
        logger.info(f"[Compile] Compile shape: {shape}, graph_name: {graph_name}")

        target_video_length = self.config.get("target_video_length", 81)
        latents_length = (target_video_length - 1) // 16 * 4 + 1
        latents_h = shape[0] // self.config.vae_stride[1]
        latents_w = shape[1] // self.config.vae_stride[2]

        new_inputs = {}
        new_inputs["text_encoder_output"] = {}
        new_inputs["text_encoder_output"]["context"] = torch.randn(1, 512, 4096, dtype=torch.bfloat16).cuda()
        new_inputs["text_encoder_output"]["context_null"] = torch.randn(1, 512, 4096, dtype=torch.bfloat16).cuda()

        new_inputs["image_encoder_output"] = {}
        new_inputs["image_encoder_output"]["clip_encoder_out"] = torch.randn(257, 1280, dtype=torch.bfloat16).cuda()
        new_inputs["image_encoder_output"]["vae_encoder_out"] = torch.randn(16, 1, latents_h, latents_w, dtype=torch.bfloat16).cuda()

        new_inputs["audio_encoder_output"] = torch.randn(1, latents_length, 128, 1024, dtype=torch.bfloat16).cuda()

        new_inputs["previmg_encoder_output"] = {}
        new_inputs["previmg_encoder_output"]["prev_latents"] = torch.randn(16, latents_length, latents_h, latents_w, dtype=torch.bfloat16).cuda()
        new_inputs["previmg_encoder_output"]["prev_mask"] = torch.randn(4, latents_length, latents_h, latents_w, dtype=torch.bfloat16).cuda()

        self.scheduler.latents = torch.randn(16, latents_length, latents_h, latents_w, dtype=torch.bfloat16).cuda()
        self.scheduler.timestep_input = torch.tensor([600.0], dtype=torch.float32).cuda()
        self.scheduler.audio_adapter_t_emb = torch.randn(1, 3, 5120, dtype=torch.bfloat16).cuda()

        self._infer_cond_uncond(new_inputs, infer_condition=True, graph_name=graph_name)

    def compile(self, compile_shapes):
        self.check_compile_shapes(compile_shapes)
        self.enable_compile_mode("_infer_cond_uncond")

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        for shape in compile_shapes:
            self.start_compile(shape)

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()

        self.disable_compile_mode("_infer_cond_uncond")
        logger.info(f"[Compile] Compile status: {self.get_compile_status()}")

    def check_compile_shapes(self, compile_shapes):
        for shape in compile_shapes:
            assert shape in [[480, 832], [544, 960], [720, 1280], [832, 480], [960, 544], [1280, 720], [480, 480], [576, 576], [704, 704], [960, 960]]

    def select_graph_for_compile(self):
        logger.info(f"tgt_h, tgt_w : {self.config.get('tgt_h')}, {self.config.get('tgt_w')}")
        self.select_graph("_infer_cond_uncond", f"graph_{self.config.get('tgt_h')}x{self.config.get('tgt_w')}")
        logger.info(f"[Compile] Compile status: {self.get_compile_status()}")
