import math

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.wan.infer.infinitetalk.pre_infer import WanInfiniteTalkPreInfer
from lightx2v.models.networks.wan.infer.infinitetalk.transformer_infer import WanInfiniteTalkTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.infinitetalk.pre_weights import WanInfiniteTalkPreWeights
from lightx2v.models.networks.wan.weights.infinitetalk.transformer_weights import WanInfiniteTalkTransformerWeights
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


class WanInfiniteTalkModel(WanModel):
    pre_weight_class = WanInfiniteTalkPreWeights
    transformer_weight_class = WanInfiniteTalkTransformerWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type="infinitetalk", lora_path=lora_path, lora_strength=lora_strength)

    def _init_infer_class(self):
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("InfiniteTalk parity path requires feature_caching=NoCaching.")
        if self.config.get("cpu_offload", False):
            raise NotImplementedError("InfiniteTalk parity path currently requires cpu_offload=false.")
        self.pre_infer_class = WanInfiniteTalkPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanInfiniteTalkTransformerInfer

    def _load_adapter_ckpt(self):
        adapter_model_path = self.config.get("adapter_model_path", None)
        if adapter_model_path is None:
            raise ValueError("InfiniteTalk requires adapter_model_path to point to the single/multi adapter checkpoint.")
        logger.info(f"Loading InfiniteTalk adapter weights from {adapter_model_path}")
        sensitive_layer = set(self.sensitive_layer)
        sensitive_layer.update({"audio_proj.norm", "norm_x"})
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        return self._load_safetensor_to_dict(adapter_model_path, unified_dtype, sensitive_layer)

    @torch.no_grad()
    def _infer_infinitetalk_branch(self, inputs, infer_condition=True, use_audio=True):
        branch_inputs = dict(inputs)
        self.scheduler.infer_condition = infer_condition
        if not use_audio:
            branch_inputs["audio_encoder_output"] = torch.zeros_like(inputs["audio_encoder_output"])[-1:]
        return self._infer_cond_uncond(branch_inputs, infer_condition=infer_condition)

    @torch.no_grad()
    def infer(self, inputs):
        if self.config.get("cfg_parallel", False):
            raise NotImplementedError("InfiniteTalk text/audio CFG is serial in the parity path.")
        if self.config.get("use_apg", False):
            raise NotImplementedError("InfiniteTalk APG is not implemented in the LightX2V parity path yet.")

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        noise_pred_cond = self._infer_infinitetalk_branch(inputs, infer_condition=True, use_audio=True)

        if math.isclose(self.scheduler.sample_text_guide_scale, 1.0):
            noise_pred_drop_audio = self._infer_infinitetalk_branch(inputs, infer_condition=True, use_audio=False)
            noise_pred_guided = noise_pred_drop_audio + self.scheduler.sample_audio_guide_scale * (noise_pred_cond - noise_pred_drop_audio)
            self.scheduler.noise_pred_uncond = noise_pred_drop_audio
        else:
            noise_pred_drop_text = self._infer_infinitetalk_branch(inputs, infer_condition=False, use_audio=True)
            noise_pred_uncond = self._infer_infinitetalk_branch(inputs, infer_condition=False, use_audio=False)
            noise_pred_guided = (
                noise_pred_uncond
                + self.scheduler.sample_text_guide_scale * (noise_pred_cond - noise_pred_drop_text)
                + self.scheduler.sample_audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
            )
            self.scheduler.noise_pred_uncond = noise_pred_uncond
            self.scheduler.noise_pred_drop_text = noise_pred_drop_text

        self.scheduler.noise_pred_cond = noise_pred_cond
        self.scheduler.noise_pred_guided = noise_pred_guided
        self.scheduler.noise_pred = -noise_pred_guided

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        if dist.is_initialized():
            raise NotImplementedError("InfiniteTalk parity path currently runs without seq_parallel.")
        return pre_infer_out
