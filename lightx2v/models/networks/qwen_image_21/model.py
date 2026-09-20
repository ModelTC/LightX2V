import torch

from lightx2v.models.networks.base_model import BaseTransformerModel

from .infer.post_infer import QwenImage21PostInfer
from .infer.pre_infer import QwenImage21PreInfer
from .infer.transformer_infer import QwenImage21TransformerInfer
from .weights.post_weights import QwenImage21PostWeights
from .weights.pre_weights import QwenImage21PreWeights
from .weights.transformer_weights import QwenImage21TransformerWeights


class QwenImage21TransformerModel(BaseTransformerModel):
    pre_weight_class = QwenImage21PreWeights
    transformer_weight_class = QwenImage21TransformerWeights
    post_weight_class = QwenImage21PostWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        self.pre_infer_class = QwenImage21PreInfer
        self.transformer_infer_class = QwenImage21TransformerInfer
        self.post_infer_class = QwenImage21PostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class()
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class()

    # Required by BaseTransformerModel; the single-GPU path never calls these hooks.
    def _seq_parallel_pre_process(self, pre_infer_out):
        raise NotImplementedError("qwen_image_21 does not support sequence parallelism")

    def _seq_parallel_post_process(self, x):
        raise NotImplementedError("qwen_image_21 does not support sequence parallelism")

    def _infer_cond_uncond(self, inputs, infer_condition=True):
        branch = inputs["cond" if infer_condition else "uncond"]
        cache = branch["cache"]
        cached = self.scheduler.step_index > 0
        state = self.pre_infer.infer(self.pre_weight, self.scheduler.latents[0], branch["prompt_embeds"], inputs.get("image_latents"), branch["layout"], cached)
        hidden = self.transformer_infer.infer(self.transformer_weights, state, cache)
        noise = self.post_infer.infer(self.post_weight, hidden, state)
        return noise[-state.layout.target_len :].unsqueeze(0)

    @torch.no_grad()
    def infer(self, inputs):
        if self.config["enable_cfg"]:
            positive = self._infer_cond_uncond(inputs, True)
            negative = self._infer_cond_uncond(inputs, False)
            self.scheduler.noise_pred = negative + self.scheduler.sample_guide_scale * (positive - negative)
        else:
            self.scheduler.noise_pred = self._infer_cond_uncond(inputs)
