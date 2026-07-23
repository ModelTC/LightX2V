import gc
import os
from contextlib import suppress

import torch
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
except ImportError:
    Qwen3VLForConditionalGeneration = None
    Qwen3VLProcessor = None


PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMG_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"


class LingBotVideoQwen3VLTextEncoder:
    def __init__(self, config):
        if Qwen3VLForConditionalGeneration is None or Qwen3VLProcessor is None:
            raise ImportError("transformers with Qwen3VL support is required for LingBot-Video text encoding.")
        self.config = config
        self.cpu_offload = config.get("qwen3vl_cpu_offload", config.get("text_encoder_cpu_offload", False))
        self.hidden_state_skip_layer = int(config.get("hidden_state_skip_layer", 0))
        self.token_length = int(config.get("token_length", 37698))
        self.prompt_template = config.get("prompt_template", PROMPT_TEMPLATE)
        self.img_prompt_template = config.get("img_prompt_template", IMG_PROMPT_TEMPLATE)
        self._crop_start = None
        self.load()

    def load(self):
        model_path = self.config.get("text_encoder_path", os.path.join(self.config["model_path"], "text_encoder"))
        processor_path = self.config.get("processor_path", os.path.join(self.config["model_path"], "processor"))
        attn_implementation = self.config.get(
            "qwen_attn_implementation",
            os.getenv("LINGBOT_QWEN_ATTN_IMPLEMENTATION", "sdpa"),
        )
        kwargs = {"torch_dtype": GET_DTYPE(), "attn_implementation": attn_implementation}
        logger.info(f"Loading LingBot-Video Qwen3VL text encoder from {model_path}")
        try:
            self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        except TypeError:
            kwargs["dtype"] = kwargs.pop("torch_dtype")
            self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        self.text_encoder.eval().requires_grad_(False)
        if self.cpu_offload:
            self.text_encoder.to("cpu")
        else:
            self.text_encoder.to(AI_DEVICE)
        self.processor = Qwen3VLProcessor.from_pretrained(processor_path)

    def _device(self):
        if self.cpu_offload:
            return torch.device(AI_DEVICE)
        return next(self.text_encoder.parameters()).device

    def _apply_text_to_template(self, text):
        return self.prompt_template.format(text)

    def _build_prompt_inputs(self, prompt, images=None):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)
        visual_template = self.img_prompt_template if images is not None else ""
        texts = [self._apply_text_to_template(visual_template + text) for text in prompts]
        return self.processor(
            text=texts,
            images=images,
            videos=None,
            video_metadata=None,
            do_resize=False,
            truncation=True,
            max_length=self.token_length,
            padding="longest",
            return_tensors="pt",
        )

    def _compute_crop_start(self):
        if self._crop_start is not None:
            return self._crop_start
        marker = "<|USER_INPUT_MARKER|>"
        marked = self.prompt_template.format(marker)
        marker_pos = marked.find(marker)
        if marker_pos < 0:
            self._crop_start = 0
            return self._crop_start
        prefix = self.processor(text=marked[:marker_pos], images=None, videos=None, return_tensors="pt")
        self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    @torch.no_grad()
    def infer(self, prompt, images=None):
        if self.cpu_offload:
            self.text_encoder.to(AI_DEVICE)
        device = self._device()
        inputs = self._build_prompt_inputs(prompt, images=images).to(device)
        outputs = self.text_encoder(
            **inputs,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )
        if self.hidden_state_skip_layer is not None:
            prompt_embeds = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
        else:
            prompt_embeds = outputs.last_hidden_state
        prompt_mask = inputs["attention_mask"]
        crop_start = self._compute_crop_start()
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]
        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]
        result = {
            "prompt_embeds": prompt_embeds.to(dtype=GET_DTYPE()),
            "prompt_mask": prompt_mask,
        }
        if self.cpu_offload:
            for key, value in list(result.items()):
                result[key] = value.to(AI_DEVICE)
            self.text_encoder.to("cpu")
            with suppress(Exception):
                torch.cuda.empty_cache()
            gc.collect()
        return result
