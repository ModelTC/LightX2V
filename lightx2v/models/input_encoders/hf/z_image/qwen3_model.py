import gc
import os

import torch
from PIL import Image

try:
    from transformers import Qwen2Tokenizer, Qwen3Model
except ImportError:
    Qwen2Tokenizer = None
    Qwen3Model = None

from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

try:
    from diffusers.image_processor import VaeImageProcessor
except ImportError:
    VaeImageProcessor = None


class Qwen3Model_TextEncoder:
    def __init__(self, config):
        self.config = config
        self.tokenizer_max_length = 512
        self.cpu_offload = config.get("qwen3_cpu_offload", config.get("cpu_offload", False))
        self.dtype = torch.bfloat16
        self.load()

    def load(self):
        self.text_encoder = Qwen3Model.from_pretrained(os.path.join(self.config["model_path"], "text_encoder"), torch_dtype=torch.bfloat16)
        if not self.cpu_offload:
            self.text_encoder = self.text_encoder.to(AI_DEVICE)

        self.tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(self.config["model_path"], "tokenizer"))

        if self.config["task"] == "i2i":
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.config.get("vae_scale_factor", 8) * 2)

    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            preprocessed_image = self.image_processor.preprocess(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            preprocessed_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return preprocessed_image

    @torch.no_grad()
    def infer(self, prompt, image_list=None):
        if self.cpu_offload:
            self.text_encoder.to(AI_DEVICE)

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [{"role": "user", "content": prompt_item}]
            prompt_tokens = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            prompt[i] = prompt_tokens

        text_inputs = self.tokenizer(prompt, max_length=self.tokenizer_max_length, padding="max_length", truncation=True, return_tensors="pt").to(AI_DEVICE)
        prompt_masks = text_inputs.attention_mask.bool().to(AI_DEVICE)

        prompt_embeds = self.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]
        embedding_list = []
        for i in range(len(prompt_embeds)):
            extracted = prompt_embeds[i][prompt_masks[i]]
            embedding_list.append(extracted)
        image_info = {}
        if self.config["task"] == "i2i" and image_list is not None:
            vae_image_list = []
            for image in image_list:
                preprocessed_image = self.preprocess_image(image)
                vae_image_list.append(preprocessed_image)

            image_info = {
                "vae_image_list": vae_image_list,
            }

        if self.cpu_offload:
            self.text_encoder.to(torch.device("cpu"))
            torch_device_module.empty_cache()
            gc.collect()

        return embedding_list, image_info
