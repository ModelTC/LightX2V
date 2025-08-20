import math
import os

import torch
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

try:
    from diffusers.image_processor import VaeImageProcessor
    from transformers import Qwen2VLProcessor
except ImportError:
    VaeImageProcessor = None
    Qwen2VLProcessor = None

PREFERRED_QWENIMAGE_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


class Qwen25_VLForConditionalGeneration_TextEncoder:
    def __init__(self, config):
        self.config = config
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(config.model_path, "text_encoder")).to(torch.device("cuda")).to(torch.bfloat16)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(config.model_path, "tokenizer"))

        self.tokenizer_max_length = 1024
        self.prompt_template_encode = config.prompt_template_encode
        self.prompt_template_encode_start_idx = config.prompt_template_encode_start_idx

        if config.task == "i2i":
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.config["vae_scale_factor"] * 2)
            self.processor = Qwen2VLProcessor.from_pretrained(os.path.join(config.model_path, "processor"))

        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def preprocess_image(self, image):
        image_size = image.size
        width, height = image_size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, width / height)

        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.config.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image_height, image_width = self.image_processor.get_default_height_width(image)
            aspect_ratio = image_width / image_height

            if self.config._auto_resize:
                _, image_width, image_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_QWENIMAGE_RESOLUTIONS)
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            prompt_image = image
            image = self.image_processor.preprocess(image, image_height, image_width)
            image = image.unsqueeze(2)
        return prompt_image, image, (image_height, image_width)

    def infer(self, text, image=None):
        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in text]

        if image is not None:
            prompt_image, image, image_info = self.preprocess_image(image)
            model_inputs = self.processor(
                text=txt,
                images=prompt_image,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            encoder_hidden_states = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )
        else:
            prompt_image, image, image_info = None, None, None
            model_inputs = self.tokenizer(txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt").to(self.device)
            encoder_hidden_states = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_hidden_states=True,
            )

        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
        encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds_mask = encoder_attention_mask

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, self.config.num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(self.config.batchsize * self.config.num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, self.config.num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(self.config.batchsize * self.config.num_images_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask, image, image_info
