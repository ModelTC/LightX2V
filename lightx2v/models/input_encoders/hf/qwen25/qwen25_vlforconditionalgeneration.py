import gc
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

    return width, height


class Qwen25_VLForConditionalGeneration_TextEncoder:
    def __init__(self, config):
        self.config = config
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = config["prompt_template_encode"]
        self.prompt_template_encode_start_idx = config["prompt_template_encode_start_idx"]
        """
        for Qwen-Image-Edit model, CONDITION_IMAGE_SIZE = 1024 * 1024
        for Qwen-Image-Edit-2509 model, CONDITION_IMAGE_SIZE = 384 * 384
        """
        self.CONDITION_IMAGE_SIZE = config.get("CONDITION_IMAGE_SIZE", 384 * 384)
        self.USE_IMAGE_ID_IN_PROMPT = config.get("USE_IMAGE_ID_IN_PROMPT", True)
        self.VAE_IMAGE_SIZE = 1024 * 1024

        self.cpu_offload = config.get("cpu_offload", False)
        self.run_device = self.config.get("run_device", "cuda")
        if self.cpu_offload:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.get("run_device", "cuda"))
        self.dtype = torch.bfloat16
        self.load()

    def load(self):
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(self.config["model_path"], "text_encoder"), torch_dtype=torch.bfloat16)
        if not self.cpu_offload:
            self.text_encoder = self.text_encoder.to(self.device)

        self.tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(self.config["model_path"], "tokenizer"))
        if self.config["task"] == "i2i":
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.config["vae_scale_factor"] * 2)
            self.processor = Qwen2VLProcessor.from_pretrained(os.path.join(self.config["model_path"], "processor"))

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def preprocess_image(self, image):
        image_width, image_height = image.size
        condition_width, condition_height = calculate_dimensions(self.CONDITION_IMAGE_SIZE, image_width / image_height)
        vae_width, vae_height = calculate_dimensions(self.VAE_IMAGE_SIZE, image_width / image_height)
        condition_image = self.image_processor.resize(image, condition_height, condition_width)
        vae_image = self.image_processor.preprocess(image, vae_height, vae_width).unsqueeze(2)

        return condition_image, vae_image, (condition_height, condition_width), (vae_height, vae_width)

    @torch.no_grad()
    def infer(self, text, image_list=None):
        if self.cpu_offload:
            self.text_encoder.to(self.device)

        if image_list is not None:
            condition_image_list = []
            vae_image_list = []
            condition_image_info_list = []
            vae_image_info_list = []
            if self.USE_IMAGE_ID_IN_PROMPT:
                base_img_prompt = ""
                img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
                for i, image in enumerate(image_list):
                    base_img_prompt += img_prompt_template.format(i + 1)
                    condition_image, vae_image, condition_image_info, vae_image_info = self.preprocess_image(image)
                    condition_image_list.append(condition_image)
                    vae_image_list.append(vae_image)
                    condition_image_info_list.append(condition_image_info)
                    vae_image_info_list.append(vae_image_info)
            else:
                base_img_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
                for i, image in enumerate(image_list):
                    condition_image, vae_image, condition_image_info, vae_image_info = self.preprocess_image(image)
                    condition_image_list.append(condition_image)
                    vae_image_list.append(vae_image)
                    condition_image_info_list.append(condition_image_info)
                    vae_image_info_list.append(vae_image_info)

            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(base_img_prompt + e) for e in text]

            model_inputs = self.processor(
                text=txt,
                images=condition_image_list,
                padding=True,
                return_tensors="pt",
            ).to(torch.device(self.device))

            encoder_hidden_states = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

            image_info = {
                "condition_image_list": condition_image_list,
                "vae_image_list": vae_image_list,
                "condition_image_info_list": condition_image_info_list,
                "vae_image_info_list": vae_image_info_list,
            }

        else:
            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(e) for e in text]

            image_info = {}
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
        prompt_embeds = prompt_embeds.repeat(1, self.config["num_images_per_prompt"], 1)
        prompt_embeds = prompt_embeds.view(self.config["batchsize"] * self.config["num_images_per_prompt"], seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, self.config["num_images_per_prompt"], 1)
        prompt_embeds_mask = prompt_embeds_mask.view(self.config["batchsize"] * self.config["num_images_per_prompt"], seq_len)

        if self.cpu_offload:
            self.text_encoder.to(torch.device("cpu"))
            if hasattr(torch, self.config.get("run_device", "cuda")):
                torch_module = getattr(torch, self.config.get("run_device", "cuda"))
                torch_module.empty_cache()
            gc.collect()

        return prompt_embeds, prompt_embeds_mask, image_info
