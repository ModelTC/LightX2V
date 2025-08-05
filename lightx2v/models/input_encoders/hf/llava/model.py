import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from loguru import logger
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration


def generate_crop_size_list(base_size=256, patch_size=32, max_ratio=4.0):
    """generate crop size list

    Args:
        base_size (int, optional): the base size for generate bucket. Defaults to 256.
        patch_size (int, optional): the stride to generate bucket. Defaults to 32.
        max_ratio (float, optional): th max ratio for h or w based on base_size . Defaults to 4.0.

    Returns:
        list: generate crop size list
    """
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
    """get the closest ratio in the buckets

    Args:
        height (float): video height
        width (float): video width
        ratios (list): video aspect ratio
        buckets (list): buckets generate by `generate_crop_size_list`

    Returns:
        the closest ratio in the buckets and the corresponding ratio
    """
    aspect_ratio = float(height) / float(width)
    diff_ratios = ratios - aspect_ratio

    if aspect_ratio >= 1:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x <= 0]
    else:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x > 0]

    closest_ratio_id = min(indices, key=lambda pair: abs(pair[1]))[0]
    closest_size = buckets[closest_ratio_id]
    closest_ratio = ratios[closest_ratio_id]

    return closest_size, closest_ratio


class TextEncoderHFLlavaModel:
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path
        self.init()
        self.load()

    def init(self):
        self.max_length = 359
        self.hidden_state_skip_layer = 2
        self.crop_start = 103
        self.double_return_token_id = 271
        self.image_emb_len = 576
        self.text_crop_start = self.crop_start - 1 + self.image_emb_len
        self.image_crop_start = 5
        self.image_crop_end = 581
        self.image_embed_interleave = 4

        self.prompt_template = (
            "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
            "1. The main content and theme of the video."
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
            "4. background environment, light, style and atmosphere."
            "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def load(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_path, low_cpu_mem_usage=True).to(torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="right")
        self.processor = CLIPImageProcessor.from_pretrained(self.model_path)

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to("cuda")

    @torch.no_grad()
    def infer(self, text, img, config):
        if config.cpu_offload:
            self.to_cuda()
        text = self.prompt_template.format(text)
        tokens = self.tokenizer(
            text,
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to("cuda")

        image_outputs = self.processor(img, return_tensors="pt")["pixel_values"].to(self.device)

        attention_mask = tokens["attention_mask"].to(self.device)

        outputs = self.model(input_ids=tokens["input_ids"], attention_mask=attention_mask, output_hidden_states=True, pixel_values=image_outputs)

        last_hidden_state = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]

        batch_indices, last_double_return_token_indices = torch.where(tokens["input_ids"] == self.double_return_token_id)

        last_double_return_token_indices = last_double_return_token_indices.reshape(1, -1)[:, -1]

        assistant_crop_start = last_double_return_token_indices - 1 + self.image_emb_len - 4
        assistant_crop_end = last_double_return_token_indices - 1 + self.image_emb_len

        attention_mask_assistant_crop_start = last_double_return_token_indices - 4
        attention_mask_assistant_crop_end = last_double_return_token_indices

        text_last_hidden_state = torch.cat([last_hidden_state[0, self.text_crop_start : assistant_crop_start[0].item()], last_hidden_state[0, assistant_crop_end[0].item() :]])
        text_attention_mask = torch.cat([attention_mask[0, self.crop_start : attention_mask_assistant_crop_start[0].item()], attention_mask[0, attention_mask_assistant_crop_end[0].item() :]])
        image_last_hidden_state = last_hidden_state[0, self.image_crop_start : self.image_crop_end]
        image_attention_mask = torch.ones(image_last_hidden_state.shape[0]).to(last_hidden_state.device).to(attention_mask.dtype)

        text_last_hidden_state.unsqueeze_(0)
        text_attention_mask.unsqueeze_(0)
        image_last_hidden_state.unsqueeze_(0)
        image_attention_mask.unsqueeze_(0)

        image_last_hidden_state = image_last_hidden_state[:, :: self.image_embed_interleave, :]
        image_attention_mask = image_attention_mask[:, :: self.image_embed_interleave]

        last_hidden_state = torch.cat([image_last_hidden_state, text_last_hidden_state], dim=1)
        attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1)

        if config.cpu_offload:
            self.to_cpu()
        return last_hidden_state, attention_mask
