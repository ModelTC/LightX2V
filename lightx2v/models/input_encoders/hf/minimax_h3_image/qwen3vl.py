"""Qwen3-VL prompt and reference-image encoding for MiniMax-H3-Image."""

import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl import MINIMAX_H3_TEXT_TAG, MiniMaxH3Qwen3VLTextEncoder
from lightx2v.models.input_encoders.hf.minimax_h3_image.qwen3vl_vision import MiniMaxH3ImageQwen3VLVisionTower
from lightx2v_platform.base.global_var import AI_DEVICE


class MiniMaxH3ImageTextEncoder(MiniMaxH3Qwen3VLTextEncoder):
    max_sequence_length = 4096

    def load_vision_encoder(self):
        if self.vision_encoder is None:
            text_encoder_path = self._component_path("text_encoder_path", "text_encoder")
            vision_config = dict(self._read_model_config(text_encoder_path)["vision_config"])
            logger.info("Building MiniMax-H3-Image Qwen3-VL vision tower from {}", text_encoder_path)
            self.vision_encoder = MiniMaxH3ImageQwen3VLVisionTower.from_pretrained(text_encoder_path, vision_config)
            if not self.cpu_offload:
                self.vision_encoder.to(AI_DEVICE)
        return self.vision_encoder

    def _prompt_token_ids(self, prompt):
        if not isinstance(prompt, str):
            raise TypeError(f"MiniMax-H3-Image expects one prompt string, got {type(prompt).__name__}")
        token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"][: self.max_sequence_length]
        if not token_ids:
            raise ValueError("MiniMax-H3-Image prompt must encode to at least one token")
        return token_ids

    def _prepare_t2av_input_ids(self, prompt, device):
        return torch.tensor(self._prompt_token_ids(prompt), dtype=torch.long, device=device)

    def _prepare_keyframe_inputs(self, prompt, images):
        processor = self.load_processor()
        vision = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = vision["image_grid_thw"]
        merge_unit = processor.image_processor.merge_size**2
        vision_start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        image_pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_end = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        token_ids, token_tags = [], []
        for index, grid in enumerate(image_grid_thw):
            label = self.tokenizer(f"Picture {index + 1}: ", add_special_tokens=False)["input_ids"]
            block = [vision_start] + [image_pad] * (int(grid.prod()) // merge_unit) + [vision_end]
            token_ids += label + block
            token_tags += [MINIMAX_H3_TEXT_TAG] * len(label) + [0] * len(block)
        prompt_ids = self._prompt_token_ids(prompt)
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
        return token_ids, token_tags, vision["pixel_values"], image_grid_thw, None, None
