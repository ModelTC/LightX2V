import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, CLIPTokenizer

from lightx2v.models.networks.native_components.checkpoint import TensorWeights, load_tensors
from lightx2v.models.networks.native_components.clip import CLIPTextInfer

from ..infer.text import Qwen3TextInfer
from ..utils.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION


class HYTextWeights:
    def __init__(self, qwen_path, clip_path):
        qwen_path, clip_path = Path(qwen_path), Path(clip_path)
        self.qwen_config = json.loads((qwen_path / "config.json").read_text())
        clip_config = json.loads((clip_path / "config.json").read_text())
        self.clip_config = clip_config.get("text_config", clip_config)
        self.qwen = TensorWeights(load_tensors(qwen_path, dtype=torch.bfloat16))
        # The language head is not part of text conditioning.
        self.qwen.tensors.pop("lm_head.weight", None)
        self.clip = TensorWeights(load_tensors(clip_path, dtype=torch.float32))
        self.clip.tensors = {k: v for k, v in self.clip.tensors.items() if k.startswith("text_model.")}
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, padding_side="right", local_files_only=True)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
        marker = "<BOC>"
        full = self.tokenizer(self.format(marker), return_tensors="pt", add_special_tokens=True).input_ids[0].tolist()
        needle = self.tokenizer(marker, add_special_tokens=False).input_ids
        self.crop_start = next((i for i in range(len(full) - len(needle) + 1) if full[i : i + len(needle)] == needle), max(0, len(full) - 1))
        self.qwen_infer = Qwen3TextInfer(self.qwen_config)
        self.clip_infer = CLIPTextInfer(self.clip_config)

    def format(self, text):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}, {"role": "user", "content": text}], tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
