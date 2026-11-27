import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, CLIPTokenizer
from transformers.models.clip.tokenization_clip import BasicTokenizer

from lightx2v.models.networks.native_components.checkpoint import TensorWeights, load_tensors
from lightx2v.models.networks.native_components.clip import CLIPTextInfer

from ..infer.text import Qwen3TextInfer
from ..utils.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION


def load_clip_tokenizer(path, cleanup):
    tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=True)
    if cleanup == "basic":
        # HY-Motion's released requirements do not install ftfy. Transformers
        # then uses this exact fallback. Make it independent of other models'
        # optional packages, especially for curly quotes and Unicode text.
        tokenizer.fix_text = None
        tokenizer.nlp = BasicTokenizer(strip_accents=False, do_split_on_punc=False)
    elif cleanup == "ftfy":
        if tokenizer.fix_text is None:
            raise ImportError("clip_text_cleanup=ftfy requires the ftfy package")
    else:
        raise ValueError("clip_text_cleanup must be basic or ftfy")
    return tokenizer


class HYTextWeights:
    def __init__(self, qwen_path, clip_path, clip_text_cleanup="basic", device="cuda"):
        qwen_path, clip_path = Path(qwen_path), Path(clip_path)
        self.qwen_config = json.loads((qwen_path / "config.json").read_text())
        clip_config = json.loads((clip_path / "config.json").read_text())
        self.clip_config = clip_config.get("text_config", clip_config)
        self.qwen = TensorWeights(load_tensors(qwen_path, dtype=torch.bfloat16, device=device))
        # The language head is not part of text conditioning.
        self.qwen.tensors.pop("lm_head.weight", None)
        self.clip = TensorWeights(load_tensors(clip_path, dtype=torch.bfloat16, device=device))
        self.clip.tensors = {k: v for k, v in self.clip.tensors.items() if k.startswith("text_model.")}
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, padding_side="right", local_files_only=True)
        self.clip_tokenizer = load_clip_tokenizer(clip_path, clip_text_cleanup)
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
