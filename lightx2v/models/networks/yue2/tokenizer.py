# Adapted from ModelTC integration reference: multimodal-art-projection/YuE
# commit 1116bea76032bee5668e5f097294d082fa0b84f0.
"""Checkpoint-native text and ABC tokenizer."""

import base64
import unicodedata
from pathlib import Path


class YuE2TextTokenizer:
    def __init__(self, merge_file):
        import tiktoken

        self.merge_file = Path(merge_file)
        ranks = {base64.b64decode(t): int(r) for t, r in (line.split() for line in self.merge_file.read_bytes().splitlines() if line)}
        if len(ranks) != 151643:
            raise ValueError("Expected checkpoint-native qwen.tiktoken (151643 ordinary tokens)")
        if set(ranks.values()) != set(range(151643)):
            raise ValueError("Expected unique contiguous tokenizer ranks in 0..151642")
        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
        specials += [f"<extra_{i}>" for i in range(200)]
        specials[204:206] = ["<abc>", "</abc>"]
        pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self._enc = tiktoken.Encoding("YuE2", pat_str=pattern, mergeable_ranks=ranks, special_tokens={s: i + len(ranks) for i, s in enumerate(specials)})

    def encode(self, text):
        return self._enc.encode_ordinary(unicodedata.normalize("NFC", text))
