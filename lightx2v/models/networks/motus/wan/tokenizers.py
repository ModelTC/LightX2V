import html
import string

import ftfy
import regex as re
from transformers import AutoTokenizer


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(part.translate(str.maketrans("", "", string.punctuation)) for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)
        local_kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            local_kwargs.update({"padding": "max_length", "truncation": True, "max_length": self.seq_len})
        local_kwargs.update(**kwargs)
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(item) for item in sequence]
        ids = self.tokenizer(sequence, **local_kwargs)
        return (ids.input_ids, ids.attention_mask) if return_mask else ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        elif self.clean == "lower":
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = canonicalize(basic_clean(text))
        return text
