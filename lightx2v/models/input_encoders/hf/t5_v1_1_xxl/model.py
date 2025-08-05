import os

import torch
from transformers import T5EncoderModel, T5Tokenizer


class T5EncoderModel_v1_1_xxl:
    def __init__(self, config):
        self.config = config
        self.model = T5EncoderModel.from_pretrained(os.path.join(config.model_path, "text_encoder")).to(torch.bfloat16).to(torch.device("cuda"))
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.model_path, "tokenizer"), padding_side="right")

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to("cuda")

    def infer(self, texts, config):
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=config.text_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to("cuda")

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(texts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, config.text_len - 1 : -1])
            print(f"The following part of your input was truncated because `max_sequence_length` is set to  {self.text_len} tokens: {removed_text}")

        prompt_embeds = self.model(text_input_ids.to(torch.device("cuda")))[0]
        return prompt_embeds
