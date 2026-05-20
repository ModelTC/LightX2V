import os

import torch

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    from transformers import AutoProcessor, Mistral3ForConditionalGeneration
except ImportError:
    AutoProcessor = None
    Mistral3ForConditionalGeneration = None

torch_device_module = getattr(torch, AI_DEVICE)

SYSTEM_MESSAGE = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation."


def format_input(prompts, system_message=SYSTEM_MESSAGE, images=None):
    """Format prompts into the conversation format expected by Mistral3's apply_chat_template."""
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    if images is None or len(images) == 0:
        return [
            [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]
    else:
        assert len(images) == len(prompts), "Number of images must match number of prompts"
        messages = [[{"role": "system", "content": [{"type": "text", "text": system_message}]}] for _ in cleaned_txt]
        for i, (el, imgs) in enumerate(zip(messages, images)):
            if imgs is not None:
                el.append({"role": "user", "content": [{"type": "image", "image": img} for img in imgs]})
            el.append({"role": "user", "content": [{"type": "text", "text": cleaned_txt[i]}]})
        return messages


class Flux2Dev_TextEncoder:
    def __init__(self, config):
        self.config = config
        self.tokenizer_max_length = config.get("tokenizer_max_length", 512)
        self.cpu_offload = config.get("mistral3_cpu_offload", config.get("cpu_offload", False))
        self.text_encoder_out_layers = config.get("text_encoder_out_layers", (10, 20, 30))
        self.load()

    def load(self):
        model_path = self.config["model_path"]
        kwargs = {}
        if not os.path.exists(model_path):
            text_encoder_path = model_path
            kwargs["subfolder"] = "text_encoder"
            tokenizer_path = model_path
            tokenizer_kwargs = {"subfolder": "tokenizer"}
        else:
            text_encoder_path = self.config.get("text_encoder_path", os.path.join(model_path, "text_encoder"))
            tokenizer_path = self.config.get("tokenizer_path", os.path.join(model_path, "tokenizer"))
            tokenizer_kwargs = {}

        if self.cpu_offload:
            self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(text_encoder_path, torch_dtype=GET_DTYPE(), device_map="cpu", **kwargs)
        else:
            self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(text_encoder_path, torch_dtype=GET_DTYPE(), device_map=AI_DEVICE, **kwargs)

        self.tokenizer = AutoProcessor.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    @torch.no_grad()
    def infer(self, prompt, image_list=None):
        if self.cpu_offload:
            self.text_encoder.to(AI_DEVICE)

        if isinstance(prompt, str):
            prompt = [prompt]

        messages_batch = format_input(prompts=prompt, system_message=SYSTEM_MESSAGE, images=image_list)

        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
        )

        input_ids = inputs["input_ids"].to(AI_DEVICE)
        attention_mask = inputs["attention_mask"].to(AI_DEVICE)

        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in self.text_encoder_out_layers], dim=1)
        out = out.to(dtype=GET_DTYPE(), device=AI_DEVICE)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        if self.cpu_offload:
            self.text_encoder.to(torch.device("cpu"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        embedding_list = [prompt_embeds[i] for i in range(batch_size)]
        return embedding_list, {}
