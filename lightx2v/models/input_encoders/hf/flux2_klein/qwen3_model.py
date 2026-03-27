import os

import torch

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
except ImportError:
    Qwen2TokenizerFast = None
    Qwen3ForCausalLM = None

torch_device_module = getattr(torch, AI_DEVICE)


class Flux2Klein_TextEncoder:
    def __init__(self, config):
        self.config = config
        self.tokenizer_max_length = config.get("tokenizer_max_length", 512)
        self.cpu_offload = config.get("qwen3_cpu_offload", config.get("cpu_offload", False))
        self.text_encoder_out_layers = config.get("text_encoder_out_layers", (9, 18, 27))
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
            self.text_encoder = Qwen3ForCausalLM.from_pretrained(text_encoder_path, torch_dtype=GET_DTYPE(), device_map="cpu", **kwargs)
        else:
            self.text_encoder = Qwen3ForCausalLM.from_pretrained(text_encoder_path, torch_dtype=GET_DTYPE(), device_map=AI_DEVICE, **kwargs)

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    @torch.no_grad()
    def infer(self, prompt, image_list=None):
        if self.cpu_offload:
            self.text_encoder.to(AI_DEVICE)

        if isinstance(prompt, str):
            prompt = [prompt]

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_max_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(AI_DEVICE)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(AI_DEVICE)

        # Forward pass
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
