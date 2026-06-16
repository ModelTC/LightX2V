import gc
import json
import os

import torch
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    from transformers import AutoTokenizer, Ministral3ForCausalLM, Mistral3Model
except ImportError:
    AutoTokenizer = None
    Mistral3Model = None
    Ministral3ForCausalLM = None

torch_device_module = getattr(torch, AI_DEVICE)


class ErnieImageTextEncoder:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("text_encoder_cpu_offload", config.get("cpu_offload", False))
        self.pe_cpu_offload = config.get("pe_cpu_offload", self.cpu_offload)
        self.use_pe = config.get("use_pe", True)
        self.pe_temperature = config.get("pe_temperature", 0.6)
        self.pe_top_p = config.get("pe_top_p", 0.95)
        self.load()

    def load(self):
        if AutoTokenizer is None or Mistral3Model is None:
            raise ImportError("ERNIE-Image text encoder requires transformers with Mistral3Model support.")

        text_encoder_path = self.config.get("text_encoder_path", os.path.join(self.config["model_path"], "text_encoder"))
        tokenizer_path = self.config.get("tokenizer_path", os.path.join(self.config["model_path"], "tokenizer"))
        text_device = "cpu" if self.cpu_offload else AI_DEVICE
        self.text_encoder = Mistral3Model.from_pretrained(
            text_encoder_path,
            torch_dtype=GET_DTYPE(),
            device_map=text_device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.pe = None
        self.pe_tokenizer = None
        if self.use_pe:
            if Ministral3ForCausalLM is None:
                logger.warning("Prompt Enhancer requested, but Ministral3ForCausalLM is unavailable. Disabling PE.")
                self.use_pe = False
                return
            pe_path = self.config.get("pe_path", os.path.join(self.config["model_path"], "pe"))
            pe_tokenizer_path = self.config.get("pe_tokenizer_path", os.path.join(self.config["model_path"], "pe_tokenizer"))
            if os.path.exists(pe_path) and os.path.exists(pe_tokenizer_path):
                pe_device = "cpu" if self.pe_cpu_offload else AI_DEVICE
                self.pe = Ministral3ForCausalLM.from_pretrained(
                    pe_path,
                    torch_dtype=GET_DTYPE(),
                    device_map=pe_device,
                )
                self.pe_tokenizer = AutoTokenizer.from_pretrained(pe_tokenizer_path)
            else:
                logger.warning("Prompt Enhancer files are missing. Disabling PE.")
                self.use_pe = False

    @torch.no_grad()
    def _enhance_prompt_with_pe(self, prompt, width, height):
        if self.pe is None or self.pe_tokenizer is None:
            return prompt
        if self.pe_cpu_offload:
            self.pe.to(AI_DEVICE)

        user_content = json.dumps(
            {"prompt": prompt, "width": int(width), "height": int(height)},
            ensure_ascii=False,
        )
        messages = [{"role": "user", "content": user_content}]
        input_text = self.pe_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.pe_tokenizer(input_text, return_tensors="pt").to(AI_DEVICE)
        output_ids = self.pe.generate(
            **inputs,
            max_new_tokens=self.pe_tokenizer.model_max_length,
            do_sample=self.pe_temperature != 1.0 or self.pe_top_p != 1.0,
            temperature=self.pe_temperature,
            top_p=self.pe_top_p,
            pad_token_id=self.pe_tokenizer.pad_token_id,
            eos_token_id=self.pe_tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        revised_prompt = self.pe_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if self.pe_cpu_offload:
            self.pe.to("cpu")
            torch_device_module.empty_cache()
            gc.collect()
        return revised_prompt

    def _encode_one(self, prompt):
        ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            padding=False,
        )["input_ids"]
        if len(ids) == 0:
            ids = [self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0]
        input_ids = torch.tensor([ids], device=AI_DEVICE)
        outputs = self.text_encoder(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-2][0].to(dtype=GET_DTYPE(), device=AI_DEVICE)

    @torch.no_grad()
    def infer(self, prompt, use_pe=None, width=1024, height=1024):
        if isinstance(prompt, str):
            prompt = [prompt]
        if use_pe is None:
            use_pe = self.use_pe

        if use_pe and self.use_pe:
            prompt = [self._enhance_prompt_with_pe(item, width=width, height=height) for item in prompt]
            revised_prompts = list(prompt)
        else:
            revised_prompts = None

        if self.cpu_offload:
            self.text_encoder.to(AI_DEVICE)

        embeddings = [self._encode_one(item) for item in prompt]

        if self.cpu_offload:
            self.text_encoder.to("cpu")
            torch_device_module.empty_cache()
            gc.collect()

        return embeddings, revised_prompts
