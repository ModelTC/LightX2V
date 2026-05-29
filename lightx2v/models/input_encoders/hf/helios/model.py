import html

import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    import ftfy
except ImportError:
    ftfy = None


def basic_clean(text):
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    return re.sub(r"\s+", " ", text).strip()


def prompt_clean(text):
    return whitespace_clean(basic_clean(text))


def pack_t5_prompt_embeds(hidden_state, attention_mask, max_sequence_length, num_videos_per_prompt=1, dtype=None, device=None):
    device = device or hidden_state.device
    dtype = dtype or hidden_state.dtype
    prompt_embeds = hidden_state.to(dtype=dtype, device=device)
    attention_mask = attention_mask.to(device=device)
    seq_lens = attention_mask.gt(0).sum(dim=1).long()
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
        dim=0,
    )
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(len(seq_lens) * num_videos_per_prompt, seq_len, -1)
    return prompt_embeds, attention_mask.bool()


class HeliosTextEncoder:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu") if config.get("t5_cpu_offload", config.get("cpu_offload", False)) else torch.device(AI_DEVICE)
        self.dtype = GET_DTYPE()
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
        self.text_encoder = UMT5EncoderModel.from_pretrained(config["text_encoder_path"], torch_dtype=self.dtype).to(self.device)

    def infer(self, prompts, max_sequence_length=None):
        max_sequence_length = max_sequence_length or self.config.get("max_sequence_length", 512)
        prompts = [prompt_clean(prompt) for prompt in prompts]
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        hidden_state = self.text_encoder(input_ids, attention_mask).last_hidden_state
        return pack_t5_prompt_embeds(
            hidden_state,
            attention_mask,
            max_sequence_length=max_sequence_length,
            num_videos_per_prompt=1,
            dtype=self.dtype,
            device=torch.device(AI_DEVICE),
        )
