import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer


class Qwen25_VLForConditionalGeneration_TextEncoder:
    def __init__(self, config):
        self.config = config
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(config.model_path, "text_encoder")
        ).to(torch.device("cuda")).to(torch.bfloat16)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            os.path.join(config.model_path, "tokenizer")
        )
        
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def infer(self, text):
        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in text]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds_mask = encoder_attention_mask
        
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, self.config.num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(self.config.batchsize * self.config.num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, self.config.num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(self.config.batchsize * self.config.num_images_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask
