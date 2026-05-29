class ErnieImageTextEncoder:
    def __init__(self, config, components, diffusers_cpu_offload=False):
        self.config = config
        self.components = components
        self.pipe = getattr(components, "pipeline", components)
        self.diffusers_cpu_offload = diffusers_cpu_offload
        self.revised_prompts = None

    def _normalize_prompt(self, prompt):
        if prompt is None:
            raise ValueError("ERNIE-Image requires input_info.prompt.")
        if isinstance(prompt, str):
            return [prompt]
        return list(prompt)

    @staticmethod
    def _normalize_negative_prompt(negative_prompt, batch_size):
        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            return [negative_prompt] * batch_size
        negative_prompt = list(negative_prompt)
        if len(negative_prompt) != batch_size:
            raise ValueError(f"negative_prompt must have same length as prompt ({batch_size})")
        return negative_prompt

    def _enhance_prompts(self, prompt, device, width, height, use_pe):
        self.revised_prompts = None
        if not use_pe:
            return prompt
        has_prompt_enhancer = getattr(
            self.components,
            "has_prompt_enhancer",
            getattr(self.pipe, "pe", None) is not None and getattr(self.pipe, "pe_tokenizer", None) is not None,
        )
        if not has_prompt_enhancer:
            return prompt

        if hasattr(self.components, "enhance_prompt_with_pe"):
            revised = [self.components.enhance_prompt_with_pe(p, device, width=width, height=height) for p in prompt]
        else:
            revised = [self.pipe._enhance_prompt_with_pe(p, device, width=width, height=height) for p in prompt]
        self.revised_prompts = list(revised)
        return revised

    def encode(self, generation_kwargs, device, dtype, do_classifier_free_guidance):
        prompt = self._normalize_prompt(generation_kwargs["prompt"])
        width = generation_kwargs["width"]
        height = generation_kwargs["height"]
        use_pe = generation_kwargs["use_pe"]
        num_images_per_prompt = int(generation_kwargs.get("num_images_per_prompt", 1))

        prompt = self._enhance_prompts(prompt, device, width, height, use_pe)
        batch_size = len(prompt)
        negative_prompt = self._normalize_negative_prompt(generation_kwargs["negative_prompt"], batch_size)

        text_hiddens = self.components.encode_prompt(prompt, device, num_images_per_prompt)
        if do_classifier_free_guidance:
            uncond_text_hiddens = self.components.encode_prompt(negative_prompt, device, num_images_per_prompt)
            text_hiddens = list(uncond_text_hiddens) + list(text_hiddens)

        if hasattr(self.components, "pad_text"):
            text_bth, text_lens = self.components.pad_text(text_hiddens, device, dtype)
        else:
            text_bth, text_lens = self.pipe._pad_text(
                text_hiddens=text_hiddens,
                device=device,
                dtype=dtype,
                text_in_dim=self.pipe.transformer.config.text_in_dim,
            )
        return batch_size, text_bth, text_lens
