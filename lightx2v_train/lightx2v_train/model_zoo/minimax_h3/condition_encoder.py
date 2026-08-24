"""Diffusers/Transformers conditioner adapter for MiniMax-H3 training."""

import torch


class MiniMaxH3ConditionEncoder:
    """Encode text with the official Qwen3-VL conditioning recipe."""

    def __init__(
        self,
        model_path,
        *,
        device,
        dtype,
        local_files_only=True,
        cpu_offload=False,
        attention_backend="torch_sdpa",
        text_encoder_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        processor_subfolder="processor",
        text_encoder_layer=50,
        text_tag=1,
    ):
        try:
            from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        except ImportError as error:
            raise ImportError("MiniMax-H3 cache construction requires Transformers with Qwen3-VL support.") from error

        self.device = torch.device(device)
        self.dtype = dtype
        self.cpu_offload = bool(cpu_offload)
        self._managed_cpu_offload = False
        self.text_encoder_layer = int(text_encoder_layer)
        self.text_tag = int(text_tag)
        if self.text_encoder_layer < 0:
            raise ValueError("MiniMax-H3 text_encoder_layer must be non-negative.")

        load_kwargs = {
            "local_files_only": bool(local_files_only),
            "attn_implementation": self._transformers_attention_backend(attention_backend),
            "low_cpu_mem_usage": True,
        }
        try:
            self.encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                subfolder=text_encoder_subfolder,
                dtype=dtype,
                **load_kwargs,
            )
        except TypeError:
            self.encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                subfolder=text_encoder_subfolder,
                torch_dtype=dtype,
                **load_kwargs,
            )
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_path,
            subfolder=tokenizer_subfolder,
            local_files_only=bool(local_files_only),
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            subfolder=processor_subfolder,
            local_files_only=bool(local_files_only),
        )
        if not hasattr(self.processor, "create_mm_token_type_ids"):
            raise RuntimeError("Installed Transformers does not expose Qwen3VLProcessor.create_mm_token_type_ids required by MiniMax-H3.")

        self.encoder.requires_grad_(False).eval()
        if self.cpu_offload and self.device.type != "cpu":
            try:
                from accelerate import cpu_offload
            except ImportError as error:
                raise ImportError("MiniMax-H3 condition encoder CPU offload requires Accelerate.") from error
            cpu_offload(self.encoder, execution_device=self.device, offload_buffers=True)
            self._managed_cpu_offload = True
        else:
            self.encoder.to(self.device)

    @staticmethod
    def _transformers_attention_backend(attention_backend):
        if attention_backend in {None, "torch_sdpa", "native"}:
            return "sdpa"
        return attention_backend

    def _activate(self):
        if self.cpu_offload and self.device.type != "cpu" and not self._managed_cpu_offload:
            self.encoder.to(self.device)

    def _offload(self):
        if self.cpu_offload and self.device.type != "cpu" and not self._managed_cpu_offload:
            self.encoder.to("cpu")

    @torch.inference_mode()
    def encode(self, prompt):
        if not isinstance(prompt, str):
            prompts = list(prompt)
            if len(prompts) != 1:
                raise ValueError(f"MiniMax-H3 requires one prompt per rank, got {len(prompts)}.")
            prompt = prompts[0]
        if not isinstance(prompt, str):
            raise TypeError(f"MiniMax-H3 prompt must be a string, got {type(prompt).__name__}.")
        if not prompt:
            raise ValueError("MiniMax-H3 prompt must contain at least one character.")

        token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError("MiniMax-H3 prompt must produce at least one token.")
        self._activate()
        try:
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            mm_token_type_ids = torch.tensor(
                self.processor.create_mm_token_type_ids([token_ids]),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            if len(outputs.hidden_states) <= self.text_encoder_layer:
                raise ValueError(f"MiniMax-H3 requires Qwen3-VL hidden_states[{self.text_encoder_layer}], but the encoder returned only {len(outputs.hidden_states)} states.")
            prompt_embeds = outputs.hidden_states[self.text_encoder_layer].to(device=self.device, dtype=self.dtype)
            del outputs
            if prompt_embeds.shape[1] != len(token_ids):
                raise ValueError(f"MiniMax-H3 Qwen3-VL returned {prompt_embeds.shape[1]} embedding rows for {len(token_ids)} tokens.")
            return {
                "prompt_embeds": prompt_embeds,
                "text_token_tags": torch.full(
                    (len(token_ids),),
                    self.text_tag,
                    dtype=torch.long,
                    device=self.device,
                ),
            }
        finally:
            self._offload()
