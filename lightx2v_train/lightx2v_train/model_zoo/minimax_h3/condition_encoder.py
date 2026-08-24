"""Training adapter for LightX2V's native MiniMax-H3 conditioner."""

import torch


class MiniMaxH3ConditionEncoder:
    """Expose the native released Qwen3-VL prefix through the training API."""

    def __init__(
        self,
        model_path,
        *,
        device,
        dtype,
        local_files_only=True,
        cpu_offload=False,
        attention_backend="torch_sdpa",
    ):
        try:
            from lightx2v.models.input_encoders.hf.minimax_h3 import (
                MiniMaxH3Qwen3VLTextEncoder,
            )
        except ImportError as error:
            raise ImportError("MiniMax-H3 cache construction requires LightX2V's native Qwen3-VL conditioner.") from error

        self.device = torch.device(device)
        self.dtype = dtype
        self.encoder = MiniMaxH3Qwen3VLTextEncoder(
            {
                "model_path": str(model_path),
                "local_files_only": bool(local_files_only),
                "text_encoder_cpu_offload": bool(cpu_offload),
                "text_encoder_tensor_parallel": False,
                "qwen3vl_attn_type": attention_backend,
            }
        )

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
        condition = self.encoder.infer(prompt)
        return {
            "prompt_embeds": condition["prompt_embeds"].to(
                device=self.device,
                dtype=self.dtype,
            ),
            "text_token_tags": condition["text_token_tags"].to(
                device=self.device,
                dtype=torch.long,
            ),
        }
