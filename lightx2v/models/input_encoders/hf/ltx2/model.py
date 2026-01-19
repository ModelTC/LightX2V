from pathlib import Path

import torch
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor

from lightx2v.models.input_encoders.hf.ltx2.gemma.encoders.av_encoder import (
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModel,
    AVGemmaTextEncoderModelConfigurator,
)
from lightx2v.models.input_encoders.hf.ltx2.gemma.tokenizer import LTXVGemmaTokenizer
from lightx2v.utils.ltx2_utils import *


def _find_matching_dir(root_path: str, pattern: str) -> str:
    """Recursively search for files matching a glob pattern and return the parent directory of the first match."""
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return str(matches[0].parent)


class LTX2TextEncoder:
    """
    Simplified text encoder loader that encapsulates all complex building logic.

    Usage:
        model = LTX2TextEncoder(
            checkpoint_path="/path/to/checkpoint.safetensors",
            gemma_root="/path/to/gemma",
            device=torch.device("cuda"),
            dtype=torch.bfloat16
        )

    This class handles:
    - Loading model configuration from checkpoint
    - Creating model structure
    - Loading Gemma model, tokenizer, and processor from gemma_root
    - Loading weights from checkpoint with key mapping
    - Moving to device and setting dtype
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the simplified text encoder loader.

        Args:
            checkpoint_path: Path to the checkpoint file containing text encoder weights
            gemma_root: Root directory containing Gemma model, tokenizer, and processor
            device: Target device for the model
            dtype: Data type for model parameters
        """
        self.checkpoint_path = checkpoint_path
        self.gemma_root = gemma_root
        self.device = device
        self.dtype = dtype
        self.loader = SafetensorsModelStateDictLoader()
        self.text_encoder = self.load()

    def _load_gemma_model(self) -> Gemma3ForConditionalGeneration:
        """Load Gemma model from gemma_root."""
        gemma_path = _find_matching_dir(self.gemma_root, "model*.safetensors")
        return Gemma3ForConditionalGeneration.from_pretrained(gemma_path, local_files_only=True, torch_dtype=torch.bfloat16)

    def _load_tokenizer(self) -> LTXVGemmaTokenizer:
        """Load tokenizer from gemma_root."""
        tokenizer_path = _find_matching_dir(self.gemma_root, "tokenizer.model")
        return LTXVGemmaTokenizer(tokenizer_path, 1024)

    def _load_processor(self, tokenizer: LTXVGemmaTokenizer) -> Gemma3Processor:
        """Load processor from gemma_root."""
        processor_path = _find_matching_dir(self.gemma_root, "preprocessor_config.json")
        image_processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True)
        return Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer.tokenizer)

    def load(self) -> AVGemmaTextEncoderModel:
        """
        Load and build the text encoder model.

        Returns:
            AVGemmaTextEncoderModel: The fully initialized text encoder model
        """
        # Step 1: Load configuration from checkpoint

        config = self.loader.metadata(self.checkpoint_path)

        # Step 2: Create model structure (meta model)
        model = AVGemmaTextEncoderModelConfigurator.from_config(config)

        # Step 3: Load Gemma model, tokenizer, and processor from gemma_root
        model.model = self._load_gemma_model()
        model.tokenizer = self._load_tokenizer()
        model.processor = self._load_processor(model.tokenizer)

        # Step 4: Load weights from checkpoint with key mapping
        state_dict_obj = self.loader.load(
            self.checkpoint_path,
            sd_ops=AV_GEMMA_TEXT_ENCODER_KEY_OPS,
            device=self.device,
        )

        # Step 5: Apply dtype conversion if needed
        state_dict = state_dict_obj.sd
        if self.dtype is not None:
            state_dict = {key: value.to(dtype=self.dtype) for key, value in state_dict.items()}

        # Step 6: Load state dict into model
        model.load_state_dict(state_dict, strict=False, assign=True)

        # Step 7: Move to device and set eval mode
        model = model.to(self.device).eval()

        return model

    def encode_text(self, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a list of prompts using the provided Gemma text encoder.

        Args:
            text_encoder: The Gemma text encoder instance.
            prompts: List of prompt strings to encode.

        Returns:
            List of tuples, each containing (v_context, a_context) tensors for each prompt.
        """
        result = []
        for prompt in prompts:
            v_context, a_context, _ = self.text_encoder(prompt)
            result.append((v_context, a_context))
        return result

    def infer(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Infer text encoder outputs for prompt and negative prompt.

        This is a convenience function that encodes both prompt and negative prompt,
        and returns the video and audio contexts for both.

        Args:
            text_encoder: The Gemma text encoder instance.
            prompt: Positive prompt string.
            negative_prompt: Negative prompt string (default: empty string).

        Returns:
            Tuple containing:
            - v_context_p: Video context for positive prompt
            - a_context_p: Audio context for positive prompt
            - v_context_n: Video context for negative prompt
            - a_context_n: Audio context for negative prompt
        """
        contexts = self.encode_text(prompts=[prompt, negative_prompt])
        context_p, context_n = contexts
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n
        return v_context_p, a_context_p, v_context_n, a_context_n


if __name__ == "__main__":
    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
        "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    )

    model = LTX2TextEncoder(
        checkpoint_path="/data/nvme0/gushiqiao/models/official_models/LTX-2/ltx-2-19b-distilled-fp8.safetensors",
        gemma_root="/data/nvme0/gushiqiao/models/official_models/LTX-2",
        device="cuda",
        dtype=torch.bfloat16,
    )

    v_context_p, a_context_p, v_context_n, a_context_n = model.infer(
        prompt="A beautiful sunset over the ocean",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    )
