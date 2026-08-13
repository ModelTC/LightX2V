import gc
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from transformers import AutoImageProcessor, Gemma3Processor

from lightx2v.models.input_encoders.hf.ltx2.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnectorConfigurator,
)
from lightx2v.models.input_encoders.hf.ltx2.gemma.embeddings_processor import (
    EmbeddingsProcessor,
)
from lightx2v.models.input_encoders.hf.ltx2.gemma.encoders.base_encoder import (
    GemmaTextEncoder,
)
from lightx2v.models.input_encoders.hf.ltx2.gemma.encoders.encoder_configurator import (
    GEMMA_MODEL_OPS,
    GemmaTextEncoderConfigurator,
    _create_feature_extractor,
)
from lightx2v.models.input_encoders.hf.ltx2.gemma.model import (
    Gemma3ForConditionalGeneration,
)
from lightx2v.models.input_encoders.hf.ltx2.gemma.tokenizer import LTXVGemmaTokenizer
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.lora_loader import LoRALoader
from lightx2v.utils.ltx2_utils import *
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

EMBEDDINGS_PROCESSOR_KEY_OPS = (
    SDOps("EMBEDDINGS_PROCESSOR_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.aggregate_embed.")
    .with_replacement("text_embedding_projection.aggregate_embed.", "feature_extractor.aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.video_aggregate_embed.")
    .with_replacement("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed.")
    .with_matching(prefix="text_embedding_projection.audio_aggregate_embed.")
    .with_replacement("text_embedding_projection.audio_aggregate_embed.", "feature_extractor.audio_aggregate_embed.")
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "embeddings_processor.video_connector.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "embeddings_processor.audio_connector.")
)


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
        cpu_offload: bool = False,
        gemma_attn_implementation: str | None = None,
    ):
        """
        Initialize the simplified text encoder loader.

        Args:
            checkpoint_path: Path to the checkpoint file containing text encoder weights
            gemma_root: Root directory containing Gemma model, tokenizer, and processor
            device: Target device for the model
            dtype: Data type for model parameters
            gemma_attn_implementation: Optional Transformers attention
                override. Registered platform backends such as
                ``"mlu_flash_attn"`` are scoped to Gemma text layers. When
                omitted, Transformers keeps its native default (currently SDPA).
        """
        self.checkpoint_path = checkpoint_path
        self.gemma_root = gemma_root
        self.device = device
        self.dtype = dtype
        self.cpu_offload = cpu_offload
        self.gemma_attn_implementation = gemma_attn_implementation
        self.loader = SafetensorsModelStateDictLoader()
        self.text_encoder = self.load()

    def _load_gemma_model(self) -> Gemma3ForConditionalGeneration:
        """Load Gemma model from gemma_root."""
        gemma_path = _find_matching_dir(self.gemma_root, "model*.safetensors")
        kwargs = {
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
        }
        if self.gemma_attn_implementation is not None:
            kwargs["attn_implementation"] = self.gemma_attn_implementation
        return Gemma3ForConditionalGeneration.from_pretrained(gemma_path, **kwargs)

    def _load_tokenizer(self) -> LTXVGemmaTokenizer:
        """Load tokenizer from gemma_root."""
        tokenizer_path = _find_matching_dir(self.gemma_root, "tokenizer.model")
        return LTXVGemmaTokenizer(tokenizer_path, 1024)

    def _load_processor(self, tokenizer: LTXVGemmaTokenizer) -> Gemma3Processor:
        """Load processor from gemma_root."""
        processor_path = _find_matching_dir(self.gemma_root, "preprocessor_config.json")
        image_processor = AutoImageProcessor.from_pretrained(processor_path, local_files_only=True)
        return Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer.tokenizer)

    def load(self) -> GemmaTextEncoder:
        """
        Load and build the text encoder model.

        Returns:
            GemmaTextEncoder: The fully initialized text encoder model
        """
        config = self.loader.metadata(self.checkpoint_path)
        model = GemmaTextEncoderConfigurator.from_config(config)

        # Latest LTX-2 loads the Gemma backbone from gemma_root
        model.model = self._load_gemma_model()
        model.tokenizer = self._load_tokenizer()
        model.processor = self._load_processor(model.tokenizer)
        model = GEMMA_MODEL_OPS.mutator(model)

        # The LTX checkpoint only provides the embeddings processor / feature extractor.
        state_dict_obj = self.loader.load(
            self.checkpoint_path,
            sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            device=self.device,
        )
        state_dict = state_dict_obj.sd
        if self.dtype is not None:
            state_dict = {key: value.to(dtype=self.dtype) for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=False, assign=True)
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
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to(AI_DEVICE)
        result = []
        for prompt in prompts:
            v_context, a_context, _ = self.text_encoder(prompt)
            result.append((v_context, a_context))
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
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

    def apply_lora(self, lora_configs):
        """
        Apply LoRA weights to text encoder's feature extractor.

        Args:
            lora_configs: List of LoRA configuration dicts, each containing:
                - path: Path to LoRA safetensors file
                - strength: LoRA strength (default: 1.0)

        Returns:
            bool: True if LoRA was successfully applied, False otherwise
        """
        if not hasattr(self, "text_encoder"):
            logger.warning("Text encoder does not have expected structure. Skipping LoRA application.")
            return False

        encoder_model = self.text_encoder

        if not hasattr(encoder_model, "feature_extractor"):
            logger.warning("Text encoder does not have feature_extractor. Skipping LoRA application.")
            return False

        feature_extractor = encoder_model.feature_extractor
        target_modules = []
        for attr in ("aggregate_embed", "video_aggregate_embed", "audio_aggregate_embed"):
            module = getattr(feature_extractor, attr, None)
            if module is not None and hasattr(module, "weight"):
                target_modules.append(attr)
        if not target_modules:
            logger.warning("feature_extractor does not expose supported projection layers. Skipping LoRA application.")
            return False

        weight_dict = {f"feature_extractor.{name}.weight": getattr(feature_extractor, name).weight.data.clone() for name in target_modules}

        key_mapping_rules = [
            (r"^text_embedding_projection\.", "feature_extractor."),
        ]
        lora_loader = LoRALoader(key_mapping_rules=key_mapping_rules)

        for lora_config in lora_configs:
            lora_path = lora_config["path"]
            lora_strength = lora_config.get("strength", 1.0)

            # Load only text_embedding_projection keys to save memory
            with safe_open(lora_path, framework="pt") as f:
                # First, get all keys and filter for text_embedding_projection
                all_keys = list(f.keys())
                text_encoder_keys = [key for key in all_keys if key.startswith("text_embedding_projection.")]

                # Only load the filtered keys
                text_encoder_lora_weights = {key: f.get_tensor(key).to(GET_DTYPE()).to(self.device) for key in text_encoder_keys}

            if text_encoder_lora_weights:
                applied_count = lora_loader.apply_lora(
                    weight_dict=weight_dict,
                    lora_weights=text_encoder_lora_weights,
                    strength=lora_strength,
                )

                if applied_count > 0:
                    for name in target_modules:
                        getattr(feature_extractor, name).weight.data = weight_dict[f"feature_extractor.{name}.weight"]
                    logger.info(f"Successfully applied {applied_count} LoRA weights to text encoder from {lora_path} (strength: {lora_strength})")
                else:
                    logger.warning(f"No LoRA weights were applied to text encoder from {lora_path}")
            else:
                logger.debug(f"No text_embedding_projection LoRA keys found in {lora_path}")

            del text_encoder_lora_weights
            gc.collect()

        return True


class LTX25TextEncoder(LTX2TextEncoder):
    """LTX-2.5 Gemma 4 unified text encoder.

    The public ``infer``/``encode_text`` interface intentionally matches
    :class:`LTX2TextEncoder`.  ``checkpoint_path`` is the split Transformer
    checkpoint (it owns the video/audio connector weights) and ``gemma_root``
    is the self-contained Gemma 4 text-encoder safetensors file.
    """

    def load(self) -> GemmaTextEncoder:
        from lightx2v.models.input_encoders.hf.ltx2.gemma.assets import (
            LTX25_EMBEDDINGS_KEY_OPS,
            LTX25_GEMMA_KEY_OPS,
            LTX25GemmaAssets,
            populate_gemma4_buffers,
            read_safetensors_metadata,
        )

        if not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(f"LTX-2.5 transformer checkpoint not found: {self.checkpoint_path}")
        if not Path(self.gemma_root).is_file():
            raise FileNotFoundError(f"LTX-2.5 text-encoder checkpoint not found: {self.gemma_root}")

        transformer_config = self.loader.metadata(self.checkpoint_path)
        transformer_metadata = read_safetensors_metadata(self.checkpoint_path)
        assets = LTX25GemmaAssets.load(self.gemma_root)
        gemma_config = assets.build_config()
        self._validate_gemma_version(transformer_metadata, gemma_config)

        gemma_model = assets.build_model(attn_implementation=self.gemma_attn_implementation)
        populate_gemma4_buffers(gemma_model)

        with torch.device("meta"):
            video_connector = Embeddings1DConnectorConfigurator.from_config(transformer_config)
            audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(transformer_config)
            embeddings_processor = EmbeddingsProcessor(
                video_connector=video_connector,
                audio_connector=audio_connector,
            )
            feature_extractor = _create_feature_extractor(
                transformer_config.get("transformer", {}),
                gemma_config.text_config,
            )
            model = GemmaTextEncoder(
                feature_extractor=feature_extractor,
                embeddings_processor=embeddings_processor,
                model=gemma_model,
                dtype=self.dtype,
            )

        gemma_state = self.loader.load(
            self.gemma_root,
            sd_ops=LTX25_GEMMA_KEY_OPS,
            device=self.device,
        ).sd
        gemma_state = self._cast_floating_state(gemma_state)
        incompatible = model.load_state_dict(gemma_state, strict=False, assign=True)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected LTX-2.5 Gemma weights: {incompatible.unexpected_keys[:10]}")
        del gemma_state

        # Connector weights are stored with the Transformer, while the dual
        # readout projections live in the text-encoder file.
        embeddings_state = self.loader.load(
            [self.checkpoint_path, self.gemma_root],
            sd_ops=LTX25_EMBEDDINGS_KEY_OPS,
            device=self.device,
        ).sd
        embeddings_state = self._cast_floating_state(embeddings_state)
        incompatible = model.load_state_dict(embeddings_state, strict=False, assign=True)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected LTX-2.5 embeddings weights: {incompatible.unexpected_keys[:10]}")
        del embeddings_state

        uninitialized = [name for name, value in (*model.named_parameters(), *model.named_buffers()) if value.is_meta]
        if uninitialized:
            raise ValueError(f"Uninitialized LTX-2.5 text-encoder tensors: {uninitialized[:20]}")

        hf_tokenizer = assets.build_tokenizer()
        model.tokenizer = LTXVGemmaTokenizer.from_tokenizer(hf_tokenizer, 1024)
        model.processor = assets.build_processor(hf_tokenizer)
        return model.to(self.device).eval()

    def _cast_floating_state(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.dtype is None:
            return state_dict
        return {key: value.to(dtype=self.dtype) if value.is_floating_point() else value for key, value in state_dict.items()}

    def _move_text_encoder(self, device: torch.device | str) -> None:
        """Move the complete Gemma/readout/connector stack as one lifecycle unit."""
        self.text_encoder = self.text_encoder.to(device)

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode a fused prompt batch exactly like the LTX-2.5 source pipeline.

        Gemma sees the full prompt batch in one forward.  Each batch slice is
        then processed independently by the readout and connector modules.  In
        addition to matching source arithmetic for CFG, this makes CPU offload
        exception-safe: a failed tokenize/forward/connector call still returns
        the complete text stack to CPU.
        """
        if not prompts:
            return []

        if self.cpu_offload:
            try:
                self._move_text_encoder(AI_DEVICE)
            except BaseException:
                # ``nn.Module.to`` mutates in place and can leave a partially
                # moved module if allocation fails.
                self._move_text_encoder("cpu")
                raise

        try:
            tokenized = [self.text_encoder.tokenizer.tokenize_with_weights(text)["gemma"] for text in prompts]
            model_device = self.text_encoder.model.device
            input_ids = torch.tensor(
                [[token for token, _ in pairs] for pairs in tokenized],
                device=model_device,
            )
            attention_mask = torch.tensor(
                [[weight for _, weight in pairs] for pairs in tokenized],
                device=model_device,
            )

            outputs = self.text_encoder.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            del outputs

            result = []
            for index in range(len(prompts)):
                per_prompt_hidden = tuple(hidden[index : index + 1] for hidden in hidden_states)
                per_prompt_mask = attention_mask[index : index + 1]
                video_features, audio_features = self.text_encoder.feature_extractor(
                    per_prompt_hidden,
                    per_prompt_mask,
                    "left",
                )
                additive_mask = self.text_encoder._convert_to_additive_mask(
                    per_prompt_mask,
                    video_features.dtype,
                )
                video_context, audio_context, _ = self.text_encoder.embeddings_processor.create_embeddings(
                    video_features,
                    audio_features,
                    additive_mask,
                )
                result.append((video_context, audio_context))
            return result
        finally:
            if self.cpu_offload:
                self._move_text_encoder("cpu")

    @staticmethod
    def _validate_gemma_version(transformer_metadata: dict, gemma_config) -> None:
        source = transformer_metadata.get("gemma_source_checkpoint")
        if not isinstance(source, dict):
            raise ValueError("LTX-2.5 Transformer metadata is missing gemma_source_checkpoint")
        expected = source.get("gemma_version")
        actual = getattr(gemma_config, "gemma_version", None)
        if expected != actual:
            raise ValueError(f"LTX-2.5 Gemma version mismatch: Transformer expects {expected!r}, text encoder declares {actual!r}")


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
        checkpoint_path="/data/nvme4/models/ltx-2.3/ltx-2.3-22b-dev.safetensors",
        gemma_root="/data/nvme0/gushiqiao/models/official_models/LTX-2",
        device="cuda",
        dtype=torch.bfloat16,
    )

    v_context_p, a_context_p, v_context_n, a_context_n = model.infer(
        prompt="A beautiful sunset over the ocean",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    )
