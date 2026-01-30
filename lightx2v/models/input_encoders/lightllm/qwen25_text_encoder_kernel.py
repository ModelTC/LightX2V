"""
Kernel-Optimized Text Encoder

Key optimizations:
1. Flash Attention
2. Fused RMSNorm - frequent operation (sgl_kernel version)
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

try:
    from diffusers.image_processor import VaeImageProcessor
except ImportError:
    try:
        from diffusers import VaeImageProcessor
    except ImportError:
        VaeImageProcessor = None


class LightLLMKernelTextEncoder:
    """
    Kernel-optimized Text Encoder

    Architecture:
    - Base: HuggingFace Qwen2_5_VLForConditionalGeneration
    - Optimizations: LightLLM Triton kernels for bottlenecks
    """

    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        from lightx2v_platform.base.global_var import AI_DEVICE

        self.config = config
        self.device = device if device is not None else AI_DEVICE
        self.dtype = torch.bfloat16

        # Configuration
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = config["prompt_template_encode"]
        self.prompt_template_encode_start_idx = config["prompt_template_encode_start_idx"]

        self.CONDITION_IMAGE_SIZE = config.get("CONDITION_IMAGE_SIZE", 384 * 384)
        self.USE_IMAGE_ID_IN_PROMPT = config.get("USE_IMAGE_ID_IN_PROMPT", True)
        self.VAE_IMAGE_SIZE = 1024 * 1024
        self.is_layered = config.get("layered", False)
        if self.is_layered:
            self.resolution = config.get("resolution", 640)
            self.VAE_IMAGE_SIZE = self.resolution * self.resolution

        self.model_path = config["model_path"]

        # Kernel optimization flags
        self.use_flash_attention_kernel = config.get("use_flash_attention_kernel", True)
        self.use_rmsnorm_kernel = config.get("use_rmsnorm_kernel", True)

        logger.info(f"Initializing Kernel-Optimized Text Encoder")
        logger.info(f"  Model Path: {self.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Flash Attention: {self.use_flash_attention_kernel}")
        logger.info(f"  RMSNorm Kernel: {self.use_rmsnorm_kernel}")

        self.load()

    def load(self):
        """Load model and apply kernel optimizations"""
        logger.info("Loading model components...")

        from transformers import Qwen2Tokenizer, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration

        # 1. Load tokenizer
        tokenizer_path = self.config.get("qwen25vl_tokenizer_path", os.path.join(self.model_path, "tokenizer"))
        self.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
        logger.info(f"  ✓ Tokenizer loaded from {tokenizer_path}")

        # 2. Load processor and image processor
        if self.config["task"] == "i2i":
            if VaeImageProcessor is None:
                raise ImportError("VaeImageProcessor could not be imported from diffusers")
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.config.get("vae_scale_factor", 8) * 2)
            processor_path = self.config.get("qwen25vl_processor_path", os.path.join(self.model_path, "processor"))
            self.processor = Qwen2VLProcessor.from_pretrained(processor_path)
            logger.info(f"  ✓ Processor loaded from {processor_path}")

        # 3. Load model - choose attn implementation based on config
        text_encoder_path = os.path.join(self.model_path, "text_encoder")

        # Select attention implementation
        # NOTE: torch.compile is incompatible with flash_attention_2
        if self.use_flash_attention_kernel:
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "eager"  # Compatible with torch.compile

        logger.info(f"  Loading model from {text_encoder_path}...")
        logger.info(f"  Attention implementation: {attn_impl}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            text_encoder_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            attn_implementation=attn_impl,
        )
        self.model.eval()

        logger.info(f"  ✓ Model loaded with {attn_impl}")

        # 4. Apply kernel optimizations (RMSNorm)
        self._apply_kernel_optimizations()

        self._is_loaded = True

    def _apply_kernel_optimizations(self):
        """Apply kernel optimizations to the model"""
        logger.info("Applying kernel optimizations...")

        # Flash Attention is already loaded with the model
        if self.use_flash_attention_kernel:
            logger.info("  ✓ Flash Attention 2 (loaded with model)")

            if self.use_rmsnorm_kernel:
                try:
                    from sgl_kernel.elementwise import rmsnorm

                    self._rmsnorm_kernel = rmsnorm
                    self._replace_rmsnorm_with_kernel()
                    logger.info("  ✓ RMSNorm kernel integrated (from sgl_kernel)")
                except ImportError as e:
                    logger.warning(f"  ✗ Failed to import sgl_kernel: {e}. RMSNorm optimization disabled.")
                    self.use_rmsnorm_kernel = False

    def _replace_rmsnorm_with_kernel(self):
        """Replace RMSNorm layers with fused kernel"""
        # Import Qwen2RMSNorm to identify layers
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        except ImportError:
            logger.warning("Could not import Qwen2RMSNorm, skipping RMSNorm optimization")
            return

        replaced_count = 0

        # Create optimized RMSNorm wrapper
        class OptimizedRMSNorm(nn.Module):
            def __init__(self, original_norm, kernel_fn):
                super().__init__()
                self.weight = original_norm.weight
                self.variance_epsilon = original_norm.variance_epsilon
                self.kernel_fn = kernel_fn

            def forward(self, hidden_states):
                orig_shape = hidden_states.shape
                # Reshape to (-1, hidden_dim) as sgl_kernel expects 2D
                x_2d = hidden_states.view(-1, orig_shape[-1])
                out_2d = self.kernel_fn(x_2d, self.weight, self.variance_epsilon)
                return out_2d.view(orig_shape)

        # Replace all RMSNorm layers
        def replace_rmsnorm_recursive(module, parent_name=""):
            nonlocal replaced_count
            for name, child in module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name

                if isinstance(child, Qwen2RMSNorm):
                    # Replace with optimized version
                    optimized = OptimizedRMSNorm(child, self._rmsnorm_kernel)
                    setattr(module, name, optimized)
                    replaced_count += 1
                else:
                    # Recursively process children
                    replace_rmsnorm_recursive(child, full_name)

        replace_rmsnorm_recursive(self.model)
        logger.info(f"    Replaced {replaced_count} RMSNorm layers with kernel version")

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Extract valid hidden states (consistent with HF baseline)"""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    @torch.inference_mode()
    def infer(self, text: List[str], image_list: Optional[List] = None) -> Tuple:
        """
        Inference method - same interface as Lite encoder

        Args:
            text: List of text prompts
            image_list: Optional list of images

        Returns:
            (prompt_embeds, prompt_embeds_mask, image_info)
        """
        from lightx2v_platform.base.global_var import AI_DEVICE

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx

        # Prepare image information
        if image_list is not None:
            condition_image_list = []
            vae_image_list = []
            condition_image_info_list = []
            vae_image_info_list = []

            if self.USE_IMAGE_ID_IN_PROMPT:
                base_img_prompt = ""
                img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
                for i, image in enumerate(image_list):
                    base_img_prompt += img_prompt_template.format(i + 1)
                    condition_image, vae_image, condition_image_info, vae_image_info = self.preprocess_image(image)
                    condition_image_list.append(condition_image)
                    vae_image_list.append(vae_image)
                    condition_image_info_list.append(condition_image_info)
                    vae_image_info_list.append(vae_image_info)
            else:
                base_img_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
                for i, image in enumerate(image_list):
                    condition_image, vae_image, condition_image_info, vae_image_info = self.preprocess_image(image)
                    condition_image_list.append(condition_image)
                    vae_image_list.append(vae_image)
                    condition_image_info_list.append(condition_image_info)
                    vae_image_info_list.append(vae_image_info)

            image_info = {
                "vae_image_list": vae_image_list,
                "vae_image_info_list": vae_image_info_list,
            }
        else:
            image_info = {}
            base_img_prompt = ""
            condition_image_list = None

        # Prepare text and model inputs
        if self.config["task"] == "i2i" and not self.is_layered and image_list is not None:
            txt = [template.format(base_img_prompt + e) for e in text]

            model_inputs = self.processor(
                text=txt,
                images=condition_image_list,
                padding=True,
                return_tensors="pt",
            ).to(AI_DEVICE)

            encoder_hidden_states = self.model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )
        else:
            txt = [template.format(e) for e in text]

            model_inputs = self.tokenizer(txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt").to(AI_DEVICE)

            encoder_hidden_states = self.model(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_hidden_states=True,
            )

        # Post-processing (same as HF baseline)
        hidden_states = encoder_hidden_states.hidden_states[-1]
        attention_mask = model_inputs.attention_mask

        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])

        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
        encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=AI_DEVICE)
        prompt_embeds_mask = encoder_attention_mask

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(1, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.view(1, seq_len)

        logger.info(f"✓ Kernel inference complete: prompt_embeds shape={prompt_embeds.shape}")

        return prompt_embeds, prompt_embeds_mask, image_info

    def _calculate_dimensions(self, target_area, ratio):
        """Calculate target dimensions"""
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height

    def preprocess_image(self, image):
        """Preprocess image"""
        image_width, image_height = image.size
        condition_width, condition_height = self._calculate_dimensions(self.CONDITION_IMAGE_SIZE, image_width / image_height)
        vae_width, vae_height = self._calculate_dimensions(self.VAE_IMAGE_SIZE, image_width / image_height)
        condition_image = self.image_processor.resize(image, condition_height, condition_width)
        vae_image = self.image_processor.preprocess(image, vae_height, vae_width).unsqueeze(2)
        return condition_image, vae_image, (condition_height, condition_width), (vae_height, vae_width)

    def offload_to_cpu(self):
        """Offload model to CPU to free GPU memory"""
        if hasattr(self, "model") and self.model is not None:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            logger.debug("Kernel encoder: model offloaded to CPU")

    def reload_to_device(self):
        """Reload model to GPU"""
        if hasattr(self, "model") and self.model is not None:
            self.model.to(self.device)
            logger.debug(f"Kernel encoder: model reloaded to {self.device}")
