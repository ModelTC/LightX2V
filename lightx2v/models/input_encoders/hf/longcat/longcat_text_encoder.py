import gc
import os
import re

import torch
from loguru import logger

try:
    from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2Tokenizer = None
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen2VLProcessor
except ImportError:
    Qwen2VLProcessor = None

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE
from .system_messages import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH

# Edit task prompt templates
EDIT_PROMPT_TEMPLATE_PREFIX = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
EDIT_PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

torch_device_module = getattr(torch, AI_DEVICE)


def get_prompt_language(text):
    """Detect if the text is primarily Chinese or English."""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.replace(' ', ''))
    if total_chars == 0:
        return "en"
    return "zh" if chinese_chars / total_chars > 0.3 else "en"


def split_quotation(text):
    """Split text by quotation marks, marking quoted sections for character-level tokenization.

    For text in quotes, each character will be tokenized separately.
    This is a key feature of LongCat for better text rendering.

    Args:
        text: Input text

    Returns:
        List of (substring, is_quoted) tuples
    """
    pattern = r'["""](.*?)["""]'
    parts = re.split(pattern, text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # Matched (inside quotes)
            if len(part):
                result.append((part, True))
        else:  # Not matched (outside quotes)
            if len(part):
                result.append((part, False))
    return result


class LongCatImageTextEncoder:
    """Text encoder for LongCat Image model.

    Supports:
    - Character-level tokenization for quoted text (split_quotation)
    - Prompt rewriting using Qwen2.5-VL
    """

    def __init__(self, config):
        self.config = config
        self.tokenizer_max_length = 512
        # Use prefix/suffix templates like diffusers
        self.prompt_template_encode_prefix = config.get(
            "prompt_template_encode_prefix",
            "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = config.get(
            "prompt_template_encode_suffix",
            "<|im_end|>\n<|im_start|>assistant\n"
        )

        self.cpu_offload = config.get("cpu_offload", False)
        self.dtype = GET_DTYPE()
        self.enable_prompt_rewrite = config.get("enable_prompt_rewrite", False)
        self.load()

    def load(self):
        """Load the text encoder and tokenizer."""
        text_encoder_path = os.path.join(self.config["model_path"], "text_encoder")
        tokenizer_path = self.config.get(
            "tokenizer_path",
            os.path.join(self.config["model_path"], "tokenizer")
        )
        processor_path = self.config.get(
            "processor_path",
            os.path.join(self.config["model_path"], "processor")
        )

        # Load text encoder
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            text_encoder_path,
            torch_dtype=torch.bfloat16
        )

        if not self.cpu_offload:
            self.text_encoder = self.text_encoder.to(AI_DEVICE)

        # Load tokenizer
        self.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)

        # Load processor (for prompt rewrite)
        if self.enable_prompt_rewrite:
            self.processor = Qwen2VLProcessor.from_pretrained(processor_path)

    def _encode_prompt(self, prompt):
        """Encode prompt with character-level tokenization for quoted text.

        This is a key feature of LongCat: text in quotes is tokenized character by character.
        """
        batch_all_tokens = []

        for each_prompt in prompt:
            all_tokens = []
            for clean_prompt_sub, matched in split_quotation(each_prompt):
                if matched:
                    # Character-level tokenization for quoted text
                    for sub_word in clean_prompt_sub:
                        tokens = self.tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                        all_tokens.extend(tokens)
                else:
                    # Normal tokenization
                    tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(tokens)

            if len(all_tokens) > self.tokenizer_max_length:
                print(f"Warning: Input truncated from {len(all_tokens)} to {self.tokenizer_max_length} tokens")
                all_tokens = all_tokens[:self.tokenizer_max_length]

            batch_all_tokens.append(all_tokens)

        return batch_all_tokens

    @torch.no_grad()
    def rewire_prompt(self, prompt, device=None):
        """Rewrite prompts using the LLM for better generation quality.

        Args:
            prompt: Single prompt string or list of prompts
            device: Device to run on (defaults to AI_DEVICE)

        Returns:
            List of rewritten prompts
        """
        device = device or AI_DEVICE
        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_text = []
        for each_prompt in prompt:
            language = get_prompt_language(each_prompt)
            if language == "zh":
                question = SYSTEM_PROMPT_ZH + f"\n用户输入为：{each_prompt}\n改写后的prompt为："
            else:
                question = SYSTEM_PROMPT_EN + f"\nUser Input: {each_prompt}\nRewritten prompt:"

            message = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                }
            ]
            text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            all_text.append(text)

        inputs = self.processor(text=all_text, padding=True, return_tensors="pt").to(device)

        if self.cpu_offload:
            self.text_encoder.to(device)

        generated_ids = self.text_encoder.generate(**inputs, max_new_tokens=self.tokenizer_max_length)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if self.cpu_offload:
            self.text_encoder.to("cpu")
            torch_device_module.empty_cache()
            gc.collect()

        return output_text

    @torch.no_grad()
    def infer(self, text, image_list=None):
        """Encode text prompts to embeddings.

        Args:
            text: List of text prompts
            image_list: Unused for T2I, kept for API compatibility

        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask, image_info)
        """
        if self.cpu_offload:
            self.text_encoder.to(AI_DEVICE)

        # Encode with character-level tokenization for quoted text
        batch_all_tokens = self._encode_prompt(text)

        # Pad token list to max_length
        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": batch_all_tokens},
            max_length=self.tokenizer_max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Tokenize prefix and suffix
        prefix_tokens = self.tokenizer(self.prompt_template_encode_prefix, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_tokens)
        suffix_len = len(suffix_tokens)

        # Create masks for prefix and suffix
        prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
        suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)

        prefix_tokens = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

        batch_size = text_tokens_and_mask.input_ids.size(0)

        # Expand prefix and suffix for batch
        prefix_tokens_batch = prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        suffix_tokens_batch = suffix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_mask_batch = prefix_tokens_mask.unsqueeze(0).expand(batch_size, -1)
        suffix_mask_batch = suffix_tokens_mask.unsqueeze(0).expand(batch_size, -1)

        # Concatenate: [prefix, content, suffix]
        input_ids = torch.cat((prefix_tokens_batch, text_tokens_and_mask.input_ids, suffix_tokens_batch), dim=-1)
        attention_mask = torch.cat((prefix_mask_batch, text_tokens_and_mask.attention_mask, suffix_mask_batch), dim=-1)

        input_ids = input_ids.to(AI_DEVICE)
        attention_mask = attention_mask.to(AI_DEVICE)

        # Get hidden states from text encoder
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract hidden states and remove prefix/suffix
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, prefix_len:-suffix_len, :]

        # Create mask for the content tokens (always tokenizer_max_length)
        prompt_embeds_mask = text_tokens_and_mask.attention_mask.to(AI_DEVICE)

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=AI_DEVICE)

        if self.cpu_offload:
            self.text_encoder.to("cpu")
            torch_device_module.empty_cache()
            gc.collect()

        return prompt_embeds, prompt_embeds_mask, {}

    @torch.no_grad()
    def infer_with_image(self, text, image):
        """Encode text + image prompts for image editing task.

        Args:
            text: List of text prompts
            image: PIL Image or tensor for the input image

        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask, image_info)
        """
        if self.cpu_offload:
            self.text_encoder.to(AI_DEVICE)

        # Load processor if not already loaded
        if not hasattr(self, 'processor') or self.processor is None:
            processor_path = self.config.get(
                "processor_path",
                os.path.join(self.config["model_path"], "text_processor")
            )
            self.processor = Qwen2VLProcessor.from_pretrained(processor_path)

        # Process image using the VL processor's image processor
        image_processor = self.processor.image_processor
        raw_vl_input = image_processor(images=image, return_tensors="pt")
        pixel_values = raw_vl_input["pixel_values"].to(AI_DEVICE, dtype=self.dtype)
        image_grid_thw = raw_vl_input["image_grid_thw"].to(AI_DEVICE)

        # Encode prompts with character-level tokenization for quoted text
        batch_all_tokens = self._encode_prompt(text)

        # Pad token list to max_length
        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": batch_all_tokens},
            max_length=self.tokenizer_max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Build the edit prompt template with image placeholder
        # Calculate number of image tokens
        merge_length = image_processor.merge_size ** 2
        num_image_tokens = image_grid_thw.prod() // merge_length

        # Replace <|image_pad|> with actual number of image tokens
        prefix_text = EDIT_PROMPT_TEMPLATE_PREFIX
        image_token = "<|image_pad|>"
        prefix_text = prefix_text.replace(image_token, "<|placeholder|>" * num_image_tokens)
        prefix_text = prefix_text.replace("<|placeholder|>", image_token)

        # Tokenize prefix and suffix
        prefix_tokens = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(EDIT_PROMPT_TEMPLATE_SUFFIX, add_special_tokens=False)["input_ids"]

        # Find vision_start position to know where image tokens start
        vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        prefix_len = prefix_tokens.index(vision_start_token_id)
        suffix_len = len(suffix_tokens)

        # Create masks
        prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
        suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)

        prefix_tokens = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

        # Concatenate: [prefix_with_image, content, suffix]
        input_ids = torch.cat((prefix_tokens, text_tokens_and_mask.input_ids[0], suffix_tokens), dim=-1)
        attention_mask = torch.cat(
            (prefix_tokens_mask, text_tokens_and_mask.attention_mask[0], suffix_tokens_mask), dim=-1
        )

        input_ids = input_ids.unsqueeze(0).to(AI_DEVICE)
        attention_mask = attention_mask.unsqueeze(0).to(AI_DEVICE)

        # Get hidden states from text encoder with image
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )

        # Extract hidden states and remove prefix/suffix
        # Note: prefix_len is position before vision_start, we want to keep only the content part
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, prefix_len:-suffix_len, :]

        # Create mask for the content tokens
        prompt_embeds_mask = text_tokens_and_mask.attention_mask.to(AI_DEVICE)

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=AI_DEVICE)

        if self.cpu_offload:
            self.text_encoder.to("cpu")
            torch_device_module.empty_cache()
            gc.collect()

        logger.info(f"Edit prompt embeds shape: {prompt_embeds.shape}")
        return prompt_embeds, prompt_embeds_mask, {}
