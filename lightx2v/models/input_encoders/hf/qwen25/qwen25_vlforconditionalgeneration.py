import gc
import math
import os

import torch

try:
    from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2Tokenizer = None
    Qwen2_5_VLForConditionalGeneration = None

from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

try:
    from diffusers.image_processor import VaeImageProcessor
    from transformers import Qwen2VLProcessor
except ImportError:
    VaeImageProcessor = None
    Qwen2VLProcessor = None

PREFERRED_QWENIMAGE_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


class Qwen25_VLForConditionalGeneration_TextEncoder:
    def __init__(self, config):
        self.config = config
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = config["prompt_template_encode"]
        self.prompt_template_encode_start_idx = config["prompt_template_encode_start_idx"]
        """
        for Qwen-Image-Edit model, CONDITION_IMAGE_SIZE = 1024 * 1024
        for Qwen-Image-Edit-2509 model, CONDITION_IMAGE_SIZE = 384 * 384
        """
        self.CONDITION_IMAGE_SIZE = config.get("CONDITION_IMAGE_SIZE", 384 * 384)
        self.USE_IMAGE_ID_IN_PROMPT = config.get("USE_IMAGE_ID_IN_PROMPT", True)
        self.VAE_IMAGE_SIZE = 1024 * 1024
        self.is_layered = self.config.get("layered", False)
        if self.is_layered:
            self.resolution = self.config.get("resolution", 640)
            self.VAE_IMAGE_SIZE = self.resolution * self.resolution

        self.cpu_offload = config.get("qwen25vl_cpu_offload", config.get("cpu_offload", False))
        self.dtype = torch.bfloat16
        self.load()

    def load(self):
        if self.config.get("qwen25vl_quantized", False):
            assert self.config["qwen25vl_quant_scheme"] == "int4"
            if self.config["cpu_offload"]:
                self.device_map = {
                    "lm_head": AI_DEVICE,
                    "model.visual": "cpu",
                    "model.language_model": "cpu",
                }
            else:
                self.device_map = "auto"
            self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.config["qwen25vl_quantized_ckpt"], dtype=torch.bfloat16, device_map=self.device_map, low_cpu_mem_usage=True)
        else:
            self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(self.config["model_path"], "text_encoder"), torch_dtype=torch.bfloat16)

        if not self.cpu_offload:
            self.text_encoder = self.text_encoder.to(AI_DEVICE)

        qwen25vl_tokenizer_path = self.config.get("qwen25vl_tokenizer_path", os.path.join(self.config["model_path"], "tokenizer"))
        self.tokenizer = Qwen2Tokenizer.from_pretrained(qwen25vl_tokenizer_path)
        if self.config["task"] == "i2i":
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.config["vae_scale_factor"] * 2)
            qwen25vl_processor_path = self.config.get("qwen25vl_processor_path", os.path.join(self.config["model_path"], "processor"))
            self.processor = Qwen2VLProcessor.from_pretrained(qwen25vl_processor_path)

        if self.is_layered:
            self.vl_processor = Qwen2VLProcessor.from_pretrained(os.path.join(self.config["model_path"], "processor"))
            self.use_en_prompt = self.config["use_en_prompt"]
            self.image_caption_prompt_cn = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n# 图像标注器\n你是一个专业的图像标注器。请基于输入图像，撰写图注:\n1.
    使用自然、描述性的语言撰写图注，不要使用结构化形式或富文本形式。\n2. 通过加入以下内容，丰富图注细节：\n - 对象的属性：如数量、颜色、形状、大小、位置、材质、状态、动作等\n -
    对象间的视觉关系：如空间关系、功能关系、动作关系、从属关系、比较关系、因果关系等\n - 环境细节：例如天气、光照、颜色、纹理、气氛等\n - 文字内容：识别图像中清晰可见的文字，不做翻译和解释，用引号在图注中强调\n3.
    保持真实性与准确性：\n - 不要使用笼统的描述\n -
    描述图像中所有可见的信息，但不要加入没有在图像中出现的内容\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"""
            self.image_caption_prompt_en = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n# Image Annotator\nYou are a professional
    image annotator. Please write an image caption based on the input image:\n1. Write the caption using natural,
    descriptive language without structured formats or rich text.\n2. Enrich caption details by including: \n - Object
    attributes, such as quantity, color, shape, size, material, state, position, actions, and so on\n - Vision Relations
    between objects, such as spatial relations, functional relations, possessive relations, attachment relations, action
    relations, comparative relations, causal relations, and so on\n - Environmental details, such as weather, lighting,
    colors, textures, atmosphere, and so on\n - Identify the text clearly visible in the image, without translation or
    explanation, and highlight it in the caption with quotation marks\n3. Maintain authenticity and accuracy:\n - Avoid
    generalizations\n - Describe all visible information in the image, while do not add information not explicitly shown in
    the image\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"""

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def preprocess_image(self, image):
        image_width, image_height = image.size
        condition_width, condition_height = calculate_dimensions(self.CONDITION_IMAGE_SIZE, image_width / image_height)
        vae_width, vae_height = calculate_dimensions(self.VAE_IMAGE_SIZE, image_width / image_height)
        condition_image = self.image_processor.resize(image, condition_height, condition_width)
        vae_image = self.image_processor.preprocess(image, vae_height, vae_width).unsqueeze(2)
        return condition_image, vae_image, (condition_height, condition_width), (vae_height, vae_width)

    @torch.no_grad()
    def get_image_caption(self, prompt_image):
        if self.use_en_prompt:
            prompt = self.image_caption_prompt_en
        else:
            prompt = self.image_caption_prompt_cn

        model_inputs = self.vl_processor(
            text=prompt,
            images=prompt_image,
            padding=True,
            return_tensors="pt",
        ).to(AI_DEVICE)

        generated_ids = self.text_encoder.generate(**model_inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
        output_text = self.vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output_text.strip()

    @torch.no_grad()
    def infer(self, text, image_list=None):
        if self.cpu_offload:
            if not hasattr(self, "device_map") or self.device_map == "auto":
                self.text_encoder.to(AI_DEVICE)

        if self.is_layered:
            text = [self.get_image_caption(image_list[0])]
            text = [
                "A charming anime character with short, light blue hair adorned with white flowers and a purple ribbon stands gracefully. She wears a detailed maid outfit featuring a white blouse with ruffled cuffs and a black apron, accessorized with a bow at the neckline. Her hands are clasped together in front of her, and she gazes slightly downward with a gentle expression. The background is a soft, light blue gradient, giving the scene a serene and ethereal atmosphere."
            ]

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

        if self.config["task"] == "i2i" and not self.is_layered:
            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(base_img_prompt + e) for e in text]

            model_inputs = self.processor(
                text=txt,
                images=condition_image_list,
                padding=True,
                return_tensors="pt",
            ).to(AI_DEVICE)

            encoder_hidden_states = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )
        else:
            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(e) for e in text]

            model_inputs = self.tokenizer(txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt").to(AI_DEVICE)
            encoder_hidden_states = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                output_hidden_states=True,
            )

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
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(1 * 1, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, 1, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(1 * 1, seq_len)

        if self.cpu_offload:
            if not hasattr(self, "device_map") or self.device_map == "auto":
                self.text_encoder.to(torch.device("cpu"))
            torch_device_module.empty_cache()
            gc.collect()

        return prompt_embeds, prompt_embeds_mask, image_info
