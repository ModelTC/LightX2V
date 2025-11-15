import json
import os
import re

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .format_prompt import MultilingualPromptFormat


def add_special_token(
    tokenizer,
    text_encoder,
    add_color,
    add_font,
    color_ann_path,
    font_ann_path,
    multilingual=False,
):
    """
    Add special tokens for color and font to tokenizer and text encoder.

    Args:
        tokenizer: Huggingface tokenizer.
        text_encoder: Huggingface T5 encoder.
        add_color (bool): Whether to add color tokens.
        add_font (bool): Whether to add font tokens.
        color_ann_path (str): Path to color annotation JSON.
        font_ann_path (str): Path to font annotation JSON.
        multilingual (bool): Whether to use multilingual font tokens.
    """
    with open(font_ann_path, "r") as f:
        idx_font_dict = json.load(f)
    with open(color_ann_path, "r") as f:
        idx_color_dict = json.load(f)

    if multilingual:
        font_token = [f"<{font_code[:2]}-font-{idx_font_dict[font_code]}>" for font_code in idx_font_dict]
    else:
        font_token = [f"<font-{i}>" for i in range(len(idx_font_dict))]
    color_token = [f"<color-{i}>" for i in range(len(idx_color_dict))]
    additional_special_tokens = []
    if add_color:
        additional_special_tokens += color_token
    if add_font:
        additional_special_tokens += font_token

    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    # Set mean_resizing=False to avoid PyTorch LAPACK dependency
    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def load_byt5_and_byt5_tokenizer(
    byt5_name="google/byt5-small",
    special_token=False,
    color_special_token=False,
    font_special_token=False,
    color_ann_path="assets/color_idx.json",
    font_ann_path="assets/font_idx_512.json",
    huggingface_cache_dir=None,
    multilingual=False,
    device=None,
):
    """
    Load ByT5 encoder and tokenizer from Huggingface, and add special tokens if needed.

    Args:
        byt5_name (str): Model name or path.
        special_token (bool): Whether to add special tokens.
        color_special_token (bool): Whether to add color tokens.
        font_special_token (bool): Whether to add font tokens.
        color_ann_path (str): Path to color annotation JSON.
        font_ann_path (str): Path to font annotation JSON.
        huggingface_cache_dir (str): Huggingface cache directory.
        multilingual (bool): Whether to use multilingual font tokens.
        device (str or torch.device): Device to load the model onto.

    Returns:
        tuple: (byt5_text_encoder, byt5_tokenizer)
    """
    byt5_tokenizer = AutoTokenizer.from_pretrained(
        byt5_name,
        cache_dir=huggingface_cache_dir,
    )
    byt5_text_encoder = T5ForConditionalGeneration.from_pretrained(
        byt5_name,
        cache_dir=huggingface_cache_dir,
    ).get_encoder()

    if "cuda" not in str(device):
        device = torch.device(device)
    else:
        device = torch.device(device)
    byt5_text_encoder = byt5_text_encoder.to(device)

    if special_token:
        add_special_token(
            byt5_tokenizer,
            byt5_text_encoder,
            add_color=color_special_token,
            add_font=font_special_token,
            color_ann_path=color_ann_path,
            font_ann_path=font_ann_path,
            multilingual=multilingual,
        )
    return byt5_text_encoder, byt5_tokenizer


class ByT5Mapper(nn.Module):
    """
    ByT5Mapper: Maps ByT5 encoder outputs to a new space, with optional residual connection.

    Args:
        in_dim (int): Input dimension (must equal out_dim if use_residual).
        out_dim (int): Output dimension after second linear layer.
        hidden_dim (int): Hidden dimension for intermediate layer.
        out_dim1 (int): Final output dimension.
        use_residual (bool): Whether to use residual connection (default: True).
    """

    def __init__(self, in_dim, out_dim, hidden_dim, out_dim1, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        """
        Forward pass for ByT5Mapper.

        Args:
            x (Tensor): Input tensor of shape (..., in_dim).

        Returns:
            Tensor: Output tensor of shape (..., out_dim1).
        """
        residual = x
        print(f"x {x.dtype}, self.layernorm : {self.layernorm.weight.dtype}")
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x2 = self.act_fn(x)
        x2 = self.fc3(x2)
        if self.use_residual:
            x2 = x2 + residual
        return x2


class ByT5TextEncoder:
    def __init__(
        self,
        config,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        byt5_max_length=256,
        enable_cfg=True,
    ):
        self.config = config
        self.device = device
        self.byt5_max_length = byt5_max_length
        self.enable_cfg = enable_cfg
        byT5_google_path = os.path.join(checkpoint_path, "text_encoder", "byt5-small")
        byT5_ckpt_path = os.path.join(checkpoint_path, "text_encoder", "Glyph-SDXL-v2", "checkpoints/byt5_model.pt")
        multilingual_prompt_format_color_path = os.path.join(checkpoint_path, "text_encoder", "Glyph-SDXL-v2", "assets/color_idx.json")
        multilingual_prompt_format_font_path = os.path.join(checkpoint_path, "text_encoder", "Glyph-SDXL-v2", "assets/multilingual_10-lang_idx.json")
        byt5_args = dict(
            byT5_google_path=byT5_google_path,
            byT5_ckpt_path=byT5_ckpt_path,
            multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
            multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
            byt5_max_length=byt5_max_length,
        )
        self.byt5_tokenizer, self.byt5_model, self.byt5_max_length = self.create_byt5(byt5_args, device)
        self.byt5_model = self.byt5_model.to(device=device)
        self.prompt_format = MultilingualPromptFormat(font_path=multilingual_prompt_format_font_path, color_path=multilingual_prompt_format_color_path)

        self.byt5_mapper = ByT5Mapper(in_dim=1472, out_dim=2048, hidden_dim=2048, out_dim1=self.config["hidden_size"], use_residual=False).to(torch.bfloat16)
        byt5_mapper_model_path = os.path.join(checkpoint_path, "transformer", self.config["transformer_model_name"], "diffusion_pytorch_model-00001-of-00004.safetensors")
        with safe_open(byt5_mapper_model_path, framework="pt", device="cpu") as f:
            byt5_mapper_state_dict = {key.replace("byt5_in.", ""): f.get_tensor(key).to(torch.bfloat16) for key in f.keys() if "byt5_in" in key}
        self.byt5_mapper.load_state_dict(byt5_mapper_state_dict)
        self.byt5_mapper.to(device=device)

    def create_byt5(self, args, device):
        """
        Create ByT5 tokenizer and encoder, load weights if provided.

        Args:
            args (dict): Configuration dictionary.
            device (str or torch.device): Device to load the model onto.

        Returns:
            tuple: (byt5_tokenizer, byt5_model, byt5_max_length)
        """
        byt5_max_length = args["byt5_max_length"]
        byt5_config = dict(
            byt5_name=args["byT5_google_path"],
            special_token=True,
            color_special_token=True,
            font_special_token=True,
            color_ann_path=args["multilingual_prompt_format_color_path"],
            font_ann_path=args["multilingual_prompt_format_font_path"],
            multilingual=True,
        )
        huggingface_cache_dir = None
        byt5_model, byt5_tokenizer = load_byt5_and_byt5_tokenizer(
            **byt5_config,
            huggingface_cache_dir=huggingface_cache_dir,
            device=device,
        )

        # Load custom checkpoint if provided
        if args["byT5_ckpt_path"] is not None:
            if "cuda" not in str(device):
                byt5_state_dict = torch.load(args["byT5_ckpt_path"], map_location=device)
            else:
                byt5_state_dict = torch.load(args["byT5_ckpt_path"], map_location=device)
            if "state_dict" in byt5_state_dict:
                sd = byt5_state_dict["state_dict"]
                newsd = {}
                for k, v in sd.items():
                    if k.startswith("module.text_tower.encoder."):
                        newsd[k[len("module.text_tower.encoder.") :]] = v
                byt5_state_dict = newsd
            byt5_model.load_state_dict(byt5_state_dict)
        byt5_model.requires_grad_(False)
        return byt5_tokenizer, byt5_model, byt5_max_length

    def _extract_glyph_texts(self, prompt):
        """
        Extract glyph texts from prompt using regex pattern.

        Args:
            prompt: Input prompt string

        Returns:
            List of extracted glyph texts
        """
        pattern = r"\"(.*?)\"|“(.*?)”"
        matches = re.findall(pattern, prompt)
        result = [match[0] or match[1] for match in matches]
        result = list(dict.fromkeys(result)) if len(result) > 1 else result
        return result

    def _process_single_byt5_prompt(self, prompt_text, device):
        """
        Process a single prompt for byT5 encoding.

        Args:
            prompt_text: The prompt text to process
            device: Target device for tensors

        Returns:
            Tuple of (byt5_embeddings, byt5_mask)
        """
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device)
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=device, dtype=torch.int64)

        glyph_texts = self._extract_glyph_texts(prompt_text)

        if len(glyph_texts) > 0:
            text_styles = [{"color": None, "font-family": None} for _ in range(len(glyph_texts))]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)

            text_ids, text_mask = self.get_byt5_text_tokens(self.byt5_tokenizer, self.byt5_max_length, formatted_text)
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)

            byt5_outputs = self.byt5_model(text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0]
            byt5_mask = text_mask

        return byt5_embeddings, byt5_mask

    def _prepare_byt5_embeddings(self, prompts):
        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings = []
        positive_masks = []
        negative_embeddings = []
        negative_masks = []

        for prompt in prompt_list:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, self.device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            if self.enable_cfg:  # TODO: 把cfg拆出去，更适合并行
                neg_emb, neg_mask = self._process_single_byt5_prompt("", self.device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)

        if self.enable_cfg:  # TODO: 把cfg拆出去，更适合并行
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)

            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return byt5_embeddings, byt5_masks

    @torch.no_grad()
    def infer(self, prompts):
        byt5_embeddings, byt5_masks = self._prepare_byt5_embeddings(prompts)
        byt5_features = self.byt5_mapper(byt5_embeddings.to(torch.bfloat16))
        return byt5_features, byt5_masks


if __name__ == "__main__":
    byt5 = ByT5TextEncoder(config={"transformer_model_name": "480p_t2v", "hidden_size": 2048}, device="cuda", checkpoint_path="/data/nvme1/yongyang/models/HunyuanVideo-1.5/ckpts/hunyuanvideo-1.5")
    prompt = "A close-up shot captures a scene on a polished, light-colored granite kitchen counter, illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, clear glass filled with golden, translucent apple juice standing next to a single, shiny red apple with a green leaf still attached to its stem. The camera moves horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the kitchen backsplash in the background remains softly blurred. The scene is in a realistic style."
    byt5_features, byt5_masks = byt5.infer(prompt)
    print(byt5_features.shape, byt5_features.sum())
    print(byt5_masks.shape, byt5_masks.sum())
