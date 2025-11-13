# Modified from ``https://github.com/openai/CLIP'' and ``https://github.com/mlfoundations/open_clip''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers.models import ModelMixin


from lightx2v.models.input_encoders.hf.wan.xlm_roberta.model import (
    VisionTransformer
)
from lightx2v.models.input_encoders.hf.wan.matrix_game2.tokenizers import (
    HuggingfaceTokenizer
)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x:   [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, s, n, d).permute(0, 2, 1, 3)

        # compute attention
        p = self.dropout.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, mask, p)
        x = x.permute(0, 2, 1, 3).reshape(b, s, c)

        # output
        x = self.o(x)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, post_norm, dropout=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.eps = eps

        # layers
        self.attn = SelfAttention(dim, num_heads, dropout, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, mask):
        if self.post_norm:
            x = self.norm1(x + self.attn(x, mask))
            x = self.norm2(x + self.ffn(x))
        else:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.ffn(self.norm2(x))
        return x


class XLMRoberta(nn.Module):
    """
    XLMRobertaModel with no pooler and no LM head.
    """

    def __init__(self, vocab_size=250002, max_seq_len=514, type_size=1, pad_id=1, dim=1024, num_heads=16, num_layers=24, post_norm=True, dropout=0.1, eps=1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.post_norm = post_norm
        self.eps = eps

        # embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.type_embedding = nn.Embedding(type_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        # blocks
        self.blocks = nn.ModuleList([AttentionBlock(dim, num_heads, post_norm, dropout, eps) for _ in range(num_layers)])

        # norm layer
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, ids):
        """
        ids: [B, L] of torch.LongTensor.
        """
        b, s = ids.shape
        mask = ids.ne(self.pad_id).long()

        # embeddings
        x = self.token_embedding(ids) + self.type_embedding(torch.zeros_like(ids)) + self.pos_embedding(self.pad_id + torch.cumsum(mask, dim=1) * mask)
        if self.post_norm:
            x = self.norm(x)
        x = self.dropout(x)

        # blocks
        mask = torch.where(mask.view(b, 1, 1, s).gt(0), 0.0, torch.finfo(x.dtype).min)
        for block in self.blocks:
            x = block(x, mask)

        # output
        if not self.post_norm:
            x = self.norm(x)
        return x


class XLMRobertaWithHead(XLMRoberta):
    def __init__(self, **kwargs):
        self.out_dim = kwargs.pop("out_dim")
        super().__init__(**kwargs)

        # head
        mid_dim = (self.dim + self.out_dim) // 2
        self.head = nn.Sequential(nn.Linear(self.dim, mid_dim, bias=False), nn.GELU(), nn.Linear(mid_dim, self.out_dim, bias=False))

    def forward(self, ids):
        # xlm-roberta
        x = super().forward(ids)

        # average pooling
        mask = ids.ne(self.pad_id).unsqueeze(-1).to(x)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        # head
        x = self.head(x)
        return x


class XLMRobertaCLIP(nn.Module):
    def __init__(
        self,
        dtype=torch.float16,
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool="token",
        vision_pre_norm=True,
        vision_post_norm=False,
        activation="gelu",
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
        norm_eps=1e-5,
        quantized=False,
        quant_scheme=None,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        text_post_norm=True,
        text_dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.norm_eps = norm_eps

        # models
        self.visual = VisionTransformer(
            dtype=dtype,
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps,
            quantized=quantized,
            quant_scheme=quant_scheme,
        )
        self.textual = XLMRobertaWithHead(
            vocab_size=vocab_size,
            max_seq_len=max_text_len,
            type_size=type_size,
            pad_id=pad_id,
            dim=text_dim,
            out_dim=embed_dim,
            num_heads=text_heads,
            num_layers=text_layers,
            post_norm=text_post_norm,
            dropout=text_dropout,
        )
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))


def _clip(pretrained=False, pretrained_name=None, model_cls=XLMRobertaCLIP, return_transforms=False, return_tokenizer=False, tokenizer_padding="eos", dtype=torch.float32, device="cpu", **kwargs):
    # init a model on device
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)
    output = (model,)

    # init transforms
    if return_transforms:
        # mean and std
        if "siglip" in pretrained_name.lower():
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]

        # transforms
        transforms = T.Compose([T.Resize((model.image_size, model.image_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=mean, std=std)])
        output += (transforms,)
    return output[0] if len(output) == 1 else output


def clip_xlm_roberta_vit_h_14(pretrained=False, pretrained_name="open-clip-xlm-roberta-large-vit-huge-14", **kwargs):
    cfg = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool="token",
        activation="gelu",
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        text_post_norm=True,
        text_dropout=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
    )
    cfg.update(**kwargs)
    return _clip(pretrained, pretrained_name, XLMRobertaCLIP, **cfg)


class CLIPModel(ModelMixin):
    def __init__(self, checkpoint_path, tokenizer_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
        )
        self.model = self.model.eval().requires_grad_(False)
        logging.info(f"loading {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=self.model.max_text_len - 2, clean="whitespace")

    def encode_video(self, video):
        # preprocess
        b, c, t, h, w = video.shape
        video = video.transpose(1, 2)
        video = video.reshape(b * t, c, h, w)
        size = (self.model.image_size,) * 2
        video = F.interpolate(video, size=size, mode="bicubic", align_corners=False)

        video = self.transforms.transforms[-1](video.mul_(0.5).add_(0.5))

        # forward
        with torch.amp.autocast(dtype=self.dtype, device_type=self.device.type):
            out = self.model.visual(video, use_31_block=True)

        return out

    def forward(self, videos):
        # preprocess
        size = (self.model.image_size,) * 2
        videos = torch.cat([F.interpolate(u.transpose(0, 1), size=size, mode="bicubic", align_corners=False) for u in videos])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        with torch.amp.autocast("cuda", dtype=self.dtype):
            out = self.model.visual(videos, use_31_block=True)
            return out