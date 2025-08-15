# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.input_encoders.hf.q_linear import Q8FQuantLinearFp8, Q8FQuantLinearInt8, TorchaoQuantLinearInt8, VllmQuantLinearFp8, VllmQuantLinearInt8
from lightx2v.utils.envs import *
from lightx2v.utils.utils import load_weights

from .tokenizer import HuggingfaceTokenizer

__all__ = [
    "T5Model",
    "T5Encoder",
    "T5Decoder",
    "T5EncoderModel",
]


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def optimize_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn) ** -0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn) ** -0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(m.embedding.weight, std=(2 * m.num_buckets * m.num_heads) ** -0.5)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, dtype=torch.float16):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1, quantized=False, quant_scheme=None, dtype=torch.bfloat16):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        if quantized:
            if quant_scheme == "int8":
                linear_cls = VllmQuantLinearInt8
            elif quant_scheme == "fp8":
                linear_cls = VllmQuantLinearFp8
            elif quant_scheme == "int8-torchao":
                linear_cls = TorchaoQuantLinearInt8
            elif quant_scheme == "int8-q8f":
                linear_cls = Q8FQuantLinearInt8
            elif quant_scheme == "fp8-q8f":
                linear_cls = Q8FQuantLinearFp8
        else:
            linear_cls = nn.Linear

        # layers
        self.q = linear_cls(dim, dim_attn, bias=False, dtype=dtype)
        self.k = linear_cls(dim, dim_attn, bias=False, dtype=dtype)
        self.v = linear_cls(dim, dim_attn, bias=False, dtype=dtype)
        self.o = linear_cls(dim_attn, dim, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias

        if hasattr(self, "cpu_offload") and self.cpu_offload:
            del attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)

        if hasattr(self, "cpu_offload") and self.cpu_offload:
            del attn
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn, dropout=0.1, quantized=False, quant_scheme=None, dtype=torch.bfloat16):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        if quantized:
            if quant_scheme == "int8":
                linear_cls = VllmQuantLinearInt8
            elif quant_scheme == "fp8":
                linear_cls = VllmQuantLinearFp8
            elif quant_scheme == "int8-torchao":
                linear_cls = TorchaoQuantLinearInt8
            elif quant_scheme == "int8-q8f":
                linear_cls = Q8FQuantLinearInt8
            elif quant_scheme == "fp8-q8f":
                linear_cls = Q8FQuantLinearFp8
        else:
            linear_cls = nn.Linear
        # layers
        self.gate = nn.Sequential(linear_cls(dim, dim_ffn, bias=False, dtype=dtype), GELU())
        self.fc1 = linear_cls(dim, dim_ffn, bias=False, dtype=dtype)
        self.fc2 = linear_cls(dim_ffn, dim, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if hasattr(self, "cpu_offload") and self.cpu_offload:
            gate_out = self.gate(x)
            fc1_out = self.fc1(x)
            x = fc1_out * gate_out
            del gate_out, fc1_out
        else:
            x = self.fc1(x) * self.gate(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos=True, dropout=0.1, quantized=False, quant_scheme=None, dtype=torch.bfloat16):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim, dtype=dtype)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout, quantized, quant_scheme, dtype)
        self.norm2 = T5LayerNorm(dim, dtype=dtype)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout, quantized, quant_scheme, dtype=dtype)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True, dtype=dtype)

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))

        if hasattr(self, "cpu_offload") and self.cpu_offload:
            attn_out = self.attn(self.norm1(x), mask=mask, pos_bias=e)
            x = fp16_clamp(x + attn_out)
            del attn_out

            ffn_out = self.ffn(self.norm2(x))
            x = fp16_clamp(x + ffn_out)
            del ffn_out
        else:
            x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
            x = fp16_clamp(x + self.ffn(self.norm2(x)))

        return x


class T5CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False)

    def forward(self, x, mask=None, encoder_states=None, encoder_mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional, dtype=torch.bfloat16, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads, dtype=dtype)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):
    def __init__(self, dtype, vocab, dim, dim_attn, dim_ffn, num_heads, num_layers, num_buckets, shared_pos=True, dropout=0.1, cpu_offload=False, quantized=False, quant_scheme=None):
        super(T5Encoder, self).__init__()

        self.cpu_offload = cpu_offload
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.quant_scheme = quant_scheme

        # layers
        self.token_embedding = vocab.to(dtype) if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim, dtype=dtype)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True, dtype=dtype) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout, quantized, quant_scheme, dtype) for _ in range(num_layers)])

        if cpu_offload:
            for block in self.blocks:
                block.cpu_offload = cpu_offload
                block.attn.cpu_offload = cpu_offload
                block.ffn.cpu_offload = cpu_offload
        self.norm = T5LayerNorm(dim, dtype=dtype)

        # initialize weights
        # self.apply(init_weights)

    def forward(self, ids, mask=None):
        if self.cpu_offload:
            self.token_embedding = self.token_embedding.cuda()
        x = self.token_embedding(ids)
        if self.cpu_offload:
            self.token_embedding = self.token_embedding.cpu()
            optimize_memory_usage()
        x = self.dropout(x)

        if self.cpu_offload and self.pos_embedding is not None:
            self.pos_embedding = self.pos_embedding.cuda()
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        if self.cpu_offload and self.pos_embedding is not None:
            self.pos_embedding = self.pos_embedding.cpu()
            optimize_memory_usage()

        for i, block in enumerate(self.blocks):
            if self.cpu_offload:
                block = block.cuda()
            x = block(x, mask, pos_bias=e)
            if self.cpu_offload:
                block = block.cpu()
                del block
                optimize_memory_usage()

        if self.cpu_offload:
            self.norm = self.norm.cuda()
        x = self.norm(x)
        if self.cpu_offload:
            self.norm = self.norm.cpu()
            optimize_memory_usage()

        x = self.dropout(x)
        return x.to(GET_DTYPE())


class T5Decoder(nn.Module):
    def __init__(
        self,
        vocab,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        num_layers,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout) for _ in range(num_layers)])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        b, s = ids.size()

        # causal mask
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        encoder_layers,
        decoder_layers,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(
            self.token_embedding,
            dim,
            dim_attn,
            dim_ffn,
            num_heads,
            encoder_layers,
            num_buckets,
            shared_pos,
            dropout,
        )
        self.decoder = T5Decoder(
            self.token_embedding,
            dim,
            dim_attn,
            dim_ffn,
            num_heads,
            decoder_layers,
            num_buckets,
            shared_pos,
            dropout,
        )
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(
    name,
    encoder_only=False,
    decoder_only=False,
    return_tokenizer=False,
    tokenizer_kwargs={},
    dtype=torch.float32,
    device="cpu",
    **kwargs,
):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs["vocab"] = kwargs.pop("vocab_size")
        kwargs["num_layers"] = kwargs.pop("encoder_layers")
        _ = kwargs.pop("decoder_layers")
    elif decoder_only:
        model_cls = T5Decoder
        kwargs["vocab"] = kwargs.pop("vocab_size")
        kwargs["num_layers"] = kwargs.pop("decoder_layers")
        _ = kwargs.pop("encoder_layers")
    else:
        model_cls = T5Model

    # init model
    with torch.device(device):
        model = model_cls(dtype=dtype, **kwargs)

    # set device
    model = model.to(device=device)
    return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1,
    )
    cfg.update(**kwargs)
    return _t5("umt5-xxl", **cfg)


class T5EncoderModel:
    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
        cpu_offload=False,
        offload_granularity="model",
        t5_quantized=False,
        t5_quantized_ckpt=None,
        quant_scheme=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        if t5_quantized_ckpt is not None and t5_quantized:
            self.checkpoint_path = t5_quantized_ckpt
        else:
            self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.offload_granularity = offload_granularity

        # sync cpu offload
        self.cpu_offload = cpu_offload
        if self.cpu_offload:
            assert self.offload_granularity in ["block", "model"]

        model = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=dtype,
                device=device,
                cpu_offload=(cpu_offload if self.offload_granularity == "block" else False),
                quantized=t5_quantized,
                quant_scheme=quant_scheme,
            )
            .eval()
            .requires_grad_(False)
        )

        weights_dict = load_weights(self.checkpoint_path, cpu_offload=cpu_offload)
        model.load_state_dict(weights_dict)

        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean="whitespace")

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to("cuda")

    def optimize_memory(self):
        """优化内存使用"""
        optimize_memory_usage()

    def infer(self, texts):
        if self.cpu_offload and self.offload_granularity == "model":
            self.to_cuda()

        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.cuda()
        mask = mask.cuda()
        seq_lens = mask.gt(0).sum(dim=1).long()

        with torch.no_grad():
            context = self.model(ids, mask)

        if self.cpu_offload and self.offload_granularity == "model":
            self.to_cpu()
            optimize_memory_usage()

        del ids, mask
        if self.cpu_offload:
            optimize_memory_usage()

        return [u[:v] for u, v in zip(context, seq_lens)]


if __name__ == "__main__":
    checkpoint_dir = ""
    t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer = "google/umt5-xxl"
    model = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        checkpoint_path=os.path.join(checkpoint_dir, t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, t5_tokenizer),
        shard_fn=None,
    )
    text = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    outputs = model.infer(text)
    logger.info(outputs)
