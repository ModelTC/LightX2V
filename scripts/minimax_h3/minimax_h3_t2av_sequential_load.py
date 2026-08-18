#!/usr/bin/env python3
"""Run H3 T2AV in two processes so text and DiT host weights never overlap."""

import os

import torch
import torch.distributed as dist
from loguru import logger

import lightx2v.infer as infer_module
from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v_platform.base.global_var import AI_DEVICE


_PHASE = os.environ.get("MINIMAX_H3_PHASE")
_PROMPT_CACHE = os.environ.get("MINIMAX_H3_PROMPT_CACHE")
if _PHASE not in {"encode", "infer"}:
    raise ValueError("MINIMAX_H3_PHASE must be 'encode' or 'infer'")
if not _PROMPT_CACHE:
    raise ValueError("MINIMAX_H3_PROMPT_CACHE must point to the prompt cache")


def _load_text_only(self):
    logger.info("MiniMax-H3 low-memory encode phase: loading only the text encoder")
    self.model = None
    self.text_encoders = self.load_text_encoder()
    self.video_vae = None
    self.audio_vae = None


def _save_prompt_cache(self):
    encoded = self.inputs["text_encoder_output"]
    if not dist.is_initialized() or dist.get_rank() == 0:
        cache = {
            "prompt": self.input_info.prompt,
            "prompt_embeds": encoded["prompt_embeds"].cpu(),
            "text_token_tags": encoded["text_token_tags"].cpu(),
        }
        torch.save(cache, _PROMPT_CACHE)
        logger.info("MiniMax-H3 prompt cache saved to {}", _PROMPT_CACHE)
    if dist.is_initialized():
        dist.barrier()
    return {"video": None, "audio": None}


def _load_generation_only(self):
    logger.info("MiniMax-H3 low-memory infer phase: loading DiT and VAEs without Qwen")
    self.model = self.load_transformer()
    self.text_encoders = []
    self.video_vae, self.audio_vae = self.load_vae()


def _load_cached_prompt(self, input_info, keyframes=None, references=None):
    if keyframes or references:
        raise ValueError("The cached low-memory path currently supports T2AV only")
    cache = torch.load(_PROMPT_CACHE, map_location="cpu", weights_only=True)
    if cache.get("prompt") != input_info.prompt:
        raise ValueError("Prompt cache does not match the requested prompt")
    logger.info("Loading MiniMax-H3 prompt cache from {}", _PROMPT_CACHE)
    return {
        "prompt_embeds": cache["prompt_embeds"].to(device=AI_DEVICE).contiguous(),
        "text_token_tags": cache["text_token_tags"].to(device=AI_DEVICE),
    }


def _infer_blocks_synchronously(self, blocks, hidden_states, pre_infer_out):
    """Avoid concurrent XPU weight copies while a block GEMM is running."""
    manager = self.offload_manager
    compute_buffer = manager.cuda_buffers[0]
    for block_index, source_block in enumerate(blocks):
        compute_buffer.load_state_dict(source_block.state_dict(), block_index)
        getattr(torch, AI_DEVICE).synchronize()
        self.block_idx = block_index
        hidden_states = self.run_block(block_index, compute_buffer, hidden_states, pre_infer_out)
        getattr(torch, AI_DEVICE).synchronize()
    manager.need_init_first_buffer = False
    return hidden_states


if _PHASE == "encode":
    MiniMaxH3Runner.load_model = _load_text_only
    MiniMaxH3Runner.run_main = _save_prompt_cache
else:
    MiniMaxH3Runner.load_model = _load_generation_only
    MiniMaxH3Runner.run_text_encoder = _load_cached_prompt
    MiniMaxH3OffloadTransformerInfer.infer_with_blocks_offload = _infer_blocks_synchronously


if __name__ == "__main__":
    infer_module.main()
