# Adapted from ModelTC integration reference: multimodal-art-projection/YuE
# commit 1116bea76032bee5668e5f097294d082fa0b84f0.
import torch

from .protocol import ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END


def window_penalty(logits, recent_ids, penalty):
    if penalty == 1.0 or len(recent_ids) == 0:
        return logits
    recent = torch.as_tensor(recent_ids, dtype=torch.long, device=logits.device).reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
    alpha = penalty**freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def distribution(logits, sampling, history, step, phase, legacy_off=False):
    # vLLM's symbolic processor receives FP32 logits; historical off uses BF16.
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores = scores + allowed
    if step < sampling.min_tokens:
        scores[..., end] = -torch.inf
    scores = window_penalty(scores, history[-sampling.penalty_window :], sampling.repetition_penalty)
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / sampling.temperature
    threshold = scores.topk(min(sampling.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if sampling.top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
        removed[..., : 3 if legacy_off else 1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores
