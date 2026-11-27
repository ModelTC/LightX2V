"""Native LightX2V YuE2 MoT: shared checkpoint, separate AR/NAR blocks.

The upstream model and pipeline are not imported. Only the separately adapted
VAE component uses the upstream dependency.
"""

import math
import time
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.utils.envs import GET_DTYPE

from .protocol import ABC_END, CODEC_OFFSET, CONTEXT, MUSIC_END, chunk_ranges
from .sampling import distribution
from .transformer_infer import YuE2TransformerInfer
from .weights import BlockOffload, YuE2Block, YuE2RMS, linear


class YuE2Model(BaseTransformerModel):
    def __init__(self, config, device):
        # Reuse checkpoint discovery/loading on CPU, including native shards.
        super().__init__(config["model_path"], config, torch.device("cpu"))
        state = self._load_ckpt(unified_dtype=True, sensitive_layer={})
        self.device = torch.device(device)
        self.dtype = GET_DTYPE()
        self.hidden = config["hidden_size"]
        self.head_dim = config["head_dim"]
        # Replicated AR preserves the single-device sampling distribution.
        # Tensor parallelism partitions the NAR acoustic transformer instead.
        self.kv_heads = config["num_key_value_heads"]
        self.nar_kv_heads = self.kv_heads // self.tp_size
        self._validate_weights(state)
        self.embedding = state["model.embed_tokens.weight"].to(self.device)
        self.position_embedding = state["latent_pos_embed.pe"].to(self.device)
        self.norm = YuE2RMS(state["model.norm.weight"].to(self.device), config["rms_norm_eps"])
        self.lm_head = linear("lm_head", state, self.device)
        self.vae2llm = linear("vae2llm", state, self.device)
        self.llm2vae = linear("llm2vae", state, self.device)
        self.time_in = linear("time_embedder.mlp.0", state, self.device)
        self.time_out = linear("time_embedder.mlp.2", state, self.device)
        block_device = torch.device("cpu") if self.cpu_offload else self.device

        def make_block(index, phase, target):
            size, rank, group = (1, 0, None) if phase == "ar" else (self.tp_size, self.tp_rank, self.tp_group)
            return YuE2Block(index, phase, state, config, target, size, rank, group)

        self.blocks = {p: [make_block(i, p, block_device) for i in range(config["num_hidden_layers"])] for p in ("ar", "nar")}
        self.offload = BlockOffload(self.blocks, lambda p: make_block(0, p, self.device), self.device) if self.cpu_offload else None
        self._init_infer()
        self.inv_freq = 1.0 / (config["rope_theta"] ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim))
        self.time_freq = torch.exp(-math.log(10000) * torch.arange(128, device=self.device, dtype=torch.float32) / 128)
        logger.info(
            "YuE2 weights loaded: rank={} NAR_TP={} AR_KV_heads={} NAR_Q_heads={} NAR_KV_heads={} core_offload={}",
            self.tp_rank,
            self.tp_size,
            self.kv_heads,
            config["num_attention_heads"] // self.tp_size,
            self.nar_kv_heads,
            self.cpu_offload,
        )

    def _validate_weights(self, state):
        c = self.config
        expected = {
            "model.embed_tokens.weight": (c["vocab_size"], c["hidden_size"]),
            "lm_head.weight": (c["vocab_size"], c["hidden_size"]),
            "latent_pos_embed.pe": (c["max_latent_frames"], c["hidden_size"]),
            "vae2llm.weight": (c["hidden_size"], 64),
            "llm2vae.weight": (64, c["hidden_size"]),
        }
        for i in range(c["num_hidden_layers"]):
            for prefix in ("self_attn", "nar_self_attn"):
                for op, heads in (("q", c["num_attention_heads"]), ("k", c["num_key_value_heads"]), ("v", c["num_key_value_heads"])):
                    expected[f"model.layers.{i}.{prefix}.{op}_proj.weight"] = (heads * c["head_dim"], c["hidden_size"])
        for key, shape in expected.items():
            if key not in state or tuple(state[key].shape) != shape:
                raise ValueError(f"Incompatible YuE2 tensor {key}; expected {shape}")

    def _init_infer_class(self):
        return YuE2TransformerInfer

    def _init_infer(self):
        cls = self._init_infer_class()
        self.prefill_infer = cls(self.config, "prefill")
        self.decode_infer = cls(self.config, "decode")
        self.transformer_infer = cls(self.config, "nar")

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        for module in (self.prefill_infer, self.decode_infer, self.transformer_infer):
            module.set_scheduler(scheduler)

    def _seq_parallel_pre_process(self, x):
        raise NotImplementedError("YuE2 uses tensor parallelism")

    def _seq_parallel_post_process(self, x):
        raise NotImplementedError("YuE2 uses tensor parallelism")

    def _infer_cond_uncond(self, inputs, infer_condition=True):
        return self.prefill(inputs["tokens"], inputs.get("capacity"))

    def infer(self, inputs):
        return self._infer_cond_uncond(inputs)

    def iter_blocks(self, phase):
        if self.offload is not None:
            yield from self.offload.iterate(phase)
        else:
            yield from enumerate(self.blocks[phase])

    def rope(self, start, length):
        angles = torch.arange(start, start + length, device=self.device).float()[:, None] * self.inv_freq
        return angles.cos(), angles.sin()

    def prefill(self, ids, capacity=None):
        if not ids or len(ids) > CONTEXT:
            raise ValueError("Empty or over-context YuE2 prefix")
        x = F.embedding(torch.tensor(ids, device=self.device), self.embedding)
        cos, sin = self.rope(0, len(ids))
        cache = []
        for i, block in self.iter_blocks("ar"):
            x, k, v = self.prefill_infer.infer(i, block, x, cos, sin)
            if capacity is not None:
                key = torch.empty((capacity, self.kv_heads, self.head_dim), dtype=self.dtype, device=self.device)
                value = torch.empty_like(key)
                key[: len(ids)].copy_(k)
                value[: len(ids)].copy_(v)
                k, v = key, value
            cache.append((k, v))
        logits = None if capacity is None else self.lm_head.apply(self.norm.apply(x[-1:]))
        return logits, cache

    def decode_token(self, token, cache, position):
        x = F.embedding(token.reshape(1), self.embedding)
        cos, sin = self.rope(position, 1)
        pos = torch.tensor([position], device=self.device)
        for i, block in self.iter_blocks("ar"):
            k, v = cache[i]
            # Dynamic used length avoids attention over the entire 9000-token
            # budget. The mutation position is a tensor, not a captured Python int.
            x = self.decode_infer.infer(i, block, x, cos, sin, k[: position + 1], v[: position + 1], pos)
        return self.lm_head.apply(self.norm.apply(x))

    def generate_tokens(self, prefix, sampling, seed, phase, check_stop, negative=None, cfg_scale=1.0, legacy_off=False):
        if max(len(prefix), len(negative or [])) + sampling.max_tokens > CONTEXT:
            raise ValueError("Prefix plus generation budget exceeds 24576; no implicit truncation")
        if cfg_scale != 1 and negative is None:
            raise ValueError("CFG requires its exact negative prefix")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        positive = negative_cache = None
        end = ABC_END if phase == "abc" else MUSIC_END
        started = time.perf_counter()
        history, eos = [], False
        try:
            check_stop()
            conditional, positive = self.prefill(prefix, len(prefix) + sampling.max_tokens)
            if cfg_scale != 1:
                unconditional, negative_cache = self.prefill(negative, len(negative) + sampling.max_tokens)
            for step in range(sampling.max_tokens):
                check_stop()
                # Preserve BF16 CFG before symbolic FP32 / legacy-off BF16 sampling.
                logits = conditional if cfg_scale == 1 else unconditional + cfg_scale * (conditional - unconditional)
                sampling_error = None
                if self.tp_rank == 0:
                    try:
                        scores = distribution(logits, sampling, history, step, phase, legacy_off)
                        if not torch.isfinite(scores.max()).item():
                            raise FloatingPointError("No finite selectable YuE2 logits")
                        next_id = scores.argmax(-1) if sampling.temperature == 0 else torch.multinomial(scores.softmax(-1), 1, generator=generator).reshape(1)
                    except (RuntimeError, ValueError, FloatingPointError) as exc:
                        # All ranks must leave sampling together, including an
                        # error originating only on the sampling rank.
                        sampling_error = f"{type(exc).__name__}: {exc}"
                        next_id = torch.full((1,), -1, dtype=torch.long, device=self.device)
                else:
                    next_id = torch.empty(1, dtype=torch.long, device=self.device)
                if self.tp_size > 1:
                    dist.broadcast(next_id, src=0, group=self.tp_group)
                token = int(next_id.item())
                if token < 0:
                    errors = [sampling_error]
                    if self.tp_size > 1:
                        dist.broadcast_object_list(errors, src=0, group=self.tp_group)
                    raise FloatingPointError(errors[0])
                if token == end:
                    eos = True
                    break
                history.append(token)
                if step + 1 < sampling.max_tokens:
                    conditional = self.decode_token(next_id, positive, len(prefix) + step)
                    if negative_cache is not None:
                        unconditional = self.decode_token(next_id, negative_cache, len(negative) + step)
            return history, {"seconds": time.perf_counter() - started, "tokens": len(history), "ended": eos}
        finally:
            positive = negative_cache = None

    def velocity(self, state, raw_time, cache, cos, sin, position_embedding):
        x = self.vae2llm.apply(F.pad(state, (0, 0, 1, 1)))
        t = torch.sigmoid(torch.tensor(raw_time, device=self.device, dtype=self.dtype))
        shift = self.config["timestep_shift"]
        t = shift * t / (1 + (shift - 1) * t)
        args = t.expand(len(x)).float()[:, None] * self.time_freq[None]
        emb = torch.cat((args.cos(), args.sin()), dim=-1).to(self.dtype)
        x = x + self.time_out.apply(F.silu(self.time_in.apply(emb)))
        x = x + position_embedding
        for i, block in self.iter_blocks("nar"):
            k, v = cache[i]
            x = self.transformer_infer.infer(i, block, x, cos, sin, k, v)
        return self.llm2vae.apply(self.norm.apply(x))[1:-1]

    def synthesize(self, prefix, codec, seed, check_stop):
        ranges = chunk_ranges(len(codec), len(prefix))
        # One CPU draw for the complete song, then slices at historical cuts.
        noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=torch.Generator(device="cpu").manual_seed(seed))
        outputs = []
        for a, b in ranges:
            check_stop()
            ids = prefix + [v + CODEC_OFFSET for v in codec[a:b]] + [MUSIC_END]
            cache = None
            try:
                _, cache = self.prefill(ids)
                if self.tp_size > 1:
                    # NAR attends to the matching head slice of the AR prefix.
                    cache = [(k.chunk(self.tp_size, dim=1)[self.tp_rank].contiguous(), v.chunk(self.tp_size, dim=1)[self.tp_rank].contiguous()) for k, v in cache]
                count = b - a + 2
                cos, sin = self.rope(len(ids), count)
                positions = torch.arange(count, device=self.device).clamp(max=self.config["max_latent_frames"] - 1)
                pe = self.position_embedding[positions]
                self.scheduler.prepare(noise[a:b], self.device, self.dtype)
                velocity = partial(self.velocity, cache=cache, cos=cos, sin=sin, position_embedding=pe)
                outputs.append(self.scheduler.solve(velocity, check_stop))
            finally:
                self.scheduler.clear()
                cache = None
        return torch.cat(outputs)

    def runtime_stats(self):
        return {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "parallel_strategy": "nar_tensor_parallel_replicated_ar",
            "ar_kv_heads": self.kv_heads,
            "nar_kv_heads_per_rank": self.nar_kv_heads,
            "blocks_executed": {m.phase: m.blocks_executed for m in (self.prefill_infer, self.decode_infer, self.transformer_infer)},
            "compiled_block_callables": {m.phase: len(m.compiled_blocks) for m in (self.prefill_infer, self.decode_infer, self.transformer_infer)},
            "offload_transfers": self.offload.transfer_count if self.offload else 0,
            "offload_bytes": self.offload.transfer_bytes if self.offload else 0,
        }

    def close(self):
        if self.offload is not None:
            self.offload.close()
        for module in (self.prefill_infer, self.decode_infer, self.transformer_infer):
            module.compiled_blocks.clear()
        self.blocks.clear()
        self.close_numa_cpu_weights()
