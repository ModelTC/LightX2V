"""Ten end-to-end latency groups using LightX2V's native FastWAM weights.

Run from the LightX2V repository using its existing uv environment (no uv sync):
    uv run --no-project --python .venv/bin/python scripts/fastwam/bench_latency.py \
        ckpt=/absolute/path/libero_uncond_2cam224.pt \
        +BENCH.groups=[1,2,3,4,5,6,7,8,9,10]

Groups: 1/2=10-step eager/graph,
3/4=1-step eager/graph, 5/6=single/split-expert asynchronous graphs,
7/8=5/6 with LightX2V kernels, 9/10=optimized SP2/TP2 with LightX2V kernels.
Groups 9/10 correspond to groups 11/12 in the former fourteen-group benchmark.
Parallel scheduling is local to this
script; the repository's FastWAM serial implementation is not changed.

Only text encoding is cached. CPU input copies, CPU RNG, VAE, preparation,
transformers, scheduling, communication and CPU action output are timed.
Graph capture and three-case correctness checks are excluded. Distributed time
is the slower rank's wall time, with timing barrier/reduction outside the timer.
Groups 7-10 use actual LightX2V fused QK RMSNorm, LayerNorm, scale/shift,
RoPE and Ulysses layout kernels, not the original benchmark's Triton copies.
They may change BF16 rounding; errors are recorded and failures are not timed.
No FP8/INT8 quantization or per-request context-KV caching is used.

Overrides use OmegaConf dotlists, without Hydra output directories. Defaults:
+BENCH.warmup=10 +BENCH.iters=100 +BENCH.graph_warmup=3
+BENCH.output_json=/tmp/fastwam_lightx2v_latency.json
+BENCH.use_random_context=false +BENCH.atol=0.02 +BENCH.rtol=0
+BENCH.kernel_affine=true +BENCH.kernel_rope=true
The action sigma shift defaults to 1 to match the original benchmark (the
LightX2V policy config defaults to 5). Actions are normalized model outputs,
before the policy's dataset denormalization or gripper postprocessing.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from datetime import timedelta
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "BF16")
os.environ.setdefault("LOGURU_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

log = logging.getLogger("bench_latency")
EXECUTIONS = {
    1: "Sequential", 2: "Sequential", 3: "Sequential", 4: "Sequential",
    5: "Asynchronous", 6: "Asynchronous",
    7: "Asynchronous", 8: "Asynchronous",
    9: "SP2, LightX2V optimized", 10: "TP2, LightX2V optimized",
}
FUSED = {7, 8, 9, 10}


def bootstrap_native_imports():
    # The public package initializer imports every model pipeline, including LTX2.
    # This standalone benchmark only needs FastWAM; actual submodules and kernel
    # registries still execute normally, without unrelated Transformers imports.
    if "lightx2v" not in sys.modules:
        import lightx2v_platform.set_ai_device  # noqa: F401
        package = ModuleType("lightx2v")
        package.__path__ = [str(ROOT / "lightx2v")]
        package.__package__ = "lightx2v"
        sys.modules["lightx2v"] = package


def config():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    defaults = {
        "ckpt": "/mnt/lm_data_afs/charles/codes/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt",
        "model_path": "/mnt/miaohua/charles/models/Wan2.2-TI2V-5B",
        "BENCH": {
            "groups": list(EXECUTIONS), "warmup": 10, "iters": 100,
            "graph_warmup": 3, "seed": 42, "device": "cuda:0",
            "action_device": "cuda:1", "action_horizon": 32,
            "height": 224, "width": 448, "sigma_shift": 1.0,
            "prompt": "pick up the object", "use_random_context": False,
            "verify": True, "atol": 0.02, "rtol": 0.0,
            "kernel_affine": True, "kernel_rope": True, "kernel_layer_norm": True,
            "output_json": "/tmp/fastwam_lightx2v_latency.json",
        },
    }
    overrides = OmegaConf.from_dotlist([arg.lstrip("+") for arg in sys.argv[1:]])
    cfg = OmegaConf.merge(defaults, overrides)
    unknown = set(cfg) - set(defaults)
    unknown |= set(cfg.BENCH) - set(defaults["BENCH"])
    if unknown:
        raise ValueError(f"Unknown overrides: {sorted(unknown)}")
    b = cfg.BENCH
    if not b.groups or len(set(b.groups)) != len(b.groups) or any(i not in EXECUTIONS for i in b.groups):
        raise ValueError("BENCH.groups must contain distinct integers in 1..10")
    if b.iters <= 0 or b.warmup < 0 or b.graph_warmup <= 0:
        raise ValueError("Invalid warmup/iteration counts")
    if b.height % 16 or b.width % 16 or b.action_horizon <= 0:
        raise ValueError("Image dimensions must be multiples of 16; horizon must be positive")
    if 9 in b.groups and b.action_horizon % 2:
        raise ValueError("SP2 requires an even action horizon")
    if 9 in b.groups and (b.height // 32 * (b.width // 32)) % 2:
        raise ValueError("SP2 requires an even number of Video tokens")
    if os.environ["DTYPE"] != "BF16" or os.environ["SENSITIVE_LAYER_DTYPE"] != "BF16":
        raise ValueError("This comparison requires DTYPE=BF16 and SENSITIVE_LAYER_DTYPE=BF16")
    cfg.ckpt = str(Path(cfg.ckpt).expanduser().resolve())
    if b.output_json and Path(b.output_json).resolve().is_relative_to(ROOT):
        raise ValueError("Write reports outside LightX2V to preserve the one-file modification constraint")
    return cfg


def sync(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    index = int(position)
    return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (position - index)


class CapturedCall:
    """Stable outputs, side-stream warmup, and a device-local CUDA Graph."""

    def __init__(self, function, device, warmup):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            for _ in range(warmup):
                self.output = function()
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(device), torch.cuda.graph(self.graph, stream=self.stream):
            self.output = function()
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            self.graph.replay()
        self.stream.synchronize()

    def __call__(self):
        with torch.cuda.device(self.device):
            self.graph.replay()
        return self.output


def load_model(cfg, device):
    bootstrap_native_imports()
    from lightx2v.models.networks.wan.fastwam_model import FastWAMNativeModel
    from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE

    native = json.loads((ROOT / "configs/fastwam/libero_i2va.json").read_text())
    native.update(adapter_model_path=cfg.ckpt, seq_parallel=False, tensor_parallel=False,
                  cpu_offload=False, action_sample_shift=float(cfg.BENCH.sigma_shift))
    log.info("Loading native FastWAM and LightX2V VAE on %s", device)
    model = FastWAMNativeModel(str(cfg.model_path), native, device)
    vae = Wan2_2_VAE(vae_path=str(Path(cfg.model_path) / "Wan2.2_VAE.pth"),
                    device=device, dtype=torch.bfloat16, cpu_offload=False)
    vae.model.eval().requires_grad_(False)
    # CPU RoPE table copies cannot be captured. Move these constant tables once.
    model.pre_infer.video_freqs = tuple(x.to(device) for x in model.pre_infer.video_freqs)
    model.pre_infer.action_freqs = model.pre_infer.action_freqs.to(device)
    return model, vae


def inputs(cfg, device):
    b = cfg.BENCH
    generator = torch.Generator().manual_seed(int(b.seed))
    image = torch.rand(1, 3, b.height, b.width, generator=generator) * 2 - 1
    proprio = torch.randn(1, 8, generator=generator)
    encoder = None
    if b.use_random_context:
        context = torch.randn(128, 4096, generator=generator).to(device=device, dtype=torch.bfloat16)
        mask = torch.ones(128, device=device, dtype=torch.bool)
    else:
        from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
        log.info("Loading and retaining the native text encoder on %s", device)
        encoder = T5EncoderModel(128, dtype=torch.bfloat16, device=device,
                                 checkpoint_path=str(Path(cfg.model_path) / "models_t5_umt5-xxl-enc-bf16.pth"),
                                 tokenizer_path=str(Path(cfg.model_path) / "google/umt5-xxl"))
        ids, mask = encoder.tokenizer([str(b.prompt)], return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(device), mask.to(device)
        context = encoder.model(ids, mask)
        context[0, int(mask.sum().item()):] = 0
        context, mask = context[0].to(torch.bfloat16), torch.ones_like(mask[0], dtype=torch.bool)
    cases = [(image, proprio, int(b.seed)), (image * 0.5, proprio + 0.25, int(b.seed)),
             (image, proprio, int(b.seed) + 1)]
    return cases, context, mask, encoder


def move_weights(module, device, seen=None):
    """LightX2V WeightModule is not torch.nn.Module; move only owned tensors."""
    seen = set() if seen is None else seen
    if id(module) in seen:
        return
    seen.add(id(module))
    for name, value in vars(module).items():
        if isinstance(value, torch.Tensor):
            setattr(module, name, value.to(device))
    for child in getattr(module, "_modules", {}).values():
        move_weights(child, device, seen)


class Operators:
    """Native transformer operations, optionally using repository kernels."""

    def __init__(self, model, bench, fused=False, tp_group=None):
        self.native = model.transformer_infer
        self.fused, self.bench, self.tp_group = fused, bench, tp_group
        self.heads = 12 if tp_group is not None else 24
        if fused:
            from lightx2v.common.ops.norm.triton_ops import (
                norm_infer, fused_qk_rms_norm, fuse_scale_shift_kernel, apply_rotary_embedding,
            )
            self.norm_kernel, self.qk_kernel = norm_infer, fused_qk_rms_norm
            self.affine_kernel, self.rope_kernel = fuse_scale_shift_kernel, apply_rotary_embedding

    def norm(self, module, x):
        if self.fused and self.bench.kernel_layer_norm:
            return self.norm_kernel(x, module.weight, module.bias, module.eps)
        return module.apply(x)

    def modulate(self, x, shift, scale):
        if self.fused and self.bench.kernel_affine:
            # The native helper treats [N,C] modulation as batch; make B,L,C explicit.
            return self.affine_kernel(x, scale.unsqueeze(0), shift.unsqueeze(0),
                                      block_l=1, block_c=256).reshape_as(x)
        return x * (1 + scale) + shift

    def qk_norm(self, q, k, attention):
        if self.tp_group is not None:
            # RMSNorm is over all 3072 attention channels, not individual heads.
            if self.fused:
                sums = torch.cat((q.float().square().sum(-1, keepdim=True),
                                  k.float().square().sum(-1, keepdim=True)), dim=0)
                dist.all_reduce(sums, group=self.tp_group)
                qs, ks = sums.split((q.shape[0], k.shape[0]), dim=0)
            else:
                qs, ks = q.float().square().sum(-1, keepdim=True), k.float().square().sum(-1, keepdim=True)
                dist.all_reduce(qs, group=self.tp_group)
                dist.all_reduce(ks, group=self.tp_group)
            q = (q.float() * torch.rsqrt(qs / 3072 + attention.norm_q.eps)).to(q.dtype) * attention.norm_q.weight
            k = (k.float() * torch.rsqrt(ks / 3072 + attention.norm_k.eps)).to(k.dtype) * attention.norm_k.weight
            return q, k
        if self.fused:
            return self.qk_kernel(q, k, attention.norm_q.weight, attention.norm_k.weight,
                                  attention.norm_q.eps, match_torch_rms_cast=True)
        return attention.norm_q.apply(q), attention.norm_k.apply(k)

    def project(self, block, x, prepared):
        if not self.fused and self.tp_group is None:
            return self.native._build_self_attention_io(block, x, prepared.freqs, prepared.t_mod)
        a = block.self_attn
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = self.native._split_modulation(a, prepared.t_mod)
        z = self.modulate(self.norm(a.norm1, x), shift, scale)
        q, k = self.qk_norm(a.q.apply(z), a.k.apply(z), a)
        q, k, v = [t.reshape(t.shape[0], self.heads, 128) for t in (q, k, a.v.apply(z))]
        if self.fused and self.bench.kernel_rope:
            # FP32 RoPE is the repository kernel's contract; validation measures its error.
            cos, sin = prepared.freqs.real.reshape(-1, 64), prepared.freqs.imag.reshape(-1, 64)
            q = self.rope_kernel(q, cos, sin, interleaved=True)
            k = self.rope_kernel(k, cos, sin, interleaved=True)
        else:
            q, k = a.rope.apply(q, k, prepared.freqs)
        return q, k, v, x, gate, shift_mlp, scale_mlp, gate_mlp

    def post(self, block, io, mixed, prepared):
        if not self.fused and self.tp_group is None:
            return self.native._post_block(block, io[3], mixed, *io[4:8], prepared.context, prepared.context_mask)
        x = io[3] + io[4] * block.self_attn.o.apply(mixed)
        a = block.cross_attn
        q, k = self.qk_norm(a.q.apply(self.norm(a.norm3, x)), a.k.apply(prepared.context), a)
        v = a.v.apply(prepared.context)
        q, k, v = [t.reshape(t.shape[0], self.heads, 128) for t in (q, k, v)]
        x = x + a.o.apply(a.attn.apply(q, k, v, attn_mask=prepared.context_mask))
        z = self.modulate(self.norm(block.ffn.norm2, x), io[5], io[6])
        z = block.ffn.fc2.apply(torch.nn.functional.gelu(block.ffn.fc0.apply(z), approximate="tanh"))
        return x + io[7] * z


class ActionRunner:
    def __init__(self, model, vae, context, mask, case, bench, group, rank=None, comms=None):
        self.model, self.vae, self.bench, self.group = model, vae, bench, group
        self.device = model.device
        self.action_device = torch.device(bench.action_device) if group in (6, 8) else self.device
        self.devices = list(dict.fromkeys([self.device, self.action_device]))
        self.steps = 10 if group in (1, 2) else 1
        self.context, self.mask, self.case = context, mask, case
        self.rank, self.comms = rank, comms
        self.sp = group == 9
        self.tp = group == 10
        self.video_ops = Operators(model, bench, group in FUSED, comms["video"] if self.tp else None)
        self.action_ops = Operators(model, bench, group in FUSED, comms["action"] if self.tp else None)
        self.image = torch.empty_like(case[0], device=self.device, dtype=torch.bfloat16)
        self.proprio = torch.empty_like(case[1], device=self.device, dtype=torch.bfloat16)
        self.noise = torch.empty((1, int(bench.action_horizon), 7), device=self.action_device, dtype=torch.bfloat16)
        self.video_stream = torch.cuda.Stream(device=self.device)
        self.action_stream = torch.cuda.Stream(device=self.action_device)
        self.ready = [torch.cuda.Event() for _ in range(30)]
        self.run_gpu = self.pipeline if group >= 5 else self.sequential
        if self.sp:
            from lightx2v.common.ops.attn.ulysses_prepost import TritonUlyssesPrePost
            self.layout = TritonUlyssesPrePost()
        self.stage_inputs()
        sync(self.devices)

    def stage_inputs(self):
        image, proprio, seed = self.case
        noise = torch.randn(self.noise.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
        self.image.copy_(image)
        self.proprio.copy_(proprio)
        self.noise.copy_(noise)

    def __call__(self):
        self.stage_inputs()
        return self.run_gpu().to(device="cpu", dtype=torch.float32)

    def partition(self, prepared):
        n = prepared.tokens.shape[0]
        if n % 2:
            raise ValueError("Sequence length must be even")
        fields = vars(prepared).copy()
        for key in ("tokens", "freqs", "context_mask"):
            fields[key] = fields[key].chunk(2, dim=0)[self.rank].contiguous()
        if fields["t_mod"].ndim == 3 and fields["t_mod"].shape[0] == n:
            fields["t_mod"] = fields["t_mod"].chunk(2, dim=0)[self.rank].contiguous()
        return SimpleNamespace(**fields)

    def prepare_video(self):
        latents = self.vae.encode(self.image.unsqueeze(2)).unsqueeze(0).to(torch.bfloat16)
        context, mask = self.model._append_robot_state_to_context(self.context, self.mask, self.proprio)
        video = self.model.pre_infer.infer_video(self.model.pre_weight, latents, context, mask)
        n = video.tokens.shape[0]
        video.attention = torch.ones((n, n), device=self.device, dtype=torch.bool)
        attention = self.model.transformer_infer.build_mot_attention_mask(n, self.noise.shape[1], video.tokens_per_frame, self.device)[n:]
        return (self.partition(video) if self.sp else video), context, mask, attention

    def schedule(self):
        u = torch.linspace(1, 0, self.steps + 1, device=self.action_device, dtype=torch.float32)
        sigmas = self.model.scheduler.phi(u, float(self.bench.sigma_shift))
        return (sigmas[:-1] * 1000).to(self.noise.dtype), (sigmas[1:] - sigmas[:-1]).to(self.noise.dtype)

    def prepare_action(self, x, timestep, context, mask, attention):
        prepared = self.model.pre_infer.infer_action(self.model.pre_weight, x, timestep, context, mask)
        prepared.attention = attention
        return self.partition(prepared) if self.sp else prepared

    def block(self, expert, index):
        return getattr(self.model.transformer_weights, expert).blocks[index]

    def project(self, expert, index, x, prepared):
        ops = self.video_ops if expert == "video" else self.action_ops
        return ops.project(self.block(expert, index), x, prepared)

    def exchange(self, tensors, group):
        result = []
        for payload, scale in tensors:
            if scale is not None:
                raise ValueError("Quantized communication is disabled")
            out = torch.empty_like(payload)
            dist.all_to_all_single(out, payload, group=group)
            result.append((out, None))
        return tuple(result)

    def sequence_to_heads(self, io, expert):
        q, k, v = io[:3]
        # The original benchmark packs QKV even in its native torch SP2 path.
        wire = self.layout.pack_qkv(q, k, v, 2, qkv_fusion=True)
        wire = self.exchange(wire, self.comms[expert])
        q, k, v = self.layout.unpack_qkv(wire, q, k, v, None, None, None, self.rank, 2, False)
        return q, k, v, *io[3:]

    def heads_to_sequence(self, mixed, prepared, expert):
        wire = self.layout.pack_attn(mixed, prepared.tokens.shape[0], 2, 12, 128)
        wire = self.exchange(wire, self.comms[expert])
        return self.layout.unpack_attn(wire, mixed.dtype, 128)

    def finish(self, expert, index, io, prepared, kv=None):
        k, v = io[1:3] if kv is None else (torch.cat((kv[0], io[1]), dim=0), torch.cat((kv[1], io[2]), dim=0))
        block = self.block(expert, index)
        mixed = block.self_attn.attn.apply(io[0], k, v, attn_mask=prepared.attention)
        if self.sp:
            mixed = self.heads_to_sequence(mixed, prepared, expert)
        return (self.video_ops if expert == "video" else self.action_ops).post(block, io, mixed, prepared)

    def post_step(self, hidden, delta, latents):
        prediction = self.model.transformer_weights.action_head.apply(hidden).unsqueeze(0)
        if self.sp:
            local = latents.chunk(2, dim=1)[self.rank] + prediction * delta
            parts = [torch.empty_like(local) for _ in range(2)]
            dist.all_gather(parts, local.contiguous(), group=self.comms["action"])
            return torch.cat(parts, dim=1)
        return latents + prediction * delta

    def sequential(self):
        video, context, mask, attention = self.prepare_video()
        # Baseline calls the repository's original serial transformer methods.
        cache = self.model.transformer_infer.prefill_video_cache(self.model.transformer_weights, video)
        ts, ds = self.schedule()
        x = self.noise
        for timestep, delta in zip(ts, ds):
            action = self.prepare_action(x, timestep, context, mask, attention)
            pred = self.model.transformer_infer.action_with_video_cache(
                self.model.transformer_weights, action, cache, video.tokens.shape[0],
                torch.cat((video.attention.new_zeros(video.tokens.shape[0], attention.shape[1]), attention)))
            x = x + pred.unsqueeze(0) * delta
        return x[0].float()

    def pipeline(self):
        video, context, mask, attention = self.prepare_video()
        ts, ds = self.schedule()
        action = self.prepare_action(self.noise, ts[0], context, mask, attention)
        caller = torch.cuda.current_stream(self.device)
        self.video_stream.wait_stream(caller)
        self.action_stream.wait_stream(caller)
        vx, ax = video.tokens, action.tokens
        for index in range(30):
            with torch.cuda.stream(self.video_stream):
                vio = self.project("video", index, vx, video)
                if self.sp:
                    vio = self.sequence_to_heads(vio, "video")
                kv = vio[1:3]
                self.ready[index].record(self.video_stream)
                vx = self.finish("video", index, vio, video)
            with torch.cuda.stream(self.action_stream):
                aio = self.project("action", index, ax, action)
                if self.sp:
                    aio = self.sequence_to_heads(aio, "action")
                self.action_stream.wait_event(self.ready[index])
                for tensor in kv:
                    tensor.record_stream(self.action_stream)
                ax = self.finish("action", index, aio, action, kv)
        caller.wait_stream(self.video_stream)
        caller.wait_stream(self.action_stream)
        ax.record_stream(caller)
        return self.post_step(ax, ds[0], self.noise)[0].float()

    def capture(self):
        self.captured = CapturedCall(self.run_gpu, self.device, self.bench.graph_warmup)
        self.run_gpu = self.captured

    def capture_two_gpu(self):
        warmup = self.bench.graph_warmup
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_ready = [torch.cuda.Event() for _ in self.ready]
        self.context_ready = torch.cuda.Event()
        self.pre_video = CapturedCall(self.prepare_video, self.device, warmup)
        video, context, mask, attention = self.pre_video.output
        self.remote = [x.to(self.action_device) for x in (context, mask, attention)]

        def prepare():
            ts, ds = self.schedule()
            action = self.prepare_action(self.noise, ts[0], *self.remote)
            return action, ds[0], self.project("action", 0, action.tokens, action)

        sync(self.devices)
        self.pre_action = CapturedCall(prepare, self.action_device, warmup)
        action, delta, aio = self.pre_action.output
        self.received = []
        self.layers = []
        vio = None
        for index in range(30):
            def video_phase(index=index, previous=vio):
                x = video.tokens if previous is None else self.finish("video", index - 1, previous, video)
                return self.project("video", index, x, video)
            vg = CapturedCall(video_phase, self.device, warmup)
            vio = vg.output
            self.received.append(tuple(t.to(self.action_device) for t in vio[1:3]))
            sync(self.devices)

            def action_phase(index=index, current=aio):
                x = self.finish("action", index, current, action, self.received[index])
                if index < 29:
                    return self.project("action", index + 1, x, action)
                return self.post_step(x, delta, self.noise)[0].float()
            ag = CapturedCall(action_phase, self.action_device, warmup)
            aio = ag.output
            self.layers.append((vg, ag))
        self.video_tail = CapturedCall(lambda: self.finish("video", 29, vio, video), self.device, warmup)
        self.run_gpu = self.two_gpu_pipeline

    def two_gpu_pipeline(self):
        caller = torch.cuda.current_stream(self.device)
        action_caller = torch.cuda.current_stream(self.action_device)
        _, context, mask, attention = self.pre_video()
        self.video_stream.wait_stream(caller)
        self.copy_stream.wait_stream(caller)
        self.action_stream.wait_stream(action_caller)
        with torch.cuda.stream(self.action_stream), torch.cuda.stream(self.copy_stream):
            for dest, source in zip(self.remote, (context, mask, attention)):
                dest.copy_(source, non_blocking=True)
            self.context_ready.record(self.copy_stream)
        with torch.cuda.stream(self.action_stream):
            self.action_stream.wait_event(self.context_ready)
            self.pre_action()
        for index, (vg, ag) in enumerate(self.layers):
            with torch.cuda.stream(self.video_stream):
                vio = vg()
                self.ready[index].record(self.video_stream)
                if index == 29:
                    self.video_tail()
            with torch.cuda.stream(self.action_stream), torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(self.ready[index])
                for dest, source in zip(self.received[index], vio[1:3]):
                    dest.copy_(source, non_blocking=True)
                self.copy_ready[index].record(self.copy_stream)
            with torch.cuda.stream(self.action_stream):
                self.action_stream.wait_event(self.copy_ready[index])
                output = ag()
        caller.wait_stream(self.video_stream)
        caller.wait_stream(self.action_stream)
        action_caller.wait_stream(self.action_stream)
        return output


def shard_tp(model, rank, comms):
    from lightx2v.common.ops.mm.mm_weight import MMWeightTP
    for name in ("video", "action"):
        for block in getattr(model.transformer_weights, name).blocks:
            modules = []
            for attention in (block.self_attn, block.cross_attn):
                modules.extend((attention, key, "col") for key in ("q", "k", "v"))
                modules.append((attention, "o", "row"))
                for key in ("norm_q", "norm_k"):
                    norm = getattr(attention, key)
                    norm.weight = norm.weight.chunk(2)[rank].contiguous()
            modules.extend(((block.ffn, "fc0", "col"), (block.ffn, "fc2", "row")))
            for parent, key, split in modules:
                old = getattr(parent, key)
                # Native MMWeight stores transposed [in,out] matrices.
                weight = old.weight.chunk(2, dim=1 if split == "col" else 0)[rank].contiguous()
                bias = old.bias.chunk(2)[rank].contiguous() if split == "col" else old.bias
                wrapper = MMWeightTP(old.weight_name, old.bias_name, tp_group=comms[name],
                                     tp_rank=rank, tp_size=2, split_dim=split)
                wrapper._mm.weight = weight
                wrapper._mm.bias = bias if split == "col" else None
                wrapper._row_split_bias = bias if split == "row" else None
                parent.add_module(key, wrapper)


def validate(runner, cases, references, atol, rtol):
    original, outputs, errors = runner.case, [], []
    try:
        for case, reference in zip(cases, references):
            runner.case = case
            actual = runner()
            sync(runner.devices)
            error = float((actual - reference).abs().max())
            passed = bool(torch.isfinite(actual).all() and torch.allclose(actual, reference, atol=atol, rtol=rtol))
            if runner.rank is not None:
                status = torch.tensor([not passed, error], device=runner.device, dtype=torch.float64)
                dist.all_reduce(status, op=dist.ReduceOp.MAX)
                passed, error = not bool(status[0].item()), status[1].item()
            errors.append(error)
            outputs.append(actual.clone())
            if not passed:
                raise AssertionError(f"Group {runner.group}: max abs error {error:.8f}, tolerance atol={atol}, rtol={rtol}")
    finally:
        runner.case = original
    return errors, outputs


def benchmark(runner):
    b, samples = runner.bench, []
    for device in runner.devices:
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(b.warmup + b.iters):
        if runner.rank is not None:
            dist.barrier()
        sync(runner.devices)
        start = time.perf_counter()
        output = runner()
        sync(runner.devices)
        elapsed = (time.perf_counter() - start) * 1000
        if runner.rank is not None:
            value = torch.tensor(elapsed, device=runner.device, dtype=torch.float64)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            elapsed = value.item()
        if index >= b.warmup:
            samples.append(elapsed)
        if runner.rank in (None, 0) and (index + 1 == b.warmup or (index + 1 - b.warmup) % 20 == 0):
            log.info("Group %s: %d/%d samples, %.3f ms", runner.group, len(samples), b.iters, elapsed)
    return {"samples_ms": samples, "action_shape": list(output.shape),
            "summary": {"mean_ms": statistics.fmean(samples), "p50_ms": percentile(samples, 50),
                        "p90_ms": percentile(samples, 90), "min_ms": min(samples), "max_ms": max(samples)},
            "peak_allocated_gib": {str(d): torch.cuda.max_memory_allocated(d) / 1024**3 for d in runner.devices}}


def run_groups(cfg, groups, model, vae, cases, context, mask, rank=None, comms=None):
    b, results = cfg.BENCH, []
    references = {}
    if b.verify:
        for steps in sorted({10 if i in (1, 2) else 1 for i in groups}):
            ref = ActionRunner(model, vae, context, mask, cases[0], b, 1 if steps == 10 else 3)
            references[steps] = []
            for case in cases:
                ref.case = case
                references[steps].append(ref())
            if torch.equal(references[steps][0], references[steps][1]):
                raise RuntimeError("Changed-observation validation did not change the action")
            del ref
    # TP changes the native weight containers; run it last and shard once.
    order = [i for i in groups if i != 10] + [i for i in groups if i == 10]
    tp_ready = False
    for group in order:
        log.info("Setting up group %d: %s", group, EXECUTIONS[group])
        if group == 10 and not tp_ready:
            shard_tp(model, rank, comms)
            tp_ready = True
        action_device = torch.device(b.action_device) if group in (6, 8) else model.device
        if rank is None:
            move_weights(model.pre_weight.action, action_device)
            move_weights(model.transformer_weights.action, action_device)
            move_weights(model.transformer_weights.action_head, action_device)
            model.pre_infer.action_freqs = model.pre_infer.action_freqs.to(action_device)
        runner = ActionRunner(model, vae, context, mask, cases[0], b, group, rank, comms)
        result = {"group": group, "execution": EXECUTIONS[group], "action_steps": runner.steps,
                  "cuda_graph": group not in (1, 3), "fusion": group in FUSED,
                  "gpu_count": 2 if group in (6, 8) or group >= 9 else 1,
                  "operator_backend": "lightx2v_kernels" if group in FUSED else "lightx2v_torch",
                  "status": "pending"}
        result["kernels"] = ([
            "lightx2v.fused_qk_rms_norm" if not runner.tp else "packed_tp_fp32_rms_reduction",
            "lightx2v.norm_infer" if b.kernel_layer_norm else "torch.layer_norm",
            "lightx2v.fuse_scale_shift_kernel" if b.kernel_affine else "torch.affine",
            "lightx2v.apply_rotary_embedding" if b.kernel_rope else "torch_complex_rope_fp64",
        ] if group in FUSED else ["lightx2v_native_torch_operators"])
        if runner.sp:
            result["kernels"].append("lightx2v.TritonUlyssesPrePost")
        result["collectives_per_rank_per_request"] = {9: 121, 10: 300}.get(group, 0)
        try:
            start = time.perf_counter()
            eager_outputs = None
            if b.verify and group not in (6, 8):
                result["eager_validation_max_abs_errors"], eager_outputs = validate(
                    runner, cases, references[runner.steps], float(b.atol), float(b.rtol))
            if group in (6, 8):
                runner.capture_two_gpu()
            elif group not in (1, 3):
                runner.capture()
            result["setup_seconds"] = time.perf_counter() - start
            if b.verify:
                result["validation_max_abs_errors"], _ = validate(
                    runner, cases, references[runner.steps], float(b.atol), float(b.rtol))
                if eager_outputs is not None and group not in (1, 3):
                    result["graph_eager_max_abs_errors"], _ = validate(runner, cases, eager_outputs, 0, 0)
            result.update(benchmark(runner), status="passed" if b.verify else "unverified")
        except AssertionError as error:
            result.update(status="validation_failed", error=str(error))
            log.error("%s", error)
        results.append(result)
        sync(runner.devices)
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    return sorted(results, key=lambda item: item["group"])


def worker(rank, raw_cfg, rendezvous, output):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = OmegaConf.create(raw_cfg)
    device = torch.device(cfg.BENCH.device if rank == 0 else cfg.BENCH.action_device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(minutes=10), device_id=device)
    try:
        comms = {name: dist.new_group([0, 1], backend="nccl") for name in ("video", "action")}
        model, vae = load_model(cfg, device)
        cases, context, mask, encoder = inputs(cfg, device)
        results = run_groups(cfg, [i for i in cfg.BENCH.groups if i >= 9], model, vae,
                             cases, context, mask, rank, comms)
        if rank == 0:
            Path(output).write_text(json.dumps(results))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def write_report(report, cfg):
    if cfg.BENCH.output_json:
        path = Path(cfg.BENCH.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


@torch.no_grad()
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = config()
    b = cfg.BENCH
    device = torch.device(b.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("BENCH.device must be cuda:N")
    torch.cuda.set_device(device)
    dual = any(i in (6, 8) or i >= 9 for i in b.groups)
    other = torch.device(b.action_device)
    if dual and (other.type != "cuda" or other.index is None or other == device or other.index >= torch.cuda.device_count()):
        raise ValueError("BENCH.action_device must be a different available CUDA device")
    report = {"implementation": "LightX2V FastWAMNativeModel", "torch": torch.__version__,
              "python": sys.executable, "checkpoint": cfg.ckpt, "dtype": "torch.bfloat16",
              "gpu_names": {str(d): torch.cuda.get_device_name(d) for d in ([device, other] if dual else [device])},
              "peer_access": torch.cuda.can_device_access_peer(device, other) if dual else None,
              "execution_mode": "no_grad", "compile_backend": "explicit_cuda_graph",
              "latency": "synchronized_end_to_end_wall_ms", "distributed_latency": "max_rank_wall_ms",
              "context_source": "synthetic" if b.use_random_context else "lightx2v_encode_prompt_once",
              "text_encoder_retained": not b.use_random_context,
              "image_shape": [1, 3, b.height, b.width], "context_shape": [128, 4096],
              "warmup": b.warmup, "iterations": b.iters, "config": OmegaConf.to_container(cfg, resolve=True),
              "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
              "excluded": ["text_encoding", "model_load", "graph_setup", "validation", "timing_barrier_and_reduction"],
              "baseline_difference": "eager uses LightX2V serial operators, not FastWAM model.infer_action per-call module traversal",
              "results": []}
    local_groups = [i for i in b.groups if i <= 8]
    if local_groups:
        model, vae = load_model(cfg, device)
        cases, context, mask, encoder = inputs(cfg, device)
        report["results"] = run_groups(cfg, local_groups, model, vae, cases, context, mask)
        write_report(report, cfg)
        del model, vae, cases, context, mask, encoder
        gc.collect()
        sync([device, other] if dual else [device])
        for d in ([device, other] if dual else [device]):
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
    if any(i >= 9 for i in b.groups):
        with tempfile.TemporaryDirectory(prefix="lightx2v_fastwam_") as temp:
            output = str(Path(temp) / "parallel.json")
            mp.spawn(worker, args=(OmegaConf.to_container(cfg, resolve=True),
                                   "file://" + str(Path(temp) / "rendezvous"), output), nprocs=2, join=True)
            report["results"].extend(json.loads(Path(output).read_text()))
    report["results"].sort(key=lambda item: item["group"])
    write_report(report, cfg)
    baseline = next((r["summary"]["mean_ms"] for r in report["results"] if r["group"] == 1 and "summary" in r), None)
    print("\n| Group | Action Steps | Compile (incl. VAE) | GPU Count | Execution | Fusion | Mean Latency | Speedup |")
    print("|---|---|---|---|---|---|---|---|")
    for r in report["results"]:
        mean = r.get("summary", {}).get("mean_ms")
        latency = f"{mean:.3f} ms" if mean is not None else r["status"]
        speedup = f"{baseline / mean:.2f}x" if baseline and mean else "-"
        print(f"| {r['group']} | {r['action_steps']} | {'CUDA Graph' if r['cuda_graph'] else 'None'} | "
              f"{r['gpu_count']} | {r['execution']} | {'Yes' if r['fusion'] else 'No'} | {latency} | {speedup} |")
    print(f"Saved: {b.output_json}")
    if any(r["status"] == "validation_failed" for r in report["results"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
