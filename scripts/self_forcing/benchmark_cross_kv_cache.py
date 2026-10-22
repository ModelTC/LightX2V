"""Measure ordinary cross-attention KV reuse with synthetic tensors and real X2V operators.

One measured request contains several chunks, denoising steps and an optional
clean rerun. Results describe one cross-attention layer, not complete DiT latency.
"""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from lightx2v.common.kvcache.condition import ConditionKVCache
from lightx2v.common.ops.attn.flash_attn import FlashAttn3Weight
from lightx2v.common.ops.mm.mm_weight import MMWeight
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightSgl
from lightx2v.models.networks.wan.infer.autoregressive import WanAutoregressiveTransformerInfer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="wan-sf-1.3b-480p")
    parser.add_argument("--source", default="Wan2.1-T2V-1.3B config; synthetic 3 latent frames at 480x832")
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--context-dim", type=int)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--query-tokens", type=int, default=4680)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--model-layers", type=int, default=30)
    parser.add_argument("--kind", choices=("text", "image"), default="text")
    parser.add_argument("--branches", type=int, choices=(1, 2), default=1)
    parser.add_argument("--condition-variants", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=12)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--clean-rerun", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.hidden_dim != args.heads * args.head_dim:
        parser.error("hidden-dim must equal heads * head-dim")
    if args.condition_variants not in (1, args.steps + int(args.clean_rerun)):
        parser.error("condition-variants must be 1 or the number of forwards per chunk")
    args.context_dim = args.hidden_dim if args.context_dim is None else args.context_dim
    return args


class CountingLinear:
    def __init__(self, name, input_dim, output_dim, generator):
        self.op = MMWeight(name + ".weight", name + ".bias")
        self.op.weight = (torch.randn(output_dim, input_dim, device="cuda", dtype=torch.bfloat16, generator=generator) / math.sqrt(input_dim)).t()
        self.op.bias = torch.randn(output_dim, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.01
        self.calls = 0

    def apply(self, x):
        self.calls += 1
        return self.op.apply(x)


class SyntheticCrossAttention:
    def __init__(self, args):
        self.args = args
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        self.key = CountingLinear("cross_attn.k", args.context_dim, args.hidden_dim, generator)
        self.value = CountingLinear("cross_attn.v", args.context_dim, args.hidden_dim, generator)
        self.query = CountingLinear("cross_attn.q", args.hidden_dim, args.hidden_dim, generator)
        self.output = CountingLinear("cross_attn.o", args.hidden_dim, args.hidden_dim, generator)
        self.key_norm = RMSWeightSgl("cross_attn.norm_k.weight")
        self.key_norm.weight = torch.ones(args.hidden_dim, device="cuda", dtype=torch.bfloat16)
        self.query_norm = RMSWeightSgl("cross_attn.norm_q.weight")
        self.query_norm.weight = torch.ones_like(self.key_norm.weight)
        self.phase = SimpleNamespace(
            cross_attn_k=self.key,
            cross_attn_v=self.value,
            cross_attn_norm_k=self.key_norm,
            cross_attn_k_img=self.key,
            cross_attn_v_img=self.value,
            cross_attn_norm_k_img=self.key_norm,
        )
        self.attention = FlashAttn3Weight()
        self.contexts = [torch.randn(args.context_tokens, args.context_dim, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(args.branches)]
        self.variant_contexts = [[context * (1 + 0.05 * variant) + 0.01 * variant for variant in range(args.condition_variants)] for context in self.contexts]
        self.queries = [torch.randn(args.query_tokens, args.hidden_dim, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(args.steps + int(args.clean_rerun))]
        config = {
            "task": "t2v",
            "seq_parallel": False,
            "num_layers": 1,
            "num_heads": args.heads,
            "dim": args.hidden_dim,
            "self_attn_1_type": "flash_attn3",
            "cross_attn_1_type": "flash_attn3",
            "mxfp8_fuse_enable": False,
        }
        self.infer = WanAutoregressiveTransformerInfer(config)
        self.infer.block_idx = 0
        self.infer.scheduler = SimpleNamespace(infer_condition=True)
        self.variant_cache = ConditionKVCache()
        self.legacy_kv = {}
        self.retained_kv = {}

    def project(self, context):
        return self.infer.project_cross_attn_kv(self.phase, context, kind=self.args.kind)

    def reset_request(self):
        self.infer.cross_attn_kv_cache.reset()
        self.variant_cache.reset()
        self.legacy_kv.clear()
        self.retained_kv.clear()
        for linear in (self.key, self.value, self.query, self.output):
            linear.calls = 0

    def get_kv(self, mode, chunk, branch, variant):
        context = self.variant_contexts[branch][variant]
        if mode == "uncached":
            return self.project(context)
        key = branch, variant
        if mode == "first_chunk":
            if chunk == 0:
                self.legacy_kv[key] = self.project(context)
            return self.legacy_kv[key]
        if self.args.condition_variants == 1:
            self.infer.scheduler.infer_condition = bool(branch)
            self.infer.set_cross_attn_context(self.contexts[branch], branch=bool(branch), kind=self.args.kind)
            kv = self.infer.get_cross_attn_kv(self.phase, context, kind=self.args.kind)
        else:
            # Prompt AdaLN varies with sigma, whose variants repeat across chunks.
            name = branch, self.args.kind
            self.variant_cache.set_context(name, self.contexts[branch])
            kv = self.variant_cache.get(name, 0, variant=variant)
            if kv is None:
                kv = self.variant_cache.store(name, 0, self.project(context), variant=variant)
        self.retained_kv[key] = kv
        return kv

    def forward(self, mode, chunk, branch, step, workload):
        variant = 0 if self.args.condition_variants == 1 else step
        key, value = self.get_kv(mode, chunk, branch, variant)
        if workload == "projection":
            return key
        query = self.query_norm.apply(self.query.apply(self.queries[step])).view(-1, self.args.heads, self.args.head_dim)
        attended = self.attention.apply(q=query, k=key, v=value)
        return self.output.apply(attended)

    def run_request(self, mode, workload):
        self.reset_request()
        for chunk in range(self.args.chunks):
            for step in range(len(self.queries)):
                for branch in range(self.args.branches):
                    output = self.forward(mode, chunk, branch, step, workload)
        return output

    def cache_bytes(self, mode):
        kv = self.retained_kv if mode == "cached" else self.legacy_kv
        return sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in kv.values())


def check_correctness(case):
    expected = {}
    case.reset_request()
    for step in range(len(case.queries)):
        for branch in range(case.args.branches):
            expected[branch, step] = case.forward("uncached", 0, branch, step, "attention").clone()
    result = {}
    for mode in ("first_chunk", "cached"):
        case.reset_request()
        for chunk in range(case.args.chunks):
            for step in range(len(case.queries)):
                for branch in range(case.args.branches):
                    output = case.forward(mode, chunk, branch, step, "attention")
                    reference = expected[branch, step]
                    if not torch.equal(output, reference):
                        max_abs = (output.float() - reference.float()).abs().max().item()
                        raise AssertionError(f"{mode} cross-attention differs from uncached: max_abs={max_abs}")
        result[mode] = {"max_abs": 0.0, "bitwise_equal": True, "kv_projection_pairs": case.key.calls, "cache_bytes": case.cache_bytes(mode)}
    return result


def measure(case, mode, workload):
    args = case.args
    for _ in range(args.warmup_requests):
        case.run_request(mode, workload)
    torch.cuda.synchronize()
    repeats = []
    for repeat in range(args.repeats):
        events = []
        wall_start = time.perf_counter()
        for _ in range(args.iterations):
            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before.record()
            case.run_request(mode, workload)
            after.record()
            events.append((before, after))
        torch.cuda.synchronize()
        gpu_ms = [before.elapsed_time(after) for before, after in events]
        repeats.append({"repeat": repeat, "request_gpu_ms": gpu_ms, "request_wall_ms_mean": (time.perf_counter() - wall_start) * 1000 / args.iterations})
    per_request = [value for repeat in repeats for value in repeat["request_gpu_ms"]]
    return {
        "request_gpu_ms_median": statistics.median(per_request),
        "request_gpu_ms_mean": statistics.mean(per_request),
        "request_gpu_ms_stdev": statistics.stdev(per_request),
        "request_wall_ms_mean": statistics.mean(repeat["request_wall_ms_mean"] for repeat in repeats),
        "kv_projection_pairs_per_request": case.key.calls,
        "q_projections_per_request": case.query.calls,
        "output_projections_per_request": case.output.calls,
        "cache_bytes_one_layer": case.cache_bytes(mode),
        "cache_bytes_all_layers_estimate": case.cache_bytes(mode) * args.model_layers,
        "repeats": repeats,
    }


@torch.no_grad()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = SyntheticCrossAttention(args)
    correctness = check_correctness(case)
    measurements = {}
    for workload in ("projection", "attention"):
        measurements[workload] = {}
        for mode in ("uncached", "first_chunk", "cached"):
            result = measure(case, mode, workload)
            measurements[workload][mode] = result
            print(f"{args.label} {workload} {mode}: {result['request_gpu_ms_median']:.4f} ms/request", flush=True)
        cached = measurements[workload]["cached"]["request_gpu_ms_median"]
        measurements[workload]["cached_reduction_vs_uncached_pct"] = (1 - cached / measurements[workload]["uncached"]["request_gpu_ms_median"]) * 100
        measurements[workload]["cached_reduction_vs_first_chunk_pct"] = (1 - cached / measurements[workload]["first_chunk"]["request_gpu_ms_median"]) * 100
    report = {
        "scope": "Synthetic inputs and random BF16 weights; one real X2V cross-attention layer, not full-model inference",
        "operators": ["MMWeight", "RMSWeightSgl", "FlashAttn3Weight", "ConditionKVCache", "WanAutoregressiveTransformerInfer.project_cross_attn_kv"],
        "cache_helper": "WanAutoregressiveTransformerInfer.get_cross_attn_kv" if args.condition_variants == 1 else "ConditionKVCache.get/store with explicit repeated variants",
        "cached_policy": "One projection pair per source branch, layer and repeated condition variant, including the first use in request timing",
        "first_chunk_policy": "Policy reference: project on every forward in chunk 0, then retain KV; not a replay of an old complete model",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": "bfloat16",
        "correctness": correctness,
        "measurements": measurements,
    }
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
