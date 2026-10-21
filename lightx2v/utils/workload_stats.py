"""Collect compute facts beside real operator calls using base formulas and actual backend routes.

Before activation, only _active is checked: no reductions, copies, or exports. One explicit session owns
the thread, per-class sample caps, and files; it neither inventories model layers nor recovers compiled calls.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

# Report-name aliases for legacy dynamic SLA entries, not a support list or switch.
# For every General mask algorithm, the operator consuming the actual mask/ranges records the facts.
_DYNAMIC_SLA_BACKEND_NAMES = {f"dynamic_sparse_{implementation}": f"sla.{implementation}.dynamic" for implementation in ("sage2", "triton", "triton_ar", "magi", "sage3", "fa4")}

SUPPORTED_FAMILIES = ("gemm", "attention", "convolution", "moe")

# Allow one explicit session per process; clear it on exit/error and keep statistics off by default.
_active = None
_session_lock = threading.Lock()


def enabled(family):
    """Use one family selector on the explicit session, without separate core/sparse collection channels."""
    return _active is not None and family in _active.families


def _dtype(tensor):
    """Record tensor storage dtype only; uint8 does not identify the NVFP4/MXFP4 quantization scheme."""
    value = str(tensor.dtype).removeprefix("torch.")
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float8_e4m3fn": "fp8_e4m3", "float8_e5m2": "fp8_e5m2"}.get(value, value)


class WorkloadStats:
    """Run a bounded diagnostic: record facts after real calls, then summarize observed calls and gaps on exit.

    observed counts visible calls, counts tracks saved samples, and gaps records coverage reasons.
    Save at most max_calls samples per backend; lightweight counts continue after the cap.
    Callers supply provenance; the tool cannot infer model or request identity.
    Use a separate eager diagnostic; compile and CUDA graph replay are unsupported.
    """

    def __init__(self, output, *, families=None, max_calls=1, moe_distribution=False, provenance=None, on_sparse_inputs=None, on_moe_inputs=None):
        families = SUPPORTED_FAMILIES if families is None else tuple(families)
        if type(max_calls) is not int or max_calls < 1 or not families or set(families) - set(SUPPORTED_FAMILIES):
            raise ValueError("positive max_calls and supported families are required")
        self.output = Path(output)
        self.families, self.max_calls = families, max_calls
        self.moe_distribution = moe_distribution
        self.provenance = provenance or {}
        self.counts = {}
        self.observed = {}
        self.gaps = {}
        self.on_sparse_inputs = on_sparse_inputs
        self.on_moe_inputs = on_moe_inputs
        self._core_stack = []
        self._core_number = 0
        self.core_counts = {}
        self.core_observed = {}

    def __enter__(self):
        global _active
        with _session_lock:
            if _active is not None:
                raise RuntimeError("a workload statistics session is already active")
            self.output.mkdir(parents=True, exist_ok=False)
            self._stream = (self.output / "workloads.jsonl").open("x", encoding="utf-8")
            try:
                self._core_stream = (self.output / "core_ops.jsonl").open("x", encoding="utf-8")
            except BaseException:
                self._stream.close()
                raise
            self._owner = threading.get_ident()
            _active = self
            return self

    def __exit__(self, exc_type, exc, tb):
        global _active
        _active = None
        try:
            self._stream.close()
        finally:
            self._core_stream.close()
        summary = {
            "schema_version": 2,
            "kind": "backend_workload_capture",
            "status": "failed" if exc_type else "complete",
            "captured": self.counts,
            "max_calls_per_backend": self.max_calls,
            "families": self.families,
            "moe_distribution": self.moe_distribution,
            "provenance": self.provenance,
            "observed_operators": self.observed,
            "coverage_gaps": self.gaps,
            "core_op_captured": self.core_counts,
            "core_op_observed": self.core_observed,
            "core_op_file": "core_ops.jsonl",
            "core_op_boundary": "Actual local apply/apply_with_lse boundary; composite parents add no FLOPs; overlapping parents/children/leaf facts must not be added; eager only; capped samples are partial.",
            "moe_input_export_callback": self.on_moe_inputs is not None,
            "coverage_boundary": "Common core-op boundaries and final sparse consumers; HF nn.Module paths bypassing these bases, independent platform bases, raw ATen and compiled interiors are not fully inventoried; capped samples are partial.",
        }
        (self.output / "session.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self._write_report()
        return False

    def _write_report(self):
        """Render a Chinese reference page from saved facts, without making collection or lifecycle decisions."""
        rows = [json.loads(line) for line in (self.output / "core_ops.jsonl").read_text().splitlines()]
        lines = [
            "# 核心算子计算量",
            "",
            "这里只统计本次本地调用的计算量，不测量性能。组合边界不另计；其他父算子、子算子和底层 kernel 可能重叠，不能相加。",
            "MM 只计主线性计算；稀疏 attention 使用声明的 shape 合同或实际 mask/ranges，未覆盖区域保留未知。MoE 总量不读专家 ID，分布按需复用生产计数。",
            "并行 attention 查看各实际子调用：head 分组、Ring 每轮和 aux 已反映在其 Q/K/V 中。采样上限可能截断；跨卡总量须收齐各 rank，不直接乘 world_size。",
            "",
            "| 调用 | 父调用 | 算子 / 入口 | 逻辑 FLOPs | 稠密等价 FLOPs | 调度 tile FLOPs | 说明 |",
            "|---:|---:|---|---:|---:|---:|---|",
        ]
        details = []
        for row in rows:
            work = row.get("work", {})
            reason = row.get("unsupported_reason", row.get("note", "已取得声明范围内的计算量"))
            scheduled = work.get("scheduled_tile_flops")
            lines.append(
                f"| {row['call']} | {row['parent_call'] or '—'} | {row['backend'].rsplit('.', 1)[-1]} / {row['entrypoint']} | {work.get('logical_flops', work.get('selected_logical_flops', '—'))} | {work.get('dense_equivalent_flops', '—')} | {scheduled if scheduled is not None else '—'} | {reason} |"
            )
            routing = row.get("routing")
            if routing:
                details += ["", f"调用 {row['call']} 的实际专家路由：{routing['expert_counts']}；活跃专家 {routing['active_experts']}；执行路由 {routing['routes']}。", ""]
            for route in row.get("actual_backend_routes", []):
                work = route["work"]
                selected = work["selected_logical_flops"]
                scheduled = work["scheduled_tile_flops"]
                scheduled_text = f"{scheduled} FLOPs" if scheduled is not None else "未知（内部 tile/padding 未提供）"
                details += ["", f"调用 {row['call']}，{route['observed_backend']}：实际选中有效计算量 {selected} FLOPs；含 padding 的调度 tile 计算量 {scheduled_text}；只计 QK/PV。", ""]
        lines += details
        if not rows:
            lines += ["", "没有采到算子层样本；查看 session.json 的选择项、可见调用与缺口。"]
        (self.output / "core_ops.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write(self, operator, facts):
        if threading.get_ident() != self._owner:
            raise RuntimeError("workload statistics require calls on the session's owning thread")
        self.counts[operator] = self.counts.get(operator, 0) + 1
        row = {"schema_version": 1, "operator": operator, "call": self.counts[operator], "provenance": self.provenance, **facts}
        self._stream.write(json.dumps(row) + "\n")
        self._stream.flush()

    def note_gap(self, operator, reason):
        key = f"{operator}: {reason}"
        self.gaps[key] = self.gaps.get(key, 0) + 1

    def observe_hook(self, operator, schema_name):
        """Count observed calls; stop reading tensors or saving facts once the sample cap is reached."""
        if threading.get_ident() != self._owner:
            raise RuntimeError("workload statistics require calls on the session's owning thread")
        self.observed[schema_name] = self.observed.get(schema_name, 0) + 1
        if self.counts.get(operator, 0) >= self.max_calls:
            self.note_gap(schema_name, "sample limit reached; full shape inventory not observed")
            return False
        return True


def record_uncovered(family, operator, reason):
    """Record a coverage gap for executed calls bypassing common bases, without guessing FLOPs or adding a channel."""
    if not enabled(family):
        return
    session = _active
    if threading.get_ident() != session._owner:
        raise RuntimeError("statistics require calls on the session's owning thread")
    if not any(frame["op"].compute_family == family for frame in session._core_stack):
        session.note_gap(operator, reason)


def capture_core_op(op, apply, parameters, args, kwargs):
    """Execute one real apply; the base supplies formulas and the session supplies caps, files, and parentage."""
    import torch

    session = _active
    if threading.get_ident() != session._owner:
        raise RuntimeError("core op statistics require calls on the session's owning thread")
    if torch.compiler.is_compiling():
        raise RuntimeError("core op statistics require a separate eager diagnostic; disable them during compile")
    if session._core_stack and session._core_stack[-1]["op"] is op:
        return apply(op, *args, **kwargs)
    arguments = parameters.bind(op, *args, **kwargs)
    arguments.apply_defaults()
    values = {k: v for k, v in arguments.arguments.items() if k != "self"}
    backend = type(op).__module__ + "." + type(op).__name__
    session.core_observed[backend] = session.core_observed.get(backend, 0) + 1
    session._core_number += 1
    # Pending parents of the same class reserve a sample slot; the finally pop releases it.
    pending = sum(frame["sample"] and frame["backend"] == backend for frame in session._core_stack)
    sample = session.core_counts.get(backend, 0) + pending < session.max_calls
    frame = {"op": op, "backend": backend, "call": session._core_number, "routes": [], "sample": sample}
    parent = session._core_stack[-1]["call"] if session._core_stack else None
    tensors = [value for value in values.values() if isinstance(value, torch.Tensor)]
    if any(value.is_meta or type(value).__name__ == "FakeTensor" for value in tensors):
        session.note_gap(backend, "fake/meta core op is not actual execution evidence")
        return apply(op, *args, **kwargs)
    if any(value.is_cuda for value in tensors) and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("core op statistics cannot run during CUDA graph capture")
    session._core_stack.append(frame)
    try:
        if not sample:
            session.note_gap(backend, "core op sample limit reached")
            return apply(op, *args, **kwargs)
        moe_inputs = None
        if session.on_moe_inputs is not None and op.compute_family == "moe":
            moe_inputs = op.replay_inputs(values)
        result = apply(op, *args, **kwargs)
        facts = op.compute_stats(values, result)
        if frame["routes"]:
            facts["actual_backend_routes"] = frame["routes"]
            # One actual attention child can supply logical work; keep multiple children separate rather than guessing overlap.
            if op.compute_family == "attention" and len(frame["routes"]) == 1:
                work = frame["routes"][0]["work"]
                facts.setdefault("work", {}).update(selected_logical_flops=work["selected_logical_flops"], scheduled_tile_flops=work["scheduled_tile_flops"])
                if facts["work"].get("dense_equivalent_flops") != work["dense_equivalent_flops"]:
                    facts["note"] = f"稠密等价量按入口shape；有效/调度量按实际backend输入（含padding/裁剪），其稠密等价量为{work['dense_equivalent_flops']} FLOPs；见actual_backend_routes。"
        row = {
            "schema_version": 1,
            "kind": "core_op_compute_stats",
            "backend": backend,
            "operator_family": op.compute_family,
            "call": frame["call"],
            "parent_call": parent,
            "entrypoint": apply.__name__,
            "weight_name": getattr(op, "weight_name", None),
            "provenance": session.provenance,
            "actual_input_dtypes": {name: _dtype(value) for name, value in values.items() if isinstance(value, torch.Tensor)},
            "actual_output_dtype": _dtype(result) if isinstance(result, torch.Tensor) else _dtype(result[0]) if isinstance(result, tuple) and result and isinstance(result[0], torch.Tensor) else None,
            "status": "unsupported" if "unsupported_reason" in facts else "complete",
            **facts,
        }
        if session.on_moe_inputs is not None and op.compute_family == "moe":
            if moe_inputs is None:
                row["input_export_gap"] = "This MoE backend has no supported replay operand layout"
                session.note_gap(backend, row["input_export_gap"])
            else:
                row["input_artifact"] = session.on_moe_inputs(
                    {"inputs": moe_inputs, "activation": op.activation, "backend": backend, "module": type(op).__module__, "call": frame["call"], "output": result}
                )
        if "moe_routing" in frame:
            row["routing"] = frame["moe_routing"]
        elif session.moe_distribution and op.compute_family == "moe":
            row["moe_distribution_gap"] = "This backend/call did not expose production expert counts; total FLOPs do not require them."
        session._core_stream.write(json.dumps(row) + "\n")
        session._core_stream.flush()
        session.core_counts[backend] = session.core_counts.get(backend, 0) + 1
        return result
    finally:
        session._core_stack.pop()


def record_moe_routing(counts):
    """Optionally reuse production expert counts instead of copying/recounting all token expert IDs."""
    session = _active
    if not enabled("moe") or not session.moe_distribution:
        return
    if threading.get_ident() != session._owner:
        raise RuntimeError("MoE routing statistics require the session's owning thread")
    if not session._core_stack or not session._core_stack[-1]["sample"]:
        return
    frame = session._core_stack[-1]
    if frame["op"].compute_family != "moe":
        return
    values = counts.detach().cpu().tolist()
    frame["moe_routing"] = {"source": "production_expert_counts", "expert_counts": values, "active_experts": sum(count > 0 for count in values), "routes": sum(values)}


def record_sparse_attn(
    q, k, v, sparse_map=None, block_q=1, block_k=1, keep_ratio=None, arch=None, *, operator="dynamic_sparse_sage2", layout="BHSD", q_block_sizes=None, k_block_sizes=None, ranges=None, range_repeat=1
):
    """Use final masks or non-overlapping ranges; a zero-argument mask factory runs only when sampled."""
    session = _active
    if not enabled("attention"):
        return
    import torch

    if threading.get_ident() != session._owner:
        raise RuntimeError("sparse statistics require calls on the session's owning thread")
    # Do not read inner regions after the parent cap; standalone consumers may still collect their own samples.
    if session._core_stack and not session._core_stack[-1]["sample"]:
        return
    core = bool(session._core_stack)
    leaf = True
    if leaf and not session.observe_hook(operator, "lightx2v::" + operator):
        leaf = False
    if not leaf and not core:
        return
    from lightx2v.common.ops.attn.template import AttnWeightTemplate

    if torch.compiler.is_compiling():
        raise RuntimeError("sparse statistics require a separate eager diagnostic")
    # FakeTensor can report is_cuda; reject it before touching the CUDA runtime.
    if any(tensor.is_meta or type(tensor).__name__ == "FakeTensor" for tensor in (q, k, v)):
        session.note_gap(operator, "fake/meta sparse input is not actual execution evidence")
        return
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("workload statistics cannot run during CUDA graph capture")
    if ranges is None:
        if callable(sparse_map):
            sparse_map = sparse_map()  # Build the final mask lazily from the actual LUT; do not allocate when disabled/capped.
        facts = AttnWeightTemplate.compute_sparse_stats(q, k, v, sparse_map, block_q, block_k, layout=layout, q_block_sizes=q_block_sizes, k_block_sizes=k_block_sizes)
    else:
        if sparse_map is not None:
            raise ValueError("choose the final mask or final ranges, not both")
        facts = AttnWeightTemplate.compute_sparse_range_stats(q, k, v, *ranges, layout=layout, range_repeat=range_repeat)
    facts.update(observed_backend=_DYNAMIC_SLA_BACKEND_NAMES.get(operator, operator), production_operator=operator, precision={"input_dtype": _dtype(q)})
    if session.on_sparse_inputs:
        session.on_sparse_inputs(
            {
                "operator": operator,
                "q": q,
                "k": k,
                "v": v,
                "mask": sparse_map,
                "q_block": block_q,
                "k_block": block_k,
                "q_block_sizes": q_block_sizes,
                "k_block_sizes": k_block_sizes,
                "layout": layout,
                "ranges": ranges,
                "facts": facts,
            }
        )
    if keep_ratio is not None:
        facts["sparse"] = {"keep_ratio": keep_ratio}
    if arch is not None:
        facts["routing"]["arch"] = arch
    if core:
        session._core_stack[-1]["routes"].append(facts)
    if leaf:
        session.write(operator, facts)
