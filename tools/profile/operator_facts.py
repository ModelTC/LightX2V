"""采集与分析共用的算子合同：识别实际库精度，计算理想字节量。

只处理已核对的布局/类型；缺少或冲突的证据返回未知，不按名称猜执行量。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache

ATEN_GEMM = {"aten::mm", "aten::bmm", "aten::addmm", "aten::_scaled_mm", "aten::_grouped_mm"}
SGL_GEMM = {"sgl_kernel::fp8_scaled_mm"}
FP8_F16_GEMM = {
    "lightx2v_kernel::cutlass_scaled_fp8_mm_f16_accum_sm90",
    "lightx2v_kernel::cutlass_scaled_fp8_mm_f16_accum_sm120",
    "lightx2v_kernel::cutlass_scaled_fp8_mm_f16_accum_with_config_sm120",
}
FP8_F32_GEMM = {"lightx2v_kernel::cutlass_scaled_fp8_mm_f32_accum_sm90"}
PACKED_FP4_GEMM = {"lightx2v_kernel::cutlass_scaled_nvfp4_mm_sm120", "lightx2v_kernel::cutlass_scaled_mxfp4_mm_sm120"}
CUSTOM_FP8_GEMM = {"lightx2v_kernel::cutlass_scaled_mxfp8_mm_sm120"}
GEMM_OPERATORS = ATEN_GEMM | SGL_GEMM | FP8_F16_GEMM | FP8_F32_GEMM | PACKED_FP4_GEMM | CUSTOM_FP8_GEMM


def normalize_dtype(value):
    """统一core短名和backend原始torch dtype字符串，不推断计算/累加类型。"""
    text = str(value).removeprefix("torch.") if value is not None else None
    if text and text.startswith("float8_"):
        return "fp8"
    return {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}.get(text, text)


@lru_cache(maxsize=512)
def demangle(name):
    if not name.startswith("_Z") or not shutil.which("c++filt"):
        return name
    try:
        return subprocess.run(["c++filt", "-r", name], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return name


def precision(name, kernel_names=()):
    """实际native名与kernel类型的源码合同；不由外层输入猜量化/累加类型。"""
    if name in FP8_F16_GEMM:
        return {"input_dtype": "fp8", "accum_dtype": "fp16", "evidence": ["lightx2v_kernel/nvidia/csrc/gemm/fp8_f16_accum_scaled_mm.cu: ElementAccumulator=cutlass::half_t"]}
    if name in FP8_F32_GEMM:
        return {
            "input_dtype": "fp8",
            "accum_dtype": "fp32",
            "evidence": [
                "lightx2v_kernel/nvidia/csrc/gemm/fp8_f16_accum_scaled_mm.cu: run_with_dtype<float>, ElementAccumulator=float",
                "FP8FastAccum schedule; fp32 accumulator type does not promise strict fp32 rounding throughout",
            ],
        }
    if name in PACKED_FP4_GEMM:
        return {"input_dtype": "fp4", "accum_dtype": "fp32", "evidence": ["lightx2v_kernel packed FP4 contract: two values per byte; ElementAccumulator=float"]}
    if name in CUSTOM_FP8_GEMM:
        return {"input_dtype": "fp8", "accum_dtype": "fp32", "evidence": ["lightx2v_kernel/nvidia/csrc/gemm/mxfp8_scaled_mm_kernels_sm120.cu: ElementAccumulator=float"]}
    inferred = set()
    outputs = set()
    evidence = []
    for kernel in kernel_names:
        kernel = demangle(kernel)
        # 首个模板参数是乘法输入，第二个是输出，不能把 half 输出当累加。
        # 此已核对的 SM90 类型将浮点 ElementAcc 固定为 float。
        if re.search(r"cutlass_3x_gemm_sm90_fp8<cutlass::float_e4m3_t\s*,", kernel):
            inferred.add(("fp8", "fp32"))
            evidence.append("SGL kernel 0.4.4 fp8_gemm_sm90_dispatch.cuh: ElementAB=float_e4m3_t, ElementAcc=float; output type is separate")
            if "FP8FastAccum" in kernel:
                evidence.append("FP8FastAccum schedule; fp32 accumulator type does not promise strict fp32 rounding throughout")
            output = re.search(r"cutlass_3x_gemm_sm90_fp8<cutlass::float_e4m3_t\s*,\s*(cutlass::half_t|cutlass::bfloat16_t)\s*,", kernel)
            if output:
                outputs.add({"cutlass::half_t": "fp16", "cutlass::bfloat16_t": "bf16"}[output[1]])
        match = re.search(r"SM120_16x8x32_TN<([^>]+)>", kernel)
        if match:
            params = [part.strip() for part in match[1].split(",")]
            if len(params) == 3 and "float_e4m3" in params[0] and "float_e4m3" in params[1]:
                accum = {"float": "fp32", "cutlass::half_t": "fp16"}.get(params[2])
                if accum:
                    inferred.add(("fp8", accum))
                    evidence.append(match[0])
        if "F32BF16BF16F32" in kernel:
            inferred.add(("bf16", "fp32"))
            evidence.append("CUTLASS BF16 FP32 accumulator type")
        elif "F32F16F16F32" in kernel:
            inferred.add(("fp16", "fp32"))
            evidence.append("CUTLASS FP16 FP32 accumulator type")
    if len(inferred) == 1:
        operand, accum = next(iter(inferred))
        known = {"input_dtype": operand, "accum_dtype": accum, "evidence": evidence}
        if len(outputs) == 1:
            known["output_dtype"] = next(iter(outputs))
        return known
    # 输入 FP32 不能说明实际用了 FP32 还是 TF32；未知 accumulator 也不能猜。
    conflict = len(inferred) > 1
    return {
        "input_dtype": None,
        "accum_dtype": None,
        "evidence": evidence if conflict else ["No supported native/kernel precision contract"],
        "conflicting_kernel_precision": conflict,
    }


def call_precision(operators, kernel_names, actual_input_dtype, context):
    """CPU native名、kernel类型与同次入口设置各有来源，冲突不挑一个发布。"""
    names = sorted(set(operators) & GEMM_OPERATORS)
    contracts = [precision(name, kernel_names) for name in names] or [precision("", kernel_names)]
    signatures = {(item.get("input_dtype"), item.get("accum_dtype")) for item in contracts if item.get("input_dtype")}
    if len(signatures) > 1:
        return {
            "input_dtype": None,
            "accum_dtype": None,
            "observed_input_dtype": actual_input_dtype,
            "conflicting_kernel_precision": True,
            "evidence": [f"Conflicting actual native contracts: {names}"],
        }
    known = next((item for item in contracts if item.get("input_dtype")), contracts[0])
    known = {**known, "observed_input_dtype": actual_input_dtype, "native_operators": names}
    before, after = context.get("before"), context.get("after")
    if before is None or before != after:
        return {**known, "runtime_evidence": "missing_or_changed_at_core_entry"}
    # 只有真实native入口才使用其运行设置；量化custom合同不按外层BF16重解释。
    if len(names) == 1 and names[0] in ATEN_GEMM and not known.get("input_dtype"):
        known = runtime_precision(names[0], known, before)
    return {**known, "runtime_evidence": "core_entry_before_and_after"}


def runtime_precision(name, known, runtime):
    """本次设置加已核对的库入口合同，补名义累加；不推断物理中间归约。

    这里只发布 PyTorch 2.11 CUDA 原生 half/BF16 GEMM 的合同。新版本或
    非 cuBLAS backend 需重新核对 owning library，不扩大到所有模型。
    """
    result = dict(known)
    dtype = known.get("observed_input_dtype") or known.get("input_dtype")
    if (
        name not in {"aten::mm", "aten::bmm", "aten::addmm"}
        or runtime.get("device_type") != "cuda"
        or runtime.get("torch_version", "").split("+", 1)[0] != "2.11.0"
        or not runtime.get("cuda_build")
        or runtime.get("preferred_blas_library") not in {"_BlasBackend.Default", "_BlasBackend.Cublas"}
        or dtype not in {"bf16", "fp16"}
        or known.get("conflicting_kernel_precision")
    ):
        return result
    accum = "fp32"
    if dtype == "fp16":
        allow = runtime.get("allow_fp16_accumulation")
        if type(allow) is not bool:
            return result
        capability = runtime.get("compute_capability")
        if allow and (not capability or capability[0] < 7):
            return result
        accum = "fp16" if allow else "fp32"
    # kernel明确类型与本次库合同冲突时，双方都保留，禁止挑一个发布。
    if known.get("accum_dtype") not in {None, accum}:
        return {**result, "input_dtype": None, "accum_dtype": None, "conflicting_kernel_precision": True}
    return {
        **result,
        "input_dtype": dtype,
        "accum_dtype": accum,
        "nominal_accumulation": True,
        "actual_partial_reduction_dtype": None,
        "evidence": [
            *(item for item in known.get("evidence", []) if item != "No supported native/kernel precision contract"),
            "PyTorch v2.11 CUDA CUDABlas.cpp native half/BF16 GEMM compute_type contract + this core entry's boundary settings; nominal accumulator, not physical partial reductions",
        ],
    }


def ideal_gemm_bytes(shape, dtype, output_dtype):
    """一次读 A/B、一次写输出的理想流量；不含量化/scale/bias/cache。"""
    sizes = {"fp4": 0.5, "fp8": 1, "bf16": 2, "fp16": 2, "fp32": 4}
    if dtype not in sizes or output_dtype not in sizes:
        return None
    b, m, n, k = (shape[key] for key in ("batch", "m", "n", "k"))
    return b * ((m * k + k * n) * sizes[dtype] + m * n * sizes[output_dtype])
