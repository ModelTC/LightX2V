"""只读精度设置；在专用 CLI 诊断进程中保存 NVIDIA 库的实际 API 日志。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path


def read_library_logs(directory):
    """进程退出后读取实际API精度字段；不按启发式分给trace中的调用。"""
    path = Path(directory)
    session = path / "session.json"
    raw = session.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("kind") != "library_precision_logs":
        raise ValueError("expected a library_precision_logs session")
    result = {"session": manifest, "session_sha256": hashlib.sha256(raw).hexdigest(), "files": [], "api_precision": [], "versions": []}
    versions = set()
    for file in sorted(path.glob("*.log")):
        data = file.read_bytes()
        result["files"].append({"path": str(file.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        api = None
        for number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
            version = re.search(r"cuBLAS \(v([\d.]+)\)", line)
            if version:
                versions.add(version[1])
            header = re.search(r"function .*?\b(cublas\w+)\(", line)
            if header:
                api = header[1]
            compute = re.search(r"computeType:.*?val=(CUBLAS_COMPUTE_\w+)", line)
            if compute and api and "Gemm" in api:
                result["api_precision"].append({"file": file.name, "line": number, "api": api, "compute_type": compute[1]})
            # Trace是实际Matmul；AlgoGetHeuristic/descriptor创建不是执行证据。
            lt_call = re.search(r"\[Trace\]\[(cublasLt\w*Matmul)\].*?computeType=(\w+)", line)
            if lt_call:
                result["api_precision"].append({"file": file.name, "line": number, "api": lt_call[1], "compute_type": lt_call[2]})
    result["versions"] = sorted(versions)
    result["evidence_status"] = "actual_api_precision_fields" if result["api_precision"] else "no_supported_api_precision_fields"
    result["association"] = "process-level library API evidence including warmup; no automatic per-call/shape attribution"
    return result


def runtime_settings(torch, device_type):
    """设置是本次调用的允许条件，不等于实际选择了低精度归约。"""
    device = torch.device(device_type)
    capability = list(torch.cuda.get_device_capability(device)) if device.type == "cuda" and torch.cuda.is_initialized() else None
    matmul = torch.backends.cuda.matmul
    names = ("allow_tf32", "allow_fp16_accumulation", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")
    flags, errors = {}, {}
    for name in names:
        try:
            flags[name] = getattr(matmul, name, None)
        except RuntimeError as error:
            # 新/旧精度API冲突时保留缺口，统计不能改变模型设置或执行结果。
            flags[name] = None
            errors[name] = str(error)
    try:
        flags["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except RuntimeError as error:
        # backend使用新API时旧全局查询可能不可用；不能使已完成算子或日志收尾失败。
        flags["float32_matmul_precision"] = None
        errors["float32_matmul_precision"] = str(error)
    return {
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "device_type": device.type,
        "compute_capability": capability,
        "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
        **flags,
        "setting_read_errors": errors,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }


@contextmanager
def library_logs(directory):
    """日志开关必须先于库初始化；只允许尚未导入 torch 的独立 CLI 进程。

    环境在退出时恢复，但已初始化的库可能继续持有日志文件，因此不作为
    可反复启停的进程内 API。原始日志不按相近时间/shape 猜配到某次算子。
    """
    if directory is None:
        yield
        return
    if "torch" in sys.modules:
        raise RuntimeError("library logging requires a fresh CLI process before importing torch")
    path = Path(directory).resolve()
    settings = {
        "CUBLAS_LOGINFO_DBG": "1",
        "CUBLAS_LOGDEST_DBG": str(path / "cublas.log"),
        "CUBLASLT_LOG_LEVEL": "5",
        "CUBLASLT_LOG_FILE": str(path / "cublaslt.%i.log"),
    }
    if any(key.startswith(("CUBLAS_LOG", "CUBLASLT_LOG")) for key in os.environ):
        raise RuntimeError("existing cuBLAS logging settings must be reconciled before capture")
    path.mkdir(parents=True, exist_ok=False)
    os.environ.update(settings)
    status = "failed"
    try:
        yield
        status = "complete"
    finally:
        for key in settings:
            os.environ.pop(key, None)
        artifacts = []
        for file in sorted(path.glob("*.log")):
            raw = file.read_bytes()
            artifacts.append({"path": file.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        loaded = []
        maps = Path("/proc/self/maps")
        if maps.exists():
            loaded = sorted({line.split()[-1] for line in maps.read_text().splitlines() if "libcublas" in line and ".so" in line})
        torch = sys.modules.get("torch")
        manifest = {
            "kind": "library_precision_logs",
            "schema_version": 1,
            "status": status,
            "pid": os.getpid(),
            "snapshot_only": True,
            "settings": settings,
            "loaded_libraries": loaded,
            "runtime": runtime_settings(torch, "cuda") if torch is not None else None,
            "artifacts": artifacts,
            "evidence_status": "logs_present" if any(row["bytes"] for row in artifacts) else "no_library_log_evidence",
            "association": "process-wide raw API logs; no automatic per-call attribution",
            "boundary": "API computeType and math permissions do not prove every physical partial reduction; logging affects timings. Files may receive a final library flush at process exit.",
        }
        (path / "session.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
