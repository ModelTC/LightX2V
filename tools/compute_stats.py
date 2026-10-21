"""Reuse production formulas for offline benchmarks without LightX2V device/model initialization."""


from pathlib import Path
from runpy import run_path

_math = run_path(str(Path(__file__).resolve().parents[1] / "lightx2v/common/ops/compute_stats.py"))
gemm_flops = _math["gemm_flops"]
attention_flops = _math["attention_flops"]

routed_matmul_flops = _math["routed_matmul_flops"]
convolution_flops = _math["convolution_flops"]
