import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as Fn
from loguru import logger
from packaging.version import parse
from scipy import integrate, special

from .kernel import fp4_dequantize

try:
    from lightx2v_kernel.kv_cache import dequantize_kv_cache_fp4
except ImportError:
    dequantize_kv_cache_fp4 = None

try:
    from fouroversix import QuantizationConfig, QuantizeBackend
    from fouroversix.quantize.quantized_tensor import QuantizedTensor, from_blocked
except ImportError:
    QuantizedTensor = None
    from_blocked = None
    QuantizationConfig = None
    QuantizeBackend = None

_KV_TORCH_VER = None
_VALID_DEQUANT_BACKENDS = frozenset({"cuda", "triton", "pytorch"})


def _kvcache_dma_stream_priority() -> int:
    """Match WeightAsyncStreamManager cuda_load_stream priority."""
    global _KV_TORCH_VER
    if not torch.cuda.is_available():
        return 0
    if _KV_TORCH_VER is None:
        _KV_TORCH_VER = parse(torch.__version__.split("+")[0])
    return 1 if _KV_TORCH_VER >= parse("2.7") else 0


def ranked_calib_path(path: str, rank: int) -> str:
    if not path:
        return path
    dot = path.rfind(".")
    if dot <= 0:
        return f"{path}.rank{rank}"
    return f"{path[:dot]}.rank{rank}{path[dot:]}"


def cdiv(n: int, m: int) -> int:
    return (n + m - 1) // m


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return max(a, b) or 1
    return a * b // math.gcd(a, b)


def compute_analytical_turboquant_codebook(head_dim: int, bits: int) -> dict:
    """Lloyd-Max codebook on the sphere marginal (Beta on [-1,1]); returns JSON-serializable dict."""

    def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
        if d <= 2:
            raise ValueError(f"head_dim d={d} too small for TurboQuant codebook (need d>=3)")
        log_const = special.gammaln(d / 2.0) - 0.5 * np.log(np.pi) - special.gammaln((d - 1) / 2.0)
        exponent = (d - 3) / 2.0
        x = np.clip(x, -1 + 1e-15, 1 - 1e-15)
        log_val = log_const + exponent * np.log(1 - x**2)
        return np.exp(log_val)

    def conditional_mean(lo: float, hi: float, d: int) -> float:
        num, _ = integrate.quad(lambda x: x * beta_pdf(np.array([x]), d)[0], lo, hi)
        den, _ = integrate.quad(lambda x: beta_pdf(np.array([x]), d)[0], lo, hi)
        if den < 1e-30:
            return (lo + hi) / 2.0
        return num / den

    def mse_cost(centroids: np.ndarray, d: int) -> float:
        n = len(centroids)
        boundaries = np.zeros(n + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0
        cost = 0.0
        for i in range(n):
            lo, hi = boundaries[i], boundaries[i + 1]
            c = centroids[i]
            val, _ = integrate.quad(lambda x: (x - c) ** 2 * beta_pdf(np.array([x]), d)[0], lo, hi)
            cost += val
        return cost

    d, n_clusters = head_dim, 2**bits
    x_grid = np.linspace(-1 + 1e-10, 1 - 1e-10, 10000)
    pdf_vals = beta_pdf(x_grid, d)
    cdf_vals = np.cumsum(pdf_vals) * (x_grid[1] - x_grid[0])
    cdf_vals /= cdf_vals[-1]
    quantile_edges = np.linspace(0, 1, n_clusters + 1)
    centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        q_lo, q_hi = quantile_edges[i], quantile_edges[i + 1]
        q_mid = (q_lo + q_hi) / 2.0
        idx = min(int(np.searchsorted(cdf_vals, q_mid)), len(x_grid) - 1)
        centroids[i] = x_grid[idx]

    prev_cost = float("inf")
    cost = 0.0
    for _ in range(200):
        boundaries = np.zeros(n_clusters + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_clusters - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0
        new_centroids = np.zeros(n_clusters)
        for i in range(n_clusters):
            new_centroids[i] = conditional_mean(boundaries[i], boundaries[i + 1], d)
        cost = mse_cost(new_centroids, d)
        centroids = new_centroids
        if abs(prev_cost - cost) < 1e-12:
            break
        prev_cost = cost

    boundaries = np.zeros(n_clusters + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n_clusters - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    return {
        "centroids": centroids.tolist(),
        "boundaries": boundaries.tolist(),
        "mse_per_coord": float(cost),
        "mse_total": float(cost * d),
        "d": d,
        "bits": bits,
        "source": "analytical",
    }


def export_turboquant_codebook_json(
    head_dim: int,
    bits: int,
    out_dir: str,
) -> str:
    """Pre-compute Lloyd-Max codebook (sphere marginal Beta on [-1,1]) and save JSON.

    Output format matches ``/turboquant`` filename ``codebook_d{d}_b{b}.json`` (loadable via inference engine).
    Requires numpy + scipy.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"codebook_d{head_dim}_b{bits}.json")
    if os.path.isfile(path):
        return path

    cb = compute_analytical_turboquant_codebook(head_dim, bits)
    cb.pop("source", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cb, f, indent=2)
    logger.info("[TurboQuant] wrote codebook {!r} (d={}, bits={})", path, head_dim, bits)
    return path


def tq_fw_load_codebook_record(
    head_dim: int,
    bits: int,
    codebook_dir: str | None,
    codebook_cache_dir: str | None,
    export_missing: bool,
) -> dict:
    """Load codebook JSON dict; optional compute+write to cache dir."""
    subdirs = [p for p in (codebook_dir, codebook_cache_dir) if p]
    name = f"codebook_d{head_dim}_b{bits}.json"
    for ddir in subdirs:
        p = os.path.join(ddir, name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    if export_missing and codebook_cache_dir:
        export_turboquant_codebook_json(head_dim, bits, codebook_cache_dir)
        p = os.path.join(codebook_cache_dir, name)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"TurboQuant codebook not found: {name} under {subdirs or '(no dirs)'}; run export_turboquant_codebook_json(...) or set codebook_cache_dir + export_missing_codebooks.")


def tq_fw_pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack integer indices (aligned with /turboquant ``quantizer._pack_indices``)."""

    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]
    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return indices.to(torch.uint8)

    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = Fn.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)
    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8) * bits
    packed = (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)
    return packed


def tq_fw_unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    batch_shape = packed.shape[:-1]
    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.long()

    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * bits
    unpacked = (packed.unsqueeze(-1) >> shifts) & mask
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


def tq_fw_packed_width(head_dim: int, bits: int) -> int:
    if bits > 4:
        return head_dim
    if bits == 1:
        vpb = 8
    elif bits == 2:
        vpb = 4
    else:
        vpb = 2
    padded_d = ((head_dim + vpb - 1) // vpb) * vpb
    return padded_d // vpb


def tq_fw_generate_rotation_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device=device, dtype=dtype)


def tq_fw_generate_qjl_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
) -> torch.Tensor:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(d, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


def tq_fw_rotate_forward(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, Pi.T)


def tq_fw_rotate_backward(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    return torch.matmul(y, Pi)


def tq_fw_pack_qjl_signs(projected: torch.Tensor) -> torch.Tensor:
    signs = (projected > 0).to(torch.uint8)
    d = signs.shape[-1]
    if d % 8 != 0:
        signs = torch.nn.functional.pad(signs, (0, 8 - d % 8), value=0)
    signs_reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=signs.device, dtype=torch.uint8)
    return (signs_reshaped * powers).sum(dim=-1, dtype=torch.uint8)


def tq_fw_unpack_qjl_signs(packed: torch.Tensor, dim: int) -> torch.Tensor:
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
    unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
    signs = unpacked.reshape(*packed.shape[:-1], -1)[..., :dim]
    return 2.0 * signs - 1.0


def tq_group_quantize_values(v: torch.Tensor, bits: int, group_size: int) -> dict:
    """Group min-max quantize V; ``v`` shape (B,H,S,D). Returns packed data + scales + zeros."""
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    if d % group_size != 0:
        raise ValueError(f"head_dim {d} must divide value_group_size {group_size}")
    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)
    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values
    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)
    zero = v_min
    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)
    if bits == 2:
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = v_4[..., 0] | (v_4[..., 1] << 2) | (v_4[..., 2] << 4) | (v_4[..., 3] << 6)
    elif bits == 4:
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = v_2[..., 0] | (v_2[..., 1] << 4)
    else:
        packed = v_q_flat
    return {
        "data": packed,
        "scales": scale.squeeze(-1).to(torch.float16),
        "zeros": zero.squeeze(-1).to(torch.float16),
        "bits": bits,
        "group_size": group_size,
        "shape": tuple(orig_shape),
    }


def tq_group_dequantize_values(comp: dict) -> torch.Tensor:
    bits = int(comp["bits"])
    group_size = int(comp["group_size"])
    packed = comp["data"]
    d = comp["shape"][-1]
    batch_shape = comp["shape"][:-1]
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        data = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*batch_shape, packed.shape[-1] * 4)
    elif bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        data = torch.stack([v0, v1], dim=-1).reshape(*batch_shape, packed.shape[-1] * 2)
    else:
        data = packed
    data = data.float()
    n_groups = d // group_size
    data = data.reshape(*batch_shape, n_groups, group_size)
    scales = comp["scales"].unsqueeze(-1).float()
    zeros = comp["zeros"].unsqueeze(-1).float()
    return (data * scales + zeros).reshape(*batch_shape, d)


def tq_value_group_packed_width(head_dim: int, bits: int) -> int:
    if bits == 2:
        return head_dim // 4
    if bits == 4:
        return head_dim // 2
    return head_dim


def tq_lloyd_max_from_histogram_counts(
    hist_counts,
    n_centroids: int,
    max_iter: int = 150,
):
    """1D Lloyd-Max on a uniform histogram over [-1, 1]. ``hist_counts`` shape (n_bins,)."""

    n_bins = int(hist_counts.shape[0])
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist_counts.astype(np.float64)
    total = w.sum()
    if total < 1e-30:
        raise ValueError("TurboQuant calib histogram is empty")
    w /= total

    cdf = np.cumsum(w)
    targets = (np.arange(n_centroids, dtype=np.float64) + 0.5) / n_centroids
    centroids = np.sort(np.interp(targets, cdf, centers))

    for _ in range(max_iter):
        boundaries = np.zeros(n_centroids + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_centroids - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

        assign = np.searchsorted(boundaries, centers, side="right") - 1
        assign = np.clip(assign, 0, n_centroids - 1)

        new_c = np.zeros(n_centroids)
        for j in range(n_centroids):
            mask = assign == j
            ww = w[mask].sum()
            if ww > 1e-30:
                new_c[j] = (w[mask] * centers[mask]).sum() / ww
            else:
                new_c[j] = centroids[j]

        if np.max(np.abs(new_c - centroids)) < 1e-9:
            centroids = new_c
            break
        centroids = new_c

    boundaries = np.zeros(n_centroids + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n_centroids - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    assign = np.searchsorted(boundaries, centers, side="right") - 1
    assign = np.clip(assign, 0, n_centroids - 1)
    mse = 0.0
    for j in range(n_centroids):
        mask = assign == j
        if w[mask].sum() > 0:
            mse += float((w[mask] * (centroids[j] - centers[mask]) ** 2).sum())

    return centroids, boundaries, mse


def turboquant_codebook_dict_from_histogram(
    hist: torch.Tensor,
    head_dim: int,
    bits: int,
    *,
    n_bins: int = 4096,
) -> dict:
    """Build TurboQuant JSON codebook dict from accumulated marginal histogram (rotated unit keys/values)."""

    hc = hist.detach().cpu().numpy().astype(np.float64)
    if hc.shape[0] != n_bins:
        raise ValueError(f"hist length {hc.shape[0]} != n_bins {n_bins}")

    n_centroids = 2**bits
    if hc.sum() < 1:
        logger.warning(
            "[TurboQuant calib] empty histogram for d={}, bits={}; using analytical codebook.",
            head_dim,
            bits,
        )
        cb = compute_analytical_turboquant_codebook(head_dim, bits)
        cb.pop("source", None)
        cb["source"] = "analytical_fallback"
        return cb

    centroids, boundaries, mse_coord = tq_lloyd_max_from_histogram_counts(hc, n_centroids)
    return {
        "centroids": centroids.tolist(),
        "boundaries": boundaries.tolist(),
        "mse_per_coord": float(mse_coord),
        "mse_total": float(mse_coord * head_dim),
        "d": head_dim,
        "bits": bits,
        "source": "empirical_histogram",
    }


def build_turboquant_codebooks_from_calib_histograms(
    hist_k: torch.Tensor,
    *,
    head_dim: int,
    key_bits: int,
    n_bins: int = 4096,
) -> dict[str, dict]:
    """Produce filename -> codebook dict for JSON export (inference loader compatible)."""
    out: dict[str, dict] = {}
    if key_bits < 2:
        raise ValueError("TurboQuantProd requires key_bits >= 2")
    b_k = key_bits - 1
    ck = turboquant_codebook_dict_from_histogram(hist_k, head_dim, b_k, n_bins=n_bins)
    out[f"codebook_d{head_dim}_b{b_k}.json"] = ck
    return out


class TurboQuantMSEInference(torch.nn.Module):
    """TurboQuant MSE stage: rotation + Lloyd-Max via ``searchsorted`` + bit-pack."""

    def __init__(
        self,
        dim: int,
        bits: int,
        device: torch.device,
        seed: int,
        codebook: dict,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.register_buffer("Pi", tq_fw_generate_rotation_matrix(dim, device, dtype, seed=seed))
        c = torch.tensor(codebook["centroids"], device=device, dtype=dtype)
        b = torch.tensor(codebook["boundaries"], device=device, dtype=dtype)
        self.register_buffer("centroids", c)
        self.register_buffer("boundaries", b)
        self.register_buffer("decision_boundaries", b[1:-1].contiguous())

    @torch.no_grad()
    def compress_bhsd(self, x: torch.Tensor) -> dict:
        norms = x.norm(dim=-1, keepdim=False)
        x_unit = x / (norms.unsqueeze(-1) + 1e-10)
        y = tq_fw_rotate_forward(x_unit.float(), self.Pi)
        indices = torch.searchsorted(self.decision_boundaries, y.contiguous())
        packed = tq_fw_pack_indices(indices, self.bits)
        B, H, S, D = x.shape
        return {
            "idx_bytes": packed,
            "vec_norms": norms.to(torch.float16),
            "shape": (B, H, S, D),
            "bits": self.bits,
        }

    @torch.no_grad()
    def decompress_bhsd(self, comp: dict) -> torch.Tensor:
        B, H, S, D = comp["shape"]
        bits = int(comp["bits"])
        idx = tq_fw_unpack_indices(comp["idx_bytes"], bits, D)
        y_hat = self.centroids[idx]
        x_hat = tq_fw_rotate_backward(y_hat, self.Pi)
        return x_hat * comp["vec_norms"].unsqueeze(-1).float()


class TurboQuantProdInference(torch.nn.Module):
    """TurboQuant inner-product path: (key_bits-1) MSE + QJL on residual."""

    def __init__(
        self,
        dim: int,
        bits: int,
        device: torch.device,
        seed: int,
        codebook_mse: dict,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert bits >= 2, "TurboQuantProd needs key_bits >= 2"
        self.dim = dim
        self.bits = bits
        self.mse_bits = bits - 1
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim
        self.mse = TurboQuantMSEInference(dim, self.mse_bits, device, seed, codebook_mse, dtype=dtype)
        self.register_buffer("S", tq_fw_generate_qjl_matrix(dim, device, dtype, seed=seed + 1000))

    @torch.no_grad()
    def compress_bhsd(self, x: torch.Tensor) -> dict:
        mse_c = self.mse.compress_bhsd(x)
        x_mse = self.mse.decompress_bhsd(mse_c)
        residual = x - x_mse
        residual_norms = residual.norm(dim=-1)
        projected = torch.matmul(residual.float(), self.S.T)
        qjl_packed = tq_fw_pack_qjl_signs(projected)
        B, H, S, D = x.shape
        return {
            "mse_idx_bytes": mse_c["idx_bytes"],
            "qjl_bytes": qjl_packed,
            "residual_norms": residual_norms.to(torch.float16),
            "vec_norms": mse_c["vec_norms"],
            "shape": (B, H, S, D),
            "mse_bits": self.mse_bits,
        }

    @torch.no_grad()
    def decompress_bhsd(self, comp: dict) -> torch.Tensor:
        B, H, S, D = comp["shape"]
        mse_c = {
            "idx_bytes": comp["mse_idx_bytes"],
            "vec_norms": comp["vec_norms"],
            "shape": (B, H, S, D),
            "bits": int(comp["mse_bits"]),
        }
        x_mse = self.mse.decompress_bhsd(mse_c)
        signs = tq_fw_unpack_qjl_signs(comp["qjl_bytes"], D)
        x_qjl = torch.matmul(signs, self.S)
        x_qjl = x_qjl * (self.qjl_scale * comp["residual_norms"].unsqueeze(-1).float())
        return x_mse + x_qjl


def normalize_dequant_backend(backend: str | None) -> str:
    """Map ``kv_quant.backend`` to a KV dequant implementation name."""
    if backend is None or not str(backend).strip():
        raise ValueError(
            f"kv_quant.backend is required for longlive_fp4 dequant. Choose one of: {', '.join(sorted(_VALID_DEQUANT_BACKENDS))}",
        )
    name = str(backend).strip().lower()
    if name == "torch":
        name = "pytorch"
    if name == "transformer_engine":
        name = "cuda"
    if name not in _VALID_DEQUANT_BACKENDS:
        allowed = ", ".join(sorted(_VALID_DEQUANT_BACKENDS))
        raise ValueError(f"Unsupported KV dequant backend {backend!r}. Expected one of: {allowed}")
    return name


def scale_rule_to_fp4_limits(scale_rule) -> tuple[float, float]:
    if hasattr(scale_rule, "max_allowed_e2m1_value") and hasattr(
        scale_rule,
        "max_allowed_e4m3_value",
    ):
        return (
            float(scale_rule.max_allowed_e2m1_value()),
            float(scale_rule.max_allowed_e4m3_value()),
        )

    normalized = str(scale_rule).lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    normalized = normalized.strip().strip("\"'")

    if normalized == "static_4":
        return 4.0, 448.0
    if normalized == "static_6":
        return 6.0, 448.0
    if normalized in {"mse", "mae", "l1_norm", "abs_max"}:
        return 6.0, 256.0

    raise ValueError(f"Unsupported FP4 scale_rule: {scale_rule}")


def _dequant_blocks_cuda(
    values: list[torch.Tensor],
    scale_factors: list[torch.Tensor],
    amax_list: list[torch.Tensor],
    *,
    num_heads: int,
    block_token_size: int,
    dtype: torch.dtype,
    scale_rule,
) -> torch.Tensor:
    """Fused parallel dequant via ``lightx2v_kernel`` (optional LongLive op fallback)."""
    if not values or values[0].device.type != "cuda":
        raise RuntimeError("KV dequant backend=cuda requires CUDA tensors.")

    e2m1_max, e4m3_max = scale_rule_to_fp4_limits(scale_rule)
    out = dequantize_kv_cache_fp4(
        values,
        scale_factors,
        amax_list,
        num_heads=num_heads,
        block_token_size=block_token_size,
        dtype=dtype,
        e2m1_max=e2m1_max,
        e4m3_max=e4m3_max,
    )
    return out[0]


def _global_scale_for_qt(qt: QuantizedTensor) -> torch.Tensor:
    e2m1_max, e4m3_max = scale_rule_to_fp4_limits(qt.scale_rule)
    return qt.amax / (e2m1_max * e4m3_max)


def _dequant_qt_triton(qt: QuantizedTensor, dtype: torch.dtype) -> torch.Tensor:
    """Per-block NVFP4 dequant via LightX2V Triton (fouroversix tensor layout)."""
    block_size = qt.dtype.block_size()
    padded_shape = qt.padded_shape
    scales_2d = from_blocked(
        qt.scale_factors,
        (padded_shape[0], padded_shape[1] // block_size),
    )
    return fp4_dequantize(
        qt.values,
        scales_2d,
        _global_scale_for_qt(qt),
        block_size=block_size,
        dtype=dtype,
    )


def _dequant_blocks_pytorch(
    blocks: list[QuantizedTensor],
    num_heads: int,
    head_dim: int,
    block_token_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    parts = [qt.dequantize(dtype).view(block_token_size, num_heads, head_dim) for qt in blocks]
    return torch.cat(parts, dim=0)


def _dequant_blocks_triton(
    blocks: list[QuantizedTensor],
    num_heads: int,
    head_dim: int,
    block_token_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    n_blks = len(blocks)
    h, d = num_heads, head_dim
    out = torch.zeros(
        [1, n_blks * block_token_size, h, d],
        dtype=dtype,
        device=device,
    )
    for block_idx, qt in enumerate(blocks):
        deq = _dequant_qt_triton(qt, dtype)
        deq = deq[: block_token_size * h]
        t_start = block_idx * block_token_size
        t_end = t_start + block_token_size
        out[0, t_start:t_end, :, :] = deq.view(block_token_size, h, d)
    return out[0]


def dequantize_kv_blocks(
    blocks: list[QuantizedTensor],
    num_heads: int,
    head_dim: int,
    block_token_size: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    backend: str,
) -> torch.Tensor:
    """
    Dequantize block list to ``[T, num_heads, head_dim]`` (T = len(blocks) * block_token_size).

    ``backend`` must be one of ``cuda``, ``triton``, ``pytorch`` (``torch`` aliases ``pytorch``).
    """
    if not blocks:
        return torch.empty(0, num_heads, head_dim, device=device, dtype=dtype)

    mode = normalize_dequant_backend(backend)
    scale_rule = blocks[0].scale_rule
    values = [qt.values for qt in blocks]
    scale_factors = [qt.scale_factors for qt in blocks]
    amax_list = [qt.amax for qt in blocks]

    if mode == "cuda":
        return _dequant_blocks_cuda(
            values,
            scale_factors,
            amax_list,
            num_heads=num_heads,
            block_token_size=block_token_size,
            dtype=dtype,
            scale_rule=scale_rule,
        )

    if mode == "triton":
        return _dequant_blocks_triton(
            blocks,
            num_heads,
            head_dim,
            block_token_size,
            dtype,
            device,
        )

    return _dequant_blocks_pytorch(blocks, num_heads, head_dim, block_token_size, dtype)


def dequantize_token_range(
    blocks: list[QuantizedTensor],
    attn_start: int,
    local_end: int,
    *,
    cache_size: int,
    num_heads: int,
    head_dim: int,
    block_token_size: int,
    dtype: torch.dtype,
    device: torch.device,
    backend: str,
) -> torch.Tensor:
    if local_end <= attn_start:
        return torch.empty(0, num_heads, head_dim, device=device, dtype=dtype)

    t0 = attn_start
    t1 = min(local_end, cache_size)
    b0 = t0 // block_token_size
    b1 = (t1 - 1) // block_token_size
    sub = blocks[b0 : b1 + 1]
    nhd = dequantize_kv_blocks(
        sub,
        num_heads,
        head_dim,
        block_token_size,
        dtype,
        device,
        backend=backend,
    )
    off0 = t0 - b0 * block_token_size
    off1 = t1 - b0 * block_token_size
    return nhd[off0:off1].contiguous()


def k_smooth(k: torch.Tensor) -> torch.Tensor:
    """Per-head mean removal before K quantization (LongLive)."""
    return k - k.mean(dim=-1, keepdim=True)


def clone_quantized_tensor(qt: QuantizedTensor) -> QuantizedTensor:
    return QuantizedTensor(
        values=qt.values.clone(),
        scale_factors=qt.scale_factors.clone(),
        amax=qt.amax.clone() if qt.amax is not None else None,
        dtype=qt.dtype,
        original_shape=qt.original_shape,
        scale_rule=qt.scale_rule,
        padded_shape=qt.padded_shape,
    )


def build_fp4_quant_config(
    *,
    scale_rule: str = "mse",
    backend: str | None = None,
) -> QuantizationConfig:
    backend_enum = QuantizeBackend(backend) if backend is not None else None
    return QuantizationConfig(scale_rule=scale_rule, backend=backend_enum)
