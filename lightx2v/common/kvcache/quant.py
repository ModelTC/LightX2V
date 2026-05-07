import torch
import math
import torch.distributed as dist
import os
import json

try:
    import fp4quant_cuda
except ImportError:
    fp4quant_cuda = None

try:
    from sageattention.triton.quant_per_thread import quant_key_per_thread_int8_kernel
except ImportError:
    quant_key_per_thread_int8_kernel = None

from .kernel import *
from .offload import KVOffloadPlugin
from .rolling import RollingKVCachePool
from loguru import logger
from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head


# --------- Vendored TurboQuant ``MSECompressor`` (from turboquant-pytorch V3 path) ---------


def _tq_beta_pdf(x: float, d: int) -> float:
    if abs(x) >= 1.0:
        return 0.0
    coeff = math.gamma(d / 2) / (math.sqrt(math.pi) * math.gamma((d - 1) / 2))
    return coeff * (1 - x * x) ** ((d - 3) / 2)


def _tq_gaussian_approx_pdf(x: float, d: float) -> float:
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def _tq_solve_lloyd_max(
    d: int,
    bits: int,
    use_exact: bool = False,
    max_iter: int = 200,
    tol: float = 1e-10,
):
    from scipy import integrate as _sci_integrate

    n_levels = 2 ** bits
    pdf = (lambda x: _tq_beta_pdf(x, d)) if use_exact else (lambda x: _tq_gaussian_approx_pdf(x, float(d)))
    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _ = _sci_integrate.quad(lambda x: x * pdf(x), a, b)
            denominator, _ = _sci_integrate.quad(pdf, a, b)
            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])
        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


def _tq_compute_expected_distortion(
    d: int,
    bits: int,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    use_exact: bool = False,
) -> float:
    from scipy import integrate as _sci_integrate

    pdf = (lambda x: _tq_beta_pdf(x, d)) if use_exact else (lambda x: _tq_gaussian_approx_pdf(x, float(d)))
    sigma = 1.0 / math.sqrt(d)
    n_levels = len(centroids)
    edges = [-3.5 * sigma * 3] + boundaries.tolist() + [3.5 * sigma * 3]
    total_distortion = 0.0
    for i in range(n_levels):
        a, b = edges[i], edges[i + 1]
        c = centroids[i].item()
        dist, _ = _sci_integrate.quad(lambda x: (x - c) ** 2 * pdf(x), a, b)
        total_distortion += dist
    return total_distortion


class _TurboquantLloydMaxCodebook:
    def __init__(self, d: int, bits: int, use_exact: bool = False):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        self.centroids, self.boundaries = _tq_solve_lloyd_max(d, bits, use_exact)
        self.distortion = _tq_compute_expected_distortion(d, bits, self.centroids, self.boundaries, use_exact)


def _tq_generate_rotation_matrix(d: int, seed: int | None = None, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


class MSECompressor:
    """Single-stage TurboQuant V3 compressor (packed indices + fp16 norms; no external repo)."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.Pi = _tq_generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.centroids = _TurboquantLloydMaxCodebook(head_dim, bits).centroids.to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        B, H, S, D = states.shape
        N = B * H * S
        flat = states.reshape(N, D).float()
        vec_norms = torch.norm(flat, dim=-1)
        flat_norm = flat / (vec_norms.unsqueeze(-1) + 1e-8)
        rotated = flat_norm @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        indices_per_byte = 8 // self.bits
        idx_pad = (indices_per_byte - D % indices_per_byte) % indices_per_byte
        idx_flat = indices.long()
        if idx_pad:
            idx_flat = torch.nn.functional.pad(idx_flat, (0, idx_pad))
        n_groups = idx_flat.shape[-1] // indices_per_byte
        idx_powers = torch.tensor(
            [2 ** (self.bits * i) for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_flat.device,
        )
        idx_bytes = (idx_flat.reshape(N, n_groups, indices_per_byte) * idx_powers).sum(-1).to(torch.uint8)

        return {
            "idx_bytes": idx_bytes.reshape(B, H, S, n_groups),
            "vec_norms": vec_norms.to(torch.float16).reshape(B, H, S),
            "shape": (B, H, S, D),
            "idx_pad": idx_pad,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        B, H, S, D = compressed["shape"]
        N = B * H * S
        idx_bytes = compressed["idx_bytes"].reshape(N, -1)
        vec_norms = compressed["vec_norms"].reshape(N, 1).float()
        idx_pad = compressed["idx_pad"]
        indices_per_byte = 8 // self.bits
        mask = (1 << self.bits) - 1
        idx_shifts = torch.tensor(
            [self.bits * i for i in range(indices_per_byte - 1, -1, -1)],
            dtype=torch.long,
            device=idx_bytes.device,
        )
        indices = ((idx_bytes.long().unsqueeze(-1) >> idx_shifts) & mask).reshape(N, -1)
        if idx_pad:
            indices = indices[:, :D]

        reconstructed = (self.centroids[indices] @ self.Pi) * vec_norms
        return reconstructed.reshape(B, H, S, D)


# --------- end vendored TurboQuant MSE ---------


# --------- TurboQuant inference engine (aligned with /turboquant: searchsorted + pack + optional QJL) ---------


def compute_analytical_turboquant_codebook(head_dim: int, bits: int) -> dict:
    """Lloyd–Max codebook on the sphere marginal (Beta on [-1,1]); returns JSON-serializable dict."""
    import numpy as np
    from scipy import integrate, special

    def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
        if d <= 2:
            raise ValueError(f"head_dim d={d} too small for TurboQuant codebook (need d>=3)")
        log_const = (
            special.gammaln(d / 2.0)
            - 0.5 * np.log(np.pi)
            - special.gammaln((d - 1) / 2.0)
        )
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
            val, _ = integrate.quad(
                lambda x: (x - c) ** 2 * beta_pdf(np.array([x]), d)[0], lo, hi
            )
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
    """Pre-compute Lloyd–Max codebook (sphere marginal Beta on [-1,1]) and save JSON.

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


def _tq_fw_load_codebook_record(
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
    raise FileNotFoundError(
        f"TurboQuant codebook not found: {name} under {subdirs or '(no dirs)'}; "
        f"run export_turboquant_codebook_json(...) or set codebook_cache_dir + export_missing_codebooks."
    )


def _tq_fw_pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack integer indices (aligned with /turboquant ``quantizer._pack_indices``)."""
    import torch.nn.functional as Fn

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


def _tq_fw_unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
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
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask)
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


def _tq_fw_packed_width(head_dim: int, bits: int) -> int:
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


def _tq_fw_generate_rotation_matrix(
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


def _tq_fw_generate_qjl_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
) -> torch.Tensor:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(d, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


def _tq_fw_rotate_forward(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, Pi.T)


def _tq_fw_rotate_backward(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    return torch.matmul(y, Pi)


def _tq_fw_pack_qjl_signs(projected: torch.Tensor) -> torch.Tensor:
    signs = (projected > 0).to(torch.uint8)
    d = signs.shape[-1]
    if d % 8 != 0:
        signs = torch.nn.functional.pad(signs, (0, 8 - d % 8), value=0)
    signs_reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=signs.device, dtype=torch.uint8)
    return (signs_reshaped * powers).sum(dim=-1, dtype=torch.uint8)


def _tq_fw_unpack_qjl_signs(packed: torch.Tensor, dim: int) -> torch.Tensor:
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
    unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
    signs = unpacked.reshape(*packed.shape[:-1], -1)[..., :dim]
    return 2.0 * signs - 1.0


class TurboQuantMSEInference(torch.nn.Module):
    """TurboQuant MSE stage: rotation + Lloyd–Max via ``searchsorted`` + bit-pack."""

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
        self.register_buffer("Pi", _tq_fw_generate_rotation_matrix(dim, device, dtype, seed=seed))
        c = torch.tensor(codebook["centroids"], device=device, dtype=dtype)
        b = torch.tensor(codebook["boundaries"], device=device, dtype=dtype)
        self.register_buffer("centroids", c)
        self.register_buffer("boundaries", b)
        self.register_buffer("decision_boundaries", b[1:-1].contiguous())

    @torch.no_grad()
    def compress_bhsd(self, x: torch.Tensor) -> dict:
        norms = x.norm(dim=-1, keepdim=False)
        x_unit = x / (norms.unsqueeze(-1) + 1e-10)
        y = _tq_fw_rotate_forward(x_unit.float(), self.Pi)
        indices = torch.searchsorted(self.decision_boundaries, y.contiguous())
        packed = _tq_fw_pack_indices(indices, self.bits)
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
        idx = _tq_fw_unpack_indices(comp["idx_bytes"], bits, D)
        y_hat = self.centroids[idx]
        x_hat = _tq_fw_rotate_backward(y_hat, self.Pi)
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
        assert bits >= 2, "Prod TurboQuant needs key_bits >= 2 when use_qjl=true"
        self.dim = dim
        self.bits = bits
        self.mse_bits = bits - 1
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim
        self.mse = TurboQuantMSEInference(dim, self.mse_bits, device, seed, codebook_mse, dtype=dtype)
        self.register_buffer("S", _tq_fw_generate_qjl_matrix(dim, device, dtype, seed=seed + 1000))

    @torch.no_grad()
    def compress_bhsd(self, x: torch.Tensor) -> dict:
        mse_c = self.mse.compress_bhsd(x)
        x_mse = self.mse.decompress_bhsd(mse_c)
        residual = x - x_mse
        residual_norms = residual.norm(dim=-1)
        projected = torch.matmul(residual.float(), self.S.T)
        qjl_packed = _tq_fw_pack_qjl_signs(projected)
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
        signs = _tq_fw_unpack_qjl_signs(comp["qjl_bytes"], D)
        x_qjl = torch.matmul(signs, self.S)
        x_qjl = x_qjl * (self.qjl_scale * comp["residual_norms"].unsqueeze(-1).float())
        return x_mse + x_qjl


def _tq_group_quantize_values(v: torch.Tensor, bits: int, group_size: int) -> dict:
    """Group min–max quantize V; ``v`` shape (B,H,S,D). Returns packed data + scales + zeros."""
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


def _tq_group_dequantize_values(comp: dict) -> torch.Tensor:
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


# --------- end TurboQuant inference engine ---------


_FP8_MAX = 448.0

_TURBOQUANT_CALIB_NBINS = 4096


def _tq_lloyd_max_from_histogram_counts(
    hist_counts,
    n_centroids: int,
    max_iter: int = 150,
):
    """1D Lloyd–Max on a uniform histogram over [-1, 1]. ``hist_counts`` shape (n_bins,)."""
    import numpy as np

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
    n_bins: int = _TURBOQUANT_CALIB_NBINS,
) -> dict:
    """Build TurboQuant JSON codebook dict from accumulated marginal histogram (rotated unit keys/values)."""
    import numpy as np

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

    centroids, boundaries, mse_coord = _tq_lloyd_max_from_histogram_counts(hc, n_centroids)
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
    hist_v: torch.Tensor | None,
    *,
    head_dim: int,
    key_bits: int,
    value_bits: int,
    use_qjl: bool,
    value_quant_mode: str,
    n_bins: int = _TURBOQUANT_CALIB_NBINS,
) -> dict[str, dict]:
    """Produce filename -> codebook dict for JSON export (inference loader compatible)."""
    out: dict[str, dict] = {}
    if use_qjl:
        b_k = key_bits - 1
        ck = turboquant_codebook_dict_from_histogram(hist_k, head_dim, b_k, n_bins=n_bins)
        out[f"codebook_d{head_dim}_b{b_k}.json"] = ck
    else:
        ck = turboquant_codebook_dict_from_histogram(hist_k, head_dim, key_bits, n_bins=n_bins)
        out[f"codebook_d{head_dim}_b{key_bits}.json"] = ck

    if value_quant_mode == "mse" and hist_v is not None:
        cv = turboquant_codebook_dict_from_histogram(hist_v, head_dim, value_bits, n_bins=n_bins)
        out[f"codebook_d{head_dim}_b{value_bits}.json"] = cv

    return out


def _ranked_calib_path(path: str, rank: int) -> str:
    if not path:
        return path
    dot = path.rfind(".")
    if dot <= 0:
        return f"{path}.rank{rank}"
    return f"{path[:dot]}.rank{rank}{path[dot:]}"

def _cdiv(n: int, m: int) -> int:
    return (n + m - 1) // m


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return max(a, b) or 1
    return a * b // math.gcd(a, b)


class CalibRollingKVCachePool(RollingKVCachePool):
    """Normal bf16 rolling cache that additionally captures the (km,
    v_channel_max, k_block_scale) that sage_attn computes internally —
    keyed by ``(step, layer)`` and **shared across all chunks**.

    Capture semantics
    -----------------
    Each chunk's call to ``capture_attn`` runs sage's K-quant kernel on
    the full attention window currently in the buffer and overwrites the
    entries at ``[step, layer]`` — but only as long as the window keeps
    growing.  Once rolling kicks in (window stops growing), captures are
    skipped: the rolled-state buffer no longer matches what early-chunk
    inference will see at those positions, so freezing the pre-roll
    snapshot gives consistent calibration.

    The scales are stored at *buffer-absolute* block positions so that
    the quant cache can index them directly when storing later chunks.

    After inference, ``export_calibration()`` returns:
        ``km``            shape [S, L,         1,        H, D]    fp32
        ``v_scale``       shape [S, L,                   H, D]    fp32
        ``k_block_scale`` shape [S, L, max_blks, H, scales_per_blk] fp32

    Set ``current_step`` before each denoising step so captures land in
    the right slot.

    **TurboQuant empirical codebooks** (optional): enable with ``turboquant_calibrate=True``
    (see ``build_self_attn_kv_cache`` when ``quant_scheme=="turboquant"`` and ``calibrate``).
    The same ``capture_attn`` hook accumulates marginal histograms of rotated unit-norm K/V
    (matching :class:`TurboQuantMSEInference` seeds). After inference,
    :meth:`export_calibration` includes ``_turboquant_hist_*`` tensors; the manager can merge
    them across ranks and write JSON files named like ``codebook_d{d}_b{b}.json`` for use with
    ``TurboQuantRollingKVCachePool(turboquant_engine="inference", codebook_dir=...)``.

    Implementation notes
    --------------------
    - The K slice fed to the calibration kernel starts at ``aligned_start
      = (attn_start // 128) * 128`` so the per-block scales line up with
      the buffer's natural 128-token blocks. The same alignment is used
      by ``QuantRollingKVCachePool.k_cache`` / ``v_cache`` with
      ``attn_start`` and ``local_end``.
    - km is captured in bf16 (matching sage's ``k.mean(...)`` dtype),
      then cast to fp32 for storage. This avoids extra mantissa bits
      that would otherwise diverge from sage at the bf16 ``k - km``
      subtraction step.
    """

    _BLKK = 128
    _SCALES_PER_BLK = 4  # WARPK=128 ⇒ 4 thread groups per block per head

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        num_steps: int = 1,
        *,
        turboquant_calibrate: bool = False,
        key_bits: int = 4,
        value_bits: int = 2,
        use_qjl: bool = False,
        turboquant_seed: int = 42,
        per_layer_compressors: bool = True,
        value_quant_mode: str = "mse",
    ) -> None:
        self._num_steps = num_steps
        self.current_step: int = 0
        self._turboquant_calibrate = bool(turboquant_calibrate)
        self._tq_key_bits = int(key_bits)
        self._tq_value_bits = int(value_bits)
        self._tq_use_qjl = bool(use_qjl)
        self._turboquant_seed = int(turboquant_seed)
        self._tq_per_layer = bool(per_layer_compressors)
        self._value_quant_mode = value_quant_mode.strip().lower()
        if self._turboquant_calibrate and self._tq_use_qjl and self._tq_key_bits < 2:
            raise ValueError("TurboQuant calib with use_qjl requires key_bits >= 2")
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device)

    def _init_kv_buffer(self) -> None:
        super()._init_kv_buffer()
        S = self._num_steps
        L, H, D = self._num_layers, self._num_heads, self._head_dim
        BLK = self._BLKK
        max_blks = (self._cache_size + BLK - 1) // BLK
        self._km = torch.zeros(S, L, 1, H, D, dtype=torch.float32, device=self._device)
        self._v_channel_max = torch.zeros(S, L, H, D, dtype=torch.float32, device=self._device)
        self._k_block_scale_calib = torch.zeros(
            S,
            L,
            max_blks,
            H,
            self._SCALES_PER_BLK,
            dtype=torch.float32,
            device=self._device,
        )
        self._capture_flag = torch.zeros(S, L, dtype=torch.bool, device=self._device)
        self._captured_window_size = torch.zeros(S, L, dtype=torch.long, device="cpu")
        if self._turboquant_calibrate:
            nb = _TURBOQUANT_CALIB_NBINS
            self._tq_hist_k = torch.zeros(nb, dtype=torch.int64, device=self._device)
            self._tq_collect_v = self._value_quant_mode == "mse"
            if self._tq_collect_v:
                self._tq_hist_v = torch.zeros(nb, dtype=torch.int64, device=self._device)

    def _quant_key(self, k: torch.Tensor, km: torch.Tensor | None = None, BLKK: int = 128, WARPK: int = 128):
        """Run sage's per_thread int8 K-quantisation kernel on ``k``.

        Returns ``(k_int8, k_scale)`` where ``k`` is ``[B, kv_len, H, D]`` (NHD).
        The km subtraction (if any) is done in ``k.dtype`` to match sage's
        behaviour exactly — sage does ``k - km`` in bf16, NOT fp32.

        This is the source-of-truth quantisation used both at calibration time
        (to capture the per-block scale we'll later replay) and as a reference
        for the preset-scale quantisation path.
        """
        if km is not None:
            km_lowp = km.to(k.dtype) if km.dtype != k.dtype else km
            k = k - km_lowp

        k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
        b, kv_len, h_kv, head_dim = k.shape

        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = (
            k_int8.stride(0),
            k_int8.stride(2),
            k_int8.stride(1),
        )

        num_blk = (kv_len + BLKK - 1) // BLKK
        scales_per_blk = (BLKK // WARPK) * 4
        k_scale = torch.empty(
            (b, h_kv, num_blk * scales_per_blk),
            device=k.device,
            dtype=torch.float32,
        )

        grid = (num_blk * scales_per_blk, h_kv, b)
        quant_key_per_thread_int8_kernel[grid](
            k,
            k_int8,
            k_scale,
            kv_len,
            stride_bz_k,
            stride_h_k,
            stride_seq_k,
            stride_bz_ko,
            stride_h_ko,
            stride_seq_ko,
            k_scale.stride(0),
            k_scale.stride(1),
            C=head_dim,
            BLK=WARPK,
        )
        return k_int8, k_scale

    def capture_attn(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
    ) -> None:
        """Capture (km, v_channel_max, k_block_scale) from the buffer's
        current state — exactly what sage_attn would see at this call.

        Parameters
        ----------
        attn_start : start position of the attention window in the buffer
                     (may not be 128-aligned).
        local_end  : end position (exclusive) — the buffer's current valid
                     length for this layer.

        The captured K slice is aligned down to the nearest 128 boundary
        so per-block scales map cleanly to buffer block indices.
        """
        BLK = self._BLKK
        aligned_start = (attn_start // BLK) * BLK
        step, layer = self.current_step, layer_id

        k_full = self._k_buffer[layer_id, aligned_start:local_end]  # [kv_len_a, H, D] bf16
        v_full = self._v_buffer[layer_id, aligned_start:local_end]  # [kv_len_a, H, D] bf16
        kv_len_a = k_full.size(0)
        if kv_len_a == 0:
            return

        prev_window = int(self._captured_window_size[step, layer].item())
        if 0 < prev_window >= kv_len_a:
            return
        self._captured_window_size[step, layer] = kv_len_a

        # ---- km (bf16 mean to match sage) ----
        km_lowp = k_full.mean(dim=0, keepdim=True)  # bf16 [1, H, D]
        self._km[step, layer] = km_lowp.to(torch.float32)

        # ---- k_block_scale via sage's quant kernel on (k - km) ----
        k_batch = k_full.unsqueeze(0).contiguous()  # [1, kv_len_a, H, D]
        _, k_scale_raw = self._quant_key(k_batch, km_lowp)  # [1, H, num_blk*4]
        num_blk_local = (kv_len_a + BLK - 1) // BLK
        k_scale_local = k_scale_raw[0].reshape(self._num_heads, num_blk_local, self._SCALES_PER_BLK).permute(1, 0, 2)  # [num_blk_local, H, 4]
        blk_offset = aligned_start // BLK
        self._k_block_scale_calib[step, layer, blk_offset : blk_offset + num_blk_local] = k_scale_local
        self._v_channel_max[step, layer] = v_full.float().abs().amax(dim=0)  # [H, D]
        self._capture_flag[step, layer] = True

        if self._turboquant_calibrate:
            self._capture_turboquant_marginals(layer_id, k_full, v_full)

    def _capture_turboquant_marginals(
        self,
        layer_id: int,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
    ) -> None:
        """Histogram rotated coordinate marginals (same convention as TurboQuant inference)."""
        D = self._head_dim
        dev = self._device
        nb = self._tq_hist_k.numel()

        seed_k = (
            self._turboquant_seed + layer_id * 1000 if self._tq_per_layer else self._turboquant_seed
        )
        Pi_k = _tq_fw_generate_rotation_matrix(D, dev, torch.float32, seed=seed_k)
        x = k_full.float()
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        x_unit = x / norms
        y = torch.matmul(x_unit, Pi_k.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        idx = ((y + 1.0) * (0.5 * (nb - 1))).long().clamp(0, nb - 1)
        ones = torch.ones(idx.numel(), dtype=torch.int64, device=dev)
        self._tq_hist_k.scatter_add_(0, idx.reshape(-1), ones)

        if self._tq_collect_v:
            seed_v = (
                self._turboquant_seed + layer_id * 1000 + 500
                if self._tq_per_layer
                else self._turboquant_seed + 500
            )
            Pi_v = _tq_fw_generate_rotation_matrix(D, dev, torch.float32, seed=seed_v)
            xv = v_full.float()
            nv = xv.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            xu = xv / nv
            yv = torch.matmul(xu, Pi_v.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            idv = ((yv + 1.0) * (0.5 * (nb - 1))).long().clamp(0, nb - 1)
            self._tq_hist_v.scatter_add_(0, idv.reshape(-1), ones)

    def export_calibration(self) -> dict[str, torch.Tensor]:
        v_scale = self._v_channel_max.clamp(min=1e-5) / _FP8_MAX
        out: dict[str, torch.Tensor] = {
            "km": self._km.clone(),
            "v_scale": v_scale,
            "k_block_scale": self._k_block_scale_calib.clone(),
        }
        if self._turboquant_calibrate:
            out["_turboquant_hist_k"] = self._tq_hist_k.clone()
            if self._tq_collect_v:
                out["_turboquant_hist_v"] = self._tq_hist_v.clone()
        return out

    def reset(self) -> None:
        super().reset()
        self._km.zero_()
        self._v_channel_max.zero_()
        self._k_block_scale_calib.zero_()
        self._capture_flag.zero_()
        self._captured_window_size.zero_()
        if self._turboquant_calibrate:
            self._tq_hist_k.zero_()
            if self._tq_collect_v:
                self._tq_hist_v.zero_()


class SageQuantRollingKVCachePool(RollingKVCachePool):
    _BLKK = 128
    _SCALES_PER_BLK = 4  # (BLKK // WARPK) * 4, WARPK=128
    _PERM_16_VAL = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
    _INV_PERM_16_VAL = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        k_cache_type: str = "int8",
        v_cache_type: str = "fp8",
        calib_path: str = None,
        kv_offload: bool = False,
    ) -> None:

        assert k_cache_type in ["int8"], f"Invalid k_cache_type: {k_cache_type}"
        assert v_cache_type in ["fp8", "fp16"], f"Invalid v_cache_type: {v_cache_type}"
        self._k_cache_type = k_cache_type
        self._v_cache_type = v_cache_type
        self._calib_path = calib_path
        self.current_step: int = 0
        self._PERM_16 = torch.tensor(self._PERM_16_VAL, dtype=torch.long, device=device)
        self._INV_PERM_16 = torch.tensor(self._INV_PERM_16_VAL, dtype=torch.long, device=device)
        self._load_calib()
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        L = self._num_layers
        N = self._cache_size
        H = self._num_heads
        D = self._head_dim
        self._k_buffer = torch.zeros(L, N, H, D, dtype=torch.int8, device=self._device)
        self._v_buffer = torch.zeros(L, N, H, D, dtype=self._v_cache_type == "fp8" and torch.float8_e4m3fn or torch.float16, device=self._device)

        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

    def _init_kv_buffer_offload(self) -> None:
        L = self._num_layers
        N = self._cache_size
        H = self._num_heads
        D = self._head_dim
        self._k_cpu = torch.zeros(L, N, H, D, dtype=torch.int8, device="cpu").pin_memory()
        self._v_cpu = torch.zeros(L, N, H, D, dtype=self._v_cache_type == "fp8" and torch.float8_e4m3fn or torch.float16, device="cpu").pin_memory()
        self._k_gpu_buf = torch.zeros(2, N, H, D, dtype=torch.int8, device=self._device)
        self._v_gpu_buf = torch.zeros(2, N, H, D, dtype=self._v_cache_type == "fp8" and torch.float8_e4m3fn or torch.float16, device=self._device)
        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

        def _async_load(layer_id: int, buf: int) -> None:
            self._k_gpu_buf[buf].copy_(self._k_cpu[layer_id], non_blocking=True)
            self._v_gpu_buf[buf].view(torch.float8_e4m3fn).copy_(self._v_cpu[layer_id], non_blocking=True)

        def _async_store(layer_id: int, buf: int, start: int, end: int) -> None:
            self._k_cpu[layer_id, start:end].copy_(
                self._k_gpu_buf[buf, start:end],
                non_blocking=True,
            )
            v_gpu_slice_u8 = self._v_gpu_buf[buf, start:end].view(torch.float8_e4m3fn)
            self._v_cpu[layer_id, start:end].copy_(v_gpu_slice_u8, non_blocking=True)

        self._offload = KVOffloadPlugin(self._device, _async_load, _async_store)
        gpu_mb = (self._k_gpu_buf.nbytes + self._v_gpu_buf.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_cpu.nbytes + self._v_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[SageQuantRollingKVCachePool+offload] GPU fixed buffer: {:.1f} MB, CPU pinned: {:.1f} MB (saved {:.1f} MB GPU)",
            gpu_mb,
            cpu_mb,
            cpu_mb - gpu_mb,
        )
        return

    def _load_calib(self, device=torch.device("cuda")) -> None:
        load_path = self._calib_path
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            rank_path = _ranked_calib_path(self._calib_path, rank)
            if os.path.exists(rank_path):
                load_path = rank_path
        calib = torch.load(load_path, map_location=device, weights_only=True)
        self._calib_km = calib["km"].to(device=device, dtype=torch.float32)
        self._calib_v_scale = calib["v_scale"].to(device=device, dtype=torch.float32)
        if "k_block_scale" not in calib:
            raise RuntimeError(f"Calibration file {load_path!r} is missing 'k_block_scale'. Re-run calibration with CalibRollingKVCachePool.")
        self._calib_k_block_scale = calib["k_block_scale"].to(
            device=device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            if load_path == self._calib_path:
                logger.warning(
                    "Sage KV calibration: loaded shared file {!r} while world_size={}. "
                    "k_block_scale is indexed by *local* rolling-buffer block id; with "
                    "seq_parallel each rank uses a shorter buffer and a different sequence "
                    "shard than a single-GPU run, so a single-GPU calib file is usually *not* "
                    "applicable. Re-run calibrate with the same world_size / seq_p as "
                    "inference (saves per-rank calib_kv.rankR.pt) or use unquantized KV to compare.",
                    self._calib_path,
                    dist.get_world_size(),
                )

    def _quant_key(
        self,
        k_smoothed: torch.Tensor,
        preset_scale: torch.Tensor,
        start_idx: int,
        BLKK: int = 128,
    ) -> torch.Tensor:
        chunk_len, H, D = k_smoothed.shape
        num_blk = preset_scale.size(0)

        k_int8 = torch.empty_like(k_smoothed, dtype=torch.int8)
        preset_scale_c = preset_scale.contiguous()
        grid = (num_blk * 4, H, 1)
        quant_key_per_thread_int8_static_scale_kernel[grid](
            k_smoothed,
            k_int8,
            preset_scale_c,
            chunk_len,
            start_idx,
            0,
            k_smoothed.stride(1),
            k_smoothed.stride(0),
            0,
            k_int8.stride(1),
            k_int8.stride(0),
            preset_scale_c.stride(0),
            preset_scale_c.stride(1),
            C=D,
            BLK=BLKK,
        )
        return k_int8

    def _lookup_km(self, layer_id: int) -> torch.Tensor | None:
        """Return km of shape [1, 1, H, D] for the current (step, layer),
        or None if K smoothing is disabled.

        Supported calibration file shapes (newest → legacy):
          [S, L, 1, H, D]  – per (step, layer)            ← preferred
          [   L, 1, H, D]  – per (layer)                  ← legacy
        """
        km_cal = self._calib_km
        if km_cal.dim() == 5:
            return km_cal[self.current_step, layer_id].unsqueeze(0)
        return km_cal[layer_id].unsqueeze(0)

    def _lookup_v_scale(self, layer_id: int) -> torch.Tensor:
        """Return v_scale of shape [H, D] for the current (step, layer).

        Supported calibration file shapes (newest → legacy):
          [S, L, H, D]  – per (step, layer)               ← preferred
          [   L, H, D]  – per (layer)                     ← legacy
        """
        vs_cal = self._calib_v_scale
        if vs_cal.dim() == 4:
            return vs_cal[self.current_step, layer_id]
        return vs_cal[layer_id]

    def _lookup_k_block_scale(
        self,
        layer_id: int,
        blk_start: int,
        num_blk: int,
    ) -> torch.Tensor:
        """Return ``[num_blk, H, scales_per_blk]`` slice of the calibrated
        k-block scale at the given absolute buffer block range.

        Calibration file shape: ``[S, L, max_blks, H, scales_per_blk]``.
        """
        return self._calib_k_block_scale[
            self.current_step,
            layer_id,
            blk_start : blk_start + num_blk,
        ]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        km = self._lookup_km(layer_id)
        if km is not None:
            km_lowp = km.to(k.dtype).squeeze(0)
            k_smoothed = k - km_lowp
        else:
            k_smoothed = k

        blk_start = start_idx // self._BLKK
        last_blk = (end_idx - 1) // self._BLKK
        num_blk = last_blk - blk_start + 1

        preset_scale = self._lookup_k_block_scale(layer_id, blk_start, num_blk)
        k_int8 = self._quant_key(k_smoothed, preset_scale, start_idx, self._BLKK)
        v_scale = self._lookup_v_scale(layer_id)
        v_fp8 = quant_value_per_channel_fp8_static_scale_kernel(v, v_scale, fp8_max=_FP8_MAX)

        if not self._kv_offload:
            self._k_buffer[layer_id, start_idx:end_idx] = k_int8
            self._v_buffer[layer_id, start_idx:end_idx] = v_fp8
            return
        buf = self._offload.cur_buf
        self._k_gpu_buf[buf, start_idx:end_idx] = k_int8
        self._v_gpu_buf[buf, start_idx:end_idx] = v_fp8
        self._mark_offload_dirty(start_idx, end_idx)

    def _gather_per_token_k_scale(
        self,
        layer_id: int,
        start_pos: int,
        num_tokens: int,
    ) -> torch.Tensor:
        positions = torch.arange(
            start_pos,
            start_pos + num_tokens,
            device=self._device,
        )
        blk_idx = positions // self._BLKK
        thread = (positions % self._BLKK // 2) % 4
        return self._calib_k_block_scale[
            self.current_step,
            layer_id,
            blk_idx,
            :,
            thread,
        ]

    def _transpose_permute_v(self, v: torch.Tensor) -> torch.Tensor:
        kv_len, H, D = v.shape
        padded_len = (kv_len + 127) // 128 * 128

        if padded_len > kv_len:
            v_t = v.new_zeros(D, H, padded_len)
            v_t[:, :, :kv_len].copy_(v.permute(2, 1, 0))
        else:
            v_t = v.permute(2, 1, 0).contiguous()

        v_t = v_t.view(D, H, -1, 16)[:, :, :, self._PERM_16].contiguous()
        v_t = v_t.view(1, D, H, padded_len)
        return v_t

    def _roll_window_on_k_v(self, kb: torch.Tensor, vb: torch.Tensor, layer_id: int, sink_tokens: int, num_evicted: int) -> None:
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        src_start = sink_tokens + num_evicted
        src_end = src_start + num_kept
        dst_start = sink_tokens
        dst_end = dst_start + num_kept
        if num_kept > 0:
            x = kb[src_start:src_end].contiguous()  # [num_kept, H, D]
            out = kb[dst_start:dst_end]
            src_scale = self._gather_per_token_k_scale(layer_id, src_start, num_kept)
            dst_scale = self._gather_per_token_k_scale(layer_id, dst_start, num_kept)
            k_int8_roll_rescale_triton(x, out, src_scale, dst_scale, scale_eps=1e-5)
        vb[dst_start:dst_end].copy_(vb[src_start:src_end].clone())

    def roll_window(self, layer_id: int, sink_tokens: int, num_evicted: int) -> None:
        if not self._kv_offload:
            self._roll_window_on_k_v(
                self._k_buffer[layer_id],
                self._v_buffer[layer_id],
                layer_id,
                sink_tokens,
                num_evicted,
            )
            return
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        dst_s = sink_tokens
        self._roll_window_on_k_v(
            self._k_gpu_buf[self._offload.cur_buf],
            self._v_gpu_buf[self._offload.cur_buf],
            layer_id,
            sink_tokens,
            num_evicted,
        )
        self._mark_offload_dirty(dst_s, dst_s + num_kept)

    def k_cache(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
    ):
        BLK = self._BLKK
        aligned_start = (attn_start // BLK) * BLK
        buf = self._offload.cur_buf if self._kv_offload else None
        kb = self._k_gpu_buf[buf] if self._kv_offload else self._k_buffer[layer_id]
        k_int8 = kb[aligned_start:local_end].unsqueeze(0).contiguous()
        blk_s = aligned_start // BLK
        blk_e = (local_end + BLK - 1) // BLK
        k_scale = self._calib_k_block_scale[self.current_step, layer_id, blk_s:blk_e].permute(1, 0, 2).reshape(1, self._num_heads, -1).contiguous()
        return k_int8, k_scale

    def v_cache(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
    ):
        BLK = self._BLKK
        aligned_start = (attn_start // BLK) * BLK
        buf = self._offload.cur_buf if self._kv_offload else None
        vb = self._v_gpu_buf[buf] if self._kv_offload else self._v_buffer[layer_id]
        v_fp8 = vb[aligned_start:local_end]
        v_fp8 = self._transpose_permute_v(v_fp8)
        v_scale = self._lookup_v_scale(layer_id).unsqueeze(0).contiguous()
        return v_fp8, v_scale

    def _sp_quant_kv_to_head_shard(
        self,
        k_cache,
        v_cache,
        shard_heads: int,
        seq_p_group,
        *,
        attn_start: int | None = None,
        local_end: int | None = None,
    ):
        if not (isinstance(k_cache, tuple) and isinstance(v_cache, tuple)):
            raise TypeError("SageQuant SP path expects tuple k_cache and v_cache.")
        if len(k_cache) != 2 or len(v_cache) != 2:
            raise ValueError("Unsupported SageQuant KV tuple format in SP path.")
        if attn_start is None or local_end is None:
            raise ValueError("SageQuant SP path requires attn_start and local_end (k_scale is buffer-aligned; see k_cache).")

        cur_rank = dist.get_rank(seq_p_group)
        hs = slice(cur_rank * shard_heads, (cur_rank + 1) * shard_heads)

        k_int8, k_scale = k_cache
        v_data, v_scale = v_cache

        # Must match k_cache: slice starts at aligned 128, not at attn_start.
        BLK = self._BLKK
        aligned_start = (attn_start // BLK) * BLK
        blk_s = aligned_start // BLK

        k_nhd = self._to_nhd(k_int8, "k_int8")
        full_k_nhd = all2all_seq2head(k_nhd, group=seq_p_group)
        full_k_int8 = full_k_nhd.unsqueeze(0).contiguous()
        full_kv_len = int(full_k_nhd.size(0))

        # Rebuild k_scale to match full-seq blocks:
        # local block-scale -> per-token (buffer-absolute) -> all2all(seq->head) -> full block-scale.
        k_scale_hs = self._to_heads_scale(k_scale, "k_scale")  # [H, local_num_blk*4]
        local_kv_len = int(k_nhd.size(0))
        local_tok_scale = self._expand_k_scale_to_tokens(
            k_scale_hs, local_kv_len, aligned_start=aligned_start, buffer_blk_s=blk_s
        )  # [local_kv_len, H]
        full_tok_scale = all2all_seq2head(local_tok_scale.unsqueeze(-1), group=seq_p_group).squeeze(-1)  # [full_kv_len, shard_heads]
        k_scale_shard = self._compress_token_scale_to_block4(full_tok_scale).unsqueeze(0).contiguous()  # [1, shard_heads, ceil(full_kv_len/128)*4]

        v_nhd = self._sage_v_layout_to_nhd(v_data, local_kv_len)
        full_v_nhd = all2all_seq2head(v_nhd, group=seq_p_group)
        full_v_data = self._nhd_to_sage_v_layout(full_v_nhd)

        v_scale_hd = self._to_heads_dim_scale(v_scale, "v_scale")
        v_scale_shard = v_scale_hd[hs, :].unsqueeze(0).contiguous()

        return (full_k_int8, k_scale_shard), (full_v_data, v_scale_shard), full_kv_len

    @staticmethod
    def _to_nhd(x, name: str):
        if x.dim() == 4 and x.size(0) == 1:
            return x[0].contiguous()
        if x.dim() == 3:
            return x.contiguous()
        raise ValueError(f"Unsupported {name} shape {tuple(x.shape)} for SP quant KV.")

    @staticmethod
    def _to_heads_scale(x, name: str):
        if x.dim() == 3 and x.size(0) == 1:
            return x[0].contiguous()
        if x.dim() == 2:
            return x.contiguous()
        raise ValueError(f"Unsupported {name} shape {tuple(x.shape)} for SP quant KV.")

    @staticmethod
    def _to_heads_dim_scale(x, name: str):
        if x.dim() == 3 and x.size(0) == 1:
            return x[0].contiguous()
        if x.dim() == 2:
            return x.contiguous()
        raise ValueError(f"Unsupported {name} shape {tuple(x.shape)} for SP quant KV.")

    def _sage_v_layout_to_nhd(self, v_data: torch.Tensor, kv_len: int) -> torch.Tensor:
        if v_data.dim() == 4 and v_data.size(0) == 1:
            v_data = v_data[0]
        if v_data.dim() != 3:
            raise ValueError(f"Unsupported v_data shape {tuple(v_data.shape)} for SP quant KV.")
        d, h, padded_len = v_data.shape
        v_unperm = v_data.view(d, h, -1, 16)[:, :, :, self._INV_PERM_16].contiguous().view(d, h, padded_len)
        v_unperm = v_unperm[:, :, :kv_len]
        return v_unperm.permute(2, 1, 0).contiguous()

    def _nhd_to_sage_v_layout(self, v_nhd: torch.Tensor) -> torch.Tensor:
        kv_len, h, d = v_nhd.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_dhp = v_nhd.permute(2, 1, 0).contiguous()
        if padded_len > kv_len:
            padded = v_dhp.new_zeros(d, h, padded_len)
            padded[:, :, :kv_len].copy_(v_dhp)
            v_dhp = padded
        return v_dhp.view(d, h, -1, 16)[:, :, :, self._PERM_16].contiguous().view(1, d, h, padded_len)

    @staticmethod
    def _expand_k_scale_to_tokens(
        k_scale_hs: torch.Tensor,
        kv_len: int,
        *,
        aligned_start: int,
        buffer_blk_s: int,
    ) -> torch.Tensor:
        """[H, (slice_num_blk*4)] -> [kv_len, H], one scale per (buffer token index).

        ``k_int8`` is ``buffer[aligned_start:aligned_start+kv_len]``; k_scale is
        ``k_block_scale[..., buffer_blk_s:buffer_blk_e]`` in the same order as
        :meth:`k_cache` — **not** 0..kv_len-1 block indices. Use global buffer
        indices ``g = aligned_start + t``.
        """
        if k_scale_hs.dim() != 2:
            raise ValueError(f"Expected k_scale_hs 2D, got {tuple(k_scale_hs.shape)}")
        h, total = k_scale_hs.shape
        if total % 4 != 0:
            raise ValueError(f"Expected k_scale last dim multiple of 4, got {total}")
        num_blk_slice = total // 4
        scales = k_scale_hs.view(h, num_blk_slice, 4)
        g = torch.arange(aligned_start, aligned_start + kv_len, device=k_scale_hs.device, dtype=torch.long)
        blk_global = g // 128
        rel_blk = blk_global - int(buffer_blk_s)
        if (rel_blk < 0).any() or (rel_blk >= num_blk_slice).any():
            raise RuntimeError(
                f"k_scale slice mismatch: buffer_blk_s={buffer_blk_s}, rel_blk in [{int(rel_blk.min())},{int(rel_blk.max())}], "
                f"num_blk_slice={num_blk_slice}, aligned_start={aligned_start}, kv_len={kv_len}."
            )
        thr = ((g % 128) // 2) % 4
        return scales[:, rel_blk, thr].transpose(0, 1).contiguous()

    @staticmethod
    def _compress_token_scale_to_block4(tok_scale: torch.Tensor) -> torch.Tensor:
        """[kv_len, shard_heads] -> [shard_heads, ceil(kv_len/128)*4]."""
        if tok_scale.dim() != 2:
            raise ValueError(f"Expected tok_scale 2D, got {tuple(tok_scale.shape)}")
        kv_len, shard_heads = tok_scale.shape
        num_blk = (kv_len + 127) // 128
        out = tok_scale.new_zeros(shard_heads, num_blk, 4)
        pos = torch.arange(kv_len, device=tok_scale.device, dtype=torch.long)
        blk = pos // 128
        thr = ((pos % 128) // 2) % 4
        out[:, blk, thr] = tok_scale.transpose(0, 1)
        return out.view(shard_heads, num_blk * 4)


class SageAttn3FP4RollingKVCachePool(RollingKVCachePool):
    _BLK = 128
    _V_GROUP = 16

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        k_cache_type: str = "fp4",
        v_cache_type: str = "fp4",
        kv_offload: bool = False,
    ) -> None:
        assert k_cache_type in ["fp4"], f"Invalid k_cache_type: {k_cache_type}"
        assert v_cache_type in ["fp4"], f"Invalid v_cache_type: {v_cache_type}"
        if fp4quant_cuda is None:
            raise ImportError("fp4quant_cuda is required for SageAttn3 FP4 KV cache.")
        self._k_cache_type = k_cache_type
        self._v_cache_type = v_cache_type
        self._N_alloc = _cdiv(int(cache_size), self._BLK) * self._BLK
        super().__init__(
            num_layers,
            self._N_alloc,
            num_heads,
            head_dim,
            dtype,
            device,
            kv_offload=kv_offload,
        )

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            raise NotImplementedError("SageAttn3FP4RollingKVCachePool does not support kv_offload yet.")

        L = self._num_layers
        N = self._N_alloc
        H = self._num_heads
        D = self._head_dim
        self._k_fp4 = torch.zeros(L, H, N, D // 2, dtype=torch.uint8, device=self._device)
        self._k_scale = torch.zeros(L, H, N, D // 16, dtype=torch.float8_e4m3fn, device=self._device)
        self._v_fp4 = torch.zeros(L, H, D, N // 2, dtype=torch.uint8, device=self._device)
        self._v_scale = torch.zeros(L, H, D, N // 16, dtype=torch.float8_e4m3fn, device=self._device)
        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        chunk_len = int(end_idx - start_idx)
        if chunk_len <= 0:
            return
        if k.size(0) != chunk_len or v.size(0) != chunk_len:
            raise ValueError(
                f"SageAttn3FP4RollingKVCachePool.store_kv shape mismatch: "
                f"chunk_len={chunk_len}, k={k.size(0)}, v={v.size(0)}."
            )
        if start_idx % self._V_GROUP != 0 or chunk_len % self._V_GROUP != 0:
            raise ValueError(
                f"SageAttn3 FP4 KV requires start/chunk aligned to {self._V_GROUP} "
                f"(got start_idx={start_idx}, chunk_len={chunk_len})."
            )

        padded_len = _cdiv(chunk_len, self._BLK) * self._BLK
        k_in = k.transpose(0, 1).unsqueeze(0).contiguous()  # [1, H, N, D]
        v_in = v.transpose(0, 1).unsqueeze(0).contiguous()  # [1, H, N, D]

        k_fp4 = torch.empty((1, self._num_heads, padded_len, self._head_dim // 2), device=self._device, dtype=torch.uint8)
        k_scale = torch.empty((1, self._num_heads, padded_len, self._head_dim // 16), device=self._device, dtype=torch.float8_e4m3fn)
        v_fp4 = torch.empty((1, self._num_heads, self._head_dim, padded_len // 2), device=self._device, dtype=torch.uint8)
        v_scale = torch.empty((1, self._num_heads, self._head_dim, padded_len // 16), device=self._device, dtype=torch.float8_e4m3fn)

        fp4quant_cuda.scaled_fp4_quant_permute(k_in, k_fp4, k_scale, 1)
        fp4quant_cuda.scaled_fp4_quant_trans(v_in, v_fp4, v_scale, 1)

        self._k_fp4[layer_id, :, start_idx:end_idx].copy_(k_fp4[0, :, :chunk_len])
        self._k_scale[layer_id, :, start_idx:end_idx].copy_(k_scale[0, :, :chunk_len])
        self._v_fp4[layer_id, :, :, start_idx // 2 : end_idx // 2].copy_(v_fp4[0, :, :, : chunk_len // 2])
        self._v_scale[layer_id, :, :, start_idx // 16 : end_idx // 16].copy_(v_scale[0, :, :, : chunk_len // 16])

    def k_cache(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
    ):
        aligned_start = (attn_start // self._BLK) * self._BLK
        kv_len = int(local_end - aligned_start)
        padded_len = _cdiv(kv_len, self._BLK) * self._BLK
        k_fp4 = self._k_fp4[layer_id, :, aligned_start : aligned_start + padded_len].unsqueeze(0).contiguous()
        k_scale = self._k_scale[layer_id, :, aligned_start : aligned_start + padded_len].unsqueeze(0).contiguous()
        return k_fp4, k_scale, kv_len

    def v_cache(
        self,
        layer_id: int,
        attn_start: int,
        local_end: int,
    ):
        aligned_start = (attn_start // self._BLK) * self._BLK
        kv_len = int(local_end - aligned_start)
        padded_len = _cdiv(kv_len, self._BLK) * self._BLK
        v_fp4 = self._v_fp4[layer_id, :, :, aligned_start // 2 : (aligned_start + padded_len) // 2].unsqueeze(0).contiguous()
        v_scale = self._v_scale[layer_id, :, :, aligned_start // 16 : (aligned_start + padded_len) // 16].unsqueeze(0).contiguous()
        return v_fp4, v_scale, kv_len

    def roll_window(
        self,
        layer_id: int,
        sink_tokens: int,
        num_evicted: int,
    ) -> None:
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        if num_kept <= 0:
            return
        if sink_tokens % self._V_GROUP != 0 or num_evicted % self._V_GROUP != 0:
            raise ValueError(
                f"SageAttn3 FP4 roll_window requires sink_tokens/num_evicted aligned to {self._V_GROUP} "
                f"(got sink_tokens={sink_tokens}, num_evicted={num_evicted})."
            )

        src_start = sink_tokens + num_evicted
        src_end = src_start + num_kept
        dst_start = sink_tokens
        dst_end = dst_start + num_kept

        self._k_fp4[layer_id, :, dst_start:dst_end].copy_(self._k_fp4[layer_id, :, src_start:src_end].clone())
        self._k_scale[layer_id, :, dst_start:dst_end].copy_(self._k_scale[layer_id, :, src_start:src_end].clone())

        self._v_fp4[layer_id, :, :, dst_start // 2 : dst_end // 2].copy_(
            self._v_fp4[layer_id, :, :, src_start // 2 : src_end // 2].clone()
        )
        self._v_scale[layer_id, :, :, dst_start // 16 : dst_end // 16].copy_(
            self._v_scale[layer_id, :, :, src_start // 16 : src_end // 16].clone()
        )

    def reset(self) -> None:
        self._k_fp4.zero_()
        self._k_scale.zero_()
        self._v_fp4.zero_()
        self._v_scale.zero_()
        self._global_end.zero_()
        self._local_end.zero_()


def _turboquant_idx_padding(head_dim: int, bits: int) -> int:
    """Match ``MSECompressor`` bit-packing alignment (compressors_v3)."""
    if bits <= 0 or bits > 8:
        raise ValueError(f"TurboQuant bits must be in 1..8, got {bits}")
    indices_per_byte = 8 // bits
    return (indices_per_byte - head_dim % indices_per_byte) % indices_per_byte


def _turboquant_n_packed_groups(head_dim: int, bits: int) -> int:
    indices_per_byte = 8 // bits
    pad = _turboquant_idx_padding(head_dim, bits)
    return (head_dim + pad) // indices_per_byte


def _tq_value_group_packed_width(head_dim: int, bits: int) -> int:
    if bits == 2:
        return head_dim // 4
    if bits == 4:
        return head_dim // 2
    return head_dim


class TurboQuantRollingKVCachePool(RollingKVCachePool):
    """Rolling KV cache using TurboQuant-style quantization.

    **legacy** (default): embedded V3 ``MSECompressor`` (argmin Lloyd–Max + packing aligned with V3).

    **inference**: aligned with ``/turboquant`` — JSON codebooks (``searchsorted`` + pack), optional QJL
    on keys (``use_qjl``), and values as sphere MSE (``value_quant_mode="mse"``) or group min–max
    (``value_quant_mode="group"``, same as upstream ``quantize_values``).

    Pre-compute codebooks with :func:`export_turboquant_codebook_json` or set ``codebook_cache_dir`` and
    ``export_missing_codebooks`` to generate missing JSON on first run (needs scipy).

    ``k_cache`` / ``v_cache`` return **dequantized** tensors in ``self._dtype``.
    """

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        key_bits: int = 4,
        value_bits: int = 2,
        seed: int = 42,
        per_layer_compressors: bool = True,
        kv_offload: bool = False,
        *,
        turboquant_engine: str = "legacy",
        use_qjl: bool = False,
        codebook_dir: str | None = None,
        codebook_cache_dir: str | None = None,
        export_missing_codebooks: bool = False,
        value_quant_mode: str = "mse",
        value_group_size: int = 32,
    ) -> None:
        eng = turboquant_engine.strip().lower()
        if eng not in ("legacy", "inference"):
            raise ValueError(f"turboquant_engine must be 'legacy' or 'inference', got {turboquant_engine!r}")

        self._turboquant_engine = eng
        self._key_bits = int(key_bits)
        self._value_bits = int(value_bits)
        self._seed_base = int(seed)
        self._per_layer_compressors = bool(per_layer_compressors)
        self._n_layers = int(num_layers)
        dev_str = str(device)

        if eng == "legacy":
            self._use_qjl = False
            self._value_quant_mode = "mse"
            ng_k = _turboquant_n_packed_groups(head_dim, self._key_bits)
            ng_v = _turboquant_n_packed_groups(head_dim, self._value_bits)
            self._k_idx_pad = _turboquant_idx_padding(head_dim, self._key_bits)
            self._v_idx_pad = _turboquant_idx_padding(head_dim, self._value_bits)

            if self._per_layer_compressors:
                self._k_compressors: list[MSECompressor] = []
                self._v_compressors: list[MSECompressor] = []
                for lid in range(self._n_layers):
                    seed_k = self._seed_base + lid * 1000
                    self._k_compressors.append(MSECompressor(head_dim, self._key_bits, seed=seed_k, device=dev_str))
                    self._v_compressors.append(MSECompressor(head_dim, self._value_bits, seed=seed_k + 500, device=dev_str))
            else:
                self._k_compressors = [
                    MSECompressor(head_dim, self._key_bits, seed=self._seed_base, device=dev_str)
                ]
                self._v_compressors = [
                    MSECompressor(head_dim, self._value_bits, seed=self._seed_base + 500, device=dev_str)
                ]

            self._ng_k = ng_k
            self._ng_v = ng_v
            self._k_inference_modules = None
            self._v_inference_modules = None
            super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device, kv_offload=kv_offload)
            return

        # ----- inference engine -----
        self._use_qjl = bool(use_qjl)
        vqm = value_quant_mode.strip().lower()
        if vqm not in ("mse", "group"):
            raise ValueError(f"value_quant_mode must be 'mse' or 'group', got {value_quant_mode!r}")
        self._value_quant_mode = vqm
        self._value_group_size = int(value_group_size)

        if self._use_qjl and self._key_bits < 2:
            raise ValueError("use_qjl requires key_bits >= 2 (inner MSE uses key_bits - 1).")
        if self._value_quant_mode == "group" and head_dim % self._value_group_size != 0:
            raise ValueError(
                f"head_dim {head_dim} must divide value_group_size {self._value_group_size} for group value quant."
            )

        device_t = torch.device(dev_str)
        inf_dtype = torch.float32
        nk_bits = self._key_bits - 1 if self._use_qjl else self._key_bits
        cb_key = _tq_fw_load_codebook_record(
            head_dim,
            nk_bits,
            codebook_dir,
            codebook_cache_dir,
            export_missing_codebooks,
        )

        self._inf_nk = _tq_fw_packed_width(head_dim, nk_bits)
        self._inf_nqjl = (head_dim + 7) // 8 if self._use_qjl else 0

        def _make_k_mod(seed_k: int) -> torch.nn.Module:
            if self._use_qjl:
                return TurboQuantProdInference(
                    head_dim, self._key_bits, device_t, seed_k, cb_key, dtype=inf_dtype
                )
            return TurboQuantMSEInference(
                head_dim, self._key_bits, device_t, seed_k, cb_key, dtype=inf_dtype
            )

        if self._per_layer_compressors:
            self._k_inference_modules = [_make_k_mod(self._seed_base + lid * 1000) for lid in range(self._n_layers)]
        else:
            _km = _make_k_mod(self._seed_base)
            self._k_inference_modules = [_km for _ in range(self._n_layers)]

        if self._value_quant_mode == "mse":
            cb_val = _tq_fw_load_codebook_record(
                head_dim,
                self._value_bits,
                codebook_dir,
                codebook_cache_dir,
                export_missing_codebooks,
            )
            self._inf_nv = _tq_fw_packed_width(head_dim, self._value_bits)

            def _make_v_mod(seed_v: int) -> TurboQuantMSEInference:
                return TurboQuantMSEInference(
                    head_dim, self._value_bits, device_t, seed_v, cb_val, dtype=inf_dtype
                )

            if self._per_layer_compressors:
                self._v_inference_modules = [
                    _make_v_mod(self._seed_base + lid * 1000 + 500) for lid in range(self._n_layers)
                ]
            else:
                _vm = _make_v_mod(self._seed_base + 500)
                self._v_inference_modules = [_vm for _ in range(self._n_layers)]
        else:
            self._v_inference_modules = None
            self._inf_nv = 0
            self._inf_v_width = _tq_value_group_packed_width(head_dim, self._value_bits)
            self._inf_v_n_groups = head_dim // self._value_group_size

        self._k_compressors = []
        self._v_compressors = []
        self._ng_k = self._inf_nk
        self._ng_v = self._inf_nv if self._value_quant_mode == "mse" else self._inf_v_width
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    def _kc(self, layer_id: int) -> MSECompressor:
        return self._k_compressors[layer_id] if self._per_layer_compressors else self._k_compressors[0]

    def _vc(self, layer_id: int) -> MSECompressor:
        return self._v_compressors[layer_id] if self._per_layer_compressors else self._v_compressors[0]

    def _k_mod_inf(self, layer_id: int) -> torch.nn.Module:
        assert self._k_inference_modules is not None
        return self._k_inference_modules[layer_id]

    def _v_mod_inf(self, layer_id: int) -> TurboQuantMSEInference:
        assert self._v_inference_modules is not None
        return self._v_inference_modules[layer_id]

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            raise NotImplementedError("TurboQuantRollingKVCachePool does not support kv_offload yet.")

        L = self._num_layers
        N = self._cache_size
        H = self._num_heads
        D = self._head_dim

        if self._turboquant_engine == "legacy":
            nk, nv = self._ng_k, self._ng_v
            self._k_packed = torch.zeros(L, N, H, nk, dtype=torch.uint8, device=self._device)
            self._k_norms = torch.zeros(L, N, H, dtype=torch.float16, device=self._device)
            self._v_packed = torch.zeros(L, N, H, nv, dtype=torch.uint8, device=self._device)
            self._v_norms = torch.zeros(L, N, H, dtype=torch.float16, device=self._device)
        else:
            self._k_packed = torch.zeros(L, N, H, self._inf_nk, dtype=torch.uint8, device=self._device)
            self._k_norms = torch.zeros(L, N, H, dtype=torch.float16, device=self._device)
            if self._use_qjl:
                self._k_qjl_packed = torch.zeros(L, N, H, self._inf_nqjl, dtype=torch.uint8, device=self._device)
                self._k_res_norms = torch.zeros(L, N, H, dtype=torch.float16, device=self._device)
            if self._value_quant_mode == "mse":
                self._v_packed = torch.zeros(L, N, H, self._inf_nv, dtype=torch.uint8, device=self._device)
                self._v_norms = torch.zeros(L, N, H, dtype=torch.float16, device=self._device)
            else:
                ng = self._inf_v_n_groups
                self._v_group_data = torch.zeros(L, N, H, self._inf_v_width, dtype=torch.uint8, device=self._device)
                self._v_group_scales = torch.zeros(L, N, H, ng, dtype=torch.float16, device=self._device)
                self._v_group_zeros = torch.zeros(L, N, H, ng, dtype=torch.float16, device=self._device)

        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

    @staticmethod
    def _bq_to_bhsg(
        slice_packed: torch.Tensor,
        norms: torch.Tensor,
        head_dim: int,
        idx_pad: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
        """(S, H, G) indices and (S, H) norms -> BHSD dict parts for ``MSECompressor.decompress``."""
        s_, h_, _g = slice_packed.shape
        b_bhsd = slice_packed.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        norms_bhs = norms.unsqueeze(0).transpose(1, 2).contiguous()
        return b_bhsd, norms_bhs, (1, h_, s_, head_dim), idx_pad

    @staticmethod
    def _sh_extra_to_bhs(extra_sh: torch.Tensor) -> torch.Tensor:
        """(S, H, G) -> (1, H, S, G)."""
        return extra_sh.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        chunk_len = int(end_idx - start_idx)
        if chunk_len <= 0:
            return
        if k.size(0) != chunk_len or v.size(0) != chunk_len:
            raise ValueError(
                f"TurboQuantRollingKVCachePool.store_kv: chunk_len={chunk_len}, k={k.size(0)}, v={v.size(0)}."
            )

        k_bhsd = k.unsqueeze(0).transpose(1, 2).contiguous()  # [1, H, S, D]
        v_bhsd = v.unsqueeze(0).transpose(1, 2).contiguous()

        if self._turboquant_engine == "legacy":
            with torch.no_grad():
                ck = self._kc(layer_id).compress(k_bhsd)
                cv = self._vc(layer_id).compress(v_bhsd)

            self._k_packed[layer_id, start_idx:end_idx].copy_(ck["idx_bytes"][0].transpose(0, 1).contiguous())
            self._k_norms[layer_id, start_idx:end_idx].copy_(ck["vec_norms"][0].transpose(0, 1).contiguous())

            self._v_packed[layer_id, start_idx:end_idx].copy_(cv["idx_bytes"][0].transpose(0, 1).contiguous())
            self._v_norms[layer_id, start_idx:end_idx].copy_(cv["vec_norms"][0].transpose(0, 1).contiguous())

            if ck["shape"][-1] != self._head_dim or cv["shape"][-1] != self._head_dim:
                raise RuntimeError("TurboQuant compress shape mismatch.")
            return

        with torch.no_grad():
            ck = self._k_mod_inf(layer_id).compress_bhsd(k_bhsd)

        if self._use_qjl:
            self._k_packed[layer_id, start_idx:end_idx].copy_(
                ck["mse_idx_bytes"][0].transpose(0, 1).contiguous()
            )
            self._k_norms[layer_id, start_idx:end_idx].copy_(ck["vec_norms"][0].transpose(0, 1).contiguous())
            self._k_qjl_packed[layer_id, start_idx:end_idx].copy_(
                ck["qjl_bytes"][0].transpose(0, 1).contiguous()
            )
            self._k_res_norms[layer_id, start_idx:end_idx].copy_(
                ck["residual_norms"][0].transpose(0, 1).contiguous()
            )
        else:
            self._k_packed[layer_id, start_idx:end_idx].copy_(ck["idx_bytes"][0].transpose(0, 1).contiguous())
            self._k_norms[layer_id, start_idx:end_idx].copy_(ck["vec_norms"][0].transpose(0, 1).contiguous())

        if ck["shape"][-1] != self._head_dim:
            raise RuntimeError("TurboQuant inference key compress shape mismatch.")

        if self._value_quant_mode == "mse":
            with torch.no_grad():
                cv = self._v_mod_inf(layer_id).compress_bhsd(v_bhsd)
            self._v_packed[layer_id, start_idx:end_idx].copy_(cv["idx_bytes"][0].transpose(0, 1).contiguous())
            self._v_norms[layer_id, start_idx:end_idx].copy_(cv["vec_norms"][0].transpose(0, 1).contiguous())
            if cv["shape"][-1] != self._head_dim:
                raise RuntimeError("TurboQuant inference value compress shape mismatch.")
        else:
            with torch.no_grad():
                cv = _tq_group_quantize_values(v_bhsd, self._value_bits, self._value_group_size)
            self._v_group_data[layer_id, start_idx:end_idx].copy_(cv["data"][0].transpose(0, 1).contiguous())
            self._v_group_scales[layer_id, start_idx:end_idx].copy_(cv["scales"][0].transpose(0, 1).contiguous())
            self._v_group_zeros[layer_id, start_idx:end_idx].copy_(cv["zeros"][0].transpose(0, 1).contiguous())

    def k_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        kv_len = local_end - attn_start
        if kv_len <= 0:
            return torch.empty(0, self._num_heads, self._head_dim, device=self._device, dtype=self._dtype)

        if self._turboquant_engine == "legacy":
            packed = self._k_packed[layer_id, attn_start:local_end]
            norms = self._k_norms[layer_id, attn_start:local_end]
            idx_bytes, norms_bhs, shape, pad = self._bq_to_bhsg(packed, norms, self._head_dim, self._k_idx_pad)
            comp = {"idx_bytes": idx_bytes, "vec_norms": norms_bhs, "shape": shape, "idx_pad": pad}
            with torch.no_grad():
                out_bhsd = self._kc(layer_id).decompress(comp)
            return out_bhsd[0].transpose(0, 1).to(dtype=self._dtype)

        packed = self._k_packed[layer_id, attn_start:local_end]
        norms = self._k_norms[layer_id, attn_start:local_end]
        idx_bytes = packed.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        norms_bhs = norms.unsqueeze(0).transpose(1, 2).contiguous()
        B, H, S, D = 1, self._num_heads, kv_len, self._head_dim

        if self._use_qjl:
            qjl_bhs = self._sh_extra_to_bhs(self._k_qjl_packed[layer_id, attn_start:local_end])
            res_bhs = self._k_res_norms[layer_id, attn_start:local_end].unsqueeze(0).transpose(1, 2).contiguous()
            comp = {
                "mse_idx_bytes": idx_bytes,
                "qjl_bytes": qjl_bhs,
                "residual_norms": res_bhs,
                "vec_norms": norms_bhs,
                "shape": (B, H, S, D),
                "mse_bits": self._key_bits - 1,
            }
            with torch.no_grad():
                out_bhsd = self._k_mod_inf(layer_id).decompress_bhsd(comp)
        else:
            comp = {
                "idx_bytes": idx_bytes,
                "vec_norms": norms_bhs,
                "shape": (B, H, S, D),
                "bits": self._key_bits,
            }
            with torch.no_grad():
                out_bhsd = self._k_mod_inf(layer_id).decompress_bhsd(comp)
        return out_bhsd[0].transpose(0, 1).to(dtype=self._dtype)

    def v_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        kv_len = local_end - attn_start
        if kv_len <= 0:
            return torch.empty(0, self._num_heads, self._head_dim, device=self._device, dtype=self._dtype)

        if self._turboquant_engine == "legacy":
            packed = self._v_packed[layer_id, attn_start:local_end]
            norms = self._v_norms[layer_id, attn_start:local_end]
            idx_bytes, norms_bhs, shape, pad = self._bq_to_bhsg(packed, norms, self._head_dim, self._v_idx_pad)
            comp = {"idx_bytes": idx_bytes, "vec_norms": norms_bhs, "shape": shape, "idx_pad": pad}
            with torch.no_grad():
                out_bhsd = self._vc(layer_id).decompress(comp)
            return out_bhsd[0].transpose(0, 1).to(dtype=self._dtype)

        if self._value_quant_mode == "mse":
            packed = self._v_packed[layer_id, attn_start:local_end]
            norms = self._v_norms[layer_id, attn_start:local_end]
            idx_bytes = packed.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            norms_bhs = norms.unsqueeze(0).transpose(1, 2).contiguous()
            comp = {
                "idx_bytes": idx_bytes,
                "vec_norms": norms_bhs,
                "shape": (1, self._num_heads, kv_len, self._head_dim),
                "bits": self._value_bits,
            }
            with torch.no_grad():
                out_bhsd = self._v_mod_inf(layer_id).decompress_bhsd(comp)
            return out_bhsd[0].transpose(0, 1).to(dtype=self._dtype)

        data = self._v_group_data[layer_id, attn_start:local_end]
        scales = self._v_group_scales[layer_id, attn_start:local_end]
        zeros = self._v_group_zeros[layer_id, attn_start:local_end]
        comp = {
            "data": data.unsqueeze(0).permute(0, 2, 1, 3).contiguous(),
            "scales": scales.unsqueeze(0).transpose(1, 2).contiguous(),
            "zeros": zeros.unsqueeze(0).transpose(1, 2).contiguous(),
            "bits": self._value_bits,
            "group_size": self._value_group_size,
            "shape": (1, self._num_heads, kv_len, self._head_dim),
        }
        with torch.no_grad():
            out_bhsd = _tq_group_dequantize_values(comp)
        return out_bhsd[0].transpose(0, 1).to(dtype=self._dtype)

    def roll_window(
        self,
        layer_id: int,
        sink_tokens: int,
        num_evicted: int,
    ) -> None:
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        if num_kept <= 0:
            return
        src_start = sink_tokens + num_evicted
        src_end = src_start + num_kept
        dst_start = sink_tokens
        dst_end = dst_start + num_kept

        self._k_packed[layer_id, dst_start:dst_end].copy_(self._k_packed[layer_id, src_start:src_end].clone())
        self._k_norms[layer_id, dst_start:dst_end].copy_(self._k_norms[layer_id, src_start:src_end].clone())
        if self._turboquant_engine == "inference" and self._use_qjl:
            self._k_qjl_packed[layer_id, dst_start:dst_end].copy_(
                self._k_qjl_packed[layer_id, src_start:src_end].clone()
            )
            self._k_res_norms[layer_id, dst_start:dst_end].copy_(
                self._k_res_norms[layer_id, src_start:src_end].clone()
            )

        if self._turboquant_engine == "legacy" or self._value_quant_mode == "mse":
            self._v_packed[layer_id, dst_start:dst_end].copy_(self._v_packed[layer_id, src_start:src_end].clone())
            self._v_norms[layer_id, dst_start:dst_end].copy_(self._v_norms[layer_id, src_start:src_end].clone())
        else:
            self._v_group_data[layer_id, dst_start:dst_end].copy_(
                self._v_group_data[layer_id, src_start:src_end].clone()
            )
            self._v_group_scales[layer_id, dst_start:dst_end].copy_(
                self._v_group_scales[layer_id, src_start:src_end].clone()
            )
            self._v_group_zeros[layer_id, dst_start:dst_end].copy_(
                self._v_group_zeros[layer_id, src_start:src_end].clone()
            )

    def reset(self) -> None:
        self._k_packed.zero_()
        self._k_norms.zero_()
        if self._turboquant_engine == "legacy" or self._value_quant_mode == "mse":
            self._v_packed.zero_()
            self._v_norms.zero_()
        if self._turboquant_engine == "inference":
            if self._use_qjl:
                self._k_qjl_packed.zero_()
                self._k_res_norms.zero_()
            if self._value_quant_mode == "group":
                self._v_group_data.zero_()
                self._v_group_scales.zero_()
                self._v_group_zeros.zero_()
        self._global_end.zero_()
        self._local_end.zero_()


class KIVIQuantRollingKVCachePool(RollingKVCachePool):
    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        k_cache_type: str = "int4",
        v_cache_type: str = "int4",
        group_size: int = 64,
        kv_offload: bool = False,
    ) -> None:
        assert k_cache_type in ["int2", "int4", "int8"], f"Invalid k_cache_type: {k_cache_type}"
        assert v_cache_type in ["int2", "int4", "int8"], f"Invalid v_cache_type: {v_cache_type}"
        assert k_cache_type == v_cache_type, "k_cache_type and v_cache_type must be the same"
        self._bits = int(k_cache_type[-1])
        self._group_size = group_size
        self._feats = 32 // self._bits
        self._align = _lcm(self._feats, group_size)
        n_alloc = _cdiv(int(cache_size), self._align) * self._align
        self.current_step: int = 0
        self._N_alloc = n_alloc
        self._kivi_io_dtype = torch.float16
        super().__init__(num_layers, n_alloc, num_heads, head_dim, dtype, device, kv_offload=kv_offload)
    
    @staticmethod
    def _nhd_to_bhdt(nhd: torch.Tensor) -> torch.Tensor:
        return nhd.permute(1, 2, 0).contiguous().unsqueeze(0)

    @staticmethod
    def _slice_token_range(nhd: torch.Tensor, t0: int, t1: int) -> torch.Tensor:
        return nhd[t0:t1, :, :].contiguous()

    def _quant_nhd(
        self,
        nhd: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], int, int]:
        T = nhd.size(0)
        if T == 0:
            raise ValueError("empty K/V chunk in KIVI store")
        T_pad = _cdiv(T, self._align) * self._align
        if nhd.size(0) < T_pad:
            pad = nhd.new_zeros((T_pad - nhd.size(0),) + nhd.shape[1:])
            nhd = torch.cat((nhd, pad), dim=0)
        elif nhd.size(0) > T_pad:
            nhd = nhd[:T_pad]
        t4 = self._nhd_to_bhdt(nhd.to(self._kivi_io_dtype))
        trip = triton_quantize_and_pack_along_last_dim(t4, self._group_size, self._bits)
        return (trip[0], trip[1], trip[2]), T, T_pad

    @staticmethod
    def _dequant_bhdn(
        code4: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        group_size: int,
        bits: int,
        as_dtype: torch.dtype,
    ) -> torch.Tensor:
        # code4 [1, H, D, n_packs], sc/mn [1, H, D, n_groups]
        # Match kernel.test_vcache: last dim of scale/mn must be 1 to broadcast
        # over the (num_groups, group_size) view inside unpack_and_dequant_cache.
        out = unpack_and_dequant_cache(
            code4, sc.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits
        )
        return out.to(as_dtype).squeeze(0)  # [H, D, T]

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        L = self._num_layers
        N = self._N_alloc
        H, D = self._num_heads, self._head_dim
        fe, G = self._feats, self._group_size
        n_packs = N // fe
        n_groups = N // G
        self._kivi_n_packs = n_packs
        self._kivi_n_groups = n_groups
        d = self._device

        self._k_code = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device=d)
        self._v_code = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device=d)
        self._k_scale = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device=d)
        self._k_mn = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_scale = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_mn = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device=d)
        self._global_end = torch.zeros(L, dtype=torch.long, device=d)
        self._local_end = torch.zeros(L, dtype=torch.long, device=d)

    def _init_kv_buffer_offload(self) -> None:
        L = self._num_layers
        N = self._N_alloc
        H, D = self._num_heads, self._head_dim
        fe, G = self._feats, self._group_size
        n_packs = N // fe
        n_groups = N // G
        self._kivi_n_packs = n_packs
        self._kivi_n_groups = n_groups
        d = self._device
        
        self._k_code_cpu = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device="cpu").pin_memory()
        self._v_code_cpu = torch.zeros(L, H, D, n_packs, dtype=torch.int32, device="cpu").pin_memory()
        self._k_scale_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._k_mn_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._v_scale_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._v_mn_cpu = torch.zeros(L, H, D, n_groups, dtype=torch.float32, device="cpu").pin_memory()
        self._k_code_gpu = torch.zeros(2, H, D, n_packs, dtype=torch.int32, device=d)
        self._v_code_gpu = torch.zeros(2, H, D, n_packs, dtype=torch.int32, device=d)
        self._k_scale_gpu = torch.zeros(2, H, D, n_groups, dtype=torch.float32, device=d)
        self._k_mn_gpu = torch.zeros(2, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_scale_gpu = torch.zeros(2, H, D, n_groups, dtype=torch.float32, device=d)
        self._v_mn_gpu = torch.zeros(2, H, D, n_groups, dtype=torch.float32, device=d)
        self._global_end = torch.zeros(L, dtype=torch.long, device=d)
        self._local_end = torch.zeros(L, dtype=torch.long, device=d)

        def _async_load(lid: int, buf: int) -> None:
            self._k_code_gpu[buf].copy_(self._k_code_cpu[lid], non_blocking=True)
            self._v_code_gpu[buf].copy_(self._v_code_cpu[lid], non_blocking=True)
            self._k_scale_gpu[buf].copy_(self._k_scale_cpu[lid], non_blocking=True)
            self._k_mn_gpu[buf].copy_(self._k_mn_cpu[lid], non_blocking=True)
            self._v_scale_gpu[buf].copy_(self._v_scale_cpu[lid], non_blocking=True)
            self._v_mn_gpu[buf].copy_(self._v_mn_cpu[lid], non_blocking=True)

        def _async_store(lid: int, buf: int, t0: int, t1: int) -> None:
            fe, G = self._feats, self._group_size
            p0, p1 = t0 // fe, _cdiv(t1, fe)
            p1 = min(p1, self._kivi_n_packs)
            g0, g1 = t0 // G, _cdiv(t1, G)
            g1 = min(g1, self._kivi_n_groups)
            if p0 < p1:
                self._k_code_cpu[lid, :, :, p0:p1].copy_(
                    self._k_code_gpu[buf, :, :, p0:p1], non_blocking=True
                )
                self._v_code_cpu[lid, :, :, p0:p1].copy_(
                    self._v_code_gpu[buf, :, :, p0:p1], non_blocking=True
                )
            if g0 < g1:
                self._k_scale_cpu[lid, :, :, g0:g1].copy_(
                    self._k_scale_gpu[buf, :, :, g0:g1], non_blocking=True
                )
                self._k_mn_cpu[lid, :, :, g0:g1].copy_(
                    self._k_mn_gpu[buf, :, :, g0:g1], non_blocking=True
                )
                self._v_scale_cpu[lid, :, :, g0:g1].copy_(
                    self._v_scale_gpu[buf, :, :, g0:g1], non_blocking=True
                )
                self._v_mn_cpu[lid, :, :, g0:g1].copy_(
                    self._v_mn_gpu[buf, :, :, g0:g1], non_blocking=True
                )

        self._offload = KVOffloadPlugin(self._device, _async_load, _async_store)
        gpu_mb = (
            self._k_code_gpu.nbytes
            + self._v_code_gpu.nbytes
            + self._k_scale_gpu.nbytes
            + self._k_mn_gpu.nbytes
            + self._v_scale_gpu.nbytes
            + self._v_mn_gpu.nbytes
        ) / (1024 * 1024)
        cpu_mb = (
            self._k_code_cpu.nbytes
            + self._v_code_cpu.nbytes
            + self._k_scale_cpu.nbytes
            + self._k_mn_cpu.nbytes
            + self._v_scale_cpu.nbytes
            + self._v_mn_cpu.nbytes
        ) / (1024 * 1024)
        logger.info(
            "[KIVIQuantRollingKVCachePool+offload] GPU fixed buffer: {:.1f} MB, CPU pinned: {:.1f} MB (saved {:.1f} MB GPU)",
            gpu_mb,
            cpu_mb,
            cpu_mb - gpu_mb,
        )

    def _kivi_k_code(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_code_gpu[self._offload.cur_buf]
        return self._k_code[_layer_id]

    def _kivi_v_code(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_code_gpu[self._offload.cur_buf]
        return self._v_code[_layer_id]

    def _kivi_k_scale(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_scale_gpu[self._offload.cur_buf]
        return self._k_scale[_layer_id]

    def _kivi_k_mn(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_mn_gpu[self._offload.cur_buf]
        return self._k_mn[_layer_id]

    def _kivi_v_scale(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_scale_gpu[self._offload.cur_buf]
        return self._v_scale[_layer_id]

    def _kivi_v_mn(self, _layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_mn_gpu[self._offload.cur_buf]
        return self._v_mn[_layer_id]

    def _write_segment(
        self,
        code: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        layer: int,
        t_start: int,
    ) -> None:
        """Write quant outputs for a chunk placed at **token** ``t_start`` (0-based)."""
        H, D = self._num_heads, self._head_dim
        # code [1, H, D, n_pl], n_pl = T_pad / fe
        b, h, d, np_l = code.shape
        assert b == 1 and h == H and d == D
        fe, G = self._feats, self._group_size
        t_pad = code.shape[3] * fe
        g_cnt = t_pad // G
        t0, t1 = t_start, t_start + t_pad
        p0, p1 = t0 // fe, t0 // fe + code.shape[3]
        g0, g1 = t0 // G, t0 // G + g_cnt
        if t1 > self._N_alloc:
            raise RuntimeError("KIVI store overflow (increase max_attention or alignment)")
        if p0 + code.shape[3] > self._kivi_n_packs:
            raise RuntimeError("KIVI pack range overflow")
        csl = code[0]
        self._kivi_k_code(layer)[:, :, p0:p1] = csl
        self._kivi_k_scale(layer)[:, :, g0:g1] = sc[0, :, :, :g_cnt]
        self._kivi_k_mn(layer)[:, :, g0:g1] = mn[0, :, :, :g_cnt]

    def _write_v_segment(
        self,
        code: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        layer: int,
        t_start: int,
    ) -> None:
        H, D = self._num_heads, self._head_dim
        b, h, d, _np = code.shape
        assert b == 1 and h == H and d == D
        fe, G = self._feats, self._group_size
        t_pad = code.shape[3] * fe
        g_cnt = t_pad // G
        t0, t1 = t_start, t_start + t_pad
        p0, p1 = t0 // fe, t0 // fe + code.shape[3]
        g0, g1 = t0 // G, t0 // G + g_cnt
        if t1 > self._N_alloc:
            raise RuntimeError("KIVI store overflow")
        csl = code[0]
        self._kivi_v_code(layer)[:, :, p0:p1] = csl
        self._kivi_v_scale(layer)[:, :, g0:g1] = sc[0, :, :, :g_cnt]
        self._kivi_v_mn(layer)[:, :, g0:g1] = mn[0, :, :, :g_cnt]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        m = self._align
        Ls = end_idx - start_idx
        if Ls == 0:
            return
        s0 = (start_idx // m) * m
        lid = layer_id
        d = self._kivi_io_dtype
        parts_k = []
        parts_v = []
        if s0 < start_idx:
            pk = self._dequant_nhd(
                self._kivi_k_code(lid),
                self._kivi_k_scale(lid).to(d),
                self._kivi_k_mn(lid).to(d),
                s0,
                start_idx,
            )
            pv = self._dequant_nhd(
                self._kivi_v_code(lid),
                self._kivi_v_scale(lid).to(d),
                self._kivi_v_mn(lid).to(d),
                s0,
                start_idx,
            )
            need = start_idx - s0
            if pk.size(0) < need:
                z = k.new_zeros(need - pk.size(0), *k.shape[1:], dtype=pk.dtype, device=pk.device)
                pk = torch.cat((pk, z), dim=0)
            if pv.size(0) < need:
                z2 = v.new_zeros(need - pv.size(0), *v.shape[1:], dtype=pv.dtype, device=pv.device)
                pv = torch.cat((pv, z2), dim=0)
            parts_k.append(pk)
            parts_v.append(pv)
        parts_k.append(self._slice_token_range(k, 0, Ls))
        parts_v.append(self._slice_token_range(v, 0, Ls))
        k_cat = torch.cat(parts_k, dim=0)
        v_cat = torch.cat(parts_v, dim=0)
        (k_code, k_sc, k_mn), _, t_pad_k = self._quant_nhd(k_cat)
        (v_code, v_sc, v_mn), _, t_pad_v = self._quant_nhd(v_cat)
        if t_pad_k != t_pad_v:
            raise RuntimeError("KIVI store: K/V padded length mismatch")
        self._write_segment(k_code, k_sc, k_mn, layer_id, s0)
        self._write_v_segment(v_code, v_sc, v_mn, layer_id, s0)
        if self._kv_offload:
            self._mark_offload_dirty(s0, s0 + t_pad_k)

    def _dequant_nhd(
        self,
        code: torch.Tensor,
        sc: torch.Tensor,
        mn: torch.Tensor,
        attn_start: int,
        local_end: int,
    ) -> torch.Tensor:
        H, D = self._num_heads, self._head_dim
        m = self._align
        t0 = (attn_start // m) * m
        t1 = min(_cdiv(max(local_end, 0), m) * m, self._N_alloc)
        if t1 <= t0 or local_end <= attn_start:
            return torch.empty(0, H, D, device=self._device, dtype=self._dtype)
        fe, G = self._feats, self._group_size
        p0, p1 = t0 // fe, t1 // fe
        g0, g1 = t0 // G, t1 // G
        c4 = code[:, :, p0:p1].unsqueeze(0)
        out = self._dequant_bhdn(
            c4, sc[:, :, g0:g1].unsqueeze(0), mn[:, :, g0:g1].unsqueeze(0), self._group_size, self._bits, self._dtype
        )
        nhd = out.permute(2, 0, 1)
        o0 = max(attn_start, t0) - t0
        o1 = o0 + (local_end - max(attn_start, t0))
        return nhd[o0:o1].contiguous()

    def k_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        d = self._kivi_io_dtype
        o = self._dequant_nhd(
            self._kivi_k_code(layer_id),
            self._kivi_k_scale(layer_id).to(d),
            self._kivi_k_mn(layer_id).to(d),
            attn_start,
            local_end,
        )
        if self._dtype in (torch.bfloat16, torch.float32) and o.dtype != self._dtype:
            return o.to(self._dtype)
        return o

    def v_cache(self, layer_id: int, attn_start: int, local_end: int) -> torch.Tensor:
        d = self._kivi_io_dtype
        o = self._dequant_nhd(
            self._kivi_v_code(layer_id),
            self._kivi_v_scale(layer_id).to(d),
            self._kivi_v_mn(layer_id).to(d),
            attn_start,
            local_end,
        )
        if self._dtype in (torch.bfloat16, torch.float32) and o.dtype != self._dtype:
            return o.to(self._dtype)
        return o

    def roll_window(
        self,
        layer_id: int,
        sink_tokens: int,
        num_evicted: int,
    ) -> None:
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        src_start = sink_tokens + num_evicted
        dst_start = sink_tokens
        if num_kept <= 0:
            return
        fe, G = self._feats, self._group_size
        t0, t1 = int(src_start), int(src_start + num_kept)
        d0, d1 = int(dst_start), int(dst_start + num_kept)
        p0, p1 = t0 // fe, _cdiv(t1, fe)
        p2, p3 = d0 // fe, _cdiv(d1, fe)
        w = p1 - p0
        if w != p3 - p2 or p0 + w > self._kivi_n_packs or p2 + w > self._kivi_n_packs:
            raise RuntimeError("KIVI roll: pack range mismatch (internal alignment).")
        g0, g1 = t0 // G, _cdiv(t1, G)
        h0, h1 = d0 // G, _cdiv(d1, G)
        w_g = g1 - g0
        if w_g != h1 - h0 or g0 + w_g > self._kivi_n_groups or h0 + w_g > self._kivi_n_groups:
            raise RuntimeError("KIVI roll: group range mismatch (internal alignment).")
        lid = layer_id
        kc, vc = self._kivi_k_code(lid), self._kivi_v_code(lid)
        kc[:, :, p2 : p2 + w] = kc[:, :, p0 : p0 + w].clone()
        vc[:, :, p2 : p2 + w] = vc[:, :, p0 : p0 + w].clone()
        for tbuf in (
            self._kivi_k_scale(lid),
            self._kivi_k_mn(lid),
            self._kivi_v_scale(lid),
            self._kivi_v_mn(lid),
        ):
            tbuf[:, :, h0 : h0 + w_g] = tbuf[:, :, g0 : g0 + w_g].clone()
        if self._kv_offload:
            self._mark_offload_dirty(dst_start, dst_start + num_kept)

    def reset(self) -> None:
        if self._kv_offload:
            self._k_code_cpu.zero_()
            self._v_code_cpu.zero_()
            self._k_scale_cpu.zero_()
            self._k_mn_cpu.zero_()
            self._v_scale_cpu.zero_()
            self._v_mn_cpu.zero_()
            self._k_code_gpu.zero_()
            self._v_code_gpu.zero_()
            self._k_scale_gpu.zero_()
            self._k_mn_gpu.zero_()
            self._v_scale_gpu.zero_()
            self._v_mn_gpu.zero_()
            self._global_end.zero_()
            self._local_end.zero_()
            self._offload.reset_state()
            return
        self._k_code.zero_()
        self._v_code.zero_()
        self._k_scale.zero_()
        self._k_mn.zero_()
        self._v_scale.zero_()
        self._v_mn.zero_()
        self._global_end.zero_()
        self._local_end.zero_()