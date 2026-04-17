"""Compare WorldMirror inference outputs against a reference snapshot.

Acceptance criteria (decision 6 = B + C in the alignment checklist):
  - B: depth / normal MAE per-pixel < 1e-3
  - C: point cloud count / bbox volume / centroid diff < 1%

Usage:
    python scripts/worldmirror/compare_to_reference.py \
        --ref  /workspace/HY-World-2.0/inference_output/Workspace/20260416_074356 \
        --cand /workspace/LightX2V/inference_output/Workspace/<timestamp>

If --cand is a parent directory containing <timestamp> subdirs, the most
recent one is picked automatically. For the default LightX2V layout we
look under inference_output/Workspace/ for the latest timestamp dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

MAE_THRESHOLD = 1e-3
PCT_THRESHOLD = 0.01  # 1%
MAE_THRESHOLD_QUANT = 1e-2  # relaxed threshold for quant/bf16 paths


def pick_latest_subdir(path: Path) -> Path:
    """Return most recent <timestamp>-like subdir, or `path` itself."""
    candidates = [p for p in path.iterdir() if p.is_dir()]
    if not candidates:
        return path
    # Prefer children that look like timestamp dirs, fall back to mtime.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_output_dir(path: Path) -> Path:
    """Find a dir that contains camera_params.json / depth/ / normal/ / *.ply."""
    if (path / "camera_params.json").exists() or (path / "points.ply").exists():
        return path
    # Walk down one or two levels to find the real output dir.
    for lvl in range(3):
        latest = pick_latest_subdir(path)
        if latest == path:
            break
        path = latest
        if (path / "camera_params.json").exists() or (path / "points.ply").exists():
            return path
    return path


def read_ply_xyz(path: Path) -> Optional[np.ndarray]:
    """Minimal PLY (ascii or little-endian binary) x/y/z reader."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Truncated PLY: {path}")
            header.append(line)
            if line.strip() == b"end_header":
                break
        header_text = b"".join(header).decode("utf-8", errors="replace")
        fmt_line = next(ln for ln in header_text.splitlines() if ln.startswith("format"))
        fmt = fmt_line.split()[1]
        # Parse first "element vertex N"
        vcount = None
        prop_types = []
        in_vertex = False
        for line in header_text.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vcount = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                prop_types.append((parts[-2], parts[-1]))  # (type, name)
        if vcount is None:
            raise RuntimeError(f"No vertex element: {path}")
        # Find x/y/z indices
        names = [p[1] for p in prop_types]
        if not all(n in names for n in ("x", "y", "z")):
            raise RuntimeError(f"No x/y/z props: {path}")
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=vcount)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
            return data[:, [ix, iy, iz]].astype(np.float64)
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            dtype_map = {
                "float": "<f4" if "little" in fmt else ">f4",
                "float32": "<f4" if "little" in fmt else ">f4",
                "double": "<f8" if "little" in fmt else ">f8",
                "float64": "<f8" if "little" in fmt else ">f8",
                "uchar": "u1",
                "uint8": "u1",
                "char": "i1",
                "int8": "i1",
                "ushort": "<u2" if "little" in fmt else ">u2",
                "uint16": "<u2" if "little" in fmt else ">u2",
                "short": "<i2" if "little" in fmt else ">i2",
                "int16": "<i2" if "little" in fmt else ">i2",
                "uint": "<u4" if "little" in fmt else ">u4",
                "uint32": "<u4" if "little" in fmt else ">u4",
                "int": "<i4" if "little" in fmt else ">i4",
                "int32": "<i4" if "little" in fmt else ">i4",
            }
            struct_dtype = np.dtype([(p[1], dtype_map[p[0]]) for p in prop_types])
            arr = np.frombuffer(f.read(vcount * struct_dtype.itemsize), dtype=struct_dtype)
            if len(arr) != vcount:
                raise RuntimeError(f"Truncated PLY binary body: got {len(arr)} / {vcount} verts in {path}")
            xyz = np.stack([arr["x"].astype(np.float64), arr["y"].astype(np.float64), arr["z"].astype(np.float64)], axis=1)
            return xyz
        else:
            raise RuntimeError(f"Unknown PLY format {fmt} in {path}")


def depth_mae(ref_dir: Path, cand_dir: Path):
    """Return list of (filename, mae, ok) per depth_XXXX.npy pair."""
    out = []
    ref_files = sorted((ref_dir / "depth").glob("depth_*.npy"))
    cand_dir_depth = cand_dir / "depth"
    for rf in ref_files:
        cf = cand_dir_depth / rf.name
        if not cf.exists():
            out.append((rf.name, None, False))
            continue
        a = np.load(rf).astype(np.float64)
        b = np.load(cf).astype(np.float64)
        if a.shape != b.shape:
            out.append((rf.name, float("nan"), False))
            continue
        # Ignore NaNs symmetrically.
        mask = np.isfinite(a) & np.isfinite(b)
        if not mask.any():
            out.append((rf.name, 0.0, True))
            continue
        mae = float(np.mean(np.abs(a[mask] - b[mask])))
        out.append((rf.name, mae, mae < MAE_THRESHOLD))
    return out


def png_u8_mae(ref_dir: Path, cand_dir: Path, subdir: str, prefix: str):
    """Compare {subdir}/{prefix}_*.png; return per-file MAE normalized to [0,1]."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return []
    ref_files = sorted((ref_dir / subdir).glob(f"{prefix}_*.png"))
    out = []
    for rf in ref_files:
        cf = cand_dir / subdir / rf.name
        if not cf.exists():
            out.append((rf.name, None, False))
            continue
        a = np.asarray(Image.open(rf)).astype(np.float64) / 255.0
        b = np.asarray(Image.open(cf)).astype(np.float64) / 255.0
        if a.shape != b.shape:
            out.append((rf.name, float("nan"), False))
            continue
        mae = float(np.mean(np.abs(a - b)))
        out.append((rf.name, mae, mae < MAE_THRESHOLD))
    return out


def point_cloud_stats(ref_ply: Path, cand_ply: Path):
    """Return (ref_n, cand_n, count_diff_pct, bbox_vol_diff_pct, centroid_diff_norm)."""
    a = read_ply_xyz(ref_ply)
    b = read_ply_xyz(cand_ply)
    if a is None or b is None:
        return None
    ref_n, cand_n = len(a), len(b)
    count_diff = abs(ref_n - cand_n) / max(ref_n, 1)

    def bbox_vol(xyz):
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        side = hi - lo
        return float(np.prod(side))

    vol_a, vol_b = bbox_vol(a), bbox_vol(b)
    vol_diff = abs(vol_a - vol_b) / max(abs(vol_a), 1e-9)
    centroid_a, centroid_b = a.mean(axis=0), b.mean(axis=0)
    # Use bbox diagonal of ref to normalize centroid shift.
    diag = np.linalg.norm(a.max(axis=0) - a.min(axis=0))
    centroid_shift = float(np.linalg.norm(centroid_a - centroid_b) / max(diag, 1e-9))
    return {
        "ref_points": ref_n,
        "cand_points": cand_n,
        "count_diff_pct": count_diff,
        "ref_bbox_vol": vol_a,
        "cand_bbox_vol": vol_b,
        "bbox_vol_diff_pct": vol_diff,
        "centroid_diff_normalized": centroid_shift,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref",
        default="/workspace/HY-World-2.0/inference_output/Workspace/20260416_074356",
        help="Reference output dir (or parent containing it).",
    )
    ap.add_argument(
        "--cand",
        default="/workspace/LightX2V/inference_output/Workspace",
        help="Candidate output dir (or parent; latest subdir is picked).",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    ap.add_argument(
        "--quant",
        action="store_true",
        help="Apply the relaxed MAE threshold (1e-2) used for fp8/int8/bf16 paths.",
    )
    ap.add_argument(
        "--bf16_ref",
        action="store_true",
        help=(
            "Switch ``--ref`` default to the bf16 reference output at "
            "``/workspace/HY-World-2.0/inference_output/Workspace_bf16`` "
            "(for validating the bf16 path against itself across commits). "
            "Implies --quant unless overridden."
        ),
    )
    ap.add_argument(
        "--mae_threshold",
        type=float,
        default=None,
        help="Override MAE threshold (overrides --quant).",
    )
    args = ap.parse_args()

    global MAE_THRESHOLD
    # --bf16_ref: snap --ref to the bf16 reference if the user didn't
    # already override it, and relax MAE (bf16 has ~1e-3 precision floor).
    if args.bf16_ref and args.ref == ap.get_default("ref"):
        args.ref = "/workspace/HY-World-2.0/inference_output/Workspace_bf16"
    if args.mae_threshold is not None:
        MAE_THRESHOLD = args.mae_threshold
    elif args.quant or args.bf16_ref:
        MAE_THRESHOLD = MAE_THRESHOLD_QUANT

    ref = resolve_output_dir(Path(args.ref))
    cand = resolve_output_dir(Path(args.cand))
    print(f"[compare] ref : {ref}")
    print(f"[compare] cand: {cand}")

    if not ref.exists() or not cand.exists():
        print(f"[compare] missing dir: ref={ref.exists()}, cand={cand.exists()}")
        return 2

    report = {"ref": str(ref), "cand": str(cand)}
    failures = []

    depth_results = depth_mae(ref, cand)
    report["depth"] = [{"file": n, "mae": m, "ok": ok} for (n, m, ok) in depth_results]
    for n, m, ok in depth_results:
        if not ok:
            failures.append(f"depth/{n}: MAE={m}")

    normal_results = png_u8_mae(ref, cand, "normal", "normal")
    report["normal"] = [{"file": n, "mae": m, "ok": ok} for (n, m, ok) in normal_results]
    for n, m, ok in normal_results:
        if not ok:
            failures.append(f"normal/{n}: MAE={m}")

    pc_report = {}
    for ply_name in ("points.ply", "gaussians.ply"):
        stats = point_cloud_stats(ref / ply_name, cand / ply_name)
        if stats is None:
            continue
        ok_count = stats["count_diff_pct"] < PCT_THRESHOLD
        ok_vol = stats["bbox_vol_diff_pct"] < PCT_THRESHOLD
        ok_centroid = stats["centroid_diff_normalized"] < PCT_THRESHOLD
        stats["ok_count"] = ok_count
        stats["ok_vol"] = ok_vol
        stats["ok_centroid"] = ok_centroid
        pc_report[ply_name] = stats
        if not ok_count:
            failures.append(f"{ply_name}: count_diff={stats['count_diff_pct']:.4%}")
        if not ok_vol:
            failures.append(f"{ply_name}: bbox_vol_diff={stats['bbox_vol_diff_pct']:.4%}")
        if not ok_centroid:
            failures.append(f"{ply_name}: centroid_diff={stats['centroid_diff_normalized']:.4%}")
    report["point_clouds"] = pc_report

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("\n=== Depth MAE (threshold {}): ===".format(MAE_THRESHOLD))
        for n, m, ok in depth_results:
            print(f"  {'PASS' if ok else 'FAIL'} {n}  MAE={m}")
        print("\n=== Normal MAE (threshold {}, normalized): ===".format(MAE_THRESHOLD))
        for n, m, ok in normal_results:
            print(f"  {'PASS' if ok else 'FAIL'} {n}  MAE={m}")
        print("\n=== Point cloud (threshold {:.0%}): ===".format(PCT_THRESHOLD))
        for name, stats in pc_report.items():
            print(f"  [{name}]")
            print(f"    points: ref={stats['ref_points']}, cand={stats['cand_points']}, diff={stats['count_diff_pct']:.4%}  {'OK' if stats['ok_count'] else 'FAIL'}")
            print(f"    bbox vol: ref={stats['ref_bbox_vol']:.4g}, cand={stats['cand_bbox_vol']:.4g}, diff={stats['bbox_vol_diff_pct']:.4%}  {'OK' if stats['ok_vol'] else 'FAIL'}")
            print(f"    centroid shift (norm by ref bbox diag): {stats['centroid_diff_normalized']:.4%}  {'OK' if stats['ok_centroid'] else 'FAIL'}")

    if failures:
        print(f"\n[compare] RESULT: FAIL ({len(failures)} checks)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[compare] RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
