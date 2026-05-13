# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# LTX-2.3 IC-LoRA reference-video preprocessing for LightX2V ``v2av``:
#   - pose   — DWPose-ONNX skeleton (Pose-Control / Union-Control; same as community DWPose path).
#   - canny  — OpenCV GaussianBlur + Canny → RGB edges (common in LTX-Video-Trainer / docs).
#   - depth  — MiDaS-small via torch.hub (monocular depth map, grayscale triple; aligns with
#              LTX docs listing monocular depth tools; for production some pipelines use DepthCrafter).
#
# Example (pose):
#   python ltx2.3_preprocess_data.py --mode pose \
#       --ckpt_path  /mnt/ckpts/dwpose_zoo \
#       --video_path /mnt/data/driving.mp4 \
#       --save_path  /mnt/data/driving_pose.mp4 \
#       --resolution_area 1280 720 --fps 24 --include_hands
#
# Example (canny / depth; no DWPose ckpt):
#   python ltx2.3_preprocess_data.py --mode canny --video_path in.mp4 --save_path out_dir/
#   python ltx2.3_preprocess_data.py --mode depth --video_path in.mp4 --save_path depth.mp4 --device cuda

import argparse
import os
import sys
from typing import List

import av
import cv2
import numpy as np
from PIL import Image
from decord import VideoReader
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dwpose_onnx import DWposeONNX
from utils import get_frame_indices, padding_resize, resize_by_area

MODE_DEFAULT_OUT = {"pose": "pose_skeleton.mp4", "canny": "canny_control.mp4", "depth": "depth_control.mp4"}


def get_preprocess_parser():
    parser = argparse.ArgumentParser(description="LTX-2.3 IC-LoRA control-video preprocessing (pose / canny / depth) for v2av.")

    parser.add_argument(
        "--mode",
        type=str,
        default="pose",
        choices=["pose", "canny", "depth"],
        help="pose: DWPose skeleton (needs --ckpt_path). canny: OpenCV Canny edges (Union-Control). depth: MiDaS-small monocular depth via torch.hub (Union-Control; first run downloads weights).",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Required when --mode pose: DWPose ONNX root with det/yolox_l.onnx and pose2d/dw-ll_ucoco_384.onnx (see yzd-v/DWPose). Ignored for canny/depth.",
    )

    parser.add_argument("--video_path", type=str, default=None, help="The path to the driving video.")
    parser.add_argument(
        "--refer_path",
        type=str,
        default=None,
        help="Optional path to the character/reference image (the human appearance the "
        "downstream LTX-2.3 v2av task will animate). When provided, the skeleton canvas "
        "aspect ratio is taken from this image; source video "
        "frames are letterboxed into the canvas before DWPose runs. This guarantees a "
        "portrait image yields a portrait output, etc. When omitted, the canvas falls "
        "back to the source video aspect ratio.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output control video (.mp4). If a directory is given, writes pose_skeleton.mp4 / canny_control.mp4 / depth_control.mp4 depending on --mode.",
    )

    parser.add_argument(
        "--resolution_area",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="The target resolution area for the skeleton video, specified as "
        "[width, height]. The video (or, if --refer_path is given, the image) is "
        "resized to have a total area equivalent to width * height while preserving "
        "its original aspect ratio. Both dimensions are snapped to a multiple of 32 "
        "for LTX-2.3.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target FPS for the output skeleton video. Set to -1 to keep the source FPS. LTX-2.3 IC-LoRA is typically trained at 24 FPS.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=-1,
        help="Maximum number of skeleton frames to write. -1 = use full video. The final count is silently snapped to LTX-2.3's `8k + 1` rule (e.g. 97, 193).",
    )

    parser.add_argument(
        "--include_hands",
        action="store_true",
        default=False,
        help="Render 21-point hand skeletons in addition to the body. Recommended for hand-intensive motions (sign language, gestures).",
    )
    parser.add_argument(
        "--include_face",
        action="store_true",
        default=False,
        help="Render the 68-point face landmarks. Usually NOT needed for body-motion transfer; enable only if the IC-LoRA explicitly requires face control.",
    )

    parser.add_argument(
        "--bg_mode",
        type=str,
        default="black",
        choices=["black", "source", "blend"],
        help="Background of the skeleton video.\n"
        "  black  - pure black background (matches LTX-2.3 Pose-Control training).\n"
        "  source - keep the source pixels where the skeleton is not drawn.\n"
        "  blend  - alpha-blend the skeleton over a dimmed source (debug only).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device: pose (DWPose ONNX), depth (torch MiDaS). Examples: cuda, cuda:0, cpu.",
    )

    parser.add_argument(
        "--canny_blur",
        type=int,
        default=5,
        help="Gaussian blur kernel size before Canny (odd ≥3; 0 disables blur). LTX docs suggest slight blur to reduce noise.",
    )
    parser.add_argument(
        "--canny_threshold1",
        type=int,
        default=100,
        help="OpenCV Canny first threshold.",
    )
    parser.add_argument(
        "--canny_threshold2",
        type=int,
        default=200,
        help="OpenCV Canny second threshold.",
    )

    return parser


def snap_to_8k_plus_1(n: int) -> int:
    """LTX-2.3 VAE-stride constraint: video length must satisfy (n - 1) % 8 == 0."""
    if n < 1:
        return 1
    k = max(0, round((n - 1) / 8))
    return 8 * k + 1


class DWPosePipeline:
    def __init__(self, det_checkpoint_path: str, pose2d_checkpoint_path: str, device: str = "cuda"):
        self.detector = DWposeONNX(
            det_onnx_path=det_checkpoint_path,
            pose_onnx_path=pose2d_checkpoint_path,
            device=device,
        )
        self.device = device
        logger.info(f"DWPose-ONNX initialized on {device}")

    def __call__(
        self,
        frames_rgb: List[np.ndarray],
        include_hands: bool = False,
        include_face: bool = False,
        bg_mode: str = "black",
    ) -> List[np.ndarray]:
        out = []
        for idx, frame in enumerate(frames_rgb):
            skeleton = self.detector(
                frame,
                include_hands=include_hands,
                include_face=include_face,
                bg_mode=bg_mode,
            )
            if skeleton.shape[:2] != frame.shape[:2]:
                skeleton = cv2.resize(skeleton, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            out.append(skeleton)

            if (idx + 1) % 50 == 0:
                logger.info(f"DWPose: {idx + 1}/{len(frames_rgb)} frames")
        return out


def run_canny_edges(
    frames_rgb: List[np.ndarray],
    blur_ksize: int,
    threshold1: int,
    threshold2: int,
) -> List[np.ndarray]:
    """Per-frame OpenCV Canny on RGB frames → RGB uint8 (white edges on black)."""
    k = int(blur_ksize)
    if k > 0 and k % 2 == 0:
        k += 1
    use_blur = k >= 3
    out: List[np.ndarray] = []
    for idx, frame in enumerate(frames_rgb):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if use_blur:
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(gray, threshold1, threshold2)
        rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        out.append(rgb)
        if (idx + 1) % 50 == 0:
            logger.info(f"Canny: {idx + 1}/{len(frames_rgb)} frames")
    return out


def run_depth_midas_small(frames_rgb: List[np.ndarray], device: str) -> List[np.ndarray]:
    try:
        import torch
    except ImportError as e:
        raise ImportError("depth mode requires PyTorch (torch). Install the project requirements.txt.") from e

    if device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
    elif device.startswith("cuda"):
        logger.warning("CUDA requested for depth but not available; using CPU.")
        dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    logger.info("Loading MiDaS-small from torch.hub (first run may download weights)...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.to(dev).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    out: List[np.ndarray] = []
    with torch.no_grad():
        for idx, frame in enumerate(frames_rgb):
            h, w = frame.shape[:2]
            im = Image.fromarray(frame)
            inp = transform(im).to(dev)
            pred = midas(inp)
            if pred.ndim == 2:
                pred = pred.unsqueeze(0)
            pred = pred.unsqueeze(1)
            pred = torch.nn.functional.interpolate(
                pred,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            d = pred.squeeze().cpu().numpy().astype(np.float32)
            dmin, dmax = float(d.min()), float(d.max())
            if dmax - dmin < 1e-6:
                u8 = np.zeros((h, w), dtype=np.uint8)
            else:
                u8 = ((d - dmin) / (dmax - dmin) * 255.0).clip(0, 255).astype(np.uint8)
            rgb = np.stack([u8, u8, u8], axis=-1)
            out.append(rgb)
            if (idx + 1) % 20 == 0:
                logger.info(f"Depth (MiDaS-small): {idx + 1}/{len(frames_rgb)} frames")

    del midas
    if dev.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return out


def _count_video_frames(path: str) -> int:
    """Count the actual decodable frame count of an mp4 (PyAV demux)."""
    c = av.open(path)
    try:
        s = next(x for x in c.streams if x.type == "video")
        n = sum(1 for _ in c.decode(s))
    finally:
        c.close()
    return n


def _save_video(frames_rgb: List[np.ndarray], out_path: str, fps: float):
    if len(frames_rgb) == 0:
        raise ValueError("No frames to save.")

    h, w = frames_rgb[0].shape[:2]
    if h % 2 == 1 or w % 2 == 1:
        h, w = h - (h % 2), w - (w % 2)
        frames_rgb = [f[:h, :w] for f in frames_rgb]

    def _encode(frames: List[np.ndarray]):
        container = av.open(out_path, mode="w")
        try:
            stream = container.add_stream("libx264", rate=int(round(fps)))
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            for frame_array in frames:
                frame_array = np.ascontiguousarray(frame_array.astype(np.uint8))
                video_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

    _encode(frames_rgb)

    written = _count_video_frames(out_path)
    expected = len(frames_rgb)
    if written == expected:
        return

    if written < expected:
        logger.warning(f"PyAV/libx264 produced {written} frames but {expected} requested. Padding by repeating the last frame and re-encoding to keep the 1+8k contract.")
        pad = expected - written
        padded = frames_rgb + [frames_rgb[-1]] * pad
        _encode(padded)
        new_written = _count_video_frames(out_path)
        if new_written != expected:
            raise RuntimeError(f"Failed to write requested frame count after padding: requested={expected}, written={new_written}.")
    else:
        raise RuntimeError(f"PyAV wrote more frames than requested: requested={expected}, written={written}.")


def process_input_video(args):
    args_dict = vars(args)
    print(args_dict)

    assert len(args.resolution_area) == 2, "resolution_area should be a list of two integers [width, height]"
    assert args.video_path is not None, "--video_path is required"
    assert args.save_path is not None, "--save_path is required"
    if args.mode == "pose":
        assert args.ckpt_path is not None, "--ckpt_path is required when --mode pose"

    pipeline = None
    if args.mode == "pose":
        det_checkpoint_path = os.path.join(args.ckpt_path, "det/yolox_l.onnx")
        pose2d_checkpoint_path = os.path.join(args.ckpt_path, "pose2d/dw-ll_ucoco_384.onnx")
        pipeline = DWPosePipeline(
            det_checkpoint_path=det_checkpoint_path,
            pose2d_checkpoint_path=pose2d_checkpoint_path,
            device=args.device,
        )

    video_reader = VideoReader(args.video_path)
    frame_num = len(video_reader)
    video_fps = video_reader.get_avg_fps()
    print(f"frame_num: {frame_num}")
    print(f"video_fps: {video_fps}")

    duration = video_reader.get_frame_timestamp(-1)[-1]
    expected_frame_num = int(duration * video_fps + 0.5)
    ratio = abs((frame_num - expected_frame_num) / max(frame_num, 1))
    if ratio > 0.1:
        print("Warning: actual frame count differs from expected by >10%; using duration-based estimate.")
        frame_num = expected_frame_num

    target_fps = video_fps if args.fps == -1 else args.fps
    if args.num_frames > 0:
        target_num = args.num_frames
    else:
        target_num = int(frame_num / video_fps * target_fps)
    target_num = max(snap_to_8k_plus_1(target_num), 1)
    upper_bound = snap_to_8k_plus_1(int(frame_num / video_fps * target_fps))
    target_num = min(target_num, upper_bound)
    print(f"target_num (snapped to 8k+1): {target_num}")

    idxs = get_frame_indices(frame_num, video_fps, target_num, target_fps)
    frames = video_reader.get_batch(idxs).asnumpy()

    target_area = args.resolution_area[0] * args.resolution_area[1]

    if args.refer_path:
        logger.info(f"Using --refer_path to drive canvas aspect: {args.refer_path}")
        refer_bgr = cv2.imread(args.refer_path)
        if refer_bgr is None:
            raise ValueError(f"Failed to read --refer_path: {args.refer_path!r}")
        refer_rgb = refer_bgr[..., ::-1]
        refer_rgb = resize_by_area(refer_rgb, target_area, divisor=32)
        height, width = refer_rgb.shape[:2]
        # Fit each driving frame into the (height, width) canvas with letterboxing.
        frames = [padding_resize(f, height=height, width=width) for f in frames]
        logger.info(f"Control canvas {width}x{height} (from reference image), {len(frames)} frames letterboxed into it.")
    else:
        frames = [resize_by_area(f, target_area, divisor=32) for f in frames]
        height, width = frames[0].shape[:2]
        logger.info(f"Control canvas {width}x{height} (from driving video), {len(frames)} frames.")

    if args.mode == "pose":
        assert pipeline is not None
        logger.info("Running DWPose skeleton extraction (--mode pose)")
        out_frames = pipeline(
            frames,
            include_hands=args.include_hands,
            include_face=args.include_face,
            bg_mode=args.bg_mode,
        )
    elif args.mode == "canny":
        logger.info(f"Running OpenCV Canny (--mode canny), blur={args.canny_blur}, thresholds=({args.canny_threshold1}, {args.canny_threshold2})")
        out_frames = run_canny_edges(
            frames,
            blur_ksize=args.canny_blur,
            threshold1=args.canny_threshold1,
            threshold2=args.canny_threshold2,
        )
    elif args.mode == "depth":
        logger.info("Running MiDaS-small depth (--mode depth)")
        out_frames = run_depth_midas_small(frames, args.device)
    else:
        raise ValueError(f"Unknown --mode: {args.mode!r}")

    default_name = MODE_DEFAULT_OUT[args.mode]
    if os.path.isdir(args.save_path) or args.save_path.endswith(os.sep):
        os.makedirs(args.save_path, exist_ok=True)
        out_path = os.path.join(args.save_path, default_name)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".", exist_ok=True)
        out_path = args.save_path

    _save_video(out_frames, out_path, fps=target_fps)

    # Sidecar with the on-disk frame count, so launch scripts can derive
    # `target_video_length` and `fps` exactly from the preprocessed mp4 without
    # having to re-probe it themselves. Format: ``<frames>\n<fps>\n``.
    written_frames = _count_video_frames(out_path)
    sidecar_path = out_path + ".meta"
    with open(sidecar_path, "w") as f:
        f.write(f"frames={written_frames}\n")
        f.write(f"fps={int(round(target_fps))}\n")
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")

    logger.info(f"Control video ({args.mode}) saved to: {out_path}  (frames={written_frames}, fps={int(round(target_fps))})")
    logger.info(f"Sidecar metadata: {sidecar_path}")
    logger.info(
        "Feed this file as `--video_path` to LightX2V v2av with e.g.\n"
        "  - ltx2.3_v2av_motion_pose.json   (Pose-Control IC-LoRA; pose mode only)\n"
        "  - ltx2.3_v2av_motion_union.json  (Union-Control IC-LoRA; pose / canny / depth)"
    )


if __name__ == "__main__":
    parser = get_preprocess_parser()
    args = parser.parse_args()
    process_input_video(args)
