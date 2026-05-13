# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

# official ONNX reference implementation:
#   https://github.com/IDEA-Research/DWPose/tree/onnx/ControlNet-v1-1-nightly/annotator/dwpose
#
# Expected checkpoint layout:
#   <ckpt_path>/det/yolox_l.onnx
#   <ckpt_path>/pose2d/dw-ll_ucoco_384.onnx
#
# Download:
#   https://huggingface.co/yzd-v/DWPose

from __future__ import annotations

import math
import os
from typing import Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    raise ImportError("onnxruntime is required by dwpose_onnx. Install via: pip install onnxruntime-gpu  (or onnxruntime on CPU-only hosts).") from e


# -----------------------------------------------------------------------------
# YOLOX detector  (person class only)
# -----------------------------------------------------------------------------


def _yolox_preprocess(img: np.ndarray, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float]:
    """Letterbox + transpose to NCHW float32 for YOLOX."""
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    rh, rw = int(img.shape[0] * r), int(img.shape[1] * r)
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded[:rh, :rw] = resized
    padded = padded.transpose(2, 0, 1)  # HWC -> CHW
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    return padded, r


def _yolox_demo_postprocess(outputs: np.ndarray, img_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Decode anchor-free YOLOX outputs to (cx, cy, w, h, obj, *cls) format."""
    grids = []
    expanded_strides = []
    strides = [8, 16, 32]
    hsizes = [img_size[0] // s for s in strides]
    wsizes = [img_size[1] // s for s in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((1, grid.shape[1], 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Pure-NumPy NMS, returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def _multiclass_nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float, num_classes: int = 80) -> np.ndarray | None:
    """Per-class NMS; returns ``[x1, y1, x2, y2, score, cls]`` array."""
    final = []
    for c in range(num_classes):
        cls_scores = scores[:, c]
        mask = cls_scores > score_thr
        if not mask.any():
            continue
        b = boxes[mask]
        s = cls_scores[mask]
        keep = _nms(b, s, nms_thr)
        if keep.size == 0:
            continue
        dets = np.concatenate([b[keep], s[keep, None], np.full((keep.size, 1), c, dtype=np.float32)], axis=1)
        final.append(dets)
    if not final:
        return None
    return np.concatenate(final, 0)


def detect_persons(
    session: ort.InferenceSession,
    img_bgr: np.ndarray,
    score_thr: float = 0.3,
    nms_thr: float = 0.45,
) -> np.ndarray:
    """Run YOLOX, keep COCO ``person`` class only.

    Returns:
        ``(N, 4)`` xyxy boxes in original image coordinates.
    """
    in_shape = session.get_inputs()[0].shape  # ['1', '3', H, W]
    H, W = int(in_shape[2]), int(in_shape[3])
    inp, ratio = _yolox_preprocess(img_bgr, (H, W))
    out = session.run(None, {session.get_inputs()[0].name: inp[None]})[0]
    out = _yolox_demo_postprocess(out, (H, W))[0]

    # (cx, cy, w, h, obj, *cls)
    boxes_xywh = out[:, :4]
    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    boxes_xyxy /= ratio

    obj = out[:, 4:5]
    cls = out[:, 5:]
    scores = obj * cls

    dets = _multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr, num_classes=cls.shape[1])
    if dets is None:
        return np.zeros((0, 4), dtype=np.float32)
    # class 0 = person
    keep = dets[:, 5] == 0
    return dets[keep, :4].astype(np.float32)


# -----------------------------------------------------------------------------
# RTMPose pose estimator  (133 keypoints, SimCC decoding)
# -----------------------------------------------------------------------------


def _get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, output_size: Tuple[int, int]) -> np.ndarray:
    """Affine matrix used by RTMPose / MMPose top-down models."""
    shift = np.zeros(2, dtype=np.float32)
    src_w = scale[0]
    dst_w, dst_h = output_size
    rot_rad = np.deg2rad(rot)
    src_dir = np.array([0.0, src_w * -0.5], dtype=np.float32)
    sn, cs = math.sin(rot_rad), math.cos(rot_rad)
    src_dir = np.array([src_dir[0] * cs - src_dir[1] * sn, src_dir[0] * sn + src_dir[1] * cs], dtype=np.float32)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center + scale * shift
    src[1] = center + src_dir + scale * shift
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2] = src[1] + np.array([-src_dir[1], src_dir[0]], dtype=np.float32)
    dst[2] = dst[1] + np.array([-dst_dir[1], dst_dir[0]], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def _bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.25) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1) * padding, (y2 - y1) * padding
    aspect = 192.0 / 256.0  # DWPose 384 input keeps 3:4 aspect (288/384)
    aspect = 288.0 / 384.0
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return np.array([cx, cy], dtype=np.float32), np.array([w, h], dtype=np.float32)


def _simcc_decode(simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio: float = 2.0):
    """Decode SimCC (1D classification) outputs to (locs, scores).

    simcc_x : (N, K, Wx),  simcc_y : (N, K, Hy)
    """
    N, K, Wx = simcc_x.shape
    locs = np.zeros((N, K, 2), dtype=np.float32)
    scores = np.zeros((N, K), dtype=np.float32)
    x_locs = simcc_x.argmax(axis=2)
    y_locs = simcc_y.argmax(axis=2)
    locs[:, :, 0] = x_locs.astype(np.float32) / simcc_split_ratio
    locs[:, :, 1] = y_locs.astype(np.float32) / simcc_split_ratio
    max_x = simcc_x.max(axis=2)
    max_y = simcc_y.max(axis=2)
    scores[:, :] = np.minimum(max_x, max_y)
    locs[scores <= 0] = -1
    return locs, scores


def estimate_pose(
    session: ort.InferenceSession,
    img_bgr: np.ndarray,
    person_boxes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run DWPose on each detected person.

    Returns:
        keypoints: ``(N, 133, 2)`` in image coords (xy).
        scores:    ``(N, 133)``.
    """
    if person_boxes.shape[0] == 0:
        return np.zeros((0, 133, 2), dtype=np.float32), np.zeros((0, 133), dtype=np.float32)

    in_shape = session.get_inputs()[0].shape
    H, W = int(in_shape[2]), int(in_shape[3])

    centers, scales, warps = [], [], []
    crops = []
    for box in person_boxes:
        c, s = _bbox_xyxy2cs(box)
        M = _get_warp_matrix(c, s, 0.0, (W, H))
        crop = cv2.warpAffine(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR)
        crops.append(crop)
        centers.append(c)
        scales.append(s)
        warps.append(M)

    batch = np.stack(crops, 0).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    batch = (batch - mean) / std
    batch = batch.transpose(0, 3, 1, 2)
    batch = np.ascontiguousarray(batch, dtype=np.float32)

    in_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    outs = session.run(out_names, {in_name: batch})
    simcc_x, simcc_y = outs[0], outs[1]
    locs_in_crop, scores = _simcc_decode(simcc_x, simcc_y, simcc_split_ratio=2.0)

    # Map crop-space keypoints back to original image space
    keypoints = np.zeros_like(locs_in_crop, dtype=np.float32)
    for i, (M, kp) in enumerate(zip(warps, locs_in_crop)):
        inv = cv2.invertAffineTransform(M)
        ones = np.ones((kp.shape[0], 1), dtype=np.float32)
        homog = np.concatenate([kp, ones], axis=1)  # (K, 3)
        keypoints[i] = homog @ inv.T

    return keypoints, scores


# -----------------------------------------------------------------------------
# OpenPose-style skeleton renderer
# -----------------------------------------------------------------------------
#
# DWPose 133 keypoints layout (COCO-WholeBody):
#   0-16    : 17 body (COCO)
#   17-22   : 6 feet
#   23-90   : 68 face
#   91-111  : 21 left hand
#   112-132 : 21 right hand

# COCO-17 -> OpenPose-18 (with computed Neck) mapping
_BODY_COCO_TO_OPENPOSE = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

_BODY_LIMBS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
]
_BODY_COLORS = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]

_HAND_LIMBS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]


def _draw_body(canvas: np.ndarray, body_kp_18: np.ndarray, body_score_18: np.ndarray, threshold: float = 0.3):
    H, W = canvas.shape[:2]
    stickwidth = 4
    for limb_idx, (a, b) in enumerate(_BODY_LIMBS):
        if body_score_18[a] < threshold or body_score_18[b] < threshold:
            continue
        x1, y1 = body_kp_18[a]
        x2, y2 = body_kp_18[b]
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            continue
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        length = math.hypot(x1 - x2, y1 - y2)
        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
        poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, _BODY_COLORS[limb_idx])
    for i in range(18):
        if body_score_18[i] < threshold:
            continue
        x, y = body_kp_18[i]
        if x < 0 or y < 0:
            continue
        cv2.circle(canvas, (int(x), int(y)), 4, _BODY_COLORS[i], thickness=-1)


def _draw_hand(canvas: np.ndarray, hand_kp_21: np.ndarray, hand_score_21: np.ndarray, threshold: float = 0.3):
    for limb_idx, (a, b) in enumerate(_HAND_LIMBS):
        if hand_score_21[a] < threshold or hand_score_21[b] < threshold:
            continue
        x1, y1 = hand_kp_21[a]
        x2, y2 = hand_kp_21[b]
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            continue
        col = _BODY_COLORS[limb_idx % len(_BODY_COLORS)]
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
    for i in range(21):
        if hand_score_21[i] < threshold:
            continue
        x, y = hand_kp_21[i]
        if x < 0 or y < 0:
            continue
        cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), thickness=-1)


def _draw_face(canvas: np.ndarray, face_kp_68: np.ndarray, face_score_68: np.ndarray, threshold: float = 0.3):
    for i in range(face_kp_68.shape[0]):
        if face_score_68[i] < threshold:
            continue
        x, y = face_kp_68[i]
        if x < 0 or y < 0:
            continue
        cv2.circle(canvas, (int(x), int(y)), 2, (255, 255, 255), thickness=-1)


def _wholebody_to_openpose(keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice the 133-keypoint output into OpenPose-style body / hand / face groups."""
    body_coco = keypoints[:, :17]  # (N, 17, 2)
    body_score = scores[:, :17]  # (N, 17)
    left_hand = keypoints[:, 91:112]
    lhand_score = scores[:, 91:112]
    right_hand = keypoints[:, 112:133]
    rhand_score = scores[:, 112:133]
    face = keypoints[:, 23:91]
    face_score = scores[:, 23:91]

    N = keypoints.shape[0]
    body_op = np.full((N, 18, 2), -1.0, dtype=np.float32)
    body_op_score = np.zeros((N, 18), dtype=np.float32)
    for op_idx, coco_idx in enumerate(_BODY_COCO_TO_OPENPOSE):
        if coco_idx == -1:
            l, r = 5, 6
            valid = (body_score[:, l] > 0.3) & (body_score[:, r] > 0.3)
            neck = (body_coco[:, l] + body_coco[:, r]) / 2.0
            body_op[valid, op_idx] = neck[valid]
            body_op_score[valid, op_idx] = np.minimum(body_score[valid, l], body_score[valid, r])
        else:
            body_op[:, op_idx] = body_coco[:, coco_idx]
            body_op_score[:, op_idx] = body_score[:, coco_idx]
    return body_op, body_op_score, left_hand, lhand_score, right_hand, rhand_score, face, face_score


# -----------------------------------------------------------------------------
# Public detector class
# -----------------------------------------------------------------------------


def _build_ort_session(onnx_path: str, device: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if device.startswith("cuda") and "CUDAExecutionProvider" in available:
        provider_ids = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
    else:
        provider_ids = [("CPUExecutionProvider", {})]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=provider_ids)


class DWposeONNX:
    """ONNX-only DWPose detector that bypasses the controlnet_aux / mmdet stack."""

    def __init__(self, det_onnx_path: str, pose_onnx_path: str, device: str = "cuda"):
        for p in (det_onnx_path, pose_onnx_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"DWPose ONNX checkpoint not found: {p}")
        self.det_session = _build_ort_session(det_onnx_path, device)
        self.pose_session = _build_ort_session(pose_onnx_path, device)
        self.device = device

    def detect(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run detector + pose. Input/Output are RGB uint8 ndarrays.

        Returns:
            keypoints : (N, 133, 2)
            scores    : (N, 133)
        """
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        boxes = detect_persons(self.det_session, img_bgr, score_thr=0.3, nms_thr=0.45)
        if boxes.shape[0] == 0:
            return np.zeros((0, 133, 2), dtype=np.float32), np.zeros((0, 133), dtype=np.float32)
        return estimate_pose(self.pose_session, img_bgr, boxes)

    def __call__(
        self,
        img_rgb: np.ndarray,
        include_hands: bool = False,
        include_face: bool = False,
        bg_mode: str = "black",
        body_threshold: float = 0.3,
        hand_threshold: float = 0.3,
        face_threshold: float = 0.5,
    ) -> np.ndarray:
        """Detect + render an OpenPose-style skeleton image.

        Args:
            img_rgb: HxWx3 uint8 RGB frame.
            include_hands: draw 21-point hand keypoints.
            include_face: draw 68-point face landmarks.
            bg_mode: 'black' / 'source' / 'blend'.

        Returns:
            HxWx3 uint8 RGB skeleton.
        """
        H, W = img_rgb.shape[:2]
        keypoints, scores = self.detect(img_rgb)

        if bg_mode == "source":
            canvas = img_rgb.copy()
        elif bg_mode == "blend":
            canvas = (img_rgb.astype(np.float32) * 0.15).clip(0, 255).astype(np.uint8)
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

        if keypoints.shape[0] == 0:
            return canvas

        body_op, body_score_op, lhand, lhand_score, rhand, rhand_score, face, face_score = _wholebody_to_openpose(keypoints, scores)

        for i in range(body_op.shape[0]):
            _draw_body(canvas, body_op[i], body_score_op[i], threshold=body_threshold)
            if include_hands:
                _draw_hand(canvas, lhand[i], lhand_score[i], threshold=hand_threshold)
                _draw_hand(canvas, rhand[i], rhand_score[i], threshold=hand_threshold)
            if include_face:
                _draw_face(canvas, face[i], face_score[i], threshold=face_threshold)
        return canvas
