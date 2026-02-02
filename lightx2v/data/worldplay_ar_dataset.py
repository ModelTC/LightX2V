"""
WorldPlay AR Dataset for autoregressive video generation training.

This dataset supports:
- Camera pose (w2c, intrinsic) loading
- Action label extraction
- Image conditioning for I2V
- Chunk-based training with memory window
- I2V masking for conditional generation
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import decord

    decord.bridge.set_bridge("torch")
except ImportError:
    decord = None


class WorldPlayARDataset(Dataset):
    """
    Dataset for WorldPlay AR model training.

    Supports loading video data with camera poses and action labels
    for autoregressive video generation training.

    Args:
        data_root: Root directory containing video data
        meta_file: Path to metadata JSON file
        video_length: Number of frames per video sample
        resolution: Target resolution (height, width)
        chunk_latent_num: Number of latent frames per chunk
        memory_window_size: Size of memory window for AR training
        select_window_out_flag: Whether to use memory window selection
        task: Task type ('t2v' or 'i2v')
        transform: Optional transform to apply to frames
        action_trans_thresh: Translation threshold for action quantization
        action_rot_thresh: Rotation threshold for action quantization
        num_action_classes: Number of discrete action classes (default 81 = 3^4)
    """

    def __init__(
        self,
        data_root: str,
        meta_file: str,
        video_length: int = 125,
        resolution: Tuple[int, int] = (480, 832),
        chunk_latent_num: int = 4,
        memory_window_size: int = 8,
        select_window_out_flag: bool = True,
        task: str = "i2v",
        transform: Optional[Any] = None,
        action_trans_thresh: float = 0.1,
        action_rot_thresh: float = 0.05,
        num_action_classes: int = 81,
    ):
        super().__init__()
        self.data_root = data_root
        self.video_length = video_length
        self.resolution = resolution
        self.chunk_latent_num = chunk_latent_num
        self.memory_window_size = memory_window_size
        self.select_window_out_flag = select_window_out_flag
        self.task = task
        self.transform = transform

        # Action quantization parameters
        self.action_trans_thresh = action_trans_thresh
        self.action_rot_thresh = action_rot_thresh
        self.num_action_classes = num_action_classes

        # Load metadata
        self.samples = self._load_metadata(meta_file)
        logger.info(f"Loaded {len(self.samples)} samples from {meta_file}")

    def _load_metadata(self, meta_file: str) -> List[Dict]:
        """Load metadata from JSON file."""
        with open(meta_file, "r") as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = {
                "video_path": os.path.join(self.data_root, item["video_path"]),
                "caption": item.get("caption", ""),
            }

            # Camera pose data
            if "w2c" in item:
                sample["w2c"] = item["w2c"]
            if "intrinsic" in item:
                sample["intrinsic"] = item["intrinsic"]
            if "action" in item:
                sample["action"] = item["action"]

            # Image conditioning for I2V
            if "image_cond" in item:
                sample["image_cond"] = os.path.join(self.data_root, item["image_cond"])

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load video frames
        video = self._load_video(sample["video_path"])

        # Load camera poses
        w2c = self._load_camera_pose(sample.get("w2c"))
        intrinsic = self._load_intrinsic(sample.get("intrinsic"))

        # Load or compute action labels
        action = self._load_action(sample.get("action"), w2c)

        # Load image condition for I2V
        image_cond = None
        if self.task == "i2v" and "image_cond" in sample:
            image_cond = self._load_image(sample["image_cond"])

        # Prepare I2V mask
        i2v_mask = self._prepare_i2v_mask(video.shape[0])

        # Select memory window for training
        if self.select_window_out_flag:
            (video, w2c, intrinsic, action, i2v_mask) = self._select_memory_window(video, w2c, intrinsic, action, i2v_mask)

        output = {
            "video": video,
            "caption": sample["caption"],
            "w2c": w2c,
            "intrinsic": intrinsic,
            "action": action,
            "i2v_mask": i2v_mask,
        }

        if image_cond is not None:
            output["image_cond"] = image_cond

        return output

    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load video frames from file."""
        if decord is None:
            raise ImportError("decord is required for video loading")

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        # Sample frames
        if total_frames >= self.video_length:
            start_idx = random.randint(0, total_frames - self.video_length)
            frame_indices = list(range(start_idx, start_idx + self.video_length))
        else:
            frame_indices = list(range(total_frames))
            # Pad with last frame
            frame_indices += [total_frames - 1] * (self.video_length - total_frames)

        frames = vr.get_batch(frame_indices)  # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

        # Resize to target resolution
        frames = F.interpolate(frames, size=self.resolution, mode="bilinear", align_corners=False)

        if self.transform is not None:
            frames = self.transform(frames)

        return frames

    def _load_camera_pose(self, w2c_data: Optional[Any]) -> torch.Tensor:
        """Load world-to-camera transformation matrices."""
        if w2c_data is None:
            # Return identity matrices
            return torch.eye(4).unsqueeze(0).repeat(self.video_length, 1, 1)

        if isinstance(w2c_data, str):
            # Load from file
            w2c = np.load(w2c_data)
        elif isinstance(w2c_data, list):
            w2c = np.array(w2c_data)
        else:
            w2c = w2c_data

        w2c = torch.from_numpy(w2c).float()

        # Ensure correct shape [T, 4, 4]
        if w2c.dim() == 2:
            w2c = w2c.unsqueeze(0).repeat(self.video_length, 1, 1)
        elif w2c.shape[0] < self.video_length:
            # Pad with last pose
            pad_size = self.video_length - w2c.shape[0]
            w2c = torch.cat([w2c, w2c[-1:].repeat(pad_size, 1, 1)], dim=0)
        elif w2c.shape[0] > self.video_length:
            w2c = w2c[: self.video_length]

        return w2c

    def _load_intrinsic(self, intrinsic_data: Optional[Any]) -> torch.Tensor:
        """Load camera intrinsic matrices."""
        if intrinsic_data is None:
            # Return default intrinsics
            K = torch.tensor([[500.0, 0.0, self.resolution[1] / 2], [0.0, 500.0, self.resolution[0] / 2], [0.0, 0.0, 1.0]])
            return K.unsqueeze(0).repeat(self.video_length, 1, 1)

        if isinstance(intrinsic_data, str):
            intrinsic = np.load(intrinsic_data)
        elif isinstance(intrinsic_data, list):
            intrinsic = np.array(intrinsic_data)
        else:
            intrinsic = intrinsic_data

        intrinsic = torch.from_numpy(intrinsic).float()

        # Ensure correct shape [T, 3, 3]
        if intrinsic.dim() == 2:
            intrinsic = intrinsic.unsqueeze(0).repeat(self.video_length, 1, 1)
        elif intrinsic.shape[0] < self.video_length:
            pad_size = self.video_length - intrinsic.shape[0]
            intrinsic = torch.cat([intrinsic, intrinsic[-1:].repeat(pad_size, 1, 1)], dim=0)
        elif intrinsic.shape[0] > self.video_length:
            intrinsic = intrinsic[: self.video_length]

        return intrinsic

    def _load_action(self, action_data: Optional[Any], w2c: torch.Tensor) -> torch.Tensor:
        """Load or compute action labels from camera poses."""
        if action_data is not None:
            if isinstance(action_data, str):
                action = np.load(action_data)
            elif isinstance(action_data, list):
                action = np.array(action_data)
            else:
                action = action_data
            action = torch.from_numpy(action).long()
        else:
            # Compute action from camera pose differences
            action = self._compute_action_from_pose(w2c)

        # Ensure correct shape [T]
        if action.shape[0] < self.video_length:
            pad_size = self.video_length - action.shape[0]
            action = torch.cat([action, action[-1:].repeat(pad_size)], dim=0)
        elif action.shape[0] > self.video_length:
            action = action[: self.video_length]

        return action

    def _compute_action_from_pose(self, w2c: torch.Tensor) -> torch.Tensor:
        """
        Compute discrete action labels from camera pose differences.

        Action space: 81 classes (3^4 for forward/backward, left/right,
        up/down, rotation)
        """
        T = w2c.shape[0]
        actions = torch.zeros(T, dtype=torch.long)

        for t in range(1, T):
            # Compute relative transformation
            rel_pose = torch.inverse(w2c[t - 1]) @ w2c[t]

            # Extract translation and rotation
            translation = rel_pose[:3, 3]
            rotation = rel_pose[:3, :3]

            # Quantize to discrete action
            action_idx = self._quantize_action(translation, rotation)
            actions[t] = action_idx

        return actions

    def _quantize_action(self, translation: torch.Tensor, rotation: torch.Tensor) -> int:
        """Quantize continuous motion to discrete action index."""
        # Use configurable thresholds
        trans_thresh = self.action_trans_thresh
        rot_thresh = self.action_rot_thresh

        # Forward/backward (z-axis)
        if translation[2] > trans_thresh:
            fb = 2  # forward
        elif translation[2] < -trans_thresh:
            fb = 0  # backward
        else:
            fb = 1  # stationary

        # Left/right (x-axis)
        if translation[0] > trans_thresh:
            lr = 2  # right
        elif translation[0] < -trans_thresh:
            lr = 0  # left
        else:
            lr = 1  # stationary

        # Up/down (y-axis)
        if translation[1] > trans_thresh:
            ud = 2  # up
        elif translation[1] < -trans_thresh:
            ud = 0  # down
        else:
            ud = 1  # stationary

        # Rotation (simplified to yaw)
        yaw = torch.atan2(rotation[0, 2], rotation[2, 2])
        if yaw > rot_thresh:
            rot = 2  # rotate right
        elif yaw < -rot_thresh:
            rot = 0  # rotate left
        else:
            rot = 1  # no rotation

        # Combine into single index (base-3 encoding)
        action_idx = fb * 27 + lr * 9 + ud * 3 + rot
        return action_idx

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load conditioning image for I2V."""
        if Image is None:
            raise ImportError("PIL is required for image loading")

        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.resolution[1], self.resolution[0]))
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        return img

    def _prepare_i2v_mask(self, num_frames: int) -> torch.Tensor:
        """
        Prepare I2V mask for conditional generation.

        For I2V task, the first frame is conditioned (mask=0),
        and remaining frames are generated (mask=1).
        """
        mask = torch.ones(num_frames)
        if self.task == "i2v":
            mask[0] = 0  # First frame is conditioned
        return mask

    def _select_memory_window(
        self,
        video: torch.Tensor,
        w2c: torch.Tensor,
        intrinsic: torch.Tensor,
        action: torch.Tensor,
        i2v_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Select a random memory window for training.

        This simulates the AR generation process where we only
        attend to a window of previous frames.
        """
        T = video.shape[0]
        window_size = self.memory_window_size * self.chunk_latent_num

        if T <= window_size:
            return video, w2c, intrinsic, action, i2v_mask

        # Random start position
        start_idx = random.randint(0, T - window_size)
        end_idx = start_idx + window_size

        return (
            video[start_idx:end_idx],
            w2c[start_idx:end_idx],
            intrinsic[start_idx:end_idx],
            action[start_idx:end_idx],
            i2v_mask[start_idx:end_idx],
        )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    output = {}

    # Stack tensors
    for key in ["video", "w2c", "intrinsic", "action", "i2v_mask"]:
        if key in batch[0]:
            output[key] = torch.stack([item[key] for item in batch])

    # Handle optional image_cond
    if "image_cond" in batch[0]:
        output["image_cond"] = torch.stack([item["image_cond"] for item in batch])

    # Collect captions
    output["caption"] = [item["caption"] for item in batch]

    return output
