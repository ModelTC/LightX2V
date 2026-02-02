# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT
# Ported from HY-WorldPlay for LightX2V integration

import json

import numpy as np
import torch

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None

# Mapping from one-hot action vectors to discrete labels
ACTION_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


def rot_x(theta):
    """Rotation matrix around X-axis (pitch)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta):
    """Rotation matrix around Y-axis (yaw)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta):
    """Rotation matrix around Z-axis (roll)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def one_hot_to_one_dimension(one_hot):
    """Convert one-hot action vectors to discrete labels."""
    y = torch.tensor([ACTION_MAPPING[tuple(row.tolist())] for row in one_hot])
    return y


def generate_camera_trajectory_local(motions):
    """
    Generate camera trajectory from motion commands.

    Args:
        motions: list of dict
            {"forward": 1.0}, {"yaw": np.pi/2}, {"pitch": np.pi/6}, {"right": 1.0}
            - forward: Translation (Forward or Backward)
            - yaw: Rotate (Left or Right)
            - pitch: Rotate (Up or Down)
            - right: Translation (Right or Left)
            - third_yaw: Third Perspective Rotate (Left or Right)

    Returns:
        list of 4x4 transformation matrices (camera-to-world)
    """
    poses = []
    T = np.eye(4)
    poses.append(T.copy())

    for move in motions:
        # Rotate (Left or Right)
        if "yaw" in move:
            R_mat = rot_y(move["yaw"])
            T[:3, :3] = T[:3, :3] @ R_mat

        # Rotate (Up or Down)
        if "pitch" in move:
            R_mat = rot_x(move["pitch"])
            T[:3, :3] = T[:3, :3] @ R_mat

        # Translation (Z-direction of the camera's local coordinate system)
        forward = move.get("forward", 0.0)
        if forward != 0:
            local_t = np.array([0, 0, forward])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Translation (X-direction of the camera's local coordinate system)
        right = move.get("right", 0.0)
        if right != 0:
            local_t = np.array([right, 0, 0])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        # Third Perspective Rotate (Left or Right)
        third_yaw = move.get("third_yaw", 0.0)
        if third_yaw != 0:
            theta = -third_yaw
            C = np.array([[1, 0.0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1.0], [0, 0, 0, 1]])
            c_origin = C.copy()
            # Rotation around the Y-axis
            R_y = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
            # Translation
            C[:3, :3] = C[:3, :3] @ R_y
            C[:3, 3] = R_y @ C[:3, 3]
            c_inv = np.linalg.inv(c_origin)
            c_relative = c_inv @ C
            T = T @ c_relative

        poses.append(T.copy())

    return poses


def parse_pose_string(pose_string):
    """
    Parse pose string to motions list.

    Format: "w-3, right-0.5, d-4"
    - w: forward movement
    - s: backward movement
    - a: left movement
    - d: right movement
    - up: pitch up rotation
    - down: pitch down rotation
    - left: yaw left rotation
    - right: yaw right rotation
    - number after dash: duration in latents

    Args:
        pose_string: str, comma-separated pose commands

    Returns:
        list of dict: motions for generate_camera_trajectory_local
    """
    # Movement amount per frame
    forward_speed = 0.08  # units per frame
    yaw_speed = np.deg2rad(3)  # radians per frame
    pitch_speed = np.deg2rad(3)  # radians per frame

    motions = []
    commands = [cmd.strip() for cmd in pose_string.split(",")]

    for cmd in commands:
        if not cmd:
            continue

        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pose command: {cmd}. Expected format: 'action-duration'")

        action = parts[0].strip()
        try:
            duration = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid duration in command: {cmd}")

        num_frames = int(duration)

        # Parse action and create motion dict
        if action == "w":
            # Forward
            for _ in range(num_frames):
                motions.append({"forward": forward_speed})
        elif action == "s":
            # Backward
            for _ in range(num_frames):
                motions.append({"forward": -forward_speed})
        elif action == "a":
            # Left
            for _ in range(num_frames):
                motions.append({"right": -forward_speed})
        elif action == "d":
            # Right
            for _ in range(num_frames):
                motions.append({"right": forward_speed})
        elif action == "up":
            # Pitch up
            for _ in range(num_frames):
                motions.append({"pitch": pitch_speed})
        elif action == "down":
            # Pitch down
            for _ in range(num_frames):
                motions.append({"pitch": -pitch_speed})
        elif action == "left":
            # Yaw left
            for _ in range(num_frames):
                motions.append({"yaw": -yaw_speed})
        elif action == "right":
            # Yaw right
            for _ in range(num_frames):
                motions.append({"yaw": yaw_speed})
        else:
            raise ValueError(f"Unknown action: {action}. Supported actions: w, s, a, d, up, down, left, right")

    return motions


def pose_string_to_json(pose_string):
    """
    Convert pose string to pose JSON format.

    Args:
        pose_string: str, comma-separated pose commands

    Returns:
        dict: pose JSON with extrinsic and intrinsic parameters
    """
    motions = parse_pose_string(pose_string)
    poses = generate_camera_trajectory_local(motions)

    # Default intrinsic matrix
    intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]

    pose_json = {}
    for i, p in enumerate(poses):
        pose_json[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}

    return pose_json


def pose_to_input(pose_data, latent_num, tps=False):
    """
    Convert pose data to input tensors.

    Args:
        pose_data: str or dict
            - If str ending with '.json': path to JSON file
            - If str: pose string (e.g., "w-3, right-0.5, d-4")
            - If dict: pose JSON data
        latent_num: int, number of latents
        tps: bool, third person mode

    Returns:
        tuple: (w2c_list, intrinsic_list, action_one_label)
            - w2c_list: torch.Tensor (batch, latent_num, 4, 4) - world-to-camera matrices
            - intrinsic_list: torch.Tensor (batch, latent_num, 3, 3) - normalized intrinsics
            - action_one_label: torch.Tensor (batch, latent_num) - discrete action labels (0-80)
    """
    # Handle different input types
    if isinstance(pose_data, str):
        if pose_data.endswith(".json"):
            # Load from JSON file
            pose_json = json.load(open(pose_data, "r"))
        else:
            # Parse pose string
            pose_json = pose_string_to_json(pose_data)
    elif isinstance(pose_data, dict):
        pose_json = pose_data
    else:
        raise ValueError(f"Invalid pose_data type: {type(pose_data)}. Expected str or dict.")

    pose_keys = list(pose_json.keys())
    latent_num_from_pose = len(pose_keys)
    assert latent_num_from_pose == latent_num, f"pose corresponds to {latent_num_from_pose * 4 - 3} frames, num_frames must be set to {latent_num_from_pose * 4 - 3} to ensure alignment."

    intrinsic_list = []
    w2c_list = []
    for i in range(latent_num):
        t_key = pose_keys[i]
        c2w = np.array(pose_json[t_key]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)
        intrinsic = np.array(pose_json[t_key]["K"])
        # Normalize intrinsics
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        intrinsic_list.append(intrinsic)

    w2c_list = np.array(w2c_list)
    intrinsic_list = torch.tensor(np.array(intrinsic_list))

    c2ws = np.linalg.inv(w2c_list)
    C_inv = np.linalg.inv(c2ws[:-1])
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0, ...] = c2ws[0, ...]
    relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

    move_norm_valid = 0.0001
    for i in range(1, relative_c2w.shape[0]):
        move_dirs = relative_c2w[i, :3, 3]  # direction vector
        move_norms = np.linalg.norm(move_dirs)
        if move_norms > move_norm_valid:  # threshold for movement
            move_norm_dirs = move_dirs / move_norms
            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
            trans_angles_deg = angles_rad * (180.0 / np.pi)  # convert to degrees
        else:
            trans_angles_deg = np.zeros(3)

        R_rel = relative_c2w[i, :3, :3]
        r = R.from_matrix(R_rel)
        rot_angles_deg = r.as_euler("xyz", degrees=True)

        # Determine movement and rotation actions
        if move_norms > move_norm_valid:  # threshold for movement
            if (not tps) or (tps and abs(rot_angles_deg[1]) < 5e-2 and abs(rot_angles_deg[0]) < 5e-2):
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1  # forward
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1  # backward

                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1  # right
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1  # left

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1  # right
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1  # left

        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1  # up
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1  # down

    trans_one_hot = torch.tensor(trans_one_hot)
    rotate_one_hot = torch.tensor(rotate_one_hot)

    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
    # Combine translation and rotation labels: 9 translation classes * 9 rotation classes = 81 total
    action_one_label = trans_one_label * 9 + rotate_one_label

    return torch.as_tensor(w2c_list), torch.as_tensor(intrinsic_list), action_one_label
