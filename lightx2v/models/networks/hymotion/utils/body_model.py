import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .geometry import (
    rot6d_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
)

# yapf: disable
LEFT_HAND_MEAN_AA = [ 0.1117,  0.0429, -0.4164,  0.1088, -0.0660, -0.7562, -0.0964, -0.0909,
        -0.1885, -0.1181,  0.0509, -0.5296, -0.1437,  0.0552, -0.7049, -0.0192,
        -0.0923, -0.3379, -0.4570, -0.1963, -0.6255, -0.2147, -0.0660, -0.5069,
        -0.3697, -0.0603, -0.0795, -0.1419, -0.0859, -0.6355, -0.3033, -0.0579,
        -0.6314, -0.1761, -0.1321, -0.3734,  0.8510,  0.2769, -0.0915, -0.4998,
        0.0266,  0.0529,  0.5356,  0.0460, -0.2774]
RIGHT_HAND_MEAN_AA = [ 0.1117, -0.0429,  0.4164,  0.1088,  0.0660,  0.7562, -0.0964,  0.0909,
        0.1885, -0.1181, -0.0509,  0.5296, -0.1437, -0.0552,  0.7049, -0.0192,
        0.0923,  0.3379, -0.4570,  0.1963,  0.6255, -0.2147,  0.0660,  0.5069,
        -0.3697,  0.0603,  0.0795, -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,
        0.6314, -0.1761,  0.1321,  0.3734,  0.8510, -0.2769,  0.0915, -0.4998,
        -0.0266, -0.0529,  0.5356, -0.0460,  0.2774]
# yapf: enable


def to_tensor(array, dtype=torch.float32, device="cpu"):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    """Calculates the rotation matrices for a batch of rotation vectors
    Parameters
    ----------
    rot_vecs: torch.tensor Nx3
        array of N axis-angle vectors
    Returns
    -------
    R: torch.tensor Nx3x3
        The rotation matrices for the given axis-angle parameters
    """
    if len(rot_vecs.shape) > 2:
        rot_vec_ori = rot_vecs
        rot_vecs = rot_vecs.view(-1, 3)
    else:
        rot_vec_ori = None
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    if rot_vec_ori is not None:
        rot_mat = rot_mat.reshape(*rot_vec_ori.shape[:-1], 3, 3)
    return rot_mat


def load_model_data(model_path):
    """
    Load wooden model data from binary files.

    Args:
        model_path: path to the directory containing .bin files

    Returns:
        dict containing:
            - v_template: (V, 3) vertex template
            - j_template: (J, 3) joint template
            - skin_weights: (V, 4) skin weights
            - skin_indices: (V, 4) skin indices
            - parents: (J,) parent indices (kintree)
            - faces: (F, 3) face indices
            - joint_names: list of joint names
    """
    model_path = Path(model_path)

    # Load vertex template: (V*3,) -> (V, 3)
    with open(model_path / "v_template.bin", "rb") as f:
        v_template_flat = np.frombuffer(f.read(), dtype=np.float32)
    num_verts = len(v_template_flat) // 3
    v_template = v_template_flat.reshape(num_verts, 3)

    # Load joint template: (J*3,) -> (J, 3)
    with open(model_path / "j_template.bin", "rb") as f:
        j_template_flat = np.frombuffer(f.read(), dtype=np.float32)
    num_joints = len(j_template_flat) // 3
    j_template = j_template_flat.reshape(num_joints, 3)

    # Load skin weights: (V*4,) -> (V, 4), 4 bones per vertex
    with open(model_path / "skinWeights.bin", "rb") as f:
        skin_weights_flat = np.frombuffer(f.read(), dtype=np.float32)
    skin_weights = skin_weights_flat.reshape(num_verts, 4)

    # Load skin indices: (V*4,) -> (V, 4), 4 bone indices per vertex
    with open(model_path / "skinIndice.bin", "rb") as f:
        skin_indices_flat = np.frombuffer(f.read(), dtype=np.uint16)
    skin_indices = skin_indices_flat.reshape(num_verts, 4).astype(np.int64)

    # Load kintree (parent indices): (J,)
    with open(model_path / "kintree.bin", "rb") as f:
        parents = np.frombuffer(f.read(), dtype=np.int32)

    # Load faces
    with open(model_path / "faces.bin", "rb") as f:
        faces_flat = np.frombuffer(f.read(), dtype=np.uint16)
    faces = faces_flat.reshape(-1, 3)

    # Load joint names
    joint_names_path = model_path / "joint_names.json"
    if joint_names_path.exists():
        with open(joint_names_path, "r") as f:
            joint_names = json.load(f)
    else:
        joint_names = [f"Joint_{i}" for i in range(num_joints)]

    return {
        "v_template": v_template,
        "j_template": j_template,
        "skin_weights": skin_weights,
        "skin_indices": skin_indices,
        "parents": parents,
        "faces": faces,
        "joint_names": joint_names,
        "num_joints": num_joints,
        "num_verts": num_verts,
    }


def simple_lbs(v_template, rot_mats, joints, parents, skin_weights, skin_indices):
    """
    Simple Linear Blend Skinning without shape blending.

    Args:
        v_template: (V, 3) template vertices
        rot_mats: (B, J, 3, 3) rotation matrices for each joint
        joints: (J, 3) joint positions in rest pose
        parents: (J,) parent indices for each joint
        skin_weights: (V, 4) skin weights for 4 bones per vertex
        skin_indices: (V, 4) bone indices for 4 bones per vertex

    Returns:
        vertices: (B, V, 3) transformed vertices
        posed_joints: (B, J, 3) transformed joint positions
    """
    batch_size = rot_mats.shape[0]
    num_joints = rot_mats.shape[1]
    num_verts = v_template.shape[0]
    device = rot_mats.device
    dtype = rot_mats.dtype

    # Compute relative joint positions
    rel_joints = joints.clone()
    rel_joints[1:] = joints[1:] - joints[parents[1:]]

    # Build transformation chain: transforms_mat (B, J, 4, 4)
    transforms_mat = torch.zeros(batch_size, num_joints, 4, 4, device=device, dtype=dtype)
    transforms_mat[..., :3, :3] = rot_mats
    transforms_mat[..., :3, 3] = rel_joints.unsqueeze(0).expand(batch_size, -1, -1)
    transforms_mat[..., 3, 3] = 1.0

    # Forward kinematics: accumulate transforms from root to each joint
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, num_joints):
        parent_idx = parents[i].item()
        curr_transform = torch.bmm(transform_chain[parent_idx], transforms_mat[:, i])
        transform_chain.append(curr_transform)

    transforms = torch.stack(transform_chain, dim=1)  # (B, J, 4, 4)

    # Get posed joint positions
    posed_joints = transforms[..., :3, 3].clone()  # (B, J, 3)

    # Compute relative transforms (for skinning)
    # We need to subtract the rest pose joint positions from the transform
    rel_transforms = transforms.clone()
    joints_homo = F.pad(joints, [0, 1], value=0)  # (J, 4)
    transformed_rest = torch.einsum("bjcd,jd->bjc", transforms[..., :3, :], joints_homo)
    rel_transforms[..., :3, 3] = transforms[..., :3, 3] - transformed_rest[..., :3]

    # Apply skinning: gather transforms for each vertex's 4 bones
    # skin_indices: (V, 4), skin_weights: (V, 4)
    vertex_transforms = torch.zeros(batch_size, num_verts, 4, 4, 4, device=device, dtype=dtype)
    for k in range(4):
        bone_idx = skin_indices[:, k].long()  # (V,)
        vertex_transforms[:, :, k] = rel_transforms[:, bone_idx]  # (B, V, 4, 4)

    # Weight the transforms
    skin_weights_expanded = skin_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, V, 4, 1, 1)
    skin_weights_expanded = skin_weights_expanded.expand(batch_size, -1, -1, 4, 4)  # (B, V, 4, 4, 4)

    weighted_transforms = (vertex_transforms * skin_weights_expanded).sum(dim=2)  # (B, V, 4, 4)

    # Apply to vertices
    v_homo = F.pad(v_template, [0, 1], value=1.0)  # (V, 4)
    vertices = torch.einsum("bvcd,vd->bvc", weighted_transforms[..., :3, :], v_homo)  # (B, V, 3)

    return vertices, posed_joints


class WoodenMesh(torch.nn.Module):
    """
    Wooden character mesh model that loads from binary files.
    Uses simple LBS without shape blending (fixed skeleton).
    """

    def __init__(self, model_path=None):
        torch.nn.Module.__init__(self)

        # Load model data from .bin files
        model = load_model_data(model_path or Path(__file__).parent / "assets/dump_wooden")

        # Register buffers like original SMPLMesh
        v_template = to_tensor(model["v_template"])
        self.register_buffer("v_template", v_template)

        j_template = to_tensor(model["j_template"])
        self.register_buffer("j_template", j_template)

        skin_weights = to_tensor(model["skin_weights"])
        self.register_buffer("skin_weights", skin_weights)

        skin_indices = to_tensor(model["skin_indices"], dtype=torch.long)
        self.register_buffer("skin_indices", skin_indices)

        parents = to_tensor(model["parents"], dtype=torch.long)
        self.register_buffer("parents", parents)

        # Store non-buffer attributes
        self.faces = model["faces"]
        self.joint_names = model["joint_names"]
        self.num_joints = model["num_joints"]
        self.num_verts = model["num_verts"]

        print(f"[WoodenMesh] Loaded model: {self.num_verts} vertices, {self.num_joints} joints")

    def forward(self, params, fast_forward=False):
        """
        Forward pass to compute deformed vertices.

        Args:
            params: dict containing:
                - 'poses': (B, J*3) axis-angle rotations, or
                - 'rot6d': (B, J, 6) 6D rotation representations
                - 'trans': (B, 3) optional translation

        Returns:
            dict with 'vertices' and 'vertices_wotrans'
        """
        if "poses" in params:
            poses = params["poses"]
            batch_size = poses.shape[0]
            rot_mats = batch_rodrigues(poses.view(-1, 3)).view([batch_size, -1, 3, 3])
        elif "rot6d" in params:
            rot6d = params["rot6d"]
            batch_size = rot6d.shape[0]
            rot_mats = rot6d_to_rotation_matrix(rot6d).view([batch_size, -1, 3, 3])
        else:
            raise ValueError("poses or rot6d must be in params")

        if rot_mats.shape[1] == 22:
            eye = torch.eye(3, device=rot_mats.device, dtype=rot_mats.dtype)[None, None, :, :].repeat(batch_size, 30, 1, 1)
            rot_mats = torch.cat([rot_mats, eye], dim=1)  # (B, 22 + 30, 3, 3)

        # Simple LBS (no shape blending, fixed skeleton)
        vertices, posed_joints = simple_lbs(
            self.v_template,
            rot_mats,
            self.j_template,
            self.parents,
            self.skin_weights,
            self.skin_indices,
        )

        # Vertices without translation (for pose-level supervision)
        vertices_wotrans = vertices

        if "trans" in params:
            trans = params["trans"]
            vertices = vertices + trans[:, None, :]

        return {
            "vertices": vertices,
            "vertices_wotrans": vertices_wotrans,
            "keypoints3d": posed_joints,
        }

    def forward_batch(self, params):
        assert "rot6d" in params and "trans" in params
        rot6d = params["rot6d"]
        trans = params["trans"]
        bs, num_frames = rot6d.shape[:2]
        rot6d_flat = rot6d.reshape(bs * num_frames, rot6d.shape[2], rot6d.shape[3])
        trans_flat = trans.reshape(bs * num_frames, trans.shape[2])
        result = self.forward(
            {
                "rot6d": rot6d_flat,
                "trans": trans_flat,
            }
        )
        out = {}
        for key in result:
            out[key] = result[key].reshape(bs, num_frames, *result[key].shape[1:])
        return out


def construct_smpl_data_dict(
    rot6d: Tensor,
    transl: Tensor,
    betas: Tensor | None = None,
    gender: str = "neutral",
    use_default_hand_mean_pose: bool = False,
) -> dict:
    rotation_matrix = rot6d_to_rotation_matrix(rot6d)
    angle_axis = rotation_matrix_to_angle_axis(rotation_matrix)
    left_hand_mean_pose = (
        torch.tensor(
            LEFT_HAND_MEAN_AA,
            device=angle_axis.device,
            dtype=angle_axis.dtype,
        )
        .unsqueeze(0)
        .repeat(angle_axis.shape[0], 1)
        .reshape(angle_axis.shape[0], -1, 3)
    )
    right_hand_mean_pose = (
        torch.tensor(
            RIGHT_HAND_MEAN_AA,
            device=angle_axis.device,
            dtype=angle_axis.dtype,
        )
        .unsqueeze(0)
        .repeat(angle_axis.shape[0], 1)
        .reshape(angle_axis.shape[0], -1, 3)
    )
    if angle_axis.shape[1] == 22:
        angle_axis = torch.cat(
            [
                angle_axis,
                left_hand_mean_pose,
                right_hand_mean_pose,
            ],
            dim=1,
        )
    elif angle_axis.shape[1] == 52:
        if use_default_hand_mean_pose:
            angle_axis = torch.cat(
                [
                    angle_axis[:, :22],
                    left_hand_mean_pose,
                    right_hand_mean_pose,
                ],
                dim=1,
            )
        else:
            pass

    assert angle_axis.shape[1] == 52, f"angle_axis should be 52, but got {angle_axis.shape[1]}"
    dump = {
        "betas": betas.cpu().numpy() if betas is not None else np.zeros((1, 16)),
        "gender": gender,
        "poses": angle_axis.cpu().numpy().reshape(angle_axis.shape[0], -1),
        "trans": transl.cpu().numpy(),
        "mocap_framerate": 30,
        "num_frames": angle_axis.shape[0],
        "Rh": angle_axis.cpu().numpy().reshape(angle_axis.shape[0], -1)[:, :3],
    }
    return dump
