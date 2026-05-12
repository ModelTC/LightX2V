# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch


def apply_transformation(Bx4x4, another_matrix):
    B = Bx4x4.shape[0]
    if another_matrix.dim() == 2:
        another_matrix = another_matrix.unsqueeze(0).expand(B, -1, -1)  # Make another_matrix compatible with batch size
    transformed_matrix = torch.bmm(Bx4x4, another_matrix)  # Shape: (B, 4, 4)

    return transformed_matrix


def look_at_matrix(camera_pos, target, invert_pos=True):
    """Creates a 4x4 look-at matrix, keeping the camera pointing towards a target."""
    forward = (target - camera_pos).float()
    forward = forward / torch.norm(forward)

    up = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device)  # assuming Y-up coordinate system
    right = torch.cross(up, forward, dim=0)
    right = right / torch.norm(right)
    up = torch.cross(forward, right, dim=0)

    look_at = torch.eye(4, device=camera_pos.device)
    look_at[0, :3] = right
    look_at[1, :3] = up
    look_at[2, :3] = forward
    if invert_pos:
        # Proper world-to-camera translation: t = -R @ C
        look_at[:3, 3] = -look_at[:3, :3] @ camera_pos
    else:
        look_at[:3, 3] = camera_pos

    return look_at


def slerp(quat1, quat2, t):
    """Spherical linear interpolation (SLERP) between two quaternions."""
    dot_product = torch.dot(quat1, quat2)

    if dot_product < 0.0:
        quat2 = -quat2
        dot_product = -dot_product

    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta_0 = torch.acos(dot_product)
    sin_theta_0 = torch.sin(theta_0)

    if sin_theta_0 > 1e-6:
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        s1 = torch.sin(theta_0 - theta) / sin_theta_0
        s2 = sin_theta / sin_theta_0
        return s1 * quat1 + s2 * quat2
    else:
        return (1.0 - t) * quat1 + t * quat2


def matrix_to_quaternion(matrix, device):
    """Converts a 3x3 rotation matrix to a quaternion."""
    m = matrix[:3, :3]
    trace = torch.trace(m)

    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return torch.tensor([w, x, y, z], device=device)


def quaternion_to_matrix(quat, device):
    """Converts a quaternion to a 4x4 rotation matrix."""
    w, x, y, z = quat
    rotation = torch.eye(4, device=device)

    rotation[0, 0] = 1 - 2 * (y * y + z * z)
    rotation[0, 1] = 2 * (x * y - z * w)
    rotation[0, 2] = 2 * (x * z + y * w)

    rotation[1, 0] = 2 * (x * y + z * w)
    rotation[1, 1] = 1 - 2 * (x * x + z * z)
    rotation[1, 2] = 2 * (y * z - x * w)

    rotation[2, 0] = 2 * (x * z - y * w)
    rotation[2, 1] = 2 * (y * z + x * w)
    rotation[2, 2] = 1 - 2 * (x * x + y * y)

    return rotation


def get_translation_matrix(dx, dy, dz, device):
    """Creates a 4x4 translation matrix."""
    translation = torch.eye(4, device=device)
    translation[0, 3] = dx
    translation[1, 3] = dy
    translation[2, 3] = dz
    return translation


def interpolate(trajectory, n_steps_per_segment, device):
    interpolated_trajectory = [trajectory[0]]

    for i in range(len(trajectory) - 1):
        start_matrix = trajectory[i]
        end_matrix = trajectory[i + 1]

        start_pos = start_matrix[:3, 3]
        end_pos = end_matrix[:3, 3]

        start_rot = matrix_to_quaternion(start_matrix, device)
        end_rot = matrix_to_quaternion(end_matrix, device)

        for t in torch.linspace(0, 1, n_steps_per_segment + 1, device=device)[1:]:
            interp_pos = (1 - t) * start_pos + t * end_pos
            interp_rot = slerp(start_rot, end_rot, t)

            interp_matrix = torch.eye(4, device=device)
            interp_matrix[:3, :3] = quaternion_to_matrix(interp_rot, device)[:3, :3]
            interp_matrix[:3, 3] = interp_pos

            interpolated_trajectory.append(interp_matrix)
    interpolated_trajectory = torch.stack(interpolated_trajectory)


def create_spiral_trajectory(
    world_to_camera_matrix,
    center_depth,
    radius_x=0.03,
    radius_y=0.02,
    radius_z=0.0,
    right=True,
    inwards=True,
    n_steps=13,
    device="cuda",
    start_from_zero=True,
    num_circles=1,
):
    """
    Create a spiral camera trajectory that follows a given motion, keeps the camera looking at a point,
    and interpolates between trajectory points to create a smooth movement between camera positions.

    Parameters:
    - world_to_camera_matrix (torch.Tensor): 4x4 camera-to-world matrix.
    - num_points (int): Number of key points in the spiral motion.
    - radius (float): Spiral radius for the camera motion.
    - look_at (torch.Tensor): 3D point the camera should look at.
    - n_steps_per_segment (int): Number of steps to interpolate between each key point.
    - device (str): The device on which to perform the calculations (e.g., 'cpu' or 'cuda').

    Returns:
    - interpolated_trajectory (list): List of 4x4 matrices representing the interpolated camera positions.
    """
    # Move all inputs to the specified device
    # world_to_camera_matrix = world_to_camera_matrix.to(device)
    look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)

    # Spiral motion key points
    trajectory = []
    spiral_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)  # world_to_camera_matrix[:3, 3].clone()

    example_scale = 1.0

    theta_max = 2 * math.pi * num_circles

    for i in range(n_steps):
        # theta = 2 * math.pi * i / (n_steps-1)  # angle for each point
        theta = theta_max * i / (n_steps - 1)  # angle for each point
        if start_from_zero:
            x = radius_x * (math.cos(theta) - 1) * (1 if right else -1) * (center_depth / example_scale)
        else:
            x = radius_x * (math.cos(theta)) * (center_depth / example_scale)

        y = radius_y * math.sin(theta) * (center_depth / example_scale)
        z = radius_z * math.sin(theta) * (center_depth / example_scale)
        spiral_positions.append(torch.tensor([x, y, z], device=device))

    for pos in spiral_positions:
        if inwards:
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at)
        else:
            view_matrix = look_at_matrix(initial_camera_pos, look_at + pos)
        trajectory.append(view_matrix)
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_orbit_trajectory(
    world_to_camera_matrix,
    center_depth,
    n_steps=13,
    angle=math.pi / 4,
    axis="y",
    direction="right",
    device="cuda",
):
    """
    Create a constant-radius orbit around the fixed look-at point [0, 0, center_depth].
    The camera always looks at the point while moving along a circular arc.

    Args:
        world_to_camera_matrix (torch.Tensor): Base 4x4 world-to-camera transform to post-multiply.
        center_depth (float or tensor): Z of the look-at point; also equals initial radius.
        n_steps (int): Number of frames in the orbit.
        angle (float): Angular sweep (radians) away from the initial pose (theta0 = pi).
        axis (str): 'y' for horizontal (left/right) orbit, 'x' for vertical (up/down) orbit.
        direction (str): 'right'/'left' for horizontal, 'up'/'down' for vertical; sets sweep sign.
        device (str): Device where tensors are created.

    Returns:
        torch.Tensor: [n_steps, 4, 4] sequence of world-to-camera matrices.
    """
    # Resolve scalar radius
    try:
        r = float(center_depth)
    except Exception:
        r = center_depth.item() if hasattr(center_depth, "item") else float(center_depth)

    look_at = torch.tensor([0.0, 0.0, r], device=device)

    # Determine sweep direction
    if axis == "y":
        sweep_sign = -1.0 if direction == "right" else 1.0
    elif axis == "x":
        sweep_sign = 1.0 if direction == "up" else -1.0
    else:
        raise ValueError("axis must be 'x' or 'y'")

    trajectory = []
    for i in range(n_steps):
        frac = 0.0 if n_steps <= 1 else (i / (n_steps - 1))
        # Sweep from 0 -> angle relative to the initial heading (theta0=pi)
        theta = math.pi + sweep_sign * (angle * frac)

        if axis == "y":
            # Horizontal orbit (rotate around Y axis)
            cx = r * math.sin(theta)
            cy = 0.0
            cz = r + r * math.cos(theta)
        else:
            # Vertical orbit (rotate around X axis)
            cx = 0.0
            cy = r * math.sin(theta)
            cz = r + r * math.cos(theta)

        camera_pos = torch.tensor([cx, cy, cz], device=device)
        view_matrix = look_at_matrix(camera_pos, look_at)
        trajectory.append(view_matrix)

    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_horizontal_trajectory(
    world_to_camera_matrix, center_depth, right=True, n_steps=13, distance=0.1, device="cuda", axis="x", outwards=False
):  
    if axis == "z":
        look_at = torch.tensor([0.0, 0.0, center_depth * (distance+1.0)]).to(device)
    else:
        look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)
    # Spiral motion key points
    trajectory = []
    translation_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)

    for i in range(n_steps):
        if axis == "x":
            x = i * distance * center_depth / n_steps * (1 if right else -1)
            y = 0
            z = 0
        elif axis == "y":
            x = 0
            y = i * distance * center_depth / n_steps * (1 if right else -1)
            z = 0
        elif axis == "z":
            x = 0
            y = 0
            z = i * distance * center_depth / n_steps * (1 if right else -1)
        else:
            raise ValueError("Axis should be x, y or z")

        translation_positions.append(torch.tensor([x, y, z], device=device))

    for pos in translation_positions:
        camera_pos = initial_camera_pos + pos
        if outwards:
            _look_at = look_at + pos * 2
        else:
            _look_at = look_at
        view_matrix = look_at_matrix(camera_pos, _look_at)
        trajectory.append(view_matrix)
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_horizontal_with_noise_trajectory(
    world_to_camera_matrix, center_depth, right=True, n_steps=13, distance=0.1, device="cuda", axis="x", outwards=False, noise_percentage=0.0001
):
    """Create a horizontal trajectory with the same primary axis movement as create_horizontal_trajectory,
    but with random noise added to the two perpendicular dimensions.
    
    Args:
        world_to_camera_matrix: Transformation matrix from world to camera space
        center_depth: Depth at the center
        right: Direction of movement along primary axis
        n_steps: Number of steps in the trajectory
        distance: Base distance (used for movement and noise scaling)
        device: Device to create tensors on
        axis: Primary axis ("x", "y", or "z") - movement along this axis
        outwards: Whether to move look_at outwards
        noise_percentage: Percentage of distance * center_depth to use as noise range (default 0.0001)
    
    Returns:
        Trajectory tensor with primary axis movement plus noise in perpendicular dimensions
    """
    if axis == "z":
        look_at = torch.tensor([0.0, 0.0, center_depth * (distance+1.0)]).to(device)
    else:
        look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)
    
    trajectory = []
    translation_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)
    
    # Calculate noise magnitude as percentage of distance * center_depth
    noise_magnitude = noise_percentage * distance * center_depth
    
    for i in range(n_steps):
        # Primary axis movement (same as create_horizontal_trajectory)
        primary_movement = i * distance * center_depth / n_steps * (1 if right else -1)
        
        # Sample random noise for the two perpendicular dimensions independently per timestep
        # Using Gaussian distribution centered around zero with std = noise_magnitude
        noise1 = torch.randn(1, device=device).item() * noise_magnitude
        noise2 = torch.randn(1, device=device).item() * noise_magnitude
        
        if axis == "x":
            x = primary_movement
            y = noise1
            z = noise2
        elif axis == "y":
            x = noise1
            y = primary_movement
            z = noise2
        elif axis == "z":
            x = noise1
            y = noise2
            z = primary_movement
        else:
            raise ValueError("Axis should be x, y or z")
        
        translation_positions.append(torch.tensor([x, y, z], device=device))
    
    for pos in translation_positions:
        camera_pos = initial_camera_pos + pos
        if outwards:
            _look_at = look_at + pos * 2
        else:
            _look_at = look_at
        view_matrix = look_at_matrix(camera_pos, _look_at)
        trajectory.append(view_matrix)
    
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_horizontal_zoom_with_bend_trajectory(
    world_to_camera_matrix, center_depth, right=True, n_steps=13, distance=0.1, device="cuda", axis="z", outwards=False, bend_percentage_in=0.04, bend_percentage_out=0.12, axis_bend="y"
):
    """Create a horizontal trajectory with bend applied to a perpendicular axis.
    
    Args:
        world_to_camera_matrix: Transformation matrix from world to camera space
        center_depth: Depth at the center
        right: Direction of movement along primary axis (True = zoom in, False = zoom out)
        n_steps: Number of steps in the trajectory
        distance: Base distance for movement
        device: Device to create tensors on
        axis: Primary axis ("x", "y", or "z") - movement along this axis
        outwards: Whether to move look_at outwards
        bend_percentage_in: Percentage of distance * center_depth to use as bend for zoom in (default 0.04)
        bend_percentage_out: Percentage of distance * center_depth to use as bend for zoom out (default 0.12)
        axis_bend: Axis to apply bend to (default "y" for vertical bend when axis="z")
                   When right=True, bend goes positive; when right=False, bend goes negative
    
    Returns:
        Trajectory tensor with primary axis movement plus bend on perpendicular axis
    """
    if axis == "z":
        look_at = torch.tensor([0.0, 0.0, center_depth * (distance+1.0)]).to(device)
    else:
        look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)
    
    trajectory = []
    translation_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)
    
    # Select bend percentage based on direction
    bend_percentage = bend_percentage_in if right else bend_percentage_out
    # Calculate bend magnitude as percentage of distance * center_depth
    bend_magnitude = bend_percentage * distance * center_depth
    
    for i in range(n_steps):
        # Primary axis movement (same as create_horizontal_trajectory)
        primary_movement = i * distance * center_depth / n_steps * (1 if right else -1)
        
        # Bend: positive when right=True, negative when right=False
        bend_base = bend_magnitude * i / n_steps
        bend_value = bend_base if right else -bend_base
        
        if axis == "x":
            x = primary_movement
            if axis_bend == "y":
                y = bend_value
                z = 0
            elif axis_bend == "z":
                y = 0
                z = bend_value
            else:
                raise ValueError(f"axis_bend must be perpendicular to axis. For axis='x', use 'y' or 'z'")
        elif axis == "y":
            y = primary_movement
            if axis_bend == "x":
                x = bend_value
                z = 0
            elif axis_bend == "z":
                x = 0
                z = bend_value
            else:
                raise ValueError(f"axis_bend must be perpendicular to axis. For axis='y', use 'x' or 'z'")
        elif axis == "z":
            z = primary_movement
            if axis_bend == "x":
                x = bend_value
                y = 0
            elif axis_bend == "y":
                x = 0
                y = bend_value
            else:
                raise ValueError(f"axis_bend must be perpendicular to axis. For axis='z', use 'x' or 'y'")
        else:
            raise ValueError("Axis should be x, y or z")
        
        translation_positions.append(torch.tensor([x, y, z], device=device))
    
    for pos in translation_positions:
        camera_pos = initial_camera_pos + pos
        if outwards:
            _look_at = look_at + pos * 2
        else:
            _look_at = look_at
        view_matrix = look_at_matrix(camera_pos, _look_at)
        trajectory.append(view_matrix)
    
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_horizontal_zoom_with_noise_and_bend_trajectory(
    world_to_camera_matrix, center_depth, right=True, n_steps=13, distance=0.1, device="cuda", axis="z", outwards=False, noise_percentage=0.0001, bend_percentage_in=0.04, bend_percentage_out=0.12, axis_bend="y"
):
    """Create a horizontal trajectory with Gaussian noise and bend applied to a perpendicular axis.
    
    Args:
        world_to_camera_matrix: Transformation matrix from world to camera space
        center_depth: Depth at the center
        right: Direction of movement along primary axis (True = zoom in, False = zoom out)
        n_steps: Number of steps in the trajectory
        distance: Base distance for movement
        device: Device to create tensors on
        axis: Primary axis ("x", "y", or "z") - movement along this axis
        outwards: Whether to move look_at outwards
        noise_percentage: Percentage of distance * center_depth to use as noise range (default 0.0001)
        bend_percentage_in: Percentage of distance * center_depth to use as bend for zoom in (default 0.04)
        bend_percentage_out: Percentage of distance * center_depth to use as bend for zoom out (default 0.12)
        axis_bend: Axis to apply bend to (default "y" for vertical bend when axis="z")
                   When right=True, bend goes positive; when right=False, bend goes negative
    
    Returns:
        Trajectory tensor with primary axis movement plus noise and bend on perpendicular axis
    """
    if axis == "z":
        look_at = torch.tensor([0.0, 0.0, center_depth * (distance+1.0)]).to(device)
    else:
        look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)
    
    trajectory = []
    translation_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)
    
    # Calculate noise magnitude
    noise_magnitude = noise_percentage * distance * center_depth
    # Select bend percentage based on direction
    bend_percentage = bend_percentage_in if right else bend_percentage_out
    # Calculate bend magnitude as percentage of distance * center_depth
    bend_magnitude = bend_percentage * distance * center_depth
    
    for i in range(n_steps):
        # Primary axis movement (same as create_horizontal_trajectory)
        primary_movement = i * distance * center_depth / n_steps * (1 if right else -1)
        
        # Bend: positive when right=True, negative when right=False
        bend_base = bend_magnitude * i / n_steps
        bend_value = bend_base if right else -bend_base
        
        # Sample random noise for the two perpendicular dimensions independently per timestep
        # Using Gaussian distribution centered around zero
        noise1 = torch.randn(1, device=device).item() * noise_magnitude
        noise2 = torch.randn(1, device=device).item() * noise_magnitude
        
        if axis == "x":
            x = primary_movement
            if axis_bend == "y":
                y = bend_value + noise1
                z = noise2
            elif axis_bend == "z":
                y = noise1
                z = bend_value + noise2
            else:
                raise ValueError(f"axis_bend must be perpendicular to axis. For axis='x', use 'y' or 'z'")
        elif axis == "y":
            y = primary_movement
            if axis_bend == "x":
                x = bend_value + noise1
                z = noise2
            elif axis_bend == "z":
                x = noise1
                z = bend_value + noise2
            else:
                raise ValueError(f"axis_bend must be perpendicular to axis. For axis='y', use 'x' or 'z'")
        elif axis == "z":
            z = primary_movement
            if axis_bend == "x":
                x = bend_value + noise1
                y = noise2
            elif axis_bend == "y":
                x = noise1
                y = bend_value + noise2
            else:
                raise ValueError(f"axis_bend must be perpendicular to axis. For axis='z', use 'x' or 'y'")
        else:
            raise ValueError("Axis should be x, y or z")
        
        translation_positions.append(torch.tensor([x, y, z], device=device))
    
    for pos in translation_positions:
        camera_pos = initial_camera_pos + pos
        if outwards:
            _look_at = look_at + pos * 2
        else:
            _look_at = look_at
        view_matrix = look_at_matrix(camera_pos, _look_at)
        trajectory.append(view_matrix)
    
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_back_trajectory(
    forward_trajectory,
    center_depth,
    radius_x=0.3,
    radius_y=0.2,
    radius_z=0.0,
    inwards=True,
    right=True,
    device="cuda",
    invert_pos=False,
):
    look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)
    if not inwards:
        look_at *= -1

    # Spiral motion key points
    trajectory = []
    spiral_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)  # world_to_camera_matrix[:3, 3].clone()
    n_steps = forward_trajectory.shape[0] - 1
    for i in range(n_steps):
        theta = 2 * math.pi * i / (n_steps - 1)  # angle for each point
        x = radius_x * (math.cos(theta) - 1) * (1 if right else -1)
        y = radius_y * math.sin(theta)
        z = radius_z * math.sin(theta)
        spiral_positions.append(torch.tensor([x, y, z], device=device))

    for pos in spiral_positions:
        camera_pos = initial_camera_pos + pos
        view_matrix = look_at_matrix(camera_pos, look_at, invert_pos=invert_pos)
        trajectory.append(view_matrix)
    trajectory = torch.stack(trajectory)
    backward_trajectory = apply_transformation(trajectory, forward_trajectory[:n_steps].flip(0))
    return torch.cat([forward_trajectory, backward_trajectory])


def create_dolly_zoom_trajectory(
    world_to_camera_matrix, intrinsic, center_depth, n_steps=13, shift_z=-3, device="cuda"
):
    center_depth = center_depth * 0.185

    look_at = torch.tensor([0.0, 0.0, center_depth], device=device)

    trajectory = []
    intrinsics_list = []
    translation_positions = []
    initial_camera_pos = torch.tensor([0.0, 0.0, 0.0], device=device)

    f0 = intrinsic[0, 0]
    f1 = intrinsic[1, 1]

    Z_subject = center_depth  # The Z position of the subject. now use 1.0 as default
    z0 = initial_camera_pos[2].item()  # also use default

    D0 = Z_subject - z0  # Initial distance to the subject

    for i in range(n_steps):
        x = 0.0
        y = 0.0
        z = shift_z * i / (n_steps - 1)
        translation_positions.append(torch.tensor([x, y, z], device=device))

    for pos in translation_positions:
        camera_pos = initial_camera_pos - pos
        zi = -camera_pos[2].item()
        Di = Z_subject - zi

        # Create new intrinsic matrix
        new_intrinsic = intrinsic.clone()
        new_intrinsic[0, 0] = f0 * (D0 / Di)
        new_intrinsic[1, 1] = f1 * (D0 / Di)
        intrinsics_list.append(new_intrinsic)

        # Create the view matrix
        view_matrix = look_at_matrix(camera_pos, look_at)
        trajectory.append(view_matrix)

    trajectory = torch.stack(trajectory)
    intrinsics_list = torch.stack(intrinsics_list)

    # Apply transformation to trajectory
    transformed_trajectory = apply_transformation(trajectory, world_to_camera_matrix)
    return transformed_trajectory, intrinsics_list


def create_spiral_horizontal_trajectory(
    world_to_camera_matrix,
    center_depth,
    radius_x=0.03,
    radius_y=0.02,
    distance=0.1,           # zoom or horizontal move distance
    right=True,
    inwards=True,
    n_steps=20,
    num_circles=1,
    device="cuda",
    axis="x",               # "x", "y", or "z" (z = zoom)
):
    """
    Combine spiral motion with horizontal/zoom translation.
    - axis="x" or "y" → spiral + horizontal pan
    - axis="z" → spiral + zoom in/out
    """
    # Match your other functions' convention
    if axis == "z":
        look_at = torch.tensor([0.0, 0.0, center_depth * (distance + 1.0)], device=device)
    else:
        look_at = torch.tensor([0.0, 0.0, center_depth], device=device)

    trajectory = []
    initial_camera_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    theta_max = 2 * math.pi * num_circles

    for i in range(n_steps):
        t = i / (n_steps - 1)
        theta = theta_max * t

        # --- Spiral (on-the-spot orbit) ---
        if axis == "z":
            # Spiral in x/y plane while zooming along z
            x_spiral = radius_x * (math.cos(theta) - 1) * (1 if right else -1) * center_depth
            y_spiral = radius_y * math.sin(theta) * center_depth
            z_spiral = 0.0
        else:
            x_spiral = radius_x * math.cos(theta) * center_depth
            y_spiral = radius_y * math.sin(theta) * center_depth
            z_spiral = 0.0

        # --- Linear motion / zoom ---
        offset = t * distance * center_depth * (1 if right else -1)
        if axis == "x":
            x = x_spiral + offset
            y = y_spiral
            z = z_spiral
        elif axis == "y":
            x = x_spiral
            y = y_spiral + offset
            z = z_spiral
        elif axis == "z":
            x = x_spiral
            y = y_spiral
            z = z_spiral + offset  # zoom motion
        else:
            raise ValueError("Axis should be 'x', 'y', or 'z'")

        camera_pos = initial_camera_pos + torch.tensor([x, y, z], device=device)

        # Keep same look_at logic as your original spiral/horizontal versions
        if inwards:
            view_matrix = look_at_matrix(camera_pos, look_at)
        else:
            view_matrix = look_at_matrix(initial_camera_pos, look_at + camera_pos)

        trajectory.append(view_matrix)

    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_rotate_then_zoom_trajectory(
    world_to_camera_matrix,
    center_depth,
    n_steps=13,
    rotate_direction="right",
    zoom_direction="right",
    rotation_angle=20.0,
    zoom_distance=0.1,
    device="cuda",
):
    """
    Create a trajectory that first rotates on the spot (no translation), then zooms in/out.
    
    Args:
        world_to_camera_matrix (torch.Tensor): Base 4x4 world-to-camera transform.
        center_depth (float): Z of the look-at point.
        n_steps (int): Number of frames in the trajectory.
        rotate_direction (str): 'right' or 'left' - rotation direction (default: 'right').
        zoom_direction (str): 'right' or 'left' - zoom direction (default: 'right').
                            'right' = zoom in (forward), 'left' = zoom out (backward).
        rotation_angle (float): Total rotation angle in degrees (default: 20.0).
        zoom_distance (float): Zoom distance as fraction of center_depth.
        device (str): Device where tensors are created.
    
    Returns:
        torch.Tensor: [n_steps, 4, 4] sequence of world-to-camera matrices.
    """
    # Convert rotation angle from degrees to radians
    rotation_angle_rad = math.radians(rotation_angle)
    
    look_at = torch.tensor([0.0, 0.0, center_depth], device=device)
    initial_camera_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    
    trajectory = []
    half_steps = n_steps // 2
    
    # Determine rotation direction
    rotation_sign = 1.0 if rotate_direction == "right" else -1.0
    
    # Determine zoom direction: right = zoom in (forward), left = zoom out (backward)
    zoom_sign = 1.0 if zoom_direction == "right" else -1.0
    
    # Pre-compute the view matrix at the transition point (end of rotation, start of zoom)
    # This ensures smooth continuity between the two phases
    final_angle = rotation_sign * rotation_angle_rad
    cos_final = math.cos(final_angle)
    sin_final = math.sin(final_angle)
    final_rotation_matrix = torch.tensor(
        [
            [cos_final, 0, sin_final, 0],
            [0, 1, 0, 0],
            [-sin_final, 0, cos_final, 0],
            [0, 0, 0, 1],
        ],
        device=device,
    )
    transition_view_matrix = look_at_matrix(initial_camera_pos, look_at)
    transition_view_matrix = final_rotation_matrix @ transition_view_matrix
    
    # Extract the forward direction from the transition view matrix
    # In look_at_matrix, the third row (index 2) is the forward direction in world space
    rotation_part = transition_view_matrix[:3, :3]
    rotated_forward = rotation_part[2, :3].clone()  # Forward direction in world space
    rotated_forward = rotated_forward / torch.norm(rotated_forward)
    
    # Compute the effective look_at point that the camera is looking at after rotation
    # The camera at origin with rotated orientation is looking along rotated_forward
    # So the look_at point is: origin + rotated_forward * distance_to_look_at
    # The distance is the same as center_depth (distance from origin to original look_at)
    zoom_look_at = initial_camera_pos + rotated_forward * center_depth
    
    for i in range(n_steps):
        if i < half_steps:
            # First half: rotate on the spot (no translation)
            # Rotate the camera's view direction around Y-axis while keeping position fixed
            # The look_at point stays fixed, but we rotate the camera's orientation
            t = i / max(half_steps - 1, 1)  # Normalize to [0, 1]
            angle = rotation_sign * rotation_angle_rad * t
            
            # Create rotation matrix around Y-axis
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rotation_matrix = torch.tensor(
                [
                    [cos_a, 0, sin_a, 0],
                    [0, 1, 0, 0],
                    [-sin_a, 0, cos_a, 0],
                    [0, 0, 0, 1],
                ],
                device=device,
            )
            
            # Camera stays at origin, but rotates its orientation
            # Start with initial view matrix
            view_matrix = look_at_matrix(initial_camera_pos, look_at)
            # Apply rotation to the view matrix (this rotates the camera's orientation)
            view_matrix = rotation_matrix @ view_matrix
            
        else:
            # Second half: zoom along forward direction
            # Keep the rotation from the last rotation state, only change translation
            # The look_at point (zoom_look_at) is computed from rotation state for reference
            
            # Zoom progress in second half (starts from 0 at transition)
            t_zoom = (i - half_steps) / max(n_steps - half_steps - 1, 1)  # Normalize to [0, 1]
            zoom_offset = zoom_sign * zoom_distance * center_depth * t_zoom
            
            # Move camera along the forward direction from the transition point
            camera_pos = initial_camera_pos + rotated_forward * zoom_offset
            
            # Keep the exact same rotation as at transition, only update translation
            # Translation in world-to-camera: -R^T @ camera_pos
            # This ensures smooth continuity: at zoom_offset=0, view_matrix = transition_view_matrix
            view_matrix = transition_view_matrix.clone()
            view_matrix[:3, 3] = -rotation_part.T @ camera_pos
        
        trajectory.append(view_matrix)
    
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_rotate_spot_trajectory(
    world_to_camera_matrix,
    center_depth,
    n_steps=13,
    rotate_direction="right",
    rotation_angle=65.0,
    device="cuda",
):
    """
    Create a trajectory that rotates on the spot (no translation).
    
    Args:
        world_to_camera_matrix (torch.Tensor): Base 4x4 world-to-camera transform.
        center_depth (float): Z of the look-at point.
        n_steps (int): Number of frames in the trajectory.
        rotate_direction (str): 'right' or 'left' - rotation direction (default: 'right').
        rotation_angle (float): Total rotation angle in degrees (default: 65.0).
        device (str): Device where tensors are created.
    
    Returns:
        torch.Tensor: [n_steps, 4, 4] sequence of world-to-camera matrices.
    """
    # Convert rotation angle from degrees to radians
    rotation_angle_rad = math.radians(rotation_angle)
    
    look_at = torch.tensor([0.0, 0.0, center_depth], device=device)
    initial_camera_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    
    trajectory = []
    
    # Determine rotation direction
    rotation_sign = 1.0 if rotate_direction == "right" else -1.0
    
    for i in range(n_steps):
        # Rotate on the spot (no translation)
        # Rotate the camera's view direction around Y-axis while keeping position fixed
        # The look_at point stays fixed, but we rotate the camera's orientation
        t = i / max(n_steps - 1, 1)  # Normalize to [0, 1]
        angle = rotation_sign * rotation_angle_rad * t
        
        # Create rotation matrix around Y-axis
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation_matrix = torch.tensor(
            [
                [cos_a, 0, sin_a, 0],
                [0, 1, 0, 0],
                [-sin_a, 0, cos_a, 0],
                [0, 0, 0, 1],
            ],
            device=device,
        )
        
        # Camera stays at origin, but rotates its orientation
        # Start with initial view matrix
        view_matrix = look_at_matrix(initial_camera_pos, look_at)
        # Apply rotation to the view matrix (this rotates the camera's orientation)
        view_matrix = rotation_matrix @ view_matrix
        
        trajectory.append(view_matrix)
    
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_rotate_spot_with_noise_trajectory(
    world_to_camera_matrix,
    center_depth,
    n_steps=13,
    rotate_direction="right",
    rotation_angle=65.0,
    device="cuda",
    noise_percentage=0.0001,
    distance=0.1,
):
    """
    Create a trajectory that mimics rotate_spot but adds tiny position offsets to avoid
    degenerate covariance errors in DA3 pose alignment during autoregressive generation.
    
    This is identical to rotate_spot except it adds minimal smooth position variation
    that's imperceptible visually but provides enough variation for pose alignment algorithms.
    
    Args:
        world_to_camera_matrix (torch.Tensor): Base 4x4 world-to-camera transform.
        center_depth (float): Z of the look-at point.
        n_steps (int): Number of frames in the trajectory.
        rotate_direction (str): 'right' or 'left' - rotation direction (default: 'right').
        rotation_angle (float): Total rotation angle in degrees (default: 65.0).
        device (str): Device where tensors are created.
        noise_percentage: Percentage of distance * center_depth for position offset (default 0.0001).
        distance: Base distance (used for offset scaling, default 0.1)
    
    Returns:
        torch.Tensor: [n_steps, 4, 4] sequence of world-to-camera matrices.
    """
    # Convert rotation angle from degrees to radians
    rotation_angle_rad = math.radians(rotation_angle)
    
    look_at = torch.tensor([0.0, 0.0, center_depth], device=device)
    initial_camera_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    
    trajectory = []
    
    # Determine rotation direction
    rotation_sign = 1.0 if rotate_direction == "right" else -1.0
    
    # Use a tiny smooth orbit instead of noise - this provides smooth, deterministic variation
    # The orbit radius is very small to avoid visible movement, but large enough for pose alignment
    # Use noise_percentage to scale the orbit radius
    orbit_radius = noise_percentage * distance * center_depth  # Small orbit radius
    
    for i in range(n_steps):
        # Rotate on the spot (with tiny orbit for pose alignment)
        # Rotate the camera's view direction around Y-axis while keeping position nearly fixed
        t = i / max(n_steps - 1, 1)  # Normalize to [0, 1]
        angle = rotation_sign * rotation_angle_rad * t
        
        # Create rotation matrix around Y-axis
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation_matrix = torch.tensor(
            [
                [cos_a, 0, sin_a, 0],
                [0, 1, 0, 0],
                [-sin_a, 0, cos_a, 0],
                [0, 0, 0, 1],
            ],
            device=device,
        )
        
        # Add tiny smooth orbit in X-Y plane (circular motion)
        # This provides smooth, deterministic variation for pose alignment
        # Use absolute frame index with very slow frequency to ensure smoothness across all frames
        # The orbit should be imperceptibly slow to be visually identical to pure rotation
        orbit_angle = 2 * math.pi * i * 0.001  # Very slow orbit based on absolute frame index
        orbit_x = orbit_radius * math.cos(orbit_angle)
        orbit_y = orbit_radius * math.sin(orbit_angle)
        orbit_z = 0.0
        
        # Camera position with tiny orbit (visually identical to pure rotation)
        camera_pos = initial_camera_pos + torch.tensor([orbit_x, orbit_y, orbit_z], device=device)
        
        # Start with view matrix at orbiting position
        view_matrix = look_at_matrix(camera_pos, look_at)
        # Apply rotation to the view matrix (this rotates the camera's orientation)
        view_matrix = rotation_matrix @ view_matrix
        
        trajectory.append(view_matrix)
    
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)
