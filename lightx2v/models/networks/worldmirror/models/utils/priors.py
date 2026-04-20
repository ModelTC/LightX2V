import torch


def normalize_poses(extrinsics, padding=0.1, return_stats=False):
    """
    Normalize camera positions to unit cube, processing each batch separately

    Args:
        extrinsics: Camera extrinsic matrices with shape (B, S, 3, 4)
        padding: Boundary space within [0,1] range to prevent values near boundaries
        return_stats: Whether to return normalization statistics

    Returns:
        normalized_extrinsics: Normalized extrinsic matrices
        (optional) stats: Dictionary containing scale and translation information
    """
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device

    # Check input validity and handle NaN/Inf values
    for i in range(B):
        if torch.isnan(extrinsics[i]).any() or torch.isinf(extrinsics[i]).any():
            print(f"Warning: dataset sample has NaN/Inf in extrinsics")
            extrinsics[i] = torch.nan_to_num(extrinsics[i], nan=0.0, posinf=1e6, neginf=-1e6)

    normalized_extrinsics = extrinsics.clone()

    # Store normalization parameters if needed
    if return_stats:
        stats = {"scale_factors": torch.zeros(B, device=device), "translation_vectors": torch.zeros(B, 3, device=device)}

    for b in range(B):
        # Extract camera positions for this batch
        positions = extrinsics[b, :, :3, 3]  # (S, 3)

        # Filter valid positions to ignore outliers
        valid_mask = torch.isfinite(positions).all(dim=1)  # (S,)

        if valid_mask.sum() == 0:
            # No valid positions, use default values
            print(f"Warning: Batch {b} has no valid camera positions")
            normalized_extrinsics[b, :, :3, 3] = 0.5  # Place at center
            if return_stats:
                stats["scale_factors"][b] = 1.0
                stats["translation_vectors"][b] = 0.0
            continue

        valid_positions = positions[valid_mask]

        # Calculate bounds using percentiles for robustness
        if valid_positions.shape[0] > 10:
            # Use 5% and 95% percentiles instead of min/max
            min_pos = torch.quantile(valid_positions, 0.05, dim=0)
            max_pos = torch.quantile(valid_positions, 0.95, dim=0)
        else:
            # Too few samples, use min/max
            min_pos = torch.min(valid_positions, dim=0)[0]
            max_pos = torch.max(valid_positions, dim=0)[0]

        # Calculate scale factor considering all dimensions
        pos_range = max_pos - min_pos

        # Add small epsilon to prevent dimension collapse
        eps = torch.maximum(torch.tensor(1e-6, device=device), torch.abs(max_pos) * 1e-6)
        pos_range = torch.maximum(pos_range, eps)

        # Use maximum range as scale factor for uniform scaling
        scale_factor = torch.max(pos_range)
        scale_factor = torch.clamp(scale_factor, min=1e-6, max=1e6)

        # Calculate center point for centering
        center = (min_pos + max_pos) / 2.0

        # Normalize: center first, then scale with padding
        actual_scale = scale_factor / (1 - 2 * padding)
        normalized_positions = (positions - center) / actual_scale + 0.5

        # Ensure all values are within valid range
        normalized_positions = torch.clamp(normalized_positions, 0.0, 1.0)

        # Handle invalid positions by setting them to scene center
        invalid_mask = ~torch.isfinite(positions).all(dim=1)
        if invalid_mask.any():
            normalized_positions[invalid_mask] = 0.5

        normalized_extrinsics[b, :, :3, 3] = normalized_positions

        if return_stats:
            stats["scale_factors"][b] = actual_scale
            stats["translation_vectors"][b] = center

    # Final validation
    assert torch.isfinite(normalized_extrinsics).all(), "Output contains non-finite values"

    if return_stats:
        return normalized_extrinsics, stats
    return normalized_extrinsics


def normalize_depth(depth, eps=1e-6, min_percentile=1, max_percentile=99):
    """
    Normalize depth values to [0, 1] range using percentile-based scaling.

    Args:
        depth: Input depth tensor with shape (B, S, H, W)
        eps: Small epsilon value to prevent division by zero
        min_percentile: Lower percentile for robust min calculation (default: 1)
        max_percentile: Upper percentile for robust max calculation (default: 99)

    Returns:
        normalized_depth: Depth tensor normalized to [0, 1] range with same shape (B, S, H, W)
    """
    B, S, H, W = depth.shape
    depth = depth.flatten(0, 1)  # [B*S, H, W]

    # Handle invalid values
    depth = torch.nan_to_num(depth, nan=0.0, posinf=1e6, neginf=0.0)

    normalized_list = []
    for i in range(depth.shape[0]):
        depth_img = depth[i]  # [H, W]
        depth_flat = depth_img.flatten()

        # Filter out zero values if needed
        non_zero_mask = depth_flat > 0
        if non_zero_mask.sum() > 0:
            values_to_use = depth_flat[non_zero_mask]
        else:
            values_to_use = depth_flat

        # Only calculate percentiles when there are enough values
        if values_to_use.numel() > 100:  # Ensure enough samples for percentile calculation
            # Calculate min and max percentiles
            depth_min = torch.quantile(values_to_use, min_percentile / 100.0)
            depth_max = torch.quantile(values_to_use, max_percentile / 100.0)
        else:
            # If too few samples, use min/max values
            depth_min = values_to_use.min()
            depth_max = values_to_use.max()

        # Handle case where max equals min
        if depth_max == depth_min:
            depth_max = depth_min + 1.0

        # Use relative epsilon
        scale = torch.abs(depth_max - depth_min)
        eps_val = max(eps, scale.item() * eps)

        # Perform normalization
        depth_norm_img = (depth_img - depth_min) / (depth_max - depth_min + eps_val)

        # Ensure output is within [0,1] range
        depth_norm_img = torch.clamp(depth_norm_img, 0.0, 1.0)

        normalized_list.append(depth_norm_img)

    # Recombine all normalized images
    depth_norm = torch.stack(normalized_list)

    return depth_norm.reshape(B, S, H, W)
