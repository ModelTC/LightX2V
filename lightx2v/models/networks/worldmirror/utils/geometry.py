"""
Utilities for geometry operations.

References: DUSt3R, MoGe
"""

from numbers import Number
from typing import Tuple, Union

import numpy as np

from .warnings import no_warnings


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5

    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5

    return K


def angle_diff_vec3_numpy(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12):
    """
    Compute angle difference between 3D vectors using NumPy.

    Args:
        v1 (np.ndarray): First vector of shape (..., 3)
        v2 (np.ndarray): Second vector of shape (..., 3)
        eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-12.

    Returns:
        np.ndarray: Angle differences in radians
    """
    return np.arctan2(np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1) + eps, (v1 * v2).sum(axis=-1))


@no_warnings(category=RuntimeWarning)
def points_to_normals(point: np.ndarray, mask: np.ndarray = None, edge_threshold: float = None) -> np.ndarray:
    """
    Calculate normal map from point map. Value range is [-1, 1].

    Args:
        point (np.ndarray): shape (height, width, 3), point map
        mask (optional, np.ndarray): shape (height, width), dtype=bool. Mask of valid depth pixels. Defaults to None.
        edge_threshold (optional, float): threshold for the angle (in degrees) between the normal and the view direction. Defaults to None.

    Returns:
        normal (np.ndarray): shape (height, width, 3), normal map.
    """
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack(
        [
            np.cross(up, left, axis=-1),
            np.cross(left, down, axis=-1),
            np.cross(down, right, axis=-1),
            np.cross(right, up, axis=-1),
        ]
    )
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    valid = (
        np.stack(
            [
                mask[:-2, 1:-1] & mask[1:-1, :-2],
                mask[1:-1, :-2] & mask[2:, 1:-1],
                mask[2:, 1:-1] & mask[1:-1, 2:],
                mask[1:-1, 2:] & mask[:-2, 1:-1],
            ]
        )
        & mask[None, 1:-1, 1:-1]
    )
    if edge_threshold is not None:
        view_angle = angle_diff_vec3_numpy(pts[None, 1:-1, 1:-1, :], normal)
        view_angle = np.minimum(view_angle, np.pi - view_angle)
        valid = valid & (view_angle < np.deg2rad(edge_threshold))

    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    if has_mask:
        normal_mask = valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    else:
        return normal


def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Create a sliding window view of the input array along a specified axis.

    This function creates a memory-efficient view of the input array with sliding windows
    of the specified size and stride. The window dimension is appended to the end of the
    output array's shape. This is useful for operations like convolution, pooling, or
    any analysis that requires examining local neighborhoods in the data.

    Args:
        x (np.ndarray): Input array with shape (..., axis_size, ...)
        window_size (int): Size of the sliding window
        stride (int): Stride of the sliding window (step size between consecutive windows)
        axis (int, optional): Axis to perform sliding window over. Defaults to -1 (last axis)

    Returns:
        np.ndarray: View of the input array with shape (..., n_windows, ..., window_size),
                   where n_windows = (axis_size - window_size + 1) // stride

    Raises:
        AssertionError: If window_size is larger than the size of the specified axis

    Example:
        >>> x = np.array([1, 2, 3, 4, 5, 6])
        >>> sliding_window_1d(x, window_size=3, stride=2)
        array([[1, 2, 3],
               [3, 4, 5]])
    """
    assert x.shape[axis] >= window_size, f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    axis = axis % x.ndim
    shape = (
        *x.shape[:axis],
        (x.shape[axis] - window_size + 1) // stride,
        *x.shape[axis + 1 :],
        window_size,
    )
    strides = (
        *x.strides[:axis],
        stride * x.strides[axis],
        *x.strides[axis + 1 :],
        x.strides[axis],
    )
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding


def sliding_window_nd(
    x: np.ndarray,
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Create sliding windows along multiple dimensions of the input array.

    This function applies sliding_window_1d sequentially along multiple axes to create
    N-dimensional sliding windows. This is useful for operations that need to examine
    local neighborhoods in multiple dimensions simultaneously.

    Args:
        x (np.ndarray): Input array
        window_size (Tuple[int, ...]): Size of the sliding window for each axis
        stride (Tuple[int, ...]): Stride of the sliding window for each axis
        axis (Tuple[int, ...]): Axes to perform sliding window over

    Returns:
        np.ndarray: Array with sliding windows along the specified dimensions.
                   The window dimensions are appended to the end of the shape.

    Note:
        The length of window_size, stride, and axis tuples must be equal.

    Example:
        >>> x = np.random.rand(10, 10)
        >>> windows = sliding_window_nd(x, window_size=(3, 3), stride=(2, 2), axis=(-2, -1))
        >>> # Creates 3x3 sliding windows with stride 2 in both dimensions
    """
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def sliding_window_2d(
    x: np.ndarray,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
) -> np.ndarray:
    """
    Create 2D sliding windows over the input array.

    Convenience function for creating 2D sliding windows, commonly used for image
    processing operations like convolution, pooling, or patch extraction.

    Args:
        x (np.ndarray): Input array
        window_size (Union[int, Tuple[int, int]]): Size of the 2D sliding window.
                                                  If int, same size is used for both dimensions.
        stride (Union[int, Tuple[int, int]]): Stride of the 2D sliding window.
                                             If int, same stride is used for both dimensions.
        axis (Tuple[int, int], optional): Two axes to perform sliding window over.
                                         Defaults to (-2, -1) (last two dimensions).

    Returns:
        np.ndarray: Array with 2D sliding windows. The window dimensions (height, width)
                   are appended to the end of the shape.

    Example:
        >>> image = np.random.rand(100, 100)
        >>> patches = sliding_window_2d(image, window_size=8, stride=4)
        >>> # Creates 8x8 patches with stride 4 from the image
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)


def max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    """
    Perform 1D max pooling on the input array.

    Max pooling reduces the dimensionality of the input by taking the maximum value
    within each sliding window. This is commonly used in neural networks and signal
    processing for downsampling and feature extraction.

    Args:
        x (np.ndarray): Input array
        kernel_size (int): Size of the pooling kernel
        stride (int): Stride of the pooling operation
        padding (int, optional): Amount of padding to add on both sides. Defaults to 0.
        axis (int, optional): Axis to perform max pooling over. Defaults to -1.

    Returns:
        np.ndarray: Max pooled array with reduced size along the specified axis

    Note:
        - For floating point arrays, padding is done with np.nan values
        - For integer arrays, padding is done with the minimum value of the dtype
        - np.nanmax is used to handle NaN values in the computation

    Example:
        >>> x = np.array([1, 3, 2, 4, 5, 1, 2])
        >>> max_pool_1d(x, kernel_size=3, stride=2)
        array([3, 5, 2])
    """
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        padding_arr = np.full(
            (*x.shape[:axis], padding, *x.shape[axis + 1 :]),
            fill_value=fill_value,
            dtype=x.dtype,
        )
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool


def max_pool_nd(
    x: np.ndarray,
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    axis: Tuple[int, ...],
) -> np.ndarray:
    """
    Perform N-dimensional max pooling on the input array.

    This function applies max_pool_1d sequentially along multiple axes to perform
    multi-dimensional max pooling. This is useful for downsampling multi-dimensional
    data while preserving the most important features.

    Args:
        x (np.ndarray): Input array
        kernel_size (Tuple[int, ...]): Size of the pooling kernel for each axis
        stride (Tuple[int, ...]): Stride of the pooling operation for each axis
        padding (Tuple[int, ...]): Amount of padding for each axis
        axis (Tuple[int, ...]): Axes to perform max pooling over

    Returns:
        np.ndarray: Max pooled array with reduced size along the specified axes

    Note:
        The length of kernel_size, stride, padding, and axis tuples must be equal.
        Max pooling is applied sequentially along each axis in the order specified.

    Example:
        >>> x = np.random.rand(10, 10, 10)
        >>> pooled = max_pool_nd(x, kernel_size=(2, 2, 2), stride=(2, 2, 2),
        ...                      padding=(0, 0, 0), axis=(-3, -2, -1))
        >>> # Reduces each dimension by half with 2x2x2 max pooling
    """
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x


def max_pool_2d(
    x: np.ndarray,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    axis: Tuple[int, int] = (-2, -1),
):
    """
    Perform 2D max pooling on the input array.

    Convenience function for 2D max pooling, commonly used in computer vision
    and image processing for downsampling images while preserving important features.

    Args:
        x (np.ndarray): Input array
        kernel_size (Union[int, Tuple[int, int]]): Size of the 2D pooling kernel.
                                                  If int, same size is used for both dimensions.
        stride (Union[int, Tuple[int, int]]): Stride of the 2D pooling operation.
                                             If int, same stride is used for both dimensions.
        padding (Union[int, Tuple[int, int]]): Amount of padding for both dimensions.
                                              If int, same padding is used for both dimensions.
        axis (Tuple[int, int], optional): Two axes to perform max pooling over.
                                         Defaults to (-2, -1) (last two dimensions).

    Returns:
        np.ndarray: 2D max pooled array with reduced size along the specified axes

    Example:
        >>> image = np.random.rand(64, 64)
        >>> pooled = max_pool_2d(image, kernel_size=2, stride=2, padding=0)
        >>> # Reduces image size from 64x64 to 32x32 with 2x2 max pooling
    """
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)


@no_warnings(category=RuntimeWarning)
def depth_edge(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.

    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff = max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
    else:
        diff = max_pool_2d(
            np.where(mask, depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ) + max_pool_2d(
            np.where(mask, -depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol

    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def depth_aliasing(
    depth: np.ndarray,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the map that indicates the aliasing of x depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff_max = max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2) + depth
    else:
        diff_max = (
            max_pool_2d(
                np.where(mask, depth, -np.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
            - depth
        )
        diff_min = (
            max_pool_2d(
                np.where(mask, -depth, -np.inf),
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
            + depth
        )
    diff = np.minimum(diff_max, diff_min)

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


@no_warnings(category=RuntimeWarning)
def normals_edge(normals: np.ndarray, tol: float, kernel_size: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute the edge mask from normal map.

    Args:
        normal (np.ndarray): shape (..., height, width, 3), normal map
        tol (float): tolerance in degrees

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    assert normals.ndim >= 3 and normals.shape[-1] == 3, "normal should be of shape (..., height, width, 3)"
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    padding = kernel_size // 2
    normals_window = sliding_window_2d(
        np.pad(
            normals,
            (
                *([(0, 0)] * (normals.ndim - 3)),
                (padding, padding),
                (padding, padding),
                (0, 0),
            ),
            mode="edge",
        ),
        window_size=kernel_size,
        stride=1,
        axis=(-3, -2),
    )
    if mask is None:
        angle_diff = np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)).max(axis=(-2, -1))
    else:
        mask_window = sliding_window_2d(
            np.pad(
                mask,
                (*([(0, 0)] * (mask.ndim - 3)), (padding, padding), (padding, padding)),
                mode="edge",
            ),
            window_size=kernel_size,
            stride=1,
            axis=(-3, -2),
        )
        angle_diff = np.where(
            mask_window,
            np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)),
            0,
        ).max(axis=(-2, -1))

    angle_diff = max_pool_2d(angle_diff, kernel_size, stride=1, padding=kernel_size // 2)
    edge = angle_diff > np.deg2rad(tol)
    return edge
