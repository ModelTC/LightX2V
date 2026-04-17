"""Visual utilities for HuggingFace integration.

References: https://github.com/facebookresearch/vggt
"""

import copy
import os
from typing import Tuple

import cv2
import matplotlib
import numpy as np
import requests
import trimesh
from scipy.spatial.transform import Rotation


def segment_sky(image_or_path, onnx_session):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_or_path: Path to input image (str) or BGR numpy array (H, W, 3)
        onnx_session: ONNX runtime session with loaded model

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    if isinstance(image_or_path, (str, os.PathLike)):
        image = cv2.imread(str(image_or_path))
    else:
        image = image_or_path
    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def create_image_mesh(
    *image_data: np.ndarray,
    mask: np.ndarray = None,
    triangulate: bool = False,
    return_vertex_indices: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Create a mesh from image data using pixel coordinates as vertices and grid connections as faces.

    Args:
        *image_data (np.ndarray): Image arrays with shape (height, width, [channels])
        mask (np.ndarray, optional): Boolean mask with shape (height, width). Defaults to None.
        triangulate (bool): Convert quad faces to triangular faces. Defaults to False.
        return_vertex_indices (bool): Include vertex indices in output. Defaults to False.

    Returns:
        faces (np.ndarray): Face connectivity array. Shape (N, 4) for quads or (N, 3) for triangles
        *vertex_data (np.ndarray): Vertex attributes corresponding to input image_data
        vertex_indices (np.ndarray, optional): Original vertex indices if return_vertex_indices=True
    """
    # Validate inputs
    assert (len(image_data) > 0) or (mask is not None), "Need at least one image or mask"

    if mask is None:
        height, width = image_data[0].shape[:2]
    else:
        height, width = mask.shape

    # Check all images have same dimensions
    for img in image_data:
        assert img.shape[:2] == (height, width), "All images must have same height and width"

    # Create quad faces connecting neighboring pixels
    base_quad = np.stack(
        [
            np.arange(0, width - 1, dtype=np.int32),  # bottom-left
            np.arange(width, 2 * width - 1, dtype=np.int32),  # top-left
            np.arange(1 + width, 2 * width, dtype=np.int32),  # top-right
            np.arange(1, width, dtype=np.int32),  # bottom-right
        ],
        axis=1,
    )

    # Replicate quad pattern for all rows
    row_offsets = np.arange(0, (height - 1) * width, width, dtype=np.int32)
    faces = (row_offsets[:, None, None] + base_quad[None, :, :]).reshape((-1, 4))

    if mask is None:
        # No masking - use all faces and vertices
        if triangulate:
            faces = _convert_quads_to_triangles(faces)

        output = [faces]
        for img in image_data:
            output.append(img.reshape(-1, *img.shape[2:]))

        if return_vertex_indices:
            output.append(np.arange(height * width, dtype=np.int32))

        return tuple(output)
    else:
        # Apply mask - only keep faces where all 4 corners are valid
        valid_quads = (mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]).ravel()
        faces = faces[valid_quads]

        if triangulate:
            faces = _convert_quads_to_triangles(faces)

        # Remove unused vertices and remap face indices
        num_face_vertices = faces.shape[-1]
        unique_vertices, remapped_indices = np.unique(faces, return_inverse=True)
        faces = remapped_indices.astype(np.int32).reshape(-1, num_face_vertices)

        output = [faces]
        for img in image_data:
            flattened_img = img.reshape(-1, *img.shape[2:])
            output.append(flattened_img[unique_vertices])

        if return_vertex_indices:
            output.append(unique_vertices)

        return tuple(output)


def _convert_quads_to_triangles(quad_faces: np.ndarray) -> np.ndarray:
    """Convert quadrilateral faces to triangular faces."""
    if quad_faces.shape[-1] == 3:
        return quad_faces  # Already triangular

    num_vertices_per_face = quad_faces.shape[-1]
    triangle_indices = np.stack(
        [
            np.zeros(num_vertices_per_face - 2, dtype=int),  # First vertex
            np.arange(1, num_vertices_per_face - 1, dtype=int),  # Sequential vertices
            np.arange(2, num_vertices_per_face, dtype=int),  # Next sequential vertices
        ],
        axis=1,
    )

    return quad_faces[:, triangle_indices].reshape((-1, 3))


def convert_predictions_to_glb_scene(
    predictions,
    filter_by_frames="all",
    show_camera=True,
    mask_sky_bg=False,
    mask_ambiguous=False,
    as_mesh=True,
) -> trimesh.Scene:
    """
    Converts model predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - images: Input images (S, H, W, 3)
            - camera_poses: Camera extrinsic matrices (S, 3, 4)
        filter_by_frames (str): Frame filter specification (default: "all")
        show_camera (bool): Include camera visualization (default: True)
        mask_sky_bg (bool): Mask out sky background pixels (default: False)
        mask_ambiguous (bool): Apply final mask to filter ambiguous predictions (default: False)
        as_mesh (bool): Represent the data as a mesh instead of point cloud (default: False)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud/mesh and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    print("Building GLB scene")

    # Parse frame selection from filter string
    target_frame_index = None
    if filter_by_frames not in ["all", "All"]:
        try:
            # Extract numeric index before colon separator
            target_frame_index = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    # Validate required data in predictions
    print("Using Pointmap Branch")
    if "world_points" not in predictions:
        raise ValueError("world_points not found in predictions. Pointmap Branch requires 'world_points' key. Depthmap and Camera branches have been removed.")

    # Extract prediction data
    point_cloud_3d = predictions["world_points"]
    input_images = predictions["images"]
    extrinsic_matrices = predictions["camera_poses"]
    ambiguity_mask = predictions["final_mask"]
    sky_region_mask = predictions["sky_mask"]

    # Filter to single frame if specified
    if target_frame_index is not None:
        point_cloud_3d = point_cloud_3d[target_frame_index][None]
        input_images = input_images[target_frame_index][None]
        extrinsic_matrices = extrinsic_matrices[target_frame_index][None]
        ambiguity_mask = ambiguity_mask[target_frame_index][None]
        sky_region_mask = sky_region_mask[target_frame_index][None]

    # Flatten 3D points to vertex array
    flattened_vertices = point_cloud_3d.reshape(-1, 3)

    # Convert images to RGB color array
    if input_images.ndim == 4 and input_images.shape[1] == 3:  # NCHW format
        rgb_colors = np.transpose(input_images, (0, 2, 3, 1))
    else:  # Already in NHWC format
        rgb_colors = input_images
    rgb_colors = (rgb_colors.reshape(-1, 3) * 255).astype(np.uint8)

    # Build composite filtering mask
    valid_points_mask = np.ones(len(flattened_vertices), dtype=bool)

    # Apply ambiguity filtering if requested
    if mask_ambiguous:
        flat_ambiguity_mask = ambiguity_mask.reshape(-1)
        valid_points_mask = valid_points_mask & flat_ambiguity_mask

    # Apply sky region filtering if requested
    if mask_sky_bg:
        flat_sky_mask = sky_region_mask.reshape(-1)
        valid_points_mask = valid_points_mask & flat_sky_mask

    # Apply mask to filter vertices and colors
    filtered_vertices = flattened_vertices[valid_points_mask].copy()
    filtered_colors = rgb_colors[valid_points_mask].copy()

    # Handle empty geometry case
    if filtered_vertices is None or np.asarray(filtered_vertices).size == 0:
        filtered_vertices = np.array([[1, 0, 0]])
        filtered_colors = np.array([[255, 255, 255]])
        scene_scale_factor = 1
    else:
        # Compute scene scale from percentile-based bounding box
        percentile_lower = np.percentile(filtered_vertices, 5, axis=0)
        percentile_upper = np.percentile(filtered_vertices, 95, axis=0)
        scene_scale_factor = np.linalg.norm(percentile_upper - percentile_lower)

    # Initialize color mapping for cameras
    color_palette = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Create empty 3D scene container
    output_scene = trimesh.Scene()

    # Add geometry to scene based on representation type
    if as_mesh:
        # Mesh representation
        if target_frame_index is not None:
            # Single frame mesh generation
            frame_height, frame_width = point_cloud_3d.shape[1:3]

            # Prepare unfiltered data for mesh construction
            structured_points = point_cloud_3d.reshape(frame_height, frame_width, 3)

            # Convert image data to proper format
            if input_images.ndim == 4 and input_images.shape[1] == 3:  # NCHW format
                structured_colors = np.transpose(input_images[0], (1, 2, 0))
            else:  # Already in HWC format
                structured_colors = input_images[0]
            structured_colors *= 255

            # Get structured mask for mesh creation
            structured_mask = predictions["final_mask"][target_frame_index].reshape(frame_height, frame_width)

            # Build filtering mask
            mesh_filter_mask = structured_mask

            # Check for normal data availability
            mesh_normals = None
            if "normal" in predictions and predictions["normal"] is not None:
                # Extract normals for selected frame
                frame_normal_data = predictions["normal"][target_frame_index] if target_frame_index is not None else predictions["normal"][0]

                # Generate mesh with normal information
                mesh_faces, mesh_vertices, mesh_colors, mesh_normals = create_image_mesh(
                    structured_points * np.array([1, -1, 1], dtype=np.float32),
                    structured_colors / 255.0,
                    frame_normal_data * np.array([1, -1, 1], dtype=np.float32),
                    mask=mesh_filter_mask,
                    triangulate=True,
                    return_vertex_indices=False,
                )

                # Apply coordinate system transformation to normals
                mesh_normals = mesh_normals * np.array([1, -1, 1], dtype=np.float32)
            else:
                # Generate mesh without normal information
                mesh_faces, mesh_vertices, mesh_colors = create_image_mesh(
                    structured_points * np.array([1, -1, 1], dtype=np.float32),
                    structured_colors / 255.0,
                    mask=mesh_filter_mask,
                    triangulate=True,
                    return_vertex_indices=False,
                )

            # Construct trimesh object with optional normals
            geometry_mesh = trimesh.Trimesh(
                vertices=mesh_vertices * np.array([1, -1, 1], dtype=np.float32),
                faces=mesh_faces,
                vertex_colors=(mesh_colors * 255).astype(np.uint8),
                vertex_normals=(mesh_normals if mesh_normals is not None else None),
                process=False,
            )
            output_scene.add_geometry(geometry_mesh)
        else:
            # Multi-frame mesh generation
            print("Creating mesh for multi-frame data...")

            for frame_idx in range(point_cloud_3d.shape[0]):
                frame_height, frame_width = point_cloud_3d.shape[1:3]

                # Extract per-frame data
                frame_point_data = point_cloud_3d[frame_idx]
                frame_ambiguity_mask = predictions["final_mask"][frame_idx]
                frame_sky_mask = predictions["sky_mask"][frame_idx]

                # Extract frame image data
                if input_images.ndim == 4 and input_images.shape[1] == 3:  # NCHW format
                    frame_image_data = np.transpose(input_images[frame_idx], (1, 2, 0))
                else:  # Already in HWC format
                    frame_image_data = input_images[frame_idx]
                frame_image_data *= 255

                # Build per-frame filtering mask
                frame_filter_mask = np.ones((frame_height, frame_width), dtype=bool)

                # Apply ambiguity filtering if enabled
                if mask_ambiguous:
                    frame_filter_mask = frame_filter_mask & frame_ambiguity_mask

                # Apply sky filtering if enabled
                if mask_sky_bg:
                    frame_filter_mask = frame_filter_mask & frame_sky_mask

                # Generate mesh for current frame
                frame_faces, frame_vertices, frame_colors = create_image_mesh(
                    frame_point_data * np.array([1, -1, 1], dtype=np.float32),
                    frame_image_data / 255.0,
                    mask=frame_filter_mask,
                    triangulate=True,
                    return_vertex_indices=False,
                )

                frame_vertices = frame_vertices * np.array([1, -1, 1], dtype=np.float32)

                # Create trimesh object for current frame
                frame_geometry = trimesh.Trimesh(
                    vertices=frame_vertices,
                    faces=frame_faces,
                    vertex_colors=(frame_colors * 255).astype(np.uint8),
                    process=False,
                )
                output_scene.add_geometry(frame_geometry)
    else:
        # Point cloud representation
        point_cloud_geometry = trimesh.PointCloud(vertices=filtered_vertices, colors=filtered_colors)
        output_scene.add_geometry(point_cloud_geometry)

    # Add camera visualizations if requested
    num_camera_views = len(extrinsic_matrices)

    if show_camera:
        # Iterate through all camera views
        for camera_idx in range(num_camera_views):
            camera_extrinsic = extrinsic_matrices[camera_idx]
            camera_color_rgba = color_palette(camera_idx / num_camera_views)
            camera_color_rgb = tuple(int(255 * x) for x in camera_color_rgba[:3])

            integrate_camera_into_scene(output_scene, camera_extrinsic, camera_color_rgb, scene_scale_factor)

    # Define coordinate system transformation matrices
    opengl_transform = np.eye(4)
    opengl_transform[1, 1] = -1  # Flip Y axis
    opengl_transform[2, 2] = -1  # Flip Z axis

    # Define alignment rotation (180 degrees around Y-axis)
    alignment_rotation = np.eye(4)
    alignment_rotation[:3, :3] = Rotation.from_euler("y", 0, degrees=True).as_matrix()

    # Compute and apply final transformation
    scene_transformation = np.linalg.inv(extrinsic_matrices[0]) @ opengl_transform @ alignment_rotation
    output_scene.apply_transform(scene_transformation)

    print("GLB Scene built")
    return output_scene


def integrate_camera_into_scene(
    scene: trimesh.Scene,
    camera_transform: np.ndarray,
    camera_color: tuple,
    scale_factor: float,
):
    """
    Adds a camera visualization mesh to the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera visualization.
        camera_transform (np.ndarray): 4x4 transformation matrix for camera positioning.
        camera_color (tuple): RGB color tuple for the camera mesh.
        scale_factor (float): Scaling factor for the camera size relative to scene.
    """
    # Define camera dimensions based on scene scale
    camera_base_width = scale_factor * 0.05
    camera_cone_height = scale_factor * 0.1

    # Create base cone geometry for camera representation
    base_cone = trimesh.creation.cone(camera_base_width, camera_cone_height, sections=4)

    # Setup rotation transformation (45 degrees around z-axis)
    z_rotation_matrix = np.eye(4)
    z_rotation_matrix[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    z_rotation_matrix[2, 3] = -camera_cone_height

    # Setup OpenGL coordinate system conversion
    opengl_coord_transform = np.eye(4)
    opengl_coord_transform[1, 1] = -1  # Flip Y axis
    opengl_coord_transform[2, 2] = -1  # Flip Z axis

    # Combine all transformations
    final_transform = camera_transform @ opengl_coord_transform @ z_rotation_matrix

    # Create slight rotation for mesh variation
    minor_rotation = np.eye(4)
    minor_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    # Generate multiple vertex sets for complex camera geometry
    original_vertices = base_cone.vertices
    scaled_vertices = 0.95 * original_vertices
    rotated_vertices = apply_transformation_to_points(minor_rotation, original_vertices)

    # Combine all vertex sets
    all_vertices = np.concatenate([original_vertices, scaled_vertices, rotated_vertices])

    # Transform vertices to final position
    transformed_vertices = apply_transformation_to_points(final_transform, all_vertices)

    # Generate faces for the complete camera mesh
    camera_faces = generate_camera_mesh_faces(base_cone)

    # Create and configure the camera mesh
    camera_mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=camera_faces)
    camera_mesh.visual.face_colors[:, :3] = camera_color

    # Add the camera mesh to the scene
    scene.add_geometry(camera_mesh)


def apply_transformation_to_points(transform_matrix: np.ndarray, point_array: np.ndarray, output_dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation matrix to a collection of 3D points.

    Args:
        transform_matrix (np.ndarray): 4x4 transformation matrix to apply.
        point_array (np.ndarray): Array of points to transform.
        output_dim (int, optional): Target dimension for output points.

    Returns:
        np.ndarray: Array of transformed points.
    """
    point_array = np.asarray(point_array)
    original_shape = point_array.shape[:-1]
    target_dim = output_dim or point_array.shape[-1]

    # Transpose transformation matrix for matrix multiplication
    transposed_transform = transform_matrix.swapaxes(-1, -2)

    # Apply rotation/scaling and translation components
    transformed_points = point_array @ transposed_transform[..., :-1, :] + transposed_transform[..., -1:, :]

    # Extract desired dimensions and restore original shape
    final_result = transformed_points[..., :target_dim].reshape(*original_shape, target_dim)
    return final_result


def generate_camera_mesh_faces(base_cone_mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Generates face indices for a complex camera mesh composed of multiple cone layers.

    Args:
        base_cone_mesh (trimesh.Trimesh): Base cone geometry used as template.

    Returns:
        np.ndarray: Array of face indices defining the camera mesh topology.
    """
    face_indices = []
    vertex_count_per_cone = len(base_cone_mesh.vertices)

    # Process each face of the base cone
    for triangle_face in base_cone_mesh.faces:
        # Skip faces that include the cone tip (vertex 0)
        if 0 in triangle_face:
            continue

        # Get vertex indices for current triangle
        vertex_a, vertex_b, vertex_c = triangle_face

        # Calculate corresponding vertices in second and third cone layers
        vertex_a_layer2, vertex_b_layer2, vertex_c_layer2 = triangle_face + vertex_count_per_cone
        vertex_a_layer3, vertex_b_layer3, vertex_c_layer3 = triangle_face + 2 * vertex_count_per_cone

        # Create connecting faces between cone layers
        connecting_faces = [
            (vertex_a, vertex_b, vertex_b_layer2),
            (vertex_a, vertex_a_layer2, vertex_c),
            (vertex_c_layer2, vertex_b, vertex_c),
            (vertex_a, vertex_b, vertex_b_layer3),
            (vertex_a, vertex_a_layer3, vertex_c),
            (vertex_c_layer3, vertex_b, vertex_c),
        ]

        face_indices.extend(connecting_faces)

    # Add reverse-winding faces for proper mesh closure
    reversed_faces = [(vertex_c, vertex_b, vertex_a) for vertex_a, vertex_b, vertex_c in face_indices]
    face_indices.extend(reversed_faces)

    return np.array(face_indices)
