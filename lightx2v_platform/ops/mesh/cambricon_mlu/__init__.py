def fill_mesh_holes(vertices, faces, max_hole_perimeter=3e-2):
    try:
        from lightx2v_kernel_mlu import fill_mesh_holes as implementation
    except ImportError as error:
        raise RuntimeError(
            "TRELLIS.2 mesh hole filling on MLU requires lightx2v-kernel-mlu. "
            "Build it from lightx2v_kernel/cambricon_mlu with the matching "
            "torch_mlu and Neuware toolchain; see that directory's README.md."
        ) from error
    return implementation(vertices, faces, max_hole_perimeter)
