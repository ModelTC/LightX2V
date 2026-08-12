def get_tensor_parallel_mm_type(use_all_gather_reduce=False):
    """Return the default tensor-parallel MM wrapper."""
    del use_all_gather_reduce
    return "TensorParallel"
