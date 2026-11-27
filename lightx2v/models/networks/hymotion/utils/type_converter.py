def get_module_device(module):
    # Bound operators retain tensors as buffers rather than nn.Parameters.
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    raise ValueError("Module has no tensors")
