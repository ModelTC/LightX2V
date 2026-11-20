_VAE_MODULE_NAMES = {
    "conv_in": "conv1",
    "norm_out": "head.0",
    "conv_out": "head.2",
}
_RESIDUAL_NAMES = {
    "norm1": "residual.0",
    "conv1": "residual.2",
    "norm2": "residual.3",
    "conv2": "residual.6",
    "conv_shortcut": "shortcut",
}


def _native_key(key):
    match key.split("."):
        case ["quant_conv", param]:
            return f"conv1.{param}"
        case ["post_quant_conv", param]:
            return f"conv2.{param}"
        case [module, layer, param] if layer in _VAE_MODULE_NAMES:
            return f"{module}.{_VAE_MODULE_NAMES[layer]}.{param}"
        case [module, "mid_block", "resnets", index, layer, *tail]:
            block = f"{module}.middle.{2 * int(index)}"
        case [module, "mid_block", "attentions", "0", *tail]:
            return ".".join((module, "middle.1", *tail))
        case ["encoder", "down_blocks", index, layer, *tail]:
            block = f"encoder.downsamples.{index}"
        # Wan 2.1 flattens each stage's three residual blocks and optional upsampler.
        case ["decoder", "up_blocks", stage, "resnets", index, layer, *tail]:
            block = f"decoder.upsamples.{4 * int(stage) + int(index)}"
        case ["decoder", "up_blocks", stage, "upsamplers", "0", *tail]:
            return ".".join((f"decoder.upsamples.{4 * int(stage) + 3}", *tail))
        case _:
            raise ValueError(f"Unrecognized Diffusers Wan 2.1 VAE weight: {key}")
    return ".".join((block, _RESIDUAL_NAMES.get(layer, layer), *tail))


def remap_diffusers_wan_vae_weights(state):
    """Translate the Diffusers Wan 2.1 VAE layout to native checkpoint names."""
    return {_native_key(key): value for key, value in state.items()}
