from importlib import import_module


_MODULES = {
    "flash_attention": ".attention",
    "CausalWanModel": ".causal_model",
    "WanModel": ".model",
    "T5Decoder": ".t5",
    "T5Encoder": ".t5",
    "T5EncoderModel": ".t5",
    "T5Model": ".t5",
    "HuggingfaceTokenizer": ".tokenizers",
    "WanVAE": ".vae",
}

__all__ = [
    "WanVAE",
    "WanModel",
    "CausalWanModel",
    "T5Model",
    "T5Encoder",
    "T5Decoder",
    "T5EncoderModel",
    "HuggingfaceTokenizer",
    "flash_attention",
]


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
