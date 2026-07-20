import sys
from pathlib import Path

_NATIVE_LTX2_ROOT = Path(__file__).resolve().parent


def ensure_ltx2_native_path():
    if not (_NATIVE_LTX2_ROOT / "ltx_core").is_dir():
        raise FileNotFoundError(f"Missing vendored LTX2 native package: {_NATIVE_LTX2_ROOT}")

    native_root = str(_NATIVE_LTX2_ROOT)
    if native_root in sys.path:
        sys.path.remove(native_root)
    sys.path.insert(0, native_root)

    loaded = sys.modules.get("ltx_core")
    if loaded is None:
        return

    origin = Path(getattr(loaded, "__file__", "")).resolve()
    try:
        origin.relative_to(_NATIVE_LTX2_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"ltx_core is already loaded from a non-native path: {origin}") from exc


ensure_ltx2_native_path()

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords  # noqa: E402
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder  # noqa: E402
from ltx_core.model.transformer.modality import Modality  # noqa: E402
from ltx_core.model.transformer.model_configurator import (  # noqa: E402
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
)
from ltx_core.text_encoders.gemma import (  # noqa: E402
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    convert_to_additive_mask,
    module_ops_from_gemma_root,
)
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape  # noqa: E402
from ltx_core.utils import find_matching_file  # noqa: E402

__all__ = [
    "AudioLatentShape",
    "AudioPatchifier",
    "EMBEDDINGS_PROCESSOR_KEY_OPS",
    "GEMMA_LLM_KEY_OPS",
    "GEMMA_MODEL_OPS",
    "EmbeddingsProcessorConfigurator",
    "GemmaTextEncoderConfigurator",
    "LTXModelConfigurator",
    "LTXV_MODEL_COMFY_RENAMING_MAP",
    "Modality",
    "SingleGPUModelBuilder",
    "SpatioTemporalScaleFactors",
    "VideoLatentPatchifier",
    "VideoLatentShape",
    "convert_to_additive_mask",
    "ensure_ltx2_native_path",
    "find_matching_file",
    "get_pixel_coords",
    "module_ops_from_gemma_root",
]
