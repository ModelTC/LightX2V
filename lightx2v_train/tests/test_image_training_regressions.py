import torch
from PIL import Image

from lightx2v_train.infer.image import ImageInferencer
from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.model_zoo.flux2.data_process import Flux2EditDataProcessor
from lightx2v_train.model_zoo.flux2.flux2_klein import Flux2KleinModel
from lightx2v_train.model_zoo.flux2.flux2_klein_edit import Flux2KleinEditModel
from lightx2v_train.model_zoo.qwen_image.data_process import QwenImageEditDataProcessor
from lightx2v_train.model_zoo.qwen_image.qwen_image import QwenImageModel
from lightx2v_train.model_zoo.qwen_image.qwen_image_edit import QwenImageEditModel


def _processor_config(target_area=64 * 96):
    return {
        "model": {
            "running_dtype": "fp32",
            "input_preprocessing": {"target_area": target_area},
        },
        "data": {
            "processor": {
                "size_multiple": 16,
                "target_area": target_area,
            },
        },
    }


def _edit_sample(*, target=True, target_size=None):
    inputs = {
        "source_images": [Image.new("RGB", (96, 64))],
    }
    if target:
        inputs["target_image"] = Image.new("RGB", (150, 100))
    meta = {}
    if target_size is not None:
        meta["target_height"], meta["target_width"] = target_size
    return {
        "inputs": inputs,
        "conditioning": {"prompt": "edit"},
        "meta": meta,
    }


def test_pcm_inference_applies_the_training_time_shift():
    config = {
        "model": {"running_dtype": "fp32"},
        "scheduler": {
            "time_shift_settings": {
                "do_time_shift": True,
                "dynamic_shift": True,
                "shift_x1": 256,
                "shift_x2": 4096,
                "shift_y1": 0.5,
                "shift_y2": 1.15,
                "patch_size": [2, 2],
            },
        },
        "inference": {"pcm_solver_steps": 100},
    }
    inferencer = ImageInferencer(config)
    latent_hw = (128, 128)
    actual = torch.tensor(
        inferencer._inference_sigmas(4, latent_hw=latent_hw),
        dtype=torch.float32,
    )
    raw_phase_boundaries = torch.tensor([1.0, 0.75, 0.5, 0.25])
    expected = inferencer.scheduler.time_shift(
        raw_phase_boundaries,
        latent_hw=latent_hw,
    )

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, raw_phase_boundaries)


def test_qwen_edit_records_actual_dmd_target_size():
    processor = QwenImageEditDataProcessor(_processor_config())
    processor._process_source_images = lambda images: ([], [])

    with_target = processor(_edit_sample(target=True))
    pixels = with_target["inputs"]["target_pixel_values"]
    assert with_target["meta"]["target_height"] == pixels.shape[-2]
    assert with_target["meta"]["target_width"] == pixels.shape[-1]

    without_target = processor(_edit_sample(target=False))
    assert without_target["meta"]["target_height"] > 0
    assert without_target["meta"]["target_width"] > 0
    assert without_target["meta"]["target_height"] % 32 == 0
    assert without_target["meta"]["target_width"] % 32 == 0


def test_flux2_edit_records_actual_dmd_target_size():
    processor = Flux2EditDataProcessor(_processor_config())
    processor._process_source = lambda image: (
        torch.zeros(3, 64, 96),
        (64, 96),
    )

    with_target = processor(_edit_sample(target=True))
    pixels = with_target["inputs"]["target_pixel_values"]
    assert with_target["meta"]["target_height"] == pixels.shape[-2]
    assert with_target["meta"]["target_width"] == pixels.shape[-1]

    without_target = processor(
        _edit_sample(target=False, target_size=(70, 100)),
    )
    assert without_target["meta"]["target_height"] % 16 == 0
    assert without_target["meta"]["target_width"] % 16 == 0


def test_qwen_and_flux2_klein_apply_dmd_cfg_norm_to_packed_tokens():
    config = {"model": {"running_dtype": "fp32"}}
    model_classes = (
        QwenImageModel,
        QwenImageEditModel,
        Flux2KleinModel,
        Flux2KleinEditModel,
    )

    for model_class in model_classes:
        model = model_class(config)
        capability = model.ensure_capabilities().require(
            DistributionMatchingCapability,
        )
        assert capability._guidance_in_denoiser_space is True
