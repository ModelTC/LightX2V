from types import SimpleNamespace

import torch
from PIL import Image

from lightx2v_train.model_zoo.longcat_image.data_process import LongCatImageEditDataProcessor
from lightx2v_train.model_zoo.longcat_image.longcat_image import LongCatImageModel
from lightx2v_train.model_zoo.longcat_image.longcat_image_edit import LongCatImageEditModel


def _config(name="longcat_image_edit"):
    return {
        "model": {
            "name": name,
            "pretrained_model_name_or_path": "/unused",
            "running_dtype": "fp32",
        },
        "data": {"processor": {"size_multiple": 16, "target_area": 1024 * 1024}},
        "inference": {"enable_cfg_renorm": True, "cfg_renorm_min": 0.0},
    }


def test_edit_processor_builds_aligned_source_conditions():
    processor = LongCatImageEditDataProcessor(_config())
    sample = {
        "inputs": {
            "target_image": Image.new("RGB", (200, 100)),
            "source_images": [Image.new("RGB", (200, 100))],
        },
        "conditioning": {"prompt": "edit"},
        "meta": {},
    }

    sample = processor(sample)
    inputs = sample["inputs"]
    height, width = sample["meta"]["target_height"], sample["meta"]["target_width"]

    assert height % 16 == width % 16 == 0
    assert inputs["target_pixel_values"].shape == (3, height, width)
    assert inputs["source_vae_pixel_values"].shape == (3, height, width)
    assert inputs["source_condition_image"].shape == (3, height // 2, width // 2)
    assert inputs["source_condition_image"].dtype == torch.uint8
    assert processor.infer_target_size(sample, 1, 1) == (height, width)


def test_edit_denoiser_input_appends_source_modality():
    model = LongCatImageEditModel(_config())
    model.vae = SimpleNamespace(config=SimpleNamespace(block_out_channels=[1, 1, 1, 1]))
    noisy_latent = torch.randn(1, 16, 8, 12)
    condition = {
        "prompt_embed": torch.zeros(1, 520, 8),
        "text_ids": torch.zeros(520, 3),
        "source_tokens": torch.randn(1, 24, 64),
        "source_height": 8,
        "source_width": 12,
    }

    denoiser_input = model.prepare_denoiser_input(noisy_latent, condition)

    assert denoiser_input.hidden_states.shape == (1, 48, 64)
    assert denoiser_input.target_token_length == 24
    assert torch.all(denoiser_input.img_ids[:24, 0] == 1)
    assert torch.all(denoiser_input.img_ids[24:, 0] == 2)
    assert torch.all(denoiser_input.img_ids[:, 1:] >= 520)


def test_cfg_renorm_is_only_used_for_text_to_image():
    positive = torch.tensor([[[3.0, 4.0]]])
    negative = torch.zeros_like(positive)

    text_to_image = LongCatImageModel(_config("longcat_image"))
    image_edit = LongCatImageEditModel(_config())

    renormed = text_to_image.apply_cfg(positive, negative, 2.0)
    plain = image_edit.apply_cfg(positive, negative, 2.0)

    assert torch.allclose(torch.linalg.vector_norm(renormed, dim=-1), torch.tensor([[5.0]]))
    assert torch.equal(plain, positive * 2)
