import unittest
from types import SimpleNamespace

import torch

from lightx2v.models.input_encoders.hf.qwen25 import qwen25_vlforconditionalgeneration as qwen25_vl


class _Batch(dict[str, torch.Tensor]):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


class _Processor:
    def __init__(self, model_inputs):
        self.model_inputs = model_inputs

    def __call__(self, **kwargs):
        return self.model_inputs


class _TextEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(hidden_states=[torch.zeros((1, 3, 2))])


class Qwen25VLInputForwardingTest(unittest.TestCase):
    def test_infer_forwards_processor_multimodal_token_types(self):
        mm_token_type_ids = torch.tensor([[0, 1, 0]])
        model_inputs = _Batch(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            pixel_values=torch.zeros((1, 3)),
            image_grid_thw=torch.tensor([[1, 1, 1]]),
            mm_token_type_ids=mm_token_type_ids,
        )
        encoder = object.__new__(qwen25_vl.Qwen25_VLForConditionalGeneration_TextEncoder)
        encoder.cpu_offload = False
        encoder.is_layered = False
        encoder.USE_IMAGE_ID_IN_PROMPT = True
        encoder.config = {"task": "i2i"}
        encoder.prompt_template_encode = "{}"
        encoder.prompt_template_encode_start_idx = 0
        encoder.dtype = torch.float32
        encoder.processor = _Processor(model_inputs)
        encoder.text_encoder = _TextEncoder()
        encoder.preprocess_image = lambda image: (image, image, (1, 1), (1, 1))

        previous_device = qwen25_vl.AI_DEVICE
        qwen25_vl.AI_DEVICE = "cpu"
        try:
            encoder.infer(["prompt"], [object()])
        finally:
            qwen25_vl.AI_DEVICE = previous_device

        self.assertIs(encoder.text_encoder.calls[0]["mm_token_type_ids"], mm_token_type_ids)

        model_inputs.pop("mm_token_type_ids")
        qwen25_vl.AI_DEVICE = "cpu"
        try:
            encoder.infer(["prompt"], [object()])
        finally:
            qwen25_vl.AI_DEVICE = previous_device

        self.assertNotIn("mm_token_type_ids", encoder.text_encoder.calls[1])


if __name__ == "__main__":
    unittest.main()
