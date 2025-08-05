import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer


class TextEncoderHFLlamaModel:
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path
        self.init()
        self.load()

    def init(self):
        self.max_length = 351
        self.hidden_state_skip_layer = 2
        self.crop_start = 95
        self.prompt_template = (
            "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
            "1. The main content and theme of the video."
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
            "4. background environment, light, style and atmosphere."
            "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        )

    def load(self):
        self.model = AutoModel.from_pretrained(self.model_path, low_cpu_mem_usage=True).to(torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="right")

    def to_cpu(self):
        self.model = self.model.to("cpu")

    def to_cuda(self):
        self.model = self.model.to("cuda")

    @torch.no_grad()
    def infer(self, text, config):
        if config.cpu_offload:
            self.to_cuda()
        text = self.prompt_template.format(text)
        tokens = self.tokenizer(
            text,
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
        )

        last_hidden_state = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)][:, self.crop_start :]
        attention_mask = tokens["attention_mask"][:, self.crop_start :]
        if config.cpu_offload:
            self.to_cpu()
        return last_hidden_state, attention_mask


if __name__ == "__main__":
    model_path = ""
    model = TextEncoderHFLlamaModel(model_path, torch.device("cuda"))
    text = "A cat walks on the grass, realistic style."
    outputs = model.infer(text)
    logger.info(outputs)
