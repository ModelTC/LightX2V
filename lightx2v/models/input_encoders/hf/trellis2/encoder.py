# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, DINOv3ViTModel

from lightx2v.common.ops.inference_module import bind_linear_operators, cast_matrix_weights


class TrellisImageEncoder:
    def __init__(self, dino_path, rembg_path, device, linear_backend="Default"):
        self.device = device
        self.dino = DINOv3ViTModel.from_pretrained(dino_path, local_files_only=True).eval().to(device)
        self.rembg = AutoModelForImageSegmentation.from_pretrained(rembg_path, trust_remote_code=True, local_files_only=True).eval().to(device)
        for model in (self.dino, self.rembg):
            cast_matrix_weights(model)
            bind_linear_operators(model, linear_backend)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.rembg_transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), self.normalize])

    def preprocess(self, image):
        has_alpha = image.mode == "RGBA" and not np.all(np.asarray(image)[:, :, 3] == 255)
        scale = min(1, 1024 / max(image.size))
        if scale < 1:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        if not has_alpha:
            image = image.convert("RGB")
            values = self.rembg_transform(image).unsqueeze(0).to(self.device)
            mask = self.rembg(values)[-1].sigmoid().float().cpu()[0].squeeze()
            image.putalpha(transforms.ToPILImage()(mask).resize(image.size))
        rgba = np.asarray(image)
        points = np.argwhere(rgba[:, :, 3] > 0.8 * 255)
        if len(points) == 0:
            raise ValueError("Background removal produced an empty foreground")
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        size = int(max(x1 - x0, y1 - y0))
        image = image.crop((cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2))
        rgba = np.asarray(image).astype(np.float32) / 255
        return Image.fromarray((rgba[:, :, :3] * rgba[:, :, 3:4] * 255).astype(np.uint8))

    def __call__(self, image):
        image = self.preprocess(image).resize((512, 512), Image.Resampling.LANCZOS)
        image = torch.from_numpy(np.asarray(image).astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        image = self.normalize(image).to(device=self.device, dtype=torch.bfloat16)
        hidden = self.dino.embeddings(image, bool_masked_pos=None)
        positions = self.dino.rope_embeddings(image)
        for layer in self.dino.layer:
            hidden = layer(hidden, position_embeddings=positions)
        features = F.layer_norm(hidden, hidden.shape[-1:])
        return {"cond": features, "neg_cond": torch.zeros_like(features)}
