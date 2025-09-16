import io
import json
import os

import torch
from PIL import Image

from lightx2v.deploy.common.utils import class_try_catch_async


class BaseDataManager:
    def __init__(self):
        self.template_images_dir = None
        self.template_audios_dir = None
        self.template_videos_dir = None
        self.template_tasks_dir = None

    async def init(self):
        pass

    async def close(self):
        pass

    def fmt_path(self, base, filename, abs_path=None):
        if abs_path:
            return abs_path
        else:
            return os.path.join(base, filename)

    def to_device(self, data, device):
        if isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    async def save_bytes(self, bytes_data, filename, abs_path=None):
        raise NotImplementedError

    async def load_bytes(self, filename, abs_path=None):
        raise NotImplementedError

    async def delete_bytes(self, filename, abs_path=None):
        raise NotImplementedError

    async def presign_url(self, filename, abs_path=None):
        return None

    async def recurrent_save(self, data, prefix):
        if isinstance(data, dict):
            return {k: await self.recurrent_save(v, f"{prefix}-{k}") for k, v in data.items()}
        elif isinstance(data, list):
            return [await self.recurrent_save(v, f"{prefix}-{idx}") for idx, v in enumerate(data)]
        elif isinstance(data, torch.Tensor):
            save_path = prefix + ".pt"
            await self.save_tensor(data, save_path)
            return save_path
        elif isinstance(data, Image.Image):
            save_path = prefix + ".png"
            await self.save_image(data, save_path)
            return save_path
        else:
            return data

    async def recurrent_load(self, data, device, prefix):
        if isinstance(data, dict):
            return {k: await self.recurrent_load(v, device, f"{prefix}-{k}") for k, v in data.items()}
        elif isinstance(data, list):
            return [await self.recurrent_load(v, device, f"{prefix}-{idx}") for idx, v in enumerate(data)]
        elif isinstance(data, str) and data == prefix + ".pt":
            return await self.load_tensor(data, device)
        elif isinstance(data, str) and data == prefix + ".png":
            return await self.load_image(data)
        else:
            return data

    async def recurrent_delete(self, data, prefix):
        if isinstance(data, dict):
            return {k: await self.recurrent_delete(v, f"{prefix}-{k}") for k, v in data.items()}
        elif isinstance(data, list):
            return [await self.recurrent_delete(v, f"{prefix}-{idx}") for idx, v in enumerate(data)]
        elif isinstance(data, str) and data == prefix + ".pt":
            await self.delete_bytes(data)
        elif isinstance(data, str) and data == prefix + ".png":
            await self.delete_bytes(data)

    @class_try_catch_async
    async def save_object(self, data, filename):
        data = await self.recurrent_save(data, filename)
        bytes_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        await self.save_bytes(bytes_data, filename)

    @class_try_catch_async
    async def load_object(self, filename, device):
        bytes_data = await self.load_bytes(filename)
        data = json.loads(bytes_data.decode("utf-8"))
        data = await self.recurrent_load(data, device, filename)
        return data

    @class_try_catch_async
    async def delete_object(self, filename):
        bytes_data = await self.load_bytes(filename)
        data = json.loads(bytes_data.decode("utf-8"))
        await self.recurrent_delete(data, filename)
        await self.delete_bytes(filename)

    @class_try_catch_async
    async def save_tensor(self, data: torch.Tensor, filename):
        buffer = io.BytesIO()
        torch.save(data.to("cpu"), buffer)
        await self.save_bytes(buffer.getvalue(), filename)

    @class_try_catch_async
    async def load_tensor(self, filename, device):
        bytes_data = await self.load_bytes(filename)
        buffer = io.BytesIO(bytes_data)
        t = torch.load(io.BytesIO(bytes_data))
        t = t.to(device)
        return t

    @class_try_catch_async
    async def save_image(self, data: Image.Image, filename):
        buffer = io.BytesIO()
        data.save(buffer, format="PNG")
        await self.save_bytes(buffer.getvalue(), filename)

    @class_try_catch_async
    async def load_image(self, filename):
        bytes_data = await self.load_bytes(filename)
        buffer = io.BytesIO(bytes_data)
        img = Image.open(buffer).convert("RGB")
        return img

    def get_delete_func(self, type):
        maps = {
            "TENSOR": self.delete_bytes,
            "IMAGE": self.delete_bytes,
            "OBJECT": self.delete_object,
            "VIDEO": self.delete_bytes,
        }
        return maps[type]

    def get_template_dir(self, template_type):
        if template_type == "audios":
            return self.template_audios_dir
        elif template_type == "images":
            return self.template_images_dir
        elif template_type == "videos":
            return self.template_videos_dir
        elif template_type == "tasks":
            return self.template_tasks_dir
        else:
            raise ValueError(f"Invalid template type: {template_type}")

    @class_try_catch_async
    async def list_template_files(self, template_type):
        template_dir = self.get_template_dir(template_type)
        if template_dir is None:
            return []
        return await self.list_files(base_dir=template_dir)

    @class_try_catch_async
    async def load_template_file(self, template_type, filename):
        template_dir = self.get_template_dir(template_type)
        if template_dir is None:
            return None
        return await self.load_bytes(None, abs_path=os.path.join(template_dir, filename))

    @class_try_catch_async
    async def template_file_exists(self, template_type, filename):
        template_dir = self.get_template_dir(template_type)
        if template_dir is None:
            return None
        return await self.file_exists(None, abs_path=os.path.join(template_dir, filename))

    @class_try_catch_async
    async def delete_template_file(self, template_type, filename):
        template_dir = self.get_template_dir(template_type)
        if template_dir is None:
            return None
        return await self.delete_bytes(None, abs_path=os.path.join(template_dir, filename))

    @class_try_catch_async
    async def save_template_file(self, template_type, filename, bytes_data):
        template_dir = self.get_template_dir(template_type)
        if template_dir is None:
            return None
        return await self.save_bytes(bytes_data, None, abs_path=os.path.join(template_dir, filename))

    @class_try_catch_async
    async def presign_template_url(self, template_type, filename):
        template_dir = self.get_template_dir(template_type)
        if template_dir is None:
            return None
        return await self.presign_url(None, abs_path=os.path.join(template_dir, filename))


# Import data manager implementations
from .local_data_manager import LocalDataManager  # noqa
from .s3_data_manager import S3DataManager  # noqa

__all__ = ["BaseDataManager", "LocalDataManager", "S3DataManager"]
