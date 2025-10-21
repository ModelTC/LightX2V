import asyncio
import hashlib
import json
import os

import aioboto3
import tos
from botocore.client import Config
from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.deploy.data_manager import BaseDataManager


class S3DataManager(BaseDataManager):
    def __init__(self, config_string, template_dir, max_retries=3):
        super().__init__()
        self.name = "s3"
        self.config = json.loads(config_string)
        self.max_retries = max_retries
        self.bucket_name = self.config["bucket_name"]
        self.aws_access_key_id = self.config["aws_access_key_id"]
        self.aws_secret_access_key = self.config["aws_secret_access_key"]
        self.endpoint_url = self.config["endpoint_url"]
        self.base_path = self.config["base_path"]
        self.connect_timeout = self.config.get("connect_timeout", 60)
        self.read_timeout = self.config.get("read_timeout", 60)
        self.write_timeout = self.config.get("write_timeout", 10)
        self.addressing_style = self.config.get("addressing_style", None)
        self.region = self.config.get("region", None)
        self.cdn_url = self.config.get("cdn_url", "")
        self.session = None
        self.s3_client = None
        self.presign_client = None
        if template_dir:
            self.template_images_dir = os.path.join(template_dir, "images")
            self.template_audios_dir = os.path.join(template_dir, "audios")
            self.template_videos_dir = os.path.join(template_dir, "videos")
            self.template_tasks_dir = os.path.join(template_dir, "tasks")

    async def init_presign_client(self):
        # init tos client for volces.com
        if "volces.com" in self.endpoint_url:
            self.presign_client = tos.TosClientV2(
                self.aws_access_key_id,
                self.aws_secret_access_key,
                self.endpoint_url.replace("tos-s3-", "tos-"),
                self.region,
            )

    async def init(self):
        for i in range(self.max_retries):
            try:
                logger.info(f"S3DataManager init with config: {self.config} (attempt {i + 1}/{self.max_retries}) ...")
                s3_config = {"payload_signing_enabled": True}
                if self.addressing_style:
                    s3_config["addressing_style"] = self.addressing_style
                self.session = aioboto3.Session()
                self.s3_client = await self.session.client(
                    "s3",
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    endpoint_url=self.endpoint_url,
                    config=Config(
                        signature_version="s3v4",
                        s3=s3_config,
                        connect_timeout=self.connect_timeout,
                        read_timeout=self.read_timeout,
                        parameter_validation=False,
                        max_pool_connections=50,
                    ),
                ).__aenter__()

                try:
                    await self.s3_client.head_bucket(Bucket=self.bucket_name)
                    logger.info(f"check bucket {self.bucket_name} success")
                except Exception as e:
                    logger.info(f"check bucket {self.bucket_name} error: {e}, try to create it...")
                    await self.s3_client.create_bucket(Bucket=self.bucket_name)

                await self.init_presign_client()
                logger.info(f"Successfully init S3 bucket: {self.bucket_name} with timeouts - connect: {self.connect_timeout}s, read: {self.read_timeout}s, write: {self.write_timeout}s")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to S3: {e}")
                await asyncio.sleep(1)

    async def close(self):
        if self.s3_client:
            await self.s3_client.__aexit__(None, None, None)
        if self.session:
            self.session = None

    @class_try_catch_async
    async def save_bytes(self, bytes_data, filename, abs_path=None):
        filename = self.fmt_path(self.base_path, filename, abs_path)
        content_sha256 = hashlib.sha256(bytes_data).hexdigest()
        await self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=filename,
            Body=bytes_data,
            ChecksumSHA256=content_sha256,
            ContentType="application/octet-stream",
        )
        return True

    @class_try_catch_async
    async def load_bytes(self, filename, abs_path=None):
        filename = self.fmt_path(self.base_path, filename, abs_path)
        response = await self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
        return await response["Body"].read()

    @class_try_catch_async
    async def delete_bytes(self, filename, abs_path=None):
        filename = self.fmt_path(self.base_path, filename, abs_path)
        await self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
        logger.info(f"deleted s3 file {filename}")
        return True

    @class_try_catch_async
    async def file_exists(self, filename, abs_path=None):
        filename = self.fmt_path(self.base_path, filename, abs_path)
        try:
            await self.s3_client.head_object(Bucket=self.bucket_name, Key=filename)
            return True
        except Exception:
            return False

    @class_try_catch_async
    async def list_files(self, base_dir=None):
        prefix = base_dir if base_dir else self.base_path
        response = await self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        files = []
        if "Contents" in response:
            for obj in response["Contents"]:
                files.append(obj["Key"].replace(prefix + "/", ""))
        return files

    @class_try_catch_async
    async def presign_url(self, filename, abs_path=None):
        filename = self.fmt_path(self.base_path, filename, abs_path)
        if self.cdn_url:
            return f"{self.cdn_url}/{filename}"

        if self.presign_client:
            expires = self.config.get("presign_expires", 24 * 60 * 60)
            out = await asyncio.to_thread(self.presign_client.pre_signed_url, tos.HttpMethodType.Http_Method_Get, self.bucket_name, filename, expires)
            return out.signed_url
        else:
            return None


async def test():
    import torch
    from PIL import Image

    s3_config = {
        "aws_access_key_id": "xxx",
        "aws_secret_access_key": "xx",
        "endpoint_url": "xxx",
        "bucket_name": "xxx",
        "base_path": "xxx",
        "connect_timeout": 10,
        "read_timeout": 10,
        "write_timeout": 10,
    }

    m = S3DataManager(json.dumps(s3_config), None)
    await m.init()

    img = Image.open("../../../assets/img_lightx2v.png")
    tensor = torch.Tensor([233, 456, 789]).to(dtype=torch.bfloat16, device="cuda:0")

    await m.save_image(img, "test_img.png")
    print(await m.load_image("test_img.png"))

    await m.save_tensor(tensor, "test_tensor.pt")
    print(await m.load_tensor("test_tensor.pt", "cuda:0"))

    await m.save_object(
        {
            "images": [img, img],
            "tensor": tensor,
            "list": [
                [2, 0, 5, 5],
                {
                    "1": "hello world",
                    "2": "world",
                    "3": img,
                    "t": tensor,
                },
                "0609",
            ],
        },
        "test_object.json",
    )
    print(await m.load_object("test_object.json", "cuda:0"))

    print("all files:", await m.list_files())
    await m.get_delete_func("OBJECT")("test_object.json")
    await m.get_delete_func("TENSOR")("test_tensor.pt")
    await m.get_delete_func("IMAGE")("test_img.png")
    print("after delete all files", await m.list_files())
    await m.close()


if __name__ == "__main__":
    asyncio.run(test())
