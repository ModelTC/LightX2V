import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import uvicorn
import json
import os
import torch
import torchvision.transforms.functional as TF

from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner
from lightx2v.models.runners.wan.wan_causvid_runner import WanCausVidRunner
from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.utils.service_utils import TaskStatusMessage, BaseServiceStatus, ProcessManager, TensorTransporter, ImageTransporter

tensor_transporter = TensorTransporter()
image_transporter = ImageTransporter()

# =========================
# FastAPI Related Code
# =========================

runner = None

app = FastAPI()


class Message(BaseModel):
    task_id: str
    task_id_must_unique: bool = False

    img: bytes

    def get(self, key, default=None):
        return getattr(self, key, default)


class ImageEncoderServiceStatus(BaseServiceStatus):
    pass


class ImageEncoderRunner:
    def __init__(self, config):
        self.config = config
        self.runner_cls = RUNNER_REGISTER[self.config.model_cls]

        self.runner = self.runner_cls(config)
        self.runner.image_encoder = self.runner.load_image_encoder()

    def _run_image_encoder(self, img):
        img = image_transporter.load_image(img)
        return self.runner.run_image_encoder(img)


def run_image_encoder(message: Message):
    try:
        global runner
        image_encoder_out = runner._run_image_encoder(message.img)
        ImageEncoderServiceStatus.complete_task(message)
        return image_encoder_out
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        ImageEncoderServiceStatus.record_failed_task(message, error=str(e))


@app.post("/v1/local/image_encoder/generate")
def v1_local_image_encoder_generate(message: Message):
    try:
        task_id = ImageEncoderServiceStatus.start_task(message)
        image_encoder_output = run_image_encoder(message)
        output = tensor_transporter.prepare_tensor(image_encoder_output)
        del image_encoder_output
        return {"task_id": task_id, "task_status": "completed", "output": output, "kwargs": None}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/v1/local/image_encoder/generate/service_status")
async def get_service_status():
    return ImageEncoderServiceStatus.get_status_service()


@app.get("/v1/local/image_encoder/generate/get_all_tasks")
async def get_all_tasks():
    return ImageEncoderServiceStatus.get_all_tasks()


@app.post("/v1/local/image_encoder/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return ImageEncoderServiceStatus.get_status_task_id(message.task_id)


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    ProcessManager.register_signal_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_distill", "wan2.1_causvid", "wan2.1_skyreels_v2_df", "cogvideox"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)

    parser.add_argument("--port", type=int, default=9003)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    assert args.task == "i2v"

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = ImageEncoderRunner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
