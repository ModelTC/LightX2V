import argparse
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from lightx2v.common.ops import *
from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_causvid_runner import WanCausVidRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner  # noqa: F401
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.service_utils import BaseServiceStatus, ImageTransporter, ProcessManager, TaskStatusMessage, TensorTransporter
from lightx2v.utils.set_config import set_config

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

    img: Optional[bytes] = None
    latents: Optional[bytes] = None

    def get(self, key, default=None):
        return getattr(self, key, default)


class VAEServiceStatus(BaseServiceStatus):
    pass


class VAERunner:
    def __init__(self, config):
        self.config = config
        self.runner_cls = RUNNER_REGISTER[self.config.model_cls]

        self.runner = self.runner_cls(config)
        self.runner.vae_encoder, self.runner.vae_decoder = self.runner.load_vae()

    def _run_vae_encoder(self, img):
        img = image_transporter.load_image(img)
        vae_encoder_out, kwargs = self.runner.run_vae_encoder(img)
        return vae_encoder_out, kwargs

    def _run_vae_decoder(self, latents):
        latents = tensor_transporter.load_tensor(latents)
        images = self.runner.vae_decoder.decode(latents, generator=None, config=self.config)
        return images


def run_vae_encoder(message: Message):
    try:
        global runner
        vae_encoder_out, kwargs = runner._run_vae_encoder(message.img)
        VAEServiceStatus.complete_task(message)
        return vae_encoder_out, kwargs
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        VAEServiceStatus.record_failed_task(message, error=str(e))


def run_vae_decoder(message: Message):
    try:
        global runner
        images = runner._run_vae_decoder(message.latents)
        VAEServiceStatus.complete_task(message)
        return images
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        VAEServiceStatus.record_failed_task(message, error=str(e))


@app.post("/v1/local/vae_model/encoder/generate")
def v1_local_vae_model_encoder_generate(message: Message):
    try:
        task_id = VAEServiceStatus.start_task(message)
        vae_encoder_out, kwargs = run_vae_encoder(message)
        output = tensor_transporter.prepare_tensor(vae_encoder_out)
        del vae_encoder_out
        return {"task_id": task_id, "task_status": "completed", "output": output, "kwargs": kwargs}
    except RuntimeError as e:
        return {"error": str(e)}


@app.post("/v1/local/vae_model/decoder/generate")
def v1_local_vae_model_decoder_generate(message: Message):
    try:
        task_id = VAEServiceStatus.start_task(message)
        vae_decode_out = run_vae_decoder(message)
        output = tensor_transporter.prepare_tensor(vae_decode_out)
        del vae_decode_out
        return {"task_id": task_id, "task_status": "completed", "output": output, "kwargs": None}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/v1/local/vae_model/generate/service_status")
async def get_service_status():
    return VAEServiceStatus.get_status_service()


@app.get("/v1/local/vae_model/encoder/generate/service_status")
async def get_service_status():
    return VAEServiceStatus.get_status_service()


@app.get("/v1/local/vae_model/decoder/generate/service_status")
async def get_service_status():
    return VAEServiceStatus.get_status_service()


@app.get("/v1/local/vae_model/encoder/generate/get_all_tasks")
async def get_all_tasks():
    return VAEServiceStatus.get_all_tasks()


@app.get("/v1/local/vae_model/decoder/generate/get_all_tasks")
async def get_all_tasks():
    return VAEServiceStatus.get_all_tasks()


@app.post("/v1/local/vae_model/encoder/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return VAEServiceStatus.get_status_task_id(message.task_id)


@app.post("/v1/local/vae_model/decoder/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return VAEServiceStatus.get_status_task_id(message.task_id)


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

    parser.add_argument("--port", type=int, default=9004)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = VAERunner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
