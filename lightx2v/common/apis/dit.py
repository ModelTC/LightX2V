import argparse
import json
import os
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from lightx2v.common.ops import *
from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner
from lightx2v.models.runners.wan.wan_causvid_runner import WanCausVidRunner
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner
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

    inputs: bytes
    kwargs: bytes

    def get(self, key, default=None):
        return getattr(self, key, default)


class DiTServiceStatus(BaseServiceStatus):
    pass


class DiTRunner:
    def __init__(self, config):
        self.config = config
        self.runner_cls = RUNNER_REGISTER[self.config.model_cls]

        self.runner = self.runner_cls(config)
        self.runner.model = self.runner.load_transformer()

    def _run_dit(self, inputs, kwargs):
        self.runner.config.update(tensor_transporter.load_tensor(kwargs))
        self.runner.inputs = tensor_transporter.load_tensor(inputs)
        self.runner.init_scheduler()
        self.runner.model.scheduler.prepare(self.runner.inputs["image_encoder_output"])
        latents, _ = self.runner.run()
        self.runner.end_run()
        return latents


def run_dit(message: Message):
    try:
        global runner
        dit_output = runner._run_dit(message.inputs, message.kwargs)
        DiTServiceStatus.complete_task(message)
        return dit_output
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        DiTServiceStatus.record_failed_task(message, error=str(e))


@app.post("/v1/local/dit/generate")
def v1_local_dit_generate(message: Message):
    try:
        task_id = DiTServiceStatus.start_task(message)
        dit_output = run_dit(message)
        output = tensor_transporter.prepare_tensor(dit_output)
        del dit_output
        return {"task_id": task_id, "task_status": "completed", "output": output, "kwargs": None}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/v1/local/dit/generate/service_status")
async def get_service_status():
    return DiTServiceStatus.get_status_service()


@app.get("/v1/local/dit/generate/get_all_tasks")
async def get_all_tasks():
    return DiTServiceStatus.get_all_tasks()


@app.post("/v1/local/dit/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return DiTServiceStatus.get_status_task_id(message.task_id)


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

    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = DiTRunner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
