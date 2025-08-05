import argparse
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

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

    text: str
    img: Optional[bytes] = None
    n_prompt: Optional[str] = None

    def get(self, key, default=None):
        return getattr(self, key, default)


class TextEncoderServiceStatus(BaseServiceStatus):
    pass


class TextEncoderRunner:
    def __init__(self, config):
        self.config = config
        self.runner_cls = RUNNER_REGISTER[self.config.model_cls]

        self.runner = self.runner_cls(config)
        self.runner.text_encoders = self.runner.load_text_encoder()

    def _run_text_encoder(self, text, img, n_prompt):
        if img is not None:
            img = image_transporter.load_image(img)
        self.runner.config["negative_prompt"] = n_prompt
        text_encoder_output = self.runner.run_text_encoder(text, img)
        return text_encoder_output


def run_text_encoder(message: Message):
    try:
        global runner
        text_encoder_output = runner._run_text_encoder(message.text, message.img, message.n_prompt)
        TextEncoderServiceStatus.complete_task(message)
        return text_encoder_output
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        TextEncoderServiceStatus.record_failed_task(message, error=str(e))


@app.post("/v1/local/text_encoders/generate")
def v1_local_text_encoder_generate(message: Message):
    try:
        task_id = TextEncoderServiceStatus.start_task(message)
        text_encoder_output = run_text_encoder(message)
        output = tensor_transporter.prepare_tensor(text_encoder_output)
        del text_encoder_output
        return {"task_id": task_id, "task_status": "completed", "output": output, "kwargs": None}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/v1/local/text_encoders/generate/service_status")
async def get_service_status():
    return TextEncoderServiceStatus.get_status_service()


@app.get("/v1/local/text_encoders/generate/get_all_tasks")
async def get_all_tasks():
    return TextEncoderServiceStatus.get_all_tasks()


@app.post("/v1/local/text_encoders/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return TextEncoderServiceStatus.get_status_task_id(message.task_id)


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

    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = TextEncoderRunner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
