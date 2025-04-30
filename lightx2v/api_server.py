import signal
import sys
import psutil
import argparse
from fastapi import FastAPI, Request
from pydantic import BaseModel
from loguru import logger
import uvicorn
import json
import asyncio

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.infer import init_runner


# =========================
# Signal & Process Control
# =========================


def kill_all_related_processes():
    """Kill the current process and all its child processes"""
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except Exception as e:
            logger.info(f"Failed to kill child process {child.pid}: {e}")
    try:
        current_process.kill()
    except Exception as e:
        logger.info(f"Failed to kill main process: {e}")


def signal_handler(sig, frame):
    logger.info("\nReceived Ctrl+C, shutting down all related processes...")
    kill_all_related_processes()
    sys.exit(0)


# =========================
# FastAPI Related Code
# =========================

runner = None

app = FastAPI()


class Message(BaseModel):
    prompt: str
    use_prompt_enhancer: bool = False
    negative_prompt: str = ""
    image_path: str = ""
    num_fragments: int = 1
    save_video_path: str

    def get(self, key, default=None):
        return getattr(self, key, default)


@app.post("/v1/local/video/generate")
async def v1_local_video_generate(message: Message):
    global runner
    runner.set_inputs(message)
    logger.info(f"message: {message}")
    await asyncio.to_thread(runner.run_pipeline)
    response = {"response": "finished", "save_video_path": message.save_video_path}
    if runner.has_prompt_enhancer and message.use_prompt_enhancer:
        response["enhanced_prompt"] = runner.config["prompt"]
    return response


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_causvid"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--prompt_enhancer", default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = init_runner(config)

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False, workers=1)
