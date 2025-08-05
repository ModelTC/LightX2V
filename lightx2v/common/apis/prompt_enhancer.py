import argparse
import json
from typing import Optional

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from vllm import LLM, SamplingParams

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.service_utils import BaseServiceStatus, ProcessManager, TaskStatusMessage

# =========================
# FastAPI Related Code
# =========================

runner = None
app = FastAPI()

sys_prompt = """
Transform the short prompt into a detailed video-generation caption using this structure:
​​Opening shot type​​ (long/medium/close-up/extreme close-up/full shot)
​​Primary subject(s)​​ with vivid attributes (colors, textures, actions, interactions)
​​Dynamic elements​​ (movement, transitions, or changes over time, e.g., 'gradually lowers,' 'begins to climb,' 'camera moves toward...')
​​Scene composition​​ (background, environment, spatial relationships)
​​Lighting/atmosphere​​ (natural/artificial, time of day, mood)
​​Camera motion​​ (zooms, pans, static/handheld shots) if applicable.

Pattern Summary from Examples:
[Shot Type] of [Subject+Action] + [Detailed Subject Description] + [Environmental Context] + [Lighting Conditions] + [Camera Movement]

​One case:
Short prompt: a person is playing football
Long prompt: Medium shot of a young athlete in a red jersey sprinting across a muddy field, dribbling a soccer ball with precise footwork. The player glances toward the goalpost, adjusts their stance, and kicks the ball forcefully into the net. Raindrops fall lightly, creating reflections under stadium floodlights. The camera follows the ball’s trajectory in a smooth pan.

Note: If the subject is stationary, incorporate camera movement to ensure the generated video remains dynamic.

​​Now expand this short prompt:​​ [{}]. Please only output the final long prompt in English.
"""


class Message(BaseModel):
    task_id: str
    task_id_must_unique: bool = False
    prompt: str

    def get(self, key, default=None):
        return getattr(self, key, default)


class PromptEnhancerServiceStatus(BaseServiceStatus):
    pass


class PromptEnhancerRunner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.get_model()
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=8192,
        )

    def get_model(self):
        model = LLM(model=self.model_path, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.95, max_model_len=16384)
        return model

    def _run_prompt_enhancer(self, prompt):
        prompt = prompt.strip()
        prompt = sys_prompt.format(prompt)
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]

        outputs = self.model.chat(
            messages=messages,
            sampling_params=self.sampling_params,
        )

        enhanced_prompt = outputs[0].outputs[0].text
        return enhanced_prompt.strip()


def run_prompt_enhancer(message: Message):
    try:
        global runner
        enhanced_prompt = runner._run_prompt_enhancer(message.prompt)
        assert enhanced_prompt is not None
        PromptEnhancerServiceStatus.complete_task(message)
        return enhanced_prompt
    except Exception as e:
        logger.error(f"task_id {message.task_id} failed: {str(e)}")
        PromptEnhancerServiceStatus.record_failed_task(message, error=str(e))


@app.post("/v1/local/prompt_enhancer/generate")
def v1_local_prompt_enhancer_generate(message: Message):
    try:
        task_id = PromptEnhancerServiceStatus.start_task(message)
        enhanced_prompt = run_prompt_enhancer(message)
        return {"task_id": task_id, "task_status": "completed", "output": enhanced_prompt, "kwargs": None}
    except RuntimeError as e:
        return {"error": str(e)}


@app.get("/v1/local/prompt_enhancer/generate/service_status")
async def get_service_status():
    return PromptEnhancerServiceStatus.get_status_service()


@app.get("/v1/local/prompt_enhancer/generate/get_all_tasks")
async def get_all_tasks():
    return PromptEnhancerServiceStatus.get_all_tasks()


@app.post("/v1/local/prompt_enhancer/generate/task_status")
async def get_task_status(message: TaskStatusMessage):
    return PromptEnhancerServiceStatus.get_status_task_id(message.task_id)


# =========================
# Main Entry
# =========================

if __name__ == "__main__":
    ProcessManager.register_signal_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    with ProfilingContext("Init Server Cost"):
        runner = PromptEnhancerRunner(args.model_path)

    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False, workers=1)
