import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/ltx2/ltx2_distill_fp8.json").read_text())
config.update(
    {
        "model_path": "Lightricks/LTX-2",
        "model_cls": "ltx2",
        "task": "t2av",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "A beautiful sunset over the ocean"
    save_result_path = "/path/to/LightX2V/save_results/output_lightx2v_ltx2_t2av_distilled_fp8.mp4"

    pipe.generate(
        seed=seed,
        prompt=prompt,
        save_result_path=save_result_path,
    )
