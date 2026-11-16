import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/ltx2/ltx2.json").read_text())
config.update(
    {
        "model_path": "Lightricks/LTX-2/",
        "model_cls": "ltx2",
        "task": "t2av",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "A beautiful sunset over the ocean"
    negative_prompt = "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    save_result_path = "/path/to/save_results/output.mp4"

    pipe.generate(
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_result_path=save_result_path,
    )
