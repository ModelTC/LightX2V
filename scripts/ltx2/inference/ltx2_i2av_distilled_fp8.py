import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/ltx2/ltx2_distill_fp8.json").read_text())
config.update(
    {
        "model_path": "Lightricks/LTX-2",
        "model_cls": "ltx2",
        "task": "i2av",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    image_path = ["/path/to/LightX2V/assets/inputs/imgs/woman.jpeg"]  # Add further paths as list items.
    image_strength = [1.0]  # One value broadcasts to every image; otherwise provide one per image.
    image_frame_indices = None  # Or e.g. [0, 60, 120] — pixel frame per image; None = evenly spaced in [0, num_frames-1]
    prompt = "A young woman with wavy, shoulder-length light brown hair is singing and dancing joyfully outdoors on a foggy day. She wears a cozy pink turtleneck sweater, swaying gracefully to the music with animated expressions and bright, piercing blue eyes. Her movements are fluid and energetic as she twirls and gestures expressively. A wooden fence and a misty, grassy field fade into the background, creating a dreamy atmosphere for her lively performance."
    save_result_path = "/path/to/LightX2V/save_results/output_lightx2v_ltx2_i2av_distilled_fp8.mp4"

    # Note: image_strength can also be set in config_json
    # A one-item list, e.g. image_strength = [1.0], applies to every image.
    # For list: image_strength = [1.0, 0.8] (must match number of images)
    pipe.generate(
        seed=seed,
        prompt=prompt,
        image_path=image_path,
        image_strength=image_strength,
        image_frame_indices=image_frame_indices,
        save_result_path=save_result_path,
    )
