# WorldPlay

Use the existing CLI launchers after [environment setup](../../docs/EN/source/getting_started/quickstart.md).
Set the repository path, HunyuanVideo-1.5 model directory and input image in the
launcher. Set `action_ckpt` in the selected JSON to the matching HY-WorldPlay weights.

| Model | Launcher | Config | Example pose |
| --- | --- | --- | --- |
| Autoregressive | [run_worldplay_ar.sh](run_worldplay_ar.sh) | [worldplay_ar_i2v_480p.json](../../configs/worldplay/worldplay_ar_i2v_480p.json) | `d-31` |
| Bidirectional | [run_worldplay_bi.sh](run_worldplay_bi.sh) | [worldplay_bi_i2v_480p.json](../../configs/worldplay/worldplay_bi_i2v_480p.json) | `s-31` |
| Distilled | [run_worldplay_distill.sh](run_worldplay_distill.sh) | [worldplay_distill_i2v_480p.json](../../configs/worldplay/worldplay_distill_i2v_480p.json) | `w-31` |
| Autoregressive, four GPUs | [run_worldplay_ar_sp4.sh](run_worldplay_ar_sp4.sh) | See launcher | `d-31` |

Run from the repository root:

```bash
bash scripts/worldplay/run_worldplay_ar.sh
```

The launchers use seed `1` and save videos under `save_results/HY-WorldPlay`.
For integration with Python, use the same config and request fields:

```python
import json

from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.set_config import build_startup_config

with open("configs/worldplay/worldplay_ar_i2v_480p.json") as f:
    config = json.load(f)
config["model_path"] = "/path/to/HunyuanVideo-1.5"
config["action_ckpt"] = "/path/to/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors"
runner = build_runner(build_startup_config(config))
request = runner.prepare_request({
    "seed": 1,
    "prompt": "A paved pathway leads towards a stone arch bridge.",
    "image_path": "/path/to/image.png",
    "pose": "d-31",
    "save_result_path": "/path/to/worldplay_ar.mp4",
    "return_result_tensor": False,
})
result = runner.run_request(request)
```

For the bidirectional and distilled models, change the config, action checkpoint and
pose together using the table above. The bidirectional example accepts an empty
`negative_prompt`; omit it for CFG-disabled models.
