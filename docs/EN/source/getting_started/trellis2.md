# TRELLIS.2-4B BF16 inference

[Official project](https://github.com/microsoft/TRELLIS.2). This entry point accepts one image and uses the existing 512 pipeline.

Prepare `pipeline.json`, `ckpts/`, `base/`, `dinov3/`, and `rembg/` under the model directory, retaining these existing revisions:

| Directory | Repository | Revision |
|---|---|---|
| root | microsoft/TRELLIS.2-4B | af44b45f2e35a493886929c6d786e563ec68364d |
| base | microsoft/TRELLIS-image-large | 459ee480d7545f873a0768364bc5f5443eb1ff5e |
| dinov3 | camenduru/dinov3-vitl16-pretrain-lvd1689m | 3c276edd87d6f6e569ff0c4400e086807d0f3881 |
| rembg | ZhengPeng7/BiRefNet | e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 |

Include BiRefNet's public Python files for the existing local model loader. Override component locations with `base_model_path`, `dino_model_path` and `rembg_model_path` in JSON.

```bash
python -m pip install -r scripts/trellis2/requirements-mlu.txt
MODEL_PATH=/path/to/TRELLIS.2 INPUT_PATH=/path/to/image.png bash scripts/trellis2/run_trellis2.sh
```

Select `PLATFORM` and device visibility before launching. The script uses BF16 and seed 42; override `SEED`, `SAVE_RESULT_PATH`, or append CLI arguments. Existing preprocessing, three-stage Euler/CFG sampling and decoding are retained.

Output is geometry GLB; the Runner also returns texture voxels. Texture baking and cascade/1024/1536 are outside this integration. Checkpoint declarations/loading live in `weights/`, computation in `infer/`, and shared sparse operators in `common/ops/sparse/`.
