# VGGT-1B BF16 inference

[Official project](https://github.com/facebookresearch/vggt). Place `model.pt` from `facebook/VGGT-1B` in the model directory. The existing weight revision is `860abec7937da0a4c03c41d3c269c366e82abdf9`.

```bash
python -m pip install -r scripts/vggt/requirements-mlu.txt
MODEL_PATH=/path/to/VGGT-1B INPUT_PATH=/path/to/images bash scripts/vggt/run_vggt.sh
```

Select `PLATFORM` and the platform's device visibility setting before launching. The script uses BF16 and seed 42; override `SEED`, `SAVE_RESULT_PATH`, or append CLI arguments.

Input is one image or a directory sorted by filename; Python requests also accept an ordered list. The number of views is variable and must be positive. The historical 32-view cohort is not an input constraint. Existing square padding and resizing to 518 × 518 are retained.

The NPZ contains camera poses, extrinsics, intrinsics, depth, point maps, confidence and original image bounds. Tracking is outside this reconstruction integration. Checkpoint declarations/loading live in `weights/`; computation lives in `infer/`, using the existing platform and registered MM, attention, LayerNorm and Conv2d operators.
