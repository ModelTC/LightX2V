# VGGT reconstruction

Prepare the LightX2V environment for your platform before running these commands.
Run all commands below from the root of the complete LightX2V repository.

## Prepare weights

The launcher loads local weights. It does not download a missing model or resolve
`MODEL_PATH` as a Hugging Face repository ID.

Download [Meta's VGGT-1B checkpoint](https://huggingface.co/facebook/VGGT-1B)
with the Hugging Face CLI (`hf`, provided by `huggingface_hub`):

```bash
hf download facebook/VGGT-1B \
  --revision 860abec7937da0a4c03c41d3c269c366e82abdf9 \
  --local-dir weights/vggt \
  --include model.pt config.json README.md
```

This revision matches the checkpoint used by the MLU functional run. The required
checkpoint is `model.pt` (about 5 GB); the current loader does not use the separate
`model.safetensors` file. The command also keeps the upstream configuration and
model card alongside the checkpoint:

```text
weights/vggt/
├── model.pt
├── config.json
└── README.md
```

Download once, then reuse this directory. To use an existing download elsewhere,
set `MODEL_PATH` to the directory containing `model.pt`.

If the Hub requests authentication, complete `hf auth login` and any access steps
on the model page before downloading. Configure your download proxy through the
standard `HTTP_PROXY` / `HTTPS_PROXY` environment variables when needed. Inference
can run offline once the local checkpoint is present.

When using Docker, mount the prepared weight directory into the container and
set `MODEL_PATH` to its container path.

## Run

The default input is the five bundled PNG views in `assets/inputs/imgs/vggt/`.
No dataset download, labels or camera files are needed for inference.

| Platform | Configuration | Launcher |
| --- | --- | --- |
| NVIDIA GPU | `configs/vggt/vggt_recon_bf16.json` (`flash_attn2`) | `scripts/vggt/run_vggt.sh` |
| MLU | `configs/platform/mlu/vggt_recon_bf16.json` (`torch_sdpa`) | `scripts/platform/mlu/run_vggt.sh` |

With the default weight directory and prepared platform environment:

```bash
# NVIDIA GPU; requires the FlashAttention v2 backend.
bash scripts/vggt/run_vggt.sh

# MLU; requires a compatible vendor Torch / torch_mlu environment.
bash scripts/platform/mlu/run_vggt.sh
```

To use your own checkpoint directory, images and output path:

```bash
MODEL_PATH=/path/to/VGGT-1B \
INPUT_PATH=/path/to/my_views \
SAVE_RESULT_PATH=/path/to/reconstruction.npz \
bash scripts/platform/mlu/run_vggt.sh
```

`INPUT_PATH` accepts an image directory or a single image. Directory inputs are
sorted by filename; use overlapping views of the same scene for reconstruction.
The default output is `save_results/vggt/reconstruction.npz`, containing poses,
intrinsics/extrinsics, depth, point maps, confidence and original image coordinates.
`CONFIG_JSON` and `SEED` can override the default configuration and seed.

If loading reports a missing `model.pt`, check `MODEL_PATH` or rerun the download
command. The launcher does not fall back to a network download.
