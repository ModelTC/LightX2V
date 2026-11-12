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

## Build the MLU environment

The MLU Dockerfile requires an explicitly selected, accessible vendor runtime
image. The base must provide Neuware under `/usr/local/neuware`, Python 3.11 or
3.12, and compatible `torch`, `torchvision`, `torchaudio`, `torch_mlu` and
`torch_mlu_ops` binary distributions in the selected interpreter. Obtain that
runtime from your hardware provider or administrator; this Dockerfile does not
fetch a proprietary driver/SDK or assume a private registry address.

```bash
MLU_BASE_IMAGE=your-registry/vendor-runtime:tag
docker build --build-arg BASE_IMAGE="$MLU_BASE_IMAGE" \
  -f dockerfiles/platforms/Dockerfile_cambricon_mlu590 \
  -t lightx2v-vggt-mlu .
```

If the base has several Python environments, pass
`--build-arg VENDOR_PYTHON=/path/to/vendor/python`. The application is installed into
`/opt/lightx2v` with system site-packages disabled. Only the five named vendor
binary distributions are copied from the base; the Dockerfile installs pinned
application dependencies and their dependencies from PyPI. It preserves the
vendor package versions, runs `pip check`, verifies package isolation and saves
`/opt/lightx2v/resolved-requirements.txt`. Use `--build-arg PYPI_INDEX_URL=...` to
select another package index, or standard Docker proxy build arguments for your
network. Do not mount a host Python environment into this image.

The host still needs a compatible MLU driver. After downloading weights into the
layout above, run the bundled example without installing packages in the container:

```bash
mkdir -p save_results/vggt
docker run --rm --shm-size 8g \
  --device /dev/cambricon_ctl \
  --device /dev/cambricon_dev0 \
  --device /dev/cambricon_ipcm0 \
  --mount "type=bind,source=$PWD/weights/vggt,target=/models/vggt,readonly" \
  -v "$PWD/save_results/vggt:/outputs" \
  -e MODEL_PATH=/models/vggt \
  -e SAVE_RESULT_PATH=/outputs/reconstruction.npz \
  lightx2v-vggt-mlu bash scripts/platform/mlu/run_vggt.sh
```

This image covers the provided BF16 MLU configuration (`torch_sdpa`). The NVIDIA
launcher uses PyTorch SDPA by default. The environment was
validated with Python 3.12, Torch 2.9.1 and torch_mlu 1.31.1; a different vendor
stack must be validated separately. A successful image build does not by itself
prove that a host driver or another vendor image is compatible.

## Run

The default input is the five bundled PNG views in `assets/inputs/imgs/vggt/`.
No dataset download, labels or camera files are needed for inference.

| Platform | Configuration | Launcher |
| --- | --- | --- |
| NVIDIA GPU | `configs/vggt/vggt_recon_bf16.json` (`torch_sdpa`) | `scripts/vggt/run_vggt.sh` |
| MLU | `configs/platform/mlu/vggt_recon_bf16.json` (`torch_sdpa`) | `scripts/platform/mlu/run_vggt.sh` |

With the default weight directory and prepared platform environment:

```bash
# NVIDIA GPU; uses PyTorch SDPA.
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
