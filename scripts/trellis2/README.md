# TRELLIS.2 image-to-3D

Prepare the LightX2V environment for your platform before running these commands.
Run all commands below from the root of the complete LightX2V repository.
This integration supports the single-image 512 pipeline.

## Prepare weights

The launcher reads local weights and does not download missing components.
DINOv3 and BiRefNet are also loaded with `local_files_only=True`.

Prepare all four components, using the Hugging Face CLI (`hf`, provided by
`huggingface_hub`). The commands below create the layout expected by the runner;
no model-path configuration changes or symbolic links are needed.

| Component | Source | Fixed revision |
| --- | --- | --- |
| Flow networks and shape/texture decoders | [Microsoft TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | `af44b45f2e35a493886929c6d786e563ec68364d` |
| Sparse structure decoder | [Microsoft TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | `459ee480d7545f873a0768364bc5f5443eb1ff5e` |
| DINOv3 ViT-L/16 image encoder | [camenduru community copy](https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m) | `3c276edd87d6f6e569ff0c4400e086807d0f3881` |
| Background removal | [Author's BiRefNet repository](https://huggingface.co/ZhengPeng7/BiRefNet) | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` |

The DINOv3 source above is a community copy, not Meta's official distribution.
These revisions identify the assets used in the MLU functional run; the BiRefNet
revision was matched to the tested weight digest and accompanying Python files.

Download the main pipeline configuration and the five checkpoints used at 512:

```bash
hf download microsoft/TRELLIS.2-4B \
  --revision af44b45f2e35a493886929c6d786e563ec68364d \
  --local-dir weights/trellis2 \
  --include pipeline.json README.md \
    'ckpts/ss_flow_img_dit_1_3B_64_bf16.*' \
    'ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.*' \
    'ckpts/shape_dec_next_dc_f16c32_fp16.*' \
    'ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.*' \
    'ckpts/tex_dec_next_dc_f16c32_fp16.*'
```

Download the structure decoder and the two image-processing models:

```bash
hf download microsoft/TRELLIS-image-large \
  --revision 459ee480d7545f873a0768364bc5f5443eb1ff5e \
  --local-dir weights/trellis2/base \
  --include 'ckpts/ss_dec_conv3d_16l8_fp16.*' README.md

hf download camenduru/dinov3-vitl16-pretrain-lvd1689m \
  --revision 3c276edd87d6f6e569ff0c4400e086807d0f3881 \
  --local-dir weights/trellis2/dinov3 \
  --include config.json model.safetensors preprocessor_config.json README.md LICENSE.md

hf download ZhengPeng7/BiRefNet \
  --revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 \
  --local-dir weights/trellis2/rembg
```

Each checkpoint prefix needs both its `.json` and `.safetensors` files. The required
layout is:

```text
weights/trellis2/
├── pipeline.json
├── ckpts/
│   ├── ss_flow_img_dit_1_3B_64_bf16.json
│   ├── ss_flow_img_dit_1_3B_64_bf16.safetensors
│   ├── slat_flow_img2shape_dit_1_3B_512_bf16.json
│   ├── slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
│   ├── shape_dec_next_dc_f16c32_fp16.json
│   ├── shape_dec_next_dc_f16c32_fp16.safetensors
│   ├── slat_flow_imgshape2tex_dit_1_3B_512_bf16.json
│   ├── slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors
│   ├── tex_dec_next_dc_f16c32_fp16.json
│   └── tex_dec_next_dc_f16c32_fp16.safetensors
├── base/
│   └── ckpts/
│       ├── ss_dec_conv3d_16l8_fp16.json
│       └── ss_dec_conv3d_16l8_fp16.safetensors
├── dinov3/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
└── rembg/
    ├── config.json
    ├── BiRefNet_config.py
    ├── birefnet.py
    └── model.safetensors
```

Keep the BiRefNet Python files with its weights: the Transformers loader uses the
model code supplied by that repository. Its downloaded `requirements.txt` is not
an extra installation step; use the dependencies declared by LightX2V.

Download once, then reuse the local directories. If you already have all components
elsewhere, either arrange them under the layout above or set `base_model_path`,
`dino_model_path` and `rembg_model_path` in your model JSON, then select it through
`CONFIG_JSON`. `MODEL_PATH` always points to the root containing `pipeline.json`.

If the Hub requests authentication, complete `hf auth login` and any access steps
on the model page before downloading. Configure `HTTP_PROXY` / `HTTPS_PROXY` for
the download command when needed. Inference can run offline after preparation.
Missing components produce a local-file loading error, not an automatic download.

When using Docker, mount the prepared weight directory into the container and
set `MODEL_PATH` to its container path. For TRELLIS.2, keep the four component
directories together under that root.

## Run

The default input is `assets/inputs/imgs/trellis2/input.png`. No dataset download,
labels, camera files or evaluation manifest are needed.

| Platform | Configuration | Launcher |
| --- | --- | --- |
| NVIDIA GPU | `configs/trellis2/trellis2_i23d_bf16.json` (`flash_attn2`) | `scripts/trellis2/run_trellis2.sh` |
| MLU | `configs/platform/mlu/trellis2_i23d_bf16.json` (`torch_sdpa`) | `scripts/platform/mlu/run_trellis2.sh` |

With the default weight directory and prepared platform environment:

```bash
# NVIDIA GPU; requires the FlashAttention v2 backend.
bash scripts/trellis2/run_trellis2.sh

# MLU; requires a compatible vendor Torch / torch_mlu environment.
bash scripts/platform/mlu/run_trellis2.sh
```

To use your own weights, image and output path:

```bash
MODEL_PATH=/path/to/TRELLIS.2 \
INPUT_PATH=/path/to/my_object.png \
SAVE_RESULT_PATH=/path/to/mesh.glb \
bash scripts/platform/mlu/run_trellis2.sh
```

The default output is `save_results/trellis2/mesh.glb`, containing geometry.
The runner also computes texture voxels, but this integration does not bake
textures into the exported GLB. It does not accept text prompts.
`CONFIG_JSON` and `SEED` can override the default configuration and seed.
