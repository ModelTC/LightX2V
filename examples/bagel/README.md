# BAGEL Image Generation

This example covers the LightX2V BAGEL image generation scope for `ByteDance-Seed/BAGEL-7B-MoT`: text-to-image and single-image editing.

## Model

Download the model to a local directory. If the Hugging Face mirror is available in your region, prefer:

```bash
mkdir -p /path/to/models/ByteDance-Seed
HF_ENDPOINT=https://hf-mirror.com hf download ByteDance-Seed/BAGEL-7B-MoT \
  --local-dir /path/to/models/ByteDance-Seed/BAGEL-7B-MoT \
  --max-workers 8
```

The model directory must contain `config.json`, `ema.safetensors`, `ae.safetensors`, tokenizer files, and the other files from the BAGEL-7B-MoT repository.

## Requirements

- CUDA GPU environment with enough memory for BAGEL-7B-MoT.
- PyTorch compatible with the current LightX2V environment.
- `flash-attn` installed for the same CUDA/PyTorch build.

If `flash-attn`, `ema.safetensors`, `ae.safetensors`, ViT weights for I2I, or required config fields are missing, the BAGEL runner raises a direct error before generation.

## Text To Image

```bash
export lightx2v_path=/path/to/LightX2V
export model_path=/path/to/BAGEL-7B-MoT

bash scripts/bagel/run_bagel_t2i.sh
```

Equivalent direct command:

```bash
python -m lightx2v.infer \
  --model_cls bagel \
  --task t2i \
  --model_path /path/to/BAGEL-7B-MoT \
  --config_json configs/bagel/bagel_t2i.json \
  --prompt "A small cabin beside a lake at sunrise, cinematic lighting" \
  --aspect_ratio 1:1 \
  --save_result_path save_results/bagel_t2i.png \
  --seed 42
```

## Shape Options

`target_shape` has priority over `aspect_ratio`. It is `[H W]`, must contain positive integers, and each dimension must be divisible by BAGEL latent downsample.

Supported `aspect_ratio` presets:

| aspect_ratio | output shape |
| --- | --- |
| `1:1` | `1024x1024` |
| `4:3` | `768x1024` |
| `3:4` | `1024x768` |
| `16:9` | `576x1024` |
| `9:16` | `1024x576` |

Custom shape example:

```bash
python -m lightx2v.infer \
  --model_cls bagel \
  --task t2i \
  --model_path /path/to/BAGEL-7B-MoT \
  --config_json configs/bagel/bagel_t2i.json \
  --prompt "A glass greenhouse in a snowy garden" \
  --target_shape 576 1024 \
  --save_result_path save_results/bagel_t2i_576x1024.png \
  --seed 42
```

## Image Editing

I2I supports a single input image. The runner converts it to RGB, resizes it to a BAGEL-aligned output shape, writes VAE and ViT image context, then applies the edit prompt.

```bash
MODEL_PATH=/path/to/BAGEL-7B-MoT \
IMAGE_PATH=assets/inputs/imgs/img_0.jpg \
PROMPT="Change the scene to golden hour while preserving the main subject." \
bash scripts/bagel/run_bagel_i2i.sh
```

Equivalent direct command:

```bash
python -m lightx2v.infer \
  --model_cls bagel \
  --task i2i \
  --model_path /path/to/BAGEL-7B-MoT \
  --config_json configs/bagel/bagel_i2i.json \
  --image_path assets/inputs/imgs/img_0.jpg \
  --prompt "Change the scene to golden hour while preserving the main subject." \
  --save_result_path save_results/bagel_i2i.png \
  --seed 42
```

For I2I, `target_shape [H W]` overrides the automatic size. Without `target_shape`, LightX2V preserves the input aspect ratio, limits the long edge to 1024, and aligns both dimensions to BAGEL latent downsample. `aspect_ratio` is ignored for I2I to avoid changing the input image shape unexpectedly.

Target shape example:

```bash
python -m lightx2v.infer \
  --model_cls bagel \
  --task i2i \
  --model_path /path/to/BAGEL-7B-MoT \
  --config_json configs/bagel/bagel_i2i.json \
  --image_path assets/inputs/imgs/img_0.jpg \
  --prompt "Make it look like a watercolor illustration." \
  --target_shape 576 1024 \
  --save_result_path save_results/bagel_i2i_576x1024.png \
  --seed 42
```

## Current Scope

Supported:

- `python -m lightx2v.infer --model_cls bagel --task t2i`
- `python -m lightx2v.infer --model_cls bagel --task i2i`
- PNG saving
- `seed`
- T2I `aspect_ratio`
- `target_shape`
- service in-memory image return through `return_result_tensor=True`

Not supported:

- mask edit
- multiple input images
- visual understanding
- thinking text output
- NF4 or INT8 quantization
- multi-GPU dispatch
