# BAGEL T2I

This example covers the first LightX2V BAGEL integration scope: text-to-image generation with `ByteDance-Seed/BAGEL-7B-MoT`.

## Model

Download the model to a local directory. If the Hugging Face mirror is available in your region, prefer:

```bash
mkdir -p /data1/lyxu18/models/ByteDance-Seed
HF_ENDPOINT=https://hf-mirror.com hf download ByteDance-Seed/BAGEL-7B-MoT \
  --local-dir /data1/lyxu18/models/ByteDance-Seed/BAGEL-7B-MoT \
  --max-workers 8
```

The model directory must contain `config.json`, `ema.safetensors`, `ae.safetensors`, tokenizer files, and the other files from the BAGEL-7B-MoT repository.

## Requirements

- CUDA GPU environment with enough memory for BAGEL-7B-MoT.
- PyTorch compatible with the current LightX2V environment.
- `flash-attn` installed for the same CUDA/PyTorch build.

If `flash-attn`, `ema.safetensors`, `ae.safetensors`, or required config fields are missing, the BAGEL runner raises a direct error before generation.

## Run

```bash
export lightx2v_path=/data1/lyxu18/opensource_proj/LightX2V
export model_path=/data1/lyxu18/models/ByteDance-Seed/BAGEL-7B-MoT

bash scripts/bagel/run_bagel_t2i.sh
```

Equivalent direct command:

```bash
python -m lightx2v.infer \
  --model_cls bagel \
  --task t2i \
  --model_path /data1/lyxu18/models/ByteDance-Seed/BAGEL-7B-MoT \
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
  --model_path /data1/lyxu18/models/ByteDance-Seed/BAGEL-7B-MoT \
  --config_json configs/bagel/bagel_t2i.json \
  --prompt "A glass greenhouse in a snowy garden" \
  --target_shape 576 1024 \
  --save_result_path save_results/bagel_t2i_576x1024.png \
  --seed 42
```

## Current Scope

Supported in this v1:

- `python -m lightx2v.infer --model_cls bagel --task t2i`
- PNG saving
- `seed`
- `aspect_ratio`
- `target_shape`
- service in-memory image return through `return_result_tensor=True`

Not supported in this v1:

- image-to-image
- image editing
- visual understanding
- thinking text output
- NF4 or INT8 quantization
- multi-GPU dispatch
