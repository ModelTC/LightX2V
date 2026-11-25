# Seed-VC V1 audio conversion

[简体中文](README_zh.md)

Content features, reference mel, speaker embeddings and optional F0 prepared by the application pass through the native LightX2V length regulator, DiT, CFM sampler and BigVGAN vocoder to produce complete WAV/FLAC audio.

Supported variants: V1 44.1 kHz F0 singing conversion, V1 22.05 kHz speech conversion, and fixed singing adapters in the local `seed-vc-lora-v1` format. V2 AR/style conversion, tiny/HiFT, live sound-card I/O and training are outside this integration.

## Weights

Download the original weights from:

- [Seed-VC V1 checkpoints and YAML](https://huggingface.co/Plachta/Seed-VC/tree/main)
- [44.1 kHz BigVGAN](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x)
- [22.05 kHz BigVGAN](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)

Set `lightx2v_path` and `model_path` at the top of the launcher for your checkout and weight root. The shared `data_dir` points to prepared inputs in the sibling `lightx2v_examples` checkout. BigVGAN inference is included in LightX2V. No checkpoint conversion is needed. The weight root contains `svc/` and `vc/`:

```text
Seed-VC/
├── svc/
│   ├── config_dit_mel_seed_uvit_whisper_base_f0_44k.yml
│   ├── DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth
│   ├── adapter.pth                 # optional, your singing adapter
│   └── vocoder/
│       ├── config.json
│       └── bigvgan_generator.pt
└── vc/
    ├── config_dit_mel_seed_uvit_whisper_small_wavenet.yml
    ├── DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth
    └── vocoder/
        ├── config.json
        └── bigvgan_generator.pt
```

## Example inputs

Samples are hosted in [lightx2v_examples/a2a/seedvc](https://github.com/ModelTC/lightx2v_examples/tree/main/a2a/seedvc). From the LightX2V repository root, clone the examples repository alongside LightX2V and check out only the Seed-VC data:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/ModelTC/lightx2v_examples.git ../lightx2v_examples
git -C ../lightx2v_examples sparse-checkout set a2a/seedvc
```

The launcher sets `data_dir` to `${lightx2v_path}/../lightx2v_examples/a2a/seedvc`. For another location, edit this variable and set `warmup_sample_path` in the deployment configuration when enabling warmup.

## End-to-end examples

Run from the LightX2V repository root. The launcher takes only a preset number, defaulting to 1. It selects the prepared inputs and model variant automatically; preset 6 loads LoRA. Every command produces playable audio in `save_results/seedvc/`.

| Preset | Input | Change from the default | Output |
|---|---|---|---|
| 1: singing conversion, preserve pitch | `../lightx2v_examples/a2a/seedvc/svc.json` | none | `function_1.wav` |
| 2: singing conversion, +2 semitones | same | `semitone_shift=2` | `function_2.wav` |
| 3: align median pitch to reference | same | `auto_f0_adjust=true` | `function_3.wav` |
| 4: 1.2× output duration | same | `length_adjust=1.2` | `function_4.wav` |
| 5: speech conversion | `../lightx2v_examples/a2a/seedvc/vc.json` | loads V1 speech weights | `function_5.wav` |
| 6: singing conversion with LoRA | `../lightx2v_examples/a2a/seedvc/svc_lora.json` | loads `svc/adapter.pth` with the Shan Yichun reference | `function_6.wav` |

```bash
bash scripts/seedvc/run_seedvc.sh 1
```

```bash
bash scripts/seedvc/run_seedvc.sh 2
```

```bash
bash scripts/seedvc/run_seedvc.sh 3
```

```bash
bash scripts/seedvc/run_seedvc.sh 4
```

```bash
bash scripts/seedvc/run_seedvc.sh 5
```

```bash
bash scripts/seedvc/run_seedvc.sh 6
```

Presets 1–4 and 6 share the singing model, with LoRA loaded for preset 6. Preset 5 uses the speech model. Singing and speech outputs are mono at 44.1 kHz and 22.05 kHz respectively. Singing samples use generation windows with overlap stitching based on source and reference duration. Listen to the FLAC files under `source/` and `reference/` in the shared data directory.

Presets 1–4 use `svc.json`; preset 6 keeps its own `svc_lora.json`. Both manifests point to the same `source/svc.flac` and `source/svc.safetensors`. Only the reference differs: the former uses the upstream example, while the latter selects the 20-second Shan Yichun reference in `reference/shan_yi_chun.*`. Preset 6 also loads the matching LoRA. When replacing the adapter with one trained for another singer, also prepare matching reference features.

Preset 6 requires a compatible adapter at `svc/adapter.pth`. The loader checks the format, base checkpoint SHA256, architecture signature and target tensors, then merges into FP32 weights at startup. The adapter format comes from a local training extension; per-request adapter switching and PEFT checkpoints are unsupported. No private adapter is included.

## Prepared sample contract

The shared `../lightx2v_examples/a2a/seedvc/` directory contains three sample manifests sharing source data, with references organized by voice.

```text
../lightx2v_examples/a2a/seedvc/
├── svc.json
├── svc_lora.json
├── vc.json
├── source/
│   ├── svc.flac
│   ├── svc.safetensors
│   ├── vc.flac
│   └── vc.safetensors
└── reference/
    ├── s2p1_svc.flac
    ├── s2p1_svc.safetensors
    ├── s2p1_vc.flac
    ├── s2p1_vc.safetensors
    ├── shan_yi_chun.flac
    └── shan_yi_chun.safetensors
```

`s2p1_svc` and `s2p1_vc` are the same upstream reference prepared for the singing and speech models respectively. A manifest includes format version, variant, sample rate, hop length, content encoder identity, source mel frame count, and relative feature filenames.

| Tensor | Shape |
|---|---|
| `source_content`, `reference_content` | `[1, T, 768]`, Whisper-small features |
| `reference_mel` | `[1, 128, T]` for singing; `[1, 80, T]` for speech |
| `style` | `[1, 192]`, CAMPPlus embedding |
| `source_f0`, `reference_f0` | `[1, T]`, raw F0 for singing |

Whisper emits FP16; storing that output as FP16 and restoring FP32 on load adds no quantization. Applications prepare new samples using the upstream resampling, mel, reference cropping and F0 conventions. The learned length regulator runs inside LightX2V.

CLI and HTTP share `sample_path`, `seed`, `diffusion_steps` (30), `cfg_rate` (0.7), `length_adjust` (1), `semitone_shift` (0), and `auto_f0_adjust` (false). A zero CFG rate uses one branch. Pitch options require singing weights. Choose WAV or FLAC through the extension of `save_result_path`.

## Deployment options: independent of the sample preset

Use `run_seedvc.sh` for one GPU and `run_seedvc_tp2.sh` for two GPUs. Both take only a preset number from 1 to 6. Select the deployment configuration with `--config_json` in the inference command.

| Configuration | Purpose |
|---|---|
| `seedvc.json` | Default single-GPU inference, without compilation or warmup |
| `seedvc_offload.json` | Keep DiT block weights on CPU to reduce GPU memory use; the vocoder stays on GPU |
| `seedvc_tp2.json` | Two-GPU tensor parallelism for the DiT; only rank 0 decodes audio |
| `seedvc_compile.json` | Compile DiT blocks and enable startup warmup |

For CPU offload or compilation, change `--config_json` in `run_seedvc.sh`:

```text
CPU offload: --config_json ${lightx2v_path}/configs/seedvc/seedvc_offload.json
Compile and warmup: --config_json ${lightx2v_path}/configs/seedvc/seedvc_compile.json
Default: --config_json ${lightx2v_path}/configs/seedvc/seedvc.json
```

Then run with only the preset number, for example:

```bash
bash scripts/seedvc/run_seedvc.sh 1
```

The two-GPU launcher already selects `seedvc_tp2.json`:

```bash
bash scripts/seedvc/run_seedvc_tp2.sh 1
```

All six presets use the selected deployment configuration. Preset 6 automatically loads `svc/adapter.pth`; set `adapter_path` in the deployment configuration to use another adapter. No separate LoRA configuration file is needed.

Single-GPU outputs are `save_results/seedvc/function_<preset>.wav`; two-GPU outputs are `save_results/seedvc/function_<preset>_tp2.wav`. Edit `--save_result_path` to keep results from different single-GPU deployment options.

Single-GPU inference and the HTTP server use GPU 0; TP2 uses GPUs 0 and 1. To select other devices, edit `export CUDA_VISIBLE_DEVICES=...` in the corresponding launcher. Seed-VC uses FP32 internally, so the launcher does not need to set `DTYPE`.

`use_compile` and `warmup` are independent configuration switches, both false by default. The compilation configurations enable both. Warmup uses the corresponding sample under the sibling `lightx2v_examples/a2a/seedvc/` directory and does not save audio. Set `warmup_sample_path` for representative custom data; new input shapes may still trigger compilation.

## HTTP

```bash
bash scripts/seedvc/server/run_server.sh svc
```

In another terminal:

```bash
python scripts/seedvc/server/post_a2a.py \
  --sample-path "$(pwd)/../lightx2v_examples/a2a/seedvc/svc.json" \
  --output save_results/seedvc/http_svc.wav
```

The default URL is `http://127.0.0.1:8018`; select another endpoint with `--url`. `sample_path` must be readable on the server. For speech, launch with `vc` and submit `vc.json`. For LoRA, launch with `svc_lora` and submit `svc_lora.json`. The client waits for task completion and downloads the final audio through the results API.
