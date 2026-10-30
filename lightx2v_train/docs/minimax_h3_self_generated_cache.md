# Download and encode the MiniMax-H3 self-generated dataset

Run once on one preprocessing machine with four GPUs. The launcher downloads
the public ModelScope dataset completely, prepares a deterministic manifest,
runs one independent encoder per GPU, then verifies all four shards and writes
the training manifests. No distributed training or DiT loading is involved.

## One command on the new machine

Sync these tools to `/mnt/lm_data_afs/gushiqiao/codes/news/LightX2V` first. Use the
existing H3 environment and local model. The launcher installs `modelscope` into
that Python environment only when it is missing; it does not upgrade Torch,
Diffusers, or Transformers.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 \
H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3 \
H3_SELFGEN_RAW=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated \
H3_SELFGEN_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16 \
bash lightx2v_train/scripts/download_encode_minimax_h3_self_generated_4gpu.sh
```

Only this one process needs to be launched. The shell stays in the foreground
until download, all four encoders, and manifest verification finish. Download
failure prevents encoding; an encoder failure stops the remaining workers and
prevents publication of a successful merged cache. Worker logs are under
`$H3_SELFGEN_CACHE/logs/`.

Both T2VA and Ref2VA subsets are processed by default, preserving their native
resolutions and 124-frame timing. The small subset of published records whose
media files are unavailable is reported by preparation and excluded, rather
than causing every valid sample to be discarded. This behavior is controlled
by `H3_SELFGEN_ALLOW_MISSING_MEDIA=1` (default); set it to `0` to require every
referenced file. Actual decoding/encoding failures remain fatal by default;
`H3_SELFGEN_ALLOW_FAILURES=1` explicitly permits a partial successful cache and
records failed samples.

## Model components and encoding

- T2AV text conditioning: `H3_MODEL_PATH/{text_encoder,tokenizer,processor}`.
- Ref2AV text conditioning: `H3_REF_MODEL_PATH`, defaulting to
  `H3_MODEL_PATH/Ref2VA`, with its own text encoder, tokenizer, and processor.
- Video/audio VAEs: `H3_MODEL_PATH/vae` and `H3_MODEL_PATH/audio_vae`.
- Text encoders run in BF16; VAEs compute in FP32. Clean AV targets use
  deterministic posterior mode. Reference conditions follow H3's native
  encoding rules with the fixed reference seed 42. Floating cache tensors are
  stored in BF16.
- The video VAE uses 512px tiles with 64px overlap to limit GPU memory usage.
- Text encoders are unloaded before VAEs are loaded. Conditioning and AV
  targets are encoded without loading an H3 DiT.
- Ref2AV image, silent-video, and audio-video reference types are retained.
- Silent video references retain their visual stream and discard embedded
  audio in a separate cached copy. PyAV remux is used if no `ffmpeg` binary is
  available; the original downloaded file is preserved.

Each complete sample provides `condition_path`, `video_latent_path`,
`audio_latent_path`, source identity, geometry, and a cache fingerprint. Latents
use the normalized H3 representation expected by the existing training code:
video `[24, 37, H/16, W/16]`, audio `[2, 32, 207]` for each 124-frame target.

## Outputs and training paths

```text
minimax_h3_self_generated_latents_124_bf16/
├── prepared.jsonl
├── all_metadata.jsonl                # every successful sample, native geometry
├── metadata.jsonl                    # T2AV, only the two existing 768p canvases
├── condition_metadata.jsonl          # corresponding conditions without AV paths
├── t2av/metadata.jsonl               # all T2AV native resolutions
├── t2av_768p/metadata.jsonl           # same default CM-compatible view
├── ref2av/metadata.jsonl             # all reference types and native resolutions
├── ref2av_imageonly/metadata.jsonl   # pure image references, all native resolutions
├── ref2av_imageonly_768p/metadata.jsonl
├── ref2av_imageonly_768p/metadata.complete.json
├── cache.complete.json               # verified counts and final completion
└── logs/
```

Use this path for the existing three-task Turbo configuration's T2AV CM stream:

```bash
H3_CM_CACHE=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_self_generated_latents_124_bf16/metadata.jsonl
```

`all_metadata.jsonl` preserves extra aspect ratios and Ref tasks. Existing
training configurations that restrict resolutions or reference types need the
corresponding filtered view above, rather than the unfiltered manifest.
The Ref image-only view contains single-image references; a training sampler
configured to require image counts 1 through 6 must be adjusted for this data.
The download/encoding command itself does not change a training configuration
or start training.

The current dCM1500 -> PDMD32 Ref launcher selects `ref2av_imageonly/` for
both the clean AV manifest and the new condition-only manifest. It automatically
builds this native-resolution view from the completed `ref2av/` cache at launch,
without re-encoding payloads. To create the view separately on an older merged
cache, run `python lightx2v_train/data_process/minimax_h3/prepare_modelscope_h3_dataset.py imageonly-view --output-dir "$H3_SELFGEN_CACHE"`.

## Resume and a small verification run

Rerun the same command to resume completed downloads and validated per-sample
encodes. A cache fingerprint prevents mixing different sources, model
identities, or preprocessing options in the same destination. Use a different
cache directory when changing the selected subset or sample limit.

Append `--dry-run` to print the exact download, preparation, four encoder, and
merge commands without downloading, installing packages, creating output
directories, or loading models.

Set `H3_SELFGEN_MAX_SAMPLES=2` and choose a new `H3_SELFGEN_CACHE` directory for
a small run with at most two samples per subset. Download still obtains the
whole source dataset; the limit applies to preprocessing. Set
`H3_SELFGEN_SUBSETS=t2av` or `ref2av` to select only that encoding subset.

Source: [MiniMax-H3-Self-Generated-Dataset](https://www.modelscope.ai/datasets/DiffSynth-Studio/MiniMax-H3-Self-Generated-Dataset).
