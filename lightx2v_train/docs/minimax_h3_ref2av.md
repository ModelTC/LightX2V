# MiniMax-H3 Ref2AV distillation

The H3 adapters use the current capability-based `dmd` trainer. They support
T2AV, cached first/last-frame conditions (`i2av`, `l2av`, `fl2av`), and ordered
image/video/audio references through the `transformer_ref` partition. Existing
condition caches and their cost-balanced samplers are supported; no cache
re-encoding is needed for the matching 124-frame recipe.

## Four ACP workers, eight GPUs each

Use the same checkout, environment, shared cache and output directory on all
four workers. ACP must supply a common `MASTER_ADDR` and `MASTER_PORT`.
The environment needs the H3-compatible Diffusers build, PEFT, and the
FlashAttention-3 kernel used by the original H3 training setup.

```bash
cd /path/to/LightX2V
export H3_PYTHON=/path/to/environment/bin/python
export H3_MODEL_PATH=/path/to/MiniMax-H3
export H3_REF2AV_CACHE=/path/to/ref2av_cache/metadata.jsonl
export H3_REF2AV_DMD_OUTPUT=/path/to/outputs/ref2av_pdmd
export H3_RDZV_ID=h3_ref2av_pdmd_run01
export H3_PDMD=true
# Optional: point to the existing offline FlashAttention-3 snapshot.
export H3_KERNEL_SNAPSHOT=/path/to/flash-attn3/snapshot
bash lightx2v_train/scripts/run_minimax_h3_ref2av_fsdp32_32gpu_acp.sh
```

Append `--dry-run` to validate paths and print the launch command without
starting training. For the unchanged DMD objective, use `H3_PDMD=false` and a
separate output directory/rendezvous ID. If needed, validate the cache row
count with `H3_REF2AV_EXPECTED_ROWS=13553`.

The config is
`configs/train/dmd/minimax_h3_ref2av_dmd_lora_match124_image_audio_1to5_uniform_fsdp32_32gpu_full_fake.yaml`.
The recipe uses:

| Setting | Value |
| --- | --- |
| Topology | FSDP2 across 32 ranks, sequence parallel size 1 |
| Conditions | Cached 1–5 image references plus audio, metadata-controlled geometry |
| Target | 124 frames, 768×1344 or 1344×768 |
| Student | LoRA rank 128, alpha 8; learning rate 5e-5 |
| Critic (`fake`) | Full model; FP32 master parameters; learning rate 4e-7 |
| Denoising | 8 evaluations; video/audio flow shifts 12/3 |
| Updates | 5 critic updates per student update; physical batch 1 per rank |
| Precision | FP32 generated latents, BF16 compute, FP32 gradient reduction |
| DMD normalization | Separate video/audio normalizers; epsilon 0; half-MSE mean |
| Saving | Every 50 outer iterations; keep 5 checkpoints; automatic resume |

FSDP initializes the transformer on the meta device and streams checkpoint
shards after partitioning. Cached-condition training does not load the VAE or
text encoder. The cost sampler is configured from the resumed outer iteration
before reading samples. Visual references receive one shared near-clean noise
augmentation per rollout; audio references stay clean.

Cache preparation tools live in `data_process/minimax_h3/`: text/keyframe
conditions, ordered Ref2AV references, teacher AV latents, and manifest
mixing/filtering/merging/rebasing. Their model, input, and output paths are
explicit command-line arguments; use each tool's `--help` for its cache schema.

## Optional projected DMD

[`PDMD`, Eq. 4 and Appendix C.2](https://arxiv.org/html/2609.35768) removes the
component of the detached DMD direction parallel to the critic–student
residual. Video and audio are projected separately in FP32, across all
non-batch axes, before their existing normalizers. A zero residual leaves the
direction unchanged. The critic regression objective and model forward count
do not change.

The option is
`model.capabilities.distribution_matching.projected_dmd`; the provided config
reads it from `H3_PDMD` and defaults to `false`. This is an application of the
paper's projection to Ref2AV, not a reproduction of a published Ref2AV recipe.
The PDMD toggle leaves all other recipe settings unchanged.

New checkpoints record the objective, modality normalization, geometry,
parallel topology and sampler recipe. Incompatible resume settings fail
explicitly. Start a fresh output directory when comparing DMD and PDMD;
`resume.allow_distribution_matching_transition` is reserved for intentional
objective changes rather than a controlled comparison.

Student-only sparse attention, adaptive video regularization, and memory
guards are also available under `model.capabilities.distribution_matching`.
They are disabled in this Ref2AV configuration; adaptive regression needs
teacher latent caches and is not supported by Ref2AV conditions-only caches.
