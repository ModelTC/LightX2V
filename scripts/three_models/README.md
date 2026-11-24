# HY-Motion, SF3D and SDXL on A100

These adapters run the released checkpoints using model implementations inside
LightX2V. They do not instantiate upstream neural models or Diffusers pipelines;
Transformers supplies local tokenizers only.

| Model | Task | Output | Setup and inference |
| --- | --- | --- | --- |
| `tencent/HY-Motion-1.0` (full) | `t2m` | Motion arrays in NPZ | [HY-Motion](../hymotion/README.md) |
| `stabilityai/stable-fast-3d` | `i23d` | Textured GLB from one image | [SF3D](../sf3d/README.md) |
| `stabilityai/stable-diffusion-xl-base-1.0` | `t2i` | PNG | [SDXL](../sdxl/README.md) |

Use an isolated Python 3.10 environment and run
`bash scripts/three_models/install_a100.sh`. Prepare the local checkpoints as
described in each model's README, including the separately compiled SF3D geometry
extensions and its checksum-verified grid. The inference code does not download
model checkpoints. Checkpoint revisions and source attribution are recorded in
the model documentation and adjacent license/NOTICE files.

Each model has a JSON configuration and launcher under
`configs/platforms/{nvidia,mlu,metax,ascend_npu}` and
`scripts/platforms/{nvidia,mlu,metax,ascend_npu}`. NVIDIA launchers use the shared
`scripts/base/base.sh` environment and the standard `lightx2v.infer` entry point.
The other three platforms are explicit placeholders and exit before inference.
Validation covers one NVIDIA A100-SXM4-40GB, without quantization, parallelism,
offload, CUDA graphs or feature caching.

`models/networks/<model>/weights` owns parameter structures, checkpoint loading
and operator binding; `infer` owns computation. Runners handle input/output and
orchestration. Model request specifications and configuration loaders use the
shared CLI/service contracts, with request defaults nested in `request_defaults`.
HY-Motion and SDXL schedulers are in `models/schedulers`. Small shape/dtype
bindings and the CLIP implementation shared by these models live under
`models/networks/three_models`; MM, Conv2d, LayerNorm, RMSNorm, attention and RoPE
reuse existing registered LightX2V operators. Model-specific masks, layouts,
precision policies and rounding order are retained.

Each output has a JSON sidecar with synchronized end-to-end request time and
peak allocated GPU memory. Time includes input preprocessing and final file
serialization; model initialization is excluded. SF3D enables process-wide
deterministic algorithms by default, and its launcher exports
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python. Use identical settings
in native SF3D when comparing full UV/texture/GLB results: setting the seed alone
does not make its CUDA reductions deterministic.

Run the metadata/CLI and A100 numerical regression checks with:

```bash
python -m pip install pytest
CUDA_VISIBLE_DEVICES=0 PLATFORM=cuda DTYPE=BF16 \
  python -m pytest -q tests/test_three_model_requests.py tests/test_three_model_a100_contracts.py
```

The GPU checks skip on machines without CUDA. They cover masked attention,
normalization, strided/autocast linear dispatch, rotary encoding, Euler scalar
promotion, native motion frame clamping and unsupported feature admission.
Metadata checks exercise the actual CLI, isolated configuration imports,
platform support declarations and atomic/checksum-verified asset preparation.

On A100 with PyTorch 2.6.0 / CUDA 12.4, nine frozen random dataset inputs
(three HumanML3D motions, one prompt each from GenEval/Parti/COCO, and the first
view of three GSO objects) were compared against independent native references.
All 150 captured tensor pairs matched exactly, including shape and dtype.
Motion NPZ arrays matched exactly; PNG and deterministic GLB files were also
byte-identical. This is sample parity validation, not a full-dataset metric run
or a guarantee of identical results across hardware/software versions.
