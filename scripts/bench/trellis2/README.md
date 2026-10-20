# TRELLIS.2-4B BF16 inference

This integration uses a native LightX2V runner and in-tree network code. It requires no external model repository checkout or experimental patch. Run the commands below from the repository root. The model integration is independent of other reconstruction models.

## Environment and precision

The validated MLU environment uses Python 3.12.3, vendor PyTorch 2.9.1, torch_mlu 1.31.1 and torchvision 0.24.1. Install the model-specific pins into that runtime:

```bash
python -m pip install -r scripts/bench/trellis2/requirements-mlu.txt
python -m pip install --no-deps -e .
export PLATFORM=cambricon_mlu DTYPE=BF16
```

The vendor runtime and driver are external prerequisites. A Docker recipe accepts an explicit compatible runtime image; registry access may require a vendor account:

```bash
docker build -f scripts/bench/reconstruction/Dockerfile.mlu \
  --build-arg BASE_IMAGE="$MLU_PYTORCH_IMAGE" --build-arg MODEL=trellis2 \
  -t lightx2v-trellis2-bf16 .
```

Matrix weights, linear/convolution operands and attention Q/K/V use BF16. Normalization statistics, RoPE phase construction, accumulation and geometry may use FP32; indices remain integers. `linear_backend` and `attn_type` select existing registered MM and attention backends. Platform selection uses LightX2V's existing `PLATFORM` / `AI_DEVICE`; model files contain no vendor branches. CUDA/A100 and MetaX performance are unmeasured. Offload, quantization and distributed inference are outside this integration.

## Weights and ordinary inference

The preparation command pins public Hugging Face revisions and retains their model cards/licenses. Weights and inputs are separate downloads with their own licenses.

```bash
python -m scripts.bench.trellis2.prepare_assets weights --output weights/trellis2
python -m lightx2v.infer \
  --model_cls trellis2 --task i23d --model_path weights/trellis2 \
  --config_json configs/trellis2/trellis2_i23d_bf16.json \
  --image_path example.png --save_result_path output/mesh.glb
```

Only the 512 pipeline is included: three 12-step Euler/CFG stages, geometry GLB and texture voxel outputs. The GLB has geometry, without baked texture. Cascade/1024/1536 and texture baking are outside this integration. The directory contains `pipeline.json`, `ckpts/`, `base/`, `dinov3/`, and `rembg/`; existing downloads can override `base_model_path`, `dino_model_path`, and `rembg_model_path` through the JSON configuration.

The input encoder uses pinned Transformers DINOv3 and BiRefNet. BiRefNet's public Python model files are downloaded into `rembg/` and loaded locally with `trust_remote_code=True`; review that pinned code before deployment. They are public model assets, not an external 3D pipeline. Encoders use their PyTorch/Hugging Face kernels; selecting the flow attention backend does not change every encoder kernel.

BiRefNet's native MLU deformable convolution upcasts to FP32, while its BF16 kernel is unsupported in the tested runtime. The registered `torch_deform_bf16` backend samples coordinates with FP32 bilinear interpolation, then projects BF16 sampled features with BF16 weights through the MM registry. Output tiling bounds temporary memory. `deform_conv_backend` selects this operator independently; it is fixed in both benchmark variants. Replacement is scoped to loaded module instances, without a global torchvision patch.

## Frozen single-device A/B benchmark

Expose exactly one idle accelerator per process and check utilization/processes before starting. Separate processes may run on two different idle cards. On MLU, expose one physical device to each container and use logical device 0. On CUDA, set `CUDA_VISIBLE_DEVICES` to a single card.

```bash
# Download the IBRNet Google Scanned Objects renderings archive from
# https://drive.google.com/uc?id=1tKHhH-L1viCvTuBO1xg--B_ioK7JUrrE
# The preparation command checks archive and extracted-image SHA256 values.
python -m scripts.bench.trellis2.prepare_assets inputs --output benchmark_inputs \
  --gso-archive /path/to/google_scanned_objects_renderings.zip
python -m scripts.bench.trellis2.benchmark \
  --model-path weights/trellis2 --input-root benchmark_inputs \
  --output benchmark_outputs/trellis2 --audit-bf16
```

`--config-json` overrides deployment paths. `--limit 2` is a diagnostic subset; omit it for the formal suite. Use a fresh output directory: reusing one replaces the JSONL record. Input byte sizes and SHA256 are checked before timing; model loading and hash verification are excluded.

The shared `scripts/bench/reconstruction/` code records events, comparisons, precision and statistics; the model-specific script supplies variant selection, request fields and timed call targets. It imports no unrelated model integration. A/B order alternates per request. Weights, BF16 policy, backend and seed remain fixed.

The two variants differ only in neighbor lookup (`cpu_hash` / `sorted`). The A100 reference cohort has 40 frozen common-success GSO objects, seed `42 + request_order`, 512 resolution, and 12 steps per stage. Exclude the first pair, leaving 39 paired observations. This is a latency cohort, not a full-dataset success-rate estimate. Events record all 36 `sample_once` calls including CFG; reduce to the median within each request, then take the median across requests. Synchronized pipeline wall time covers image reading through geometry GLB export, including texture voxel generation. Some `segment_reduce` calls fall back to CPU in the tested torch_mlu version; that cost is included.

Report pipeline p50 and median paired percentage change, with at least 31 effective pairs for `formal_candidate`. Failures are saved and counted; inspect them, all returned-tensor comparisons and raw event records before accepting a result. The tool saves environment/configuration/manifest identity, raw requests, comparisons and summary. `--audit-bf16` audits actual matrix operands during excluded warmup; require observed matrix calls, `passed=true`, and zero violations. Integer coordinates/topology are compared too.

## Validation and recorded measurements

The original complete integration run on MLU590-M9D (80 GB, driver 6.5.42) had 39 effective pairs, zero failures, pipeline p50 39.752405 / 12.215323 seconds (baseline / optimized), and median paired change -69.384%. The reduced sampler statistic was 0.212628 / 0.212445 seconds; lookup savings occur in sparse decoding outside those calls. All A/B output tensors were exactly equal and both warmup audits passed.

That implementation fixed the historical experiment's `-1/0/1` kernel-offset indexing, incomplete decoder BF16 conversion and FP32 BiRefNet deformable projection. Kernel indices now use `0/1/2`, matching dense convolution and the official FlexGEMM offset order. These corrections change historical mesh outputs, so older experiment quality/timing numbers are not acceptance results. Full-dataset ground-truth quality metrics have not been rerun for the corrected graph. A/B consistency proves the lookup optimization preserves outputs, not ground-truth quality.

Those full-suite measurements were recorded before the PR split (`c20b99d`). Splitting retains the network/operator math and fixed inputs; model-specific entry points and dense/sparse adapter loading are separated. Run the commands above to obtain timings for this checkout. Latency depends on CPU decoding, storage, runtime versions, clocks and contention; A100 statistics define the protocol and do not make MLU timings interchangeable with A100 timings.

```bash
python -m pytest -q tests/test_inference_modules.py tests/test_trellis2_ops.py
```

The new numerical tests run without an accelerator: the import context isolates eager device bootstrap, restores package namespaces, and uses real CPU operations. Source licenses are retained in the model directory and included in built packages.
