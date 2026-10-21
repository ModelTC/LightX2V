# VGGT-1B BF16 inference

This integration uses a native LightX2V runner and in-tree network code. It requires no external model repository checkout or experimental patch. Run the commands below from the repository root. The model integration is independent of other reconstruction models.

## Environment and precision

The validated MLU environment uses Python 3.12.3, vendor PyTorch 2.9.1, torch_mlu 1.31.1 and torchvision 0.24.1. Install the model-specific pins into that runtime:

```bash
python -m pip install -r scripts/bench/vggt/requirements-mlu.txt
python -m pip install --no-deps -e .
export PLATFORM=cambricon_mlu DTYPE=BF16
```

The vendor runtime and driver are external prerequisites. A Docker recipe accepts an explicit compatible runtime image; registry access may require a vendor account:

```bash
docker build -f scripts/bench/reconstruction/Dockerfile.mlu \
  --build-arg BASE_IMAGE="$MLU_PYTORCH_IMAGE" --build-arg MODEL=vggt \
  -t lightx2v-vggt-bf16 .
```

Matrix weights, linear/convolution operands and attention Q/K/V use BF16. Normalization statistics, RoPE phase construction, accumulation and geometry may use FP32; indices remain integers. `linear_backend` and `attn_type` select existing registered MM and attention backends. Platform selection uses LightX2V's existing `PLATFORM` / `AI_DEVICE`; model files contain no vendor branches. CUDA/A100 and MetaX performance are unmeasured. Offload, quantization and distributed inference are outside this integration.

## Weights and ordinary inference

The preparation command pins public Hugging Face revisions and retains their model cards/licenses. Weights and inputs are separate downloads with their own licenses.

```bash
python -m scripts.bench.vggt.prepare_assets weights --output weights/vggt
python -m lightx2v.infer \
  --model_cls vggt --task recon --model_path weights/vggt \
  --config_json configs/vggt/vggt_recon_bf16.json \
  --input_path example_images --save_result_path output/reconstruction.npz
```

Inputs are ordered images, transformed to 518 × 518. Outputs include cameras, depth, point maps, confidence and original image bounds; tracking is not included. The CLI sorts a directory lexicographically. The runner's Python request can pass an ordered list as `input_path`; returned tensors are on CPU. `preprocess_workers=8` uses an ordered thread pool; `1` uses the identical serial pixel transform.

## Frozen single-device A/B benchmark

Expose exactly one idle accelerator per process and check utilization/processes before starting. Separate processes may run on two different idle cards. On MLU, expose one physical device to each container and use logical device 0. On CUDA, set `CUDA_VISIBLE_DEVICES` to a single card.

```bash
python -m scripts.bench.vggt.prepare_assets inputs --output benchmark_inputs
python -m scripts.bench.vggt.benchmark \
  --model-path weights/vggt --input-root benchmark_inputs \
  --output benchmark_outputs/vggt --audit-bf16
```

`--config-json` overrides deployment paths. `--limit 2` is a diagnostic subset; omit it for the formal suite. Use a fresh output directory: reusing one replaces the JSONL record. Input byte sizes and SHA256 are checked before timing; model loading and hash verification are excluded.

The shared `scripts/bench/reconstruction/` code records events, comparisons, precision and statistics; the model-specific script supplies variant selection, request fields and timed call targets. It imports no unrelated model integration. A/B order alternates per request. Weights, BF16 policy, backend and seed remain fixed.

The two variants differ only in preprocessing workers (1 / 8). The A100 reference cohort has 36 requests: courtyard with 32 views, electro with 32, office with 26, each repeated 12 times. Use the original high-resolution JPEG bytes; resized replacements change the workload. Exclude the first pair, leaving 35 paired observations. Events time one model forward; synchronized wall time covers image reading through all CPU predictions, without dataset metric computation or NPZ serialization.

Report pipeline p50 and median paired percentage change, with at least 31 effective pairs for `formal_candidate`. Failures are saved and counted; inspect them, all returned-tensor comparisons and raw event records before accepting a result. The tool saves environment/configuration/manifest identity, raw requests, comparisons and summary. `--audit-bf16` audits actual matrix operands during excluded warmup; require observed matrix calls, `passed=true`, and zero violations. Integer coordinates/topology are compared too.

## Validation and recorded measurements

The original complete integration run on MLU590-M9D (80 GB, driver 6.5.42) had 35 effective pairs, zero failures, pipeline p50 25.106818 / 8.343892 seconds (baseline / optimized), and median paired change -66.812%. Forward time was 5.101998 / 5.102104 seconds: the gain comes from CPU preprocessing. All A/B output tensors were exactly equal and both warmup audits passed.

Those full-suite measurements were recorded before the PR split (`c20b99d`). Splitting retains the network/operator math and fixed inputs; model-specific entry points and dense/sparse adapter loading are separated. Run the commands above to obtain timings for this checkout. Latency depends on CPU decoding, storage, runtime versions, clocks and contention; A100 statistics define the protocol and do not make MLU timings interchangeable with A100 timings.

```bash
python -m pytest -q tests/test_inference_modules.py tests/test_vggt_preprocessing.py
```

The new numerical tests run without an accelerator: the import context isolates eager device bootstrap, restores package namespaces, and uses real CPU operations. Source licenses are retained in the model directory and included in built packages.
