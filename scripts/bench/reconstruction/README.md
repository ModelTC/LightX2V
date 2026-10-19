# VGGT and TRELLIS.2 BF16 inference

Both models use LightX2V runners and in-tree network code. No VGGT or TRELLIS repository checkout, local experiment patch, or external pipeline is required.

## Scope

| Model | Task | Inputs and outputs | Optimization |
| --- | --- | --- | --- |
| VGGT-1B | `recon` | Ordered images at 518 × 518; cameras, depth, point maps, confidence, original image bounds | Ordered CPU preprocessing with eight workers; the serial path uses the same pixel transform |
| TRELLIS.2-4B | `i23d` | One image; the 512 pipeline with three 12-step samplers; geometry GLB and texture voxel attributes | Sorted integer neighbor lookup for portable submanifold convolution; CPU hash lookup remains the reference |

Matrix weights, linear/conv operands, and attention Q/K/V use BF16. Normalization, RoPE phase construction, sampling accumulation, and geometry decoding may use FP32. Coordinates and indices remain integers. TRELLIS checkpoint metadata requesting FP16 decoder weights is overridden at loading, including custom sparse convolutions.

BiRefNet's native torchvision deformable convolution upcasts to FP32 in the validated MLU runtime; calling its kernel directly with BF16 is unsupported there. The `torch_deform_bf16` backend instead samples coordinates with FP32 bilinear interpolation and sends BF16 sampled features and BF16 weights to the registered MM backend. Output tiles bound temporary memory. This backend is fixed in both TRELLIS benchmark variants, and `deform_conv_backend` can select another implementation of the same operator contract. Model code contains no vendor branch or global torchvision patch.

VGGT tracking, TRELLIS cascade/1024/1536 pipelines, texture baking, CPU offload, quantization, and distributed inference are outside this integration. A geometry GLB does not contain a baked texture; texture voxel attributes are returned separately. Model weights and input assets are downloaded separately, under their respective licenses. VGGT architecture files retain the VGGT license and Apache-2.0 notices; TRELLIS-derived files retain the MIT license.

## Environment

For the validated MLU path, start from the vendor PyTorch 2.9.1 / `torch_mlu` 1.31.1 runtime, Python 3.12.3, and torchvision 0.24.1. Install LightX2V's normal dependencies and the matching extra pins:

```bash
# VGGT environment
python -m pip install -r scripts/bench/reconstruction/requirements-vggt-mlu.txt

# TRELLIS environment
python -m pip install -r scripts/bench/reconstruction/requirements-mlu.txt

python -m pip install --no-deps -e .
export PLATFORM=cambricon_mlu DTYPE=BF16
```

Install each requirements file in its own environment: the validated Pillow/NumPy versions differ. CPU image decoding is part of VGGT's latency, so its package versions matter. The vendor runtime and device driver must be installed independently; these are not supplied by a Python package from this repository. The TRELLIS input encoder uses Transformers DINOv3 and pinned BiRefNet custom model code; neither is an external TRELLIS pipeline.

An installation recipe is also provided for a vendor MLU runtime image. Supply an image containing the versions above; registry access may require your vendor account. The base image is an explicit prerequisite and is not a project-specific experiment image.

```bash
docker build -f scripts/bench/reconstruction/Dockerfile.mlu \
  --build-arg BASE_IMAGE="$MLU_PYTORCH_IMAGE" --build-arg MODEL=vggt \
  -t lightx2v-vggt-bf16 .
# Use MODEL=trellis2 for the TRELLIS environment.
```

`PLATFORM=cuda` and `PLATFORM=metax_cuda` use their existing LightX2V device setup. The new model files contain no vendor imports. CUDA/A100 and MetaX performance have not been measured for this integration. `attn_type` selects a registered backend, such as `torch_sdpa` or `mlu_flash_attn` when its dependencies are installed. `linear_backend` selects a registered MM backend that preserves BF16 weights. Other model encoders and convolutions use their PyTorch device kernels; changing an attention backend does not change those components.

## Weights and ordinary inference

The preparation tool pins each Hugging Face repository to a commit. It downloads only the components used by this integration:

```bash
python scripts/bench/reconstruction/prepare_assets.py weights vggt --output weights/vggt
python scripts/bench/reconstruction/prepare_assets.py weights trellis2 --output weights/trellis2

python -m lightx2v.infer \
  --model_cls vggt --task recon --model_path weights/vggt \
  --config_json configs/vggt/vggt_recon_bf16.json \
  --input_path example_images --save_result_path output/reconstruction.npz

python -m lightx2v.infer \
  --model_cls trellis2 --task i23d --model_path weights/trellis2 \
  --config_json configs/trellis2/trellis2_i23d_bf16.json \
  --image_path example.png --save_result_path output/mesh.glb
```

The TRELLIS model directory contains `pipeline.json`, `ckpts/`, `base/`, `dinov3/`, and `rembg/`. Existing downloads can be used by setting `base_model_path`, `dino_model_path`, and `rembg_model_path` in the deployment JSON. BiRefNet is loaded from that local directory with `trust_remote_code=True`; review the pinned upstream code before deployment.

The same runners are available through `LightX2VPipeline` or `runner_factory.build_runner`. VGGT's Python request may provide an ordered list as `input_path`; the CLI directory input is sorted lexicographically. Both return CPU tensors. Errors such as an empty TRELLIS sparse structure are reported, not silently replaced by a different sample.

## Frozen A/B benchmark

Use one exposed device per process and check device utilization/processes before starting. Two separate processes may use two idle cards. Do not run both processes on the same card. For MLU, bind one physical device into each container, then use its logical device 0. For CUDA, set `CUDA_VISIBLE_DEVICES` to one card.

```bash
python scripts/bench/reconstruction/prepare_assets.py inputs vggt --output benchmark_inputs

# Download the IBRNet Google Scanned Objects renderings archive from the URL
# printed by the preparation tool; the archive SHA256 is checked before extraction.
python scripts/bench/reconstruction/prepare_assets.py inputs trellis2 \
  --output benchmark_inputs --gso-archive /path/to/google_scanned_objects_renderings.zip

python scripts/bench/reconstruction/benchmark.py vggt \
  --model-path weights/vggt --input-root benchmark_inputs \
  --output benchmark_outputs/vggt --audit-bf16

python scripts/bench/reconstruction/benchmark.py trellis2 \
  --model-path weights/trellis2 --input-root benchmark_inputs \
  --output benchmark_outputs/trellis2 --audit-bf16
```

`--config-json` supplies deployment overrides for existing weight directories. `--limit 4` runs a diagnostic subset. Each output directory is a fresh run: invoking the tool again replaces its JSONL record. Use a different directory to retain previous runs.

The two variants differ only in preprocessing workers (VGGT: 1 versus 8) or neighbor lookup (TRELLIS: CPU hash versus sorted). Each request runs both variants, with order reversed on alternating pairs. Weights, BF16 policy, inputs, seeds, attention backend, and sampler settings remain fixed. Inputs are checked against size and SHA256 before timing. Model loading and these checks are excluded.

The A100 reference statistics are preserved:

- VGGT uses 36 requests: courtyard with 32 views, electro with 32, office with 26; each scene is repeated 12 times. Original high-resolution JPEG bytes are retained; replacing them with resized images changes the workload. First request is excluded, leaving 35 paired observations.
- TRELLIS uses the frozen 40 common-success GSO objects, seed `42 + request_order`, the 512 pipeline, and 12 steps per stage. First request is excluded, leaving 39 paired observations. This selection is a latency cohort, not a full-dataset success-rate estimate.
- Device events measure VGGT's single forward or each TRELLIS `sampler.sample_once` call, including that step's CFG work. All 36 TRELLIS calls and their stage labels are saved. Each request is reduced to its call median; cross-request statistics use those request medians.
- Synchronized wall time measures the complete runner: image read through CPU predictions for VGGT; image read through geometry GLB export for TRELLIS. VGGT returns all reconstruction heads; TRELLIS also returns texture voxels. Neither boundary includes model initialization or dataset metric computation.
- Report pipeline p50 and the median paired percentage change. At least 31 effective pairs are required for `formal_candidate`; a shorter run remains diagnostic. Failures are counted and excluded from successful timing pairs. Inspect raw records and output comparisons before accepting a result.

The tool saves environment/config/manifest identity, per-request events and wall times, tensor comparisons, and a summary. `--audit-bf16` also records matrix operand dtypes during excluded warmup. A valid audit has observed matrix calls, `passed=true`, and no violations. Output comparisons check all returned tensors, including integer coordinates/topology.

## Numerical checks and historical results

```bash
python -m pytest -q tests/test_reconstruction_ops.py tests/test_vggt_preprocessing.py
```

The tests compare neighbor lookup against CPU hash, sparse convolution against dense `conv3d`, deformable sampling against torchvision, attention layouts/ragged sample boundaries against SDPA, CFG intervals, geometry boundary handling, and preprocessing pixels/order. The new unit tests run without an accelerator: their import context isolates LightX2V's eager device bootstrap and restores package namespaces after import. Their numerical operations use the real CPU implementations. A/B comparisons are additionally produced for actual model requests.

The old experimental TRELLIS fallback indexed its kernel with coordinate offsets `-1/0/1`, so negative indices selected a different weight slice. This implementation uses kernel indices `0/1/2`, matching the official FlexGEMM offset order and dense convolution. It also fixes the decoder BF16 storage gap and BiRefNet's FP32 deformable-convolution path. These corrections change geometry relative to that experiment; old mesh metrics and old latency numbers must not be used as acceptance results for this port. Measure both variants from this checkout with the frozen manifests. The lookup optimization itself preserves outputs between the corrected BF16 variants.

Latency depends on CPU decoding, storage, runtime versions, device clocks, and contention. A100 statistics define the measurement protocol; they do not make MLU times directly interchangeable with A100 times.
