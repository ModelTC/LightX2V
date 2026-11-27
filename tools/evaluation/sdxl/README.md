# sdxl original-project / LightX2V evaluation

The MLU590 entry runs each full SDXL fixture through inference PR #1754 on one physical card. The supported fixtures are GenEval (553 prompts), Parti (1,632 prompts), and COCO (5,000 prompts); each campaign reports the PR's quality metric and per-request latency summary.

```bash
cp tools/evaluation/sdxl/configs/mlu/paths.json tools/evaluation/sdxl/configs/mlu/paths.local.json
# Replace the path placeholders and set the assigned host card number.
MLU_PHYSICAL_CARD=0 PYTHON_BIN=/path/to/torch_mlu/python bash tools/evaluation/sdxl/scripts/platform/mlu/run_eval.sh \
  --paths tools/evaluation/sdxl/configs/mlu/paths.local.json \
  --dataset geneval --output /absolute/results/sdxl-geneval-mlu --count 64
```

This developer-only package calls the pinned original implementation and inference PR from `configs/nvidia/source-lock.json`. It does not copy model networks or modify the user-facing inference API. Each implementation runs the full frozen dataset once at batch size one.

## Reproduce

Use Python 3.10 on an otherwise idle NVIDIA A100 40 GB. Native and LightX2V can have separate Python environments; do not mix their torch/transformers installations. Keep outputs on a filesystem with sufficient space. CPU-only source preparation and input checking use the standard library.

```bash
EVAL=tools/evaluation/sdxl
python3 "$EVAL/evaluate.py" prepare-sources --workspace /absolute/eval-sources
python3 "$EVAL/prepare_assets.py" --source-root /absolute/eval-sources/geneval
# Provision weights/data/evaluator assets according to DATA.md, install the
# native upstream dependencies plus runtime-requirements-native.txt.
# For LightX2V use the pinned inference source's scripts/sdxl/install_a100.sh.
# Copy configs/nvidia/paths.json to paths.local.json and fill all absolute roots.
python3 "$EVAL/evaluate.py" check-inputs --paths paths.local.json
bash "$EVAL/scripts/platform/nvidia/run_eval.sh" campaign \
  --paths paths.local.json --dataset geneval \
  --gpu GPU-REPLACE-WITH-FULL-UUID --output /absolute/results/sdxl --count 256
```

Run each model–dataset campaign on exactly one fixed GPU. Different campaigns may run on different idle GPUs. Native and LightX2V chunks alternate serially on that same device; no cross-card sharding. Resource capture/check reuse the pinned LightX2V `tools/runtime/gpu_watch.py`; the host must expose real GPU PIDs and their /proc ancestry. MPS, MIG and proxy attribution are unsupported. Execute the campaign from the GPU host, not an isolated PID namespace.

Each child workload has GNU timeout 3300 s, kill-after 20 s, and live resource checks. A campaign consists of successive bounded chunks; the parent may live longer than one hour. Keep the parent terminal/job alive. Resume with the same dataset, GPU, source and paths: completed verified input ranges are skipped. `--count` may be reduced for future workers (for example to 64 after a monitoring interruption); the original plan and every subsequent requested chunk size are retained. Existing outputs, timing records and source fingerprints are preserved. Failed, partial or resource-invalid chunks stop the campaign and retain evidence; investigate before attempting a fresh campaign. Formal samples are never silently retried or dropped. `--skip-score` stops after generation; `score` and `compare` are also available independently.

## Protocol

GenEval: all 553 prompts. PartiPrompts: all 1,632 prompts. COCO val2014: the locked project fixture of 5,000 sorted image IDs and first caption by caption ID. One 1024×1024 image per prompt, Euler 30 steps, CFG 5, seed 42.

Both generation backends set cuDNN `benchmark=False` and `deterministic=True`. The numerical environment is pinned in `configs/nvidia/runtime.json`: `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1`, and `CUBLAS_WORKSPACE_CONFIG` is removed before CUDA initialization so both implementations use the upstream workspace default. Inherited workspace overrides are removed too. Each worker records its actual environment and cuDNN flags; completed results with a different or missing environment profile are rejected. PyTorch's global deterministic-algorithms flag is reported separately. Changing the numerical profile requires a new campaign; timings from another profile are not reused.

GenEval: six published dimension accuracies and their unweighted mean, using the official Mask2Former/CLIP evaluator. Parti and COCO: OpenAI CLIP ViT-L/14 FP32 normalized cosine ×100 and 2.5×max(cosine,0). Native and adapted images are scored with the same metric runtime and weights. This is not FID.

End-to-end latency begins before request decoding and input file read, and ends after inference, CPU conversion, final PNG serialization, fsync and GPU synchronization. Model loading, RNG reset, harness directory creation, SHA hashing, output validation, logs and quality scoring are outside this interval. Loading is recorded separately. Each loaded process uses one separate synthetic/upstream-demo warmup, never a dataset sample. Report mean, median, P95 and every sample’s latency. No five-repeat averaging. Final artifact durability measures the filesystem’s fsync contract, not independent storage replication. No auxiliary preview is required; old timings with preview/export or different boundaries are not interchangeable.

The comparison requires complete one-to-one coverage, verified resource isolation/cleanup, exact final PNG bytes, finite quality metrics matching within 1e-5 absolute roundoff, and default latency ratios mean≤1.25/P95≤1.5. These configurable latency gates are engineering criteria, not official dataset thresholds. A failure remains a failure and blocks an alignment claim. Discrete GPU sampling cannot establish absence of sub-sample interference or CPU/network contention.

## Evidence and platforms

`run.json`, per-sample JSONL, immutable final outputs and hashes, commands, environments, original-source pins, resource samples and PID/start-tick/task cleanup records are retained outside git. `quality.json`, `latency.json` and `comparison.json` are generated from checked complete coverage. Large artifacts, datasets, weights, upstream repositories and local absolute paths never enter this PR. Sanitized full-run summaries may be published under `results/nvidia/` only after actual validation. CPU tests: `python -m unittest discover -s "$EVAL/tests"`.

NVIDIA A100 wiring is implemented. MetaX C500 is documented below; see PLATFORMS.md for the other platform entries. No target-device results are inferred from A100.

Resource completion uses an explicit worker/controller acknowledgement: after all work and outputs complete, the worker retains its PID until capture obtains a post-workload sample. Process/CUDA teardown then occurs with separate cleanup verification. This prevents disappearing /proc lineage at interpreter exit from being mistaken for a competing process; unknown lineage within the measured workload still fails. The acknowledgement wait is outside per-input end-to-end latency.

NVIDIA capture uses a 0.25 s polling interval. The default query budget is 2.5 s and the default maximum accepted completed-sample gap is 3 s. Explicit local overrides use `LIGHTX2V_EVAL_RESOURCE_QUERY_TIMEOUT` and `LIGHTX2V_EVAL_RESOURCE_MAX_GAP`; the gap must cover the query budget plus the polling interval. Each capture records the actual settings. Query failures, missing coverage and unknown process ancestry invalidate that attempt; partial timing records are retained separately. Changing a later capture's settings does not validate an earlier failed attempt.

NVIDIA telemetry, worker logs and completion acknowledgements use local `/tmp` during execution (`LIGHTX2V_EVAL_RESOURCE_SCRATCH` can select another fast local directory). Evidence is archived to the configured result directory before a completed chunk becomes resumable. Failed archives retain local files and record their location in `resources/*/staging.json`. Model artifacts and their fsync are timed on the configured output filesystem.

For bounded minimal NVML acquisition, install `nvidia-ml-py==12.550.52` in a separate Python environment, export `LIGHTX2V_EVAL_RESOURCE_BACKEND=nvml` and `LIGHTX2V_EVAL_RESOURCE_PYTHON=/absolute/monitor-env/bin/python`. The shared `tools/runtime/nvml_capture.py` calls the pinned inference source’s existing `gpu_watch` capture/check and ancestry policy. It reads UUID, memory, utilization, compute/graphics and MPS inventories; missing APIs, MIG or MPS fail closed. It uses the query-budget and maximum-gap settings described above. Include `tools/runtime/nvml_capture.py` and its pinned requirements when copying this evaluation package. The default XML backend needs no extra binding.

Generation workers initialize their single visible CUDA primary context before CPU model preparation. This reduces the window in which another automatic device selector can mistake the loading worker for an idle card. Resource attribution and early/late samples still apply; model preparation remains outside request timing.

GenEval scoring runs under `geneval_python` with its original literal evaluator launcher in the held worker PID. This preserves the pinned evaluator and arguments while keeping CUDA process ancestry available until the post-workload sample. COCO/Parti scoring continues under `metric_python`.

## BF16 inference profile

Set `LIGHTX2V_EVAL_PRECISION_PROFILE=bf16` and pass `campaign --backend lightx2v`
to evaluate the updated inference PR with `configs/nvidia/bf16/` source/model/runtime
locks. Use a new result directory and a local paths file without `worker_script`.
`prepare-sources`, `check-inputs`, `run` and `score` select the same profile via
that environment variable. The original default profile and its source locks remain
available for reproducing the earlier native-precision comparison.

The BF16 campaign generates each complete frozen dataset once and produces
`summary.json`, per-input output/timing records, quality metrics and resource
evidence. It runs no native-precision generation and makes no exact-output parity
or cross-precision latency-ratio claim. Quality metric networks and dataset
selection are unchanged; retained FP32 inference operations are recorded in
`precision_policy`. `--backend` also selects a single backend in the default
profile; omitting it preserves the original serial two-backend comparison.

## MetaX C500

The MetaX entry evaluates the pinned BF16 LightX2V adapter on GenEval (553), Parti (1,632) and the locked COCO fixture (5,000). It reuses the frozen manifests, model requests, quality functions and request timing boundary. Each dataset stays on one physical card.

Run the controller on a host with Docker, `mx-smi` and a compatible C500 driver. Provision an inference image with the matching model PR's MetaX installer and install `configs/metax/requirements.txt` in its metric environment. The controller uses `--image` or the image in `configs/metax/runtime.json`. Copy `configs/metax/paths.json` to a local paths file and replace its absolute paths. Every container mount listed in `mounts` is read-only, with `output_root` mounted writable. Preserve the same absolute paths inside the containers.

```bash
PYTHON_BIN=python3 bash tools/evaluation/sdxl/scripts/platform/metax/run_eval.sh \
  --paths /absolute/paths.local.json --dataset geneval \
  --card 0 --output /absolute/results/new-run
# Add --smoke in a separate directory to verify generation and actual scoring.
```

The resource observer runs as a non-root user in the host PID namespace, with a read-only filesystem and all capabilities dropped. It receives only its output directory, telemetry executable, driver library and read-only observation source. The host controller launches dedicated workers, collects before/after samples, acknowledges completion while the worker PID is alive, and verifies cleanup. Resource or coverage failures stop the campaign and preserve evidence. Resume accepts complete verified chunks with identical source, image, configuration, paths and card identity.

The smoke plan uses one geometry/CLIPScore input, six GenEval prompts covering its six dimensions, or enough HumanML3D references for the unchanged 300-pair diversity metric. Smoke summaries explicitly retain the formal fixture count. Full summaries contain every frozen input, quality metrics, mean/median/P95 end-to-end latency, source/image fingerprints and resource evidence.

GenEval uses the pinned official evaluator and weights with MMCV 1.7.2 CPU extensions. Its released `multi_scale_deformable_attn_pytorch` executes on MetaX tensors. The metric Dockerfile builds the released MMCV source against the installed MetaX PyTorch; an external MMCV source checkout is not required. A separate `metric_image` provides the FP32 Transformers CLIP evaluator through PyTorch SDPA. Both scoring backends retain their original metric definitions.

Build the metric environment with `docker build --build-arg INFERENCE_IMAGE=<model-metax-image> -f tools/evaluation/sdxl/scripts/platform/metax/Dockerfile.metrics -t lightx2v-metax-metrics:20261127 .` and set `metric_image` in the local paths file. The image fixes the SDK OpenMP runtime for metric dependencies. GenEval additionally builds MMCV CPU extensions against that same MetaX PyTorch.

<!-- metax-measured-results:start -->
Complete C500 BF16 measurements:

| Dataset | Inputs | Quality metric | Mean E2E (s) | P95 E2E (s) |
| --- | ---: | --- | ---: | ---: |
| GenEval | 553 | Overall 0.549463 | 5.688347 | 5.747663 |
| PartiPrompts | 1,632 | CLIP cosine ×100 27.727707 | 5.629281 | 5.742524 |
| COCO | 5,000 | CLIP cosine ×100 26.593067 | 5.722419 | 5.841624 |

[Full results](results/metax/bf16/full-results.json) retain the exact measured revisions, per-input timing/output hashes and closed resource-capture audits for all three complete fixtures. Each dataset stayed on one fixed physical C500; every formal input ran once.

COCO retains one invalid worker-initialization attempt (exit 135) separately. It started no formal inputs in that chunk. After verified cleanup, the same source, image and card resumed with the first 256 valid inputs preserved. No failed formal inputs were omitted from the denominator.
<!-- metax-measured-results:end -->

## Ascend NPU

The Ascend entry preserves the inference PR's BF16/FP32 policy and the frozen
dataset, quality definitions and end-to-end boundary above. Provision Python 3.10,
a matching torch/torch_npu/CANN installation, and the inference PR's
`scripts/sdxl/install_ascend.sh`. Install metric dependencies with
`scripts/platform/ascend_npu/install_metrics.sh` in the metric environment.
Source CANN's `set_env.sh` before running the host controller.

```bash
EVAL=tools/evaluation/sdxl
PLATFORM=ascend_npu python "$EVAL/evaluate.py" prepare-sources --workspace /absolute/sources
# Fill configs/ascend_npu/paths.json in a separate paths.local.json.
PYTHON_BIN=/absolute/inference-env/bin/python bash "$EVAL/scripts/platform/ascend_npu/run_eval.sh" \
  campaign-ascend --paths /absolute/paths.local.json --dataset geneval \
  --backend lightx2v --card 0 --count 128 --output /absolute/results/new-campaign
# Add --smoke, using a separate output directory, for actual quality/latency validation.
```

Include the four shared `tools/runtime/ascend_*.py` files with this evaluation
directory. Source preparation reconstructs one pinned main checkout with the
three slim inference overlays; it checks every runtime file hash. Inference and
metric Python paths may differ. Each complete dataset stays on one physical NPU;
different datasets can run concurrently on distinct idle cards. The host must
expose `npu-smi` and real process ancestry. Virtual partitions, device sharing,
foreign processes, query failures and missing samples stop the campaign.
The 3-second query budget, 5-second maximum sample gap and 3,600 MiB idle-memory
ceiling are explicit Ascend 910B settings, including the firmware reservation.
All workers have a 3,300-second timeout plus verified cleanup. The host controller
may live longer, retaining every bounded chunk and sample.

Outputs include full coverage, quality, mean/median/P95 request latency, source
and configuration fingerprints, physical PCI identity and closed resource
evidence. Smoke reports remain labeled as smoke. Ascend runs the BF16 LightX2V
backend; it does not assert native equality or A100 latency ratios. Publish
sanitized completed measurements under `results/ascend_npu/` only after the full
dataset has finished. Original NVIDIA, MLU and MetaX entries remain available.

Run separate campaigns for `geneval`, `parti` and `coco`. GenEval retains the pinned official FP32 Mask2Former/CLIP evaluator, its six dimensions and weights. The scoped adapter changes only the evaluator device selection and availability assertion to NPU. MMCV 1.7.2 uses its released equivalent PyTorch grid-sample attention implementation; its original CPU extension is built from the SHA-checked source archive. Parti/COCO retain the original FP32 CLIP scoring.


### MLU host resource evidence

Run `scripts/platform/mlu/run_observed.sh` on the host with Docker and cnmon.
Pass `--dataset`, `--paths /absolute/host-paths.json`, `--output`, `--card`, and
optionally `--image`. `--smoke` selects a declared partial fixture; it never
claims full dataset completion. `--smoke --skip-score` checks one generation
request only. Full runs retain the frozen dataset, request and scoring code.

The paths file uses host absolute paths visible unchanged in the worker. Add
`output_root` and `mounts`, a map of absolute host directories to `ro`/`rw`;
mount source/data read-only and the chosen output root writable. `python` names
the interpreter inside the image. Build the MLU inference environment described
in the inference PR; SF3D requires its full CPU UV/texture extensions.

The host cnmon observer records card UUID/PCI identity, host PID/start-tick
ancestry and bounded queries. A held worker allows a post-workload observation;
the controller checks the closed capture and task cleanup before accepting an
interval. Missing/unknown inventory, partitions, foreign PIDs and observation
gaps reject resource evidence. The local legacy MLU entry remains a measurement
without this resource isolation claim. Native comparison is still a separately
tracked capability; no BF16-versus-native exact equality is inferred.


### Independent upstream native comparison on MLU

The observed host entry requires an explicit `--backend lightx2v` or
`--backend native`. Use a separate output directory for each backend and the
same physical card for a model/dataset comparison. `paths.native_image` selects
a separate native software environment; keep the A100 native model's
Transformers version (SF3D 4.42.3, HY-Motion 4.53.3, SDXL 5.17.0) and install
`runtime-requirements-native-mlu.txt` without replacing torch/torchvision or
torch_mlu. The native model keeps its original precision, while the PR keeps
its BF16/FP32 precision boundaries. They are separate precision experiments.

`PLATFORM=cambricon_mlu ... prepare-sources --workspace ...` reconstructs the
pinned sources. SF3D's explicit hashed source overlay adds MLU device selection,
the original FP32 geometry/material scope and CPU UV transport. CPU UV and
baking retain upstream algorithms. Use `prepare_runtime.py` to provision the
same offline constructor paths as the A100 native adapter. HY-Motion prepares
its original graph on CPU and then moves it to MLU without dtype conversion.
Its hashed device-helper overlay recognizes MLU instead of misclassifying it as CPU.

For completed full campaigns, `evaluate.py compare-mlu --dataset ...
--native-run .../native --lightx2v-run .../lightx2v --output report.json` reports
both qualities and the mean end-to-end latency ratio. It requires full coverage
and valid host resource/cleanup evidence, and does not claim exact equality
between native precision and PR BF16. Smoke results remain explicitly partial.


GenEval's MLU source lock reconstructs the previously validated device-placement
shim from the pinned upstream evaluator. Its exact runtime hash is preserved;
matching/counting rules are unchanged. Do not replace this overlay with the
CUDA evaluator file hash when importing NVIDIA package/evaluator locks.


For GenEval use the dedicated scoring image/paths template
`configs/mlu/paths-geneval.json` and provision the locked evaluator dependencies.
`metric_image` selects this environment without changing the generation image.
Build `scripts/platform/mlu/Dockerfile.geneval` with the existing MLU scoring
base as `BASE_IMAGE`. It pins Pillow 12.3.0, matching A100: Pillow 13 rejects the
upstream NumPy crop box through an ambiguous-truth-value API change. Keep the
six-dimension check; a one-image smoke does not satisfy the GenEval aggregate
protocol. Frozen existing images can be rescored separately with their origin
recorded, without claiming new generation or full-dataset latency evidence.

<!-- ascend-full-results-20261127 -->

Complete Ascend 910B2C BF16 measurement (original FP32 scopes retained):

- geneval: 553 inputs, mean end-to-end 4.641460 s, P95 4.749956 s. GenEval 0.534247, including all six dimensions.
- parti: 1,632 inputs, mean end-to-end 4.594178 s, P95 4.733826 s. CLIP cosine ×100 27.815064, CLIPScore 0.695377.
- coco: 5,000 inputs, mean end-to-end 4.648103 s, P95 4.741775 s. CLIP cosine ×100 26.619274, CLIPScore 0.665482.

[Full results, measured revisions and evidence fingerprints](results/ascend_npu/bf16/full-results.json) include every formal input’s latency and output SHA. Independent verification checks complete coverage, output hashes, original metric aggregation, NPU process attribution and cleanup. Raw artifacts and device observations remain outside git. No native equality or A100 latency ratio is claimed.

<!-- /ascend-full-results-20261127 -->
