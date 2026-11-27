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
