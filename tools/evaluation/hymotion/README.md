# hymotion original-project / LightX2V evaluation

The MLU590 entry runs all 4,384 frozen HumanML3D test motions through inference PR #1752 on one physical card, then computes the official feature metrics and per-request latency summary. The 4,198 evaluator-eligible references remain explicitly identified by the scoring protocol.

```bash
cp tools/evaluation/hymotion/configs/mlu/paths.json tools/evaluation/hymotion/configs/mlu/paths.local.json
# Replace the path placeholders and set the assigned host card number.
MLU_PHYSICAL_CARD=0 PYTHON_BIN=/path/to/torch_mlu/python bash tools/evaluation/hymotion/scripts/platform/mlu/run_eval.sh \
  --paths tools/evaluation/hymotion/configs/mlu/paths.local.json \
  --output /absolute/results/hymotion-mlu --count 64
```

This developer-only package calls the pinned original implementation and inference PR from `configs/nvidia/source-lock.json`. It does not copy model networks or modify the user-facing inference API. Each implementation runs the full frozen dataset once at batch size one.

## Reproduce

Use Python 3.10 on an otherwise idle NVIDIA A100 40 GB. Native and LightX2V can have separate Python environments; do not mix their torch/transformers installations. Keep outputs on a filesystem with sufficient space. CPU-only source preparation and input checking use the standard library.

```bash
EVAL=tools/evaluation/hymotion
python3 "$EVAL/evaluate.py" prepare-sources --workspace /absolute/eval-sources
# Provision weights/data/evaluator assets according to DATA.md, install the
# native upstream dependencies plus runtime-requirements-native.txt.
# For LightX2V use the pinned inference source's scripts/hymotion/install_a100.sh.
# Copy configs/nvidia/paths.json to paths.local.json and fill all absolute roots.
python3 "$EVAL/evaluate.py" check-inputs --paths paths.local.json
bash "$EVAL/scripts/platform/nvidia/run_eval.sh" campaign \
  --paths paths.local.json --dataset humanml3d \
  --gpu GPU-REPLACE-WITH-FULL-UUID --output /absolute/results/hymotion --count 256
```

Run each model–dataset campaign on exactly one fixed GPU. Different campaigns may run on different idle GPUs. Native and LightX2V chunks alternate serially on that same device; no cross-card sharding. Resource capture/check reuse the pinned LightX2V `tools/runtime/gpu_watch.py`; the host must expose real GPU PIDs and their /proc ancestry. MPS, MIG and proxy attribution are unsupported. Execute the campaign from the GPU host, not an isolated PID namespace.

Each child workload has GNU timeout 3300 s, kill-after 20 s, and live resource checks. A campaign consists of successive bounded chunks; the parent may live longer than one hour. Keep the parent terminal/job alive. Resume with the same dataset, GPU, source and paths: completed verified input ranges are skipped. `--count` may change within [1,256] for future workers; it bounds process lifetime while per-request batch size remains one. The original plan and every subsequent requested chunk size are retained. Existing outputs, timing records and source fingerprints are preserved. Failed, partial or resource-invalid chunks stop the campaign and retain evidence; investigate before attempting a fresh campaign. Formal samples are never silently retried or dropped. `--skip-score` stops after generation; `score` and `compare` are also available independently.

## Protocol

HumanML3D official test split: 4,384 motions, first caption per motion; 4,198 references satisfy the official 40 <= frames < 200 feature-evaluator interval. All 4,384 are generated and timed. Original 20–360-frame model clamping is retained; requested-length completeness is reported separately.

Authors’ Momask/HumanML3D evaluator: R-Precision@1/2/3, matching score, FID, diversity. Native fixed-skeleton FK, linear 30→20 FPS resampling and authors’ canonical 263-D extraction. Paired central crop to min(prediction, reference, 196), multiple of 4; original evaluator normalization; retrieval batch size 32 including last batch; diversity seed 42, 300 pairs. Length exclusions are explicit, never removed from generation coverage.

End-to-end latency begins before request decoding and input file read, and ends after inference, CPU conversion, final NPZ serialization, fsync and GPU synchronization. Model loading, RNG reset, harness directory creation, SHA hashing, output validation, logs and quality scoring are outside this interval. Loading is recorded separately. Each loaded process uses one separate synthetic/upstream-demo warmup, never a dataset sample. Report mean, median, P95 and every sample’s latency. No five-repeat averaging. Final artifact durability measures the filesystem’s fsync contract, not independent storage replication. No auxiliary preview is required; old timings with preview/export or different boundaries are not interchangeable.

The comparison requires complete one-to-one coverage, verified resource isolation/cleanup, exact final PNG/GLB bytes or exact NPZ array keys/dtypes/shapes/elements, finite quality metrics matching within 1e-5 absolute roundoff, and default latency ratios mean≤1.25/P95≤1.5. These configurable latency gates are engineering criteria, not official dataset thresholds. A failure remains a failure and blocks an alignment claim. Discrete GPU sampling cannot establish absence of sub-sample interference or CPU/network contention.

## Evidence and platforms

`run.json`, per-sample JSONL, immutable final outputs and hashes, commands, environments, original-source pins, resource samples and PID/start-tick/task cleanup records are retained outside git. `quality.json`, `latency.json` and `comparison.json` are generated from checked complete coverage. Large artifacts, datasets, weights, upstream repositories and local absolute paths never enter this PR. Sanitized full-run summaries may be published under `results/nvidia/` only after actual validation. CPU tests: `python -m unittest discover -s "$EVAL/tests"`.

NVIDIA A100 wiring is implemented. MetaX C500 is documented below; see PLATFORMS.md for the other platform entries. No target-device results are inferred from A100.

Resource completion uses an explicit worker/controller acknowledgement: after all work and outputs complete, the worker retains its PID until capture obtains a post-workload sample. Process/CUDA teardown then occurs with separate cleanup verification. This prevents disappearing /proc lineage at interpreter exit from being mistaken for a competing process; unknown lineage within the measured workload still fails. The acknowledgement wait is outside per-input end-to-end latency.

NVIDIA capture defaults to a 0.25 s polling interval, a bounded 2.5 s query budget and a 3 s maximum completed-sample gap. `LIGHTX2V_EVAL_RESOURCE_QUERY_TIMEOUT` and `LIGHTX2V_EVAL_RESOURCE_MAX_GAP` select explicit per-run limits. Every archived capture retains its actual limits and sampler fingerprint; published full-run summaries group those profiles under `resources.*.capture_sources`. An earlier failed attempt is never accepted retroactively under different limits. Query failures within the checked coverage window, missing coverage and unknown process ancestry invalidate that attempt; partial timing records are retained separately.

NVIDIA telemetry, worker logs and completion acknowledgements use local `/tmp` during execution (`LIGHTX2V_EVAL_RESOURCE_SCRATCH` can select another fast local directory). Evidence is archived to the configured result directory before a completed chunk becomes resumable. Failed archives retain local files and record their location in `resources/*/staging.json`. Model artifacts and their fsync are timed on the configured output filesystem.

For bounded minimal NVML acquisition, install `nvidia-ml-py==12.550.52` in a separate Python environment, export `LIGHTX2V_EVAL_RESOURCE_BACKEND=nvml` and `LIGHTX2V_EVAL_RESOURCE_PYTHON=/absolute/monitor-env/bin/python`. The shared `tools/runtime/nvml_capture.py` calls the pinned inference source's existing `gpu_watch` capture/check and ancestry policy. It reads UUID, memory, utilization, compute/graphics and MPS inventories; missing APIs, MIG or MPS fail closed. Query and gap limits use the explicit settings above, including the 2.5 s / 3 s defaults. Include that shared file when copying this evaluation package. The default XML backend needs no extra binding.

Generation workers initialize their single visible CUDA primary context before CPU model preparation. This reduces the window in which another automatic device selector can mistake the loading worker for an idle card. Resource attribution and early/late samples still apply; model preparation remains outside request timing.

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

HY-Motion generation workers retain one byte in the selected CUDA primary context before CPU checkpoint loading. This makes the loading task visible to NVML; it changes no RNG state and remains outside request timing. The original exclusive-process/resource checks still reject external GPU processes.

## MetaX C500

The MetaX entry evaluates the pinned BF16 LightX2V adapter on HumanML3D (4,384 generated inputs and 4,198 evaluator-eligible references). It reuses the frozen manifests, model requests, quality functions and request timing boundary. Each dataset stays on one physical card.

Run the controller on a host with Docker, `mx-smi` and a compatible C500 driver. Provision an inference image with the matching model PR's MetaX installer and install `configs/metax/requirements.txt` in its metric environment. The controller uses `--image` or the image in `configs/metax/runtime.json`. Copy `configs/metax/paths.json` to a local paths file and replace its absolute paths. Every container mount listed in `mounts` is read-only, with `output_root` mounted writable. Preserve the same absolute paths inside the containers.

```bash
PYTHON_BIN=python3 bash tools/evaluation/hymotion/scripts/platform/metax/run_eval.sh \
  --paths /absolute/paths.local.json --dataset humanml3d \
  --card 0 --output /absolute/results/new-run
# Add --smoke in a separate directory to verify generation and actual scoring.
```

The resource observer runs as a non-root user in the host PID namespace, with a read-only filesystem and all capabilities dropped. It receives only its output directory, telemetry executable, driver library and read-only observation source. The host controller launches dedicated workers, collects before/after samples, acknowledges completion while the worker PID is alive, and verifies cleanup. Resource or coverage failures stop the campaign and preserve evidence. Resume accepts complete verified chunks with identical source, image, configuration, paths and card identity.

The smoke plan uses one geometry/CLIPScore input, six GenEval prompts covering its six dimensions, or enough HumanML3D references for the unchanged 300-pair diversity metric. Smoke summaries explicitly retain the formal fixture count. Full summaries contain every frozen input, quality metrics, mean/median/P95 end-to-end latency, source/image fingerprints and resource evidence.

Build the metric environment with `docker build --build-arg INFERENCE_IMAGE=<model-metax-image> -f tools/evaluation/hymotion/scripts/platform/metax/Dockerfile.metrics -t lightx2v-metax-metrics:20261127 .` and set `metric_image` in the local paths file. The metric image fixes the SDK OpenMP runtime for its dependencies.

## Ascend NPU

The Ascend entry preserves the inference PR's BF16/FP32 policy and the frozen
dataset, quality definitions and end-to-end boundary above. Provision Python 3.10,
a matching torch/torch_npu/CANN installation, and the inference PR's
`scripts/hymotion/install_ascend.sh`. Install metric dependencies with
`scripts/platform/ascend_npu/install_metrics.sh` in the metric environment.
Source CANN's `set_env.sh` before running the host controller.

```bash
EVAL=tools/evaluation/hymotion
PLATFORM=ascend_npu python "$EVAL/evaluate.py" prepare-sources --workspace /absolute/sources
# Fill configs/ascend_npu/paths.json in a separate paths.local.json.
PYTHON_BIN=/absolute/inference-env/bin/python bash "$EVAL/scripts/platform/ascend_npu/run_eval.sh" \
  campaign-ascend --paths /absolute/paths.local.json --dataset humanml3d \
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

The original learned HumanML3D evaluator runs in FP32 on NPU, including its variable-length bidirectional GRUs. Preserve the `t2m/text_mot_match/model/finest.tar` layout under `metric_checkpoints`; it must resolve to the same hashed checkpoint under `metric_assets`. All 4,384 requests remain in latency coverage; the original 4,198-reference quality eligibility rule is unchanged.

Complete C500 BF16 HumanML3D measurement: all 4,384 inputs generated and timed; 4,198 eligible references scored with the original 186 explicit reference-length exclusions. R-Precision@1 is 0.536922, FID 0.491118, mean end-to-end 2.841573 s and P95 3.103997 s. [Full results and exact measured revisions](results/metax/bf16/full-results.json) include per-input timing/output hashes and closed-capture audit fingerprints.
