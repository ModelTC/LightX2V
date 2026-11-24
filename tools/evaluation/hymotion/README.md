# hymotion original-project / LightX2V evaluation

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

Each child workload has GNU timeout 3300 s, kill-after 20 s, and live resource checks. A campaign consists of successive bounded chunks; the parent may live longer than one hour. Keep the parent terminal/job alive. Resume with the same dataset, GPU, source and paths: completed verified input ranges are skipped. `--count` may be reduced for future workers (for example to 64 after a monitoring interruption); the original plan and every subsequent requested chunk size are retained. Existing outputs, timing records and source fingerprints are preserved. Failed, partial or resource-invalid chunks stop the campaign and retain evidence; investigate before attempting a fresh campaign. Formal samples are never silently retried or dropped. `--skip-score` stops after generation; `score` and `compare` are also available independently.

## Protocol

HumanML3D official test split: 4,384 motions, first caption per motion; 4,198 references satisfy the official 40 <= frames < 200 feature-evaluator interval. All 4,384 are generated and timed. Original 20–360-frame model clamping is retained; requested-length completeness is reported separately.

Authors’ Momask/HumanML3D evaluator: R-Precision@1/2/3, matching score, FID, diversity. Native fixed-skeleton FK, linear 30→20 FPS resampling and authors’ canonical 263-D extraction. Paired central crop to min(prediction, reference, 196), multiple of 4; original evaluator normalization; retrieval batch size 32 including last batch; diversity seed 42, 300 pairs. Length exclusions are explicit, never removed from generation coverage.

End-to-end latency begins before request decoding and input file read, and ends after inference, CPU conversion, final NPZ serialization, fsync and GPU synchronization. Model loading, RNG reset, harness directory creation, SHA hashing, output validation, logs and quality scoring are outside this interval. Loading is recorded separately. Each loaded process uses one separate synthetic/upstream-demo warmup, never a dataset sample. Report mean, median, P95 and every sample’s latency. No five-repeat averaging. Final artifact durability measures the filesystem’s fsync contract, not independent storage replication. No auxiliary preview is required; old timings with preview/export or different boundaries are not interchangeable.

The comparison requires complete one-to-one coverage, verified resource isolation/cleanup, exact final PNG/GLB bytes or exact NPZ array keys/dtypes/shapes/elements, finite quality metrics matching within 1e-5 absolute roundoff, and default latency ratios mean≤1.25/P95≤1.5. These configurable latency gates are engineering criteria, not official dataset thresholds. A failure remains a failure and blocks an alignment claim. Discrete GPU sampling cannot establish absence of sub-sample interference or CPU/network contention.

## Evidence and platforms

`run.json`, per-sample JSONL, immutable final outputs and hashes, commands, environments, original-source pins, resource samples and PID/start-tick/task cleanup records are retained outside git. `quality.json`, `latency.json` and `comparison.json` are generated from checked complete coverage. Large artifacts, datasets, weights, upstream repositories and local absolute paths never enter this PR. Sanitized full-run summaries may be published under `results/nvidia/` only after actual validation. CPU tests: `python -m unittest discover -s "$EVAL/tests"`.

NVIDIA A100 wiring is implemented. MLU, MetaX and Ascend NPU are explicit placeholders whose launchers exit 2; see PLATFORMS.md. No target-device results are inferred from A100.

Resource completion uses an explicit worker/controller acknowledgement: after all work and outputs complete, the worker retains its PID until capture obtains a post-workload sample. Process/CUDA teardown then occurs with separate cleanup verification. This prevents disappearing /proc lineage at interpreter exit from being mistaken for a competing process; unknown lineage within the measured workload still fails. The acknowledgement wait is outside per-input end-to-end latency.

NVIDIA capture uses a 0.25 s polling interval and a bounded 2.5 s query budget; the maximum accepted completed-sample gap remains 3 s. Query failures, missing coverage and unknown process ancestry invalidate that attempt; partial timing records are retained separately.
