# Configuration ownership

common/suite.json defines immutable coverage, fixture SHA256 and timing boundaries. The sibling JSONL manifests fix exact input selection. common/weight-lock.json fixes model revision and every required weight/evaluator/data asset hash.

nvidia/model.json is the inference PR’s supported configuration. nvidia/source-lock.json pins external original/LightX2V/evaluator code. nvidia/runtime.json documents one-device admission, limits and alignment gates. nvidia/paths.json is a portable template, never a ready-to-run machine configuration: copy it to an ignored *.local.json and replace placeholders with absolute paths. native_python/lightx2v_python/metric_python select independent environments. The native and adapted runs share inputs, model checkpoint, seed and final artifact format.

For GenEval, geneval_python selects the legacy detector environment; geneval_weights holds the detector .pth; geneval_configs is the mmdetection config tree; geneval_config is its complete mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py filename. geneval_clip is the cache directory holding the locked OpenAI ViT-L-14.pt checkpoint. metric_python for CLIPScore must have the necessary torch/transformers packages and the metric assets. Local configuration files are never imported as Python.

`mlu/` contains a functional-smoke config and local-path template for one inference-PR request. It is not a full evaluation environment or metric campaign. The MLU files do not validate the native Diffusers runtime. MetaX and Ascend remain unimplemented.

`nvidia/runtime.json` also pins `generation_environment` for both implementations. String values are exported before worker/CUDA startup; `null` removes a variable, including any inherited value. The explicit campaign GPU overrides a parent device selection. In particular, SDXL uses the upstream default cuBLAS workspace and eight OMP/MKL threads. Actual values are retained in worker provenance and checked before generation and before accepting completed results. A numerical environment change requires a new campaign and fresh timings.

For recovery of an interrupted campaign after a controller-only fix, an ignored local paths file may optionally set `worker_script` to the archived measured evaluate.py. The controller requires identical non-controller Python AST and identical JSON/JSONL configurations, retains that worker’s original SHA in measurement records and campaign identity, and records the new resource-bootstrap SHA separately. Changed model, timing, input or metric functions are rejected. This option does not rewrite or rebrand previous measurements. Fresh campaigns use the submitted evaluate.py itself.

For updated BF16 inference, set `LIGHTX2V_EVAL_PRECISION_PROFILE=bf16`; the model, source and runtime locks are selected from `nvidia/bf16/`. Run `campaign --backend lightx2v` in a fresh output directory. The original default locks remain unchanged for earlier native-precision reproduction.

MetaX uses its own `configs/metax/` model, runtime, source lock and paths. The source lock pins the matching inference PR commit. `mounts` and `output_root` configure host/container path visibility. These deployment paths belong in a local paths file.

`ascend_npu/` owns the NPU-only source/model/runtime/dependency profile. It preserves the original precision policy and common dataset/weight locks. Machine paths stay in untracked local JSON files.
