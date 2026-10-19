---
name: support-new-model
description: "Add native LightX2V support for a new model or task from an upstream inference implementation. Map runner/model/weight/infer/scheduler components, add configs and launchers, implement required offload or parallel paths, and verify output parity."
---

# LightX2V Native Model Porting

Goal: build a **native LightX2V runner** whose output matches the upstream model inference. Do not wrap the upstream repo at runtime unless the user explicitly asks for a temporary bridge.

Hard rule: do **not** depend on diffusers/transformers/third-party model or pipeline classes to execute the core model. Rebuild the model structure with LightX2V's data structures and ops (`weights`, `infer`, `common/ops/mm`, norm, attention, conv, scheduler, KV cache), or reuse an existing LightX2V implementation that already does so.

## Core Principle

Always map the upstream pipeline onto LightX2V's existing architecture:

```text
CLI/config/script
  -> build_cli_inputs / build_startup_config
  -> runner_factory.build_runner (init_modules, optional warmup)
  -> prepare_request -> InputInfo -> run_request
  -> runner.run_pipeline
  -> input encoder(s)
  -> scheduler.step_pre()
  -> model.infer()
       -> pre_infer
       -> transformer_infer
       -> post_infer
       -> weights/common ops
  -> scheduler.step_post()
  -> VAE decode/postprocess/save
```

Prefer reusing existing LightX2V families (`wan`, `qwen_image`, `hunyuan_video`, `ltx2`, etc.) over creating a parallel implementation. New code should look like a small specialization of an existing family, not a copied upstream pipeline.

For a new DiT/video model, the desired ownership chain is strict:

```text
new runner
  -> creates/owns scheduler(s)
  -> loads new model
new model
  -> owns new/reused weights
  -> owns new/reused infer modules
weights
  -> describe x2v tensor layout using common ops
infer
  -> executes DiT math with those weights
```

Read model paths from startup config in the loader; do not hard-code them in a runner. Do not put checkpoint mapping in infer. Do not call the upstream repo from runner/model to get a result. The result should come from LightX2V-native modules.

Treat base inference as the first milestone, not the finish line. When a model family supports block/model offload, sequence parallel, or CFG parallel, the new model should support the corresponding LightX2V modes too unless there is a specific architectural blocker documented in the final report. Verify inherited support against custom blocks and state; do not assume the whole family matrix works without validation.

Native means:

- model parameters are loaded into LightX2V weight wrappers
- linear projections use LightX2V MM ops, not upstream `nn.Linear` modules
- normalization/attention/conv use LightX2V common ops or existing family wrappers
- scheduler state is managed by LightX2V scheduler classes
- third-party libraries may be used for tokenizers, file IO, or conversion scripts when necessary, but not as the runtime DiT/pipeline implementation

## First Inspection

Before editing, inspect the current repo and the upstream model:

- LightX2V entrypoints:
  - `lightx2v/infer.py`
  - `lightx2v/utils/set_config.py`
  - `lightx2v/utils/input_info.py`
  - `lightx2v/utils/registry_factory.py`
  - `lightx2v/models/runners/runner_factory.py`
  - `lightx2v/models/runners/request_fields.py`
  - `lightx2v/models/runners/base_runner.py`
  - `lightx2v/models/runners/default_runner.py`
  - nearest family runner, for example `lightx2v/models/runners/wan/wan_runner.py`
- Model implementation:
  - `lightx2v/models/networks/<family>/model.py`
  - `lightx2v/models/networks/<family>/weights/`
  - `lightx2v/models/networks/<family>/infer/`
  - `lightx2v/models/schedulers/<family>/`
  - `lightx2v/common/ops/`
  - `lightx2v/common/kvcache/`
- Upstream repo:
  - README model list and quickstart
  - config registry and launch scripts
  - model definition
  - weight loading path and checkpoint format
  - scheduler/solver code
  - CFG logic
  - KV cache logic
  - input preprocessing and output postprocessing

Use `rg` first. Do not guess from filenames when config files or README disagree.

When upstream has several config names or model names, cross-check:

- README model table and release notes
- config registry, for example `VA_CONFIGS`
- per-config camera keys, action channel ids, normalization stats, resolution, and chunk settings
- local model directory contents
- actual launch script defaults

Treat edited local paths as weak evidence. Dataset/domain fields and README model descriptions are stronger evidence for deciding which weight belongs to which task.

## Model Components

Read the reference for the component being implemented; keep these constraints in view:

| Component and reference | Key constraints |
| --- | --- |
| [Base family](references/model-components.md#decide-the-base-family) | Choose the nearest architectural family from tensor layout, attention, conditioning and scheduling; similar model names are insufficient. |
| [Weights](references/model-components.md#weight-strategy) | Keep checkpoint mapping in converters/loaders and weight containers. Inspect real checkpoint keys and preserve required non-quantized dtypes before choosing conversion or direct loading. |
| [DiT tensor layout](references/model-components.md#dit-tensor-layout) | Follow the selected family's weight/infer/attention contract. Wan-style DiTs use unbatched token tensors; Bagel and SeedVR have different layouts, so this is not a repository-wide batch prohibition. |
| [Input encoders](references/model-components.md#input-encoders) | Reuse the family's encoder interfaces and reproduce upstream conditioning, masks and sequence order. |
| [VAE and decoding](references/model-components.md#vae-and-decoding) | Match latent scaling, layout, temporal behavior and output decoding to the source pipeline. |
| [Model layout](references/model-components.md#model-layout) | Keep orchestration in the model, weight storage/loading in weights, and tensor math in infer. Reuse family modules rather than calling the upstream pipeline for the result. |
| [Infer implementation](references/model-components.md#infer-implementation) | Match source math and branch ordering first; use the family's registered operators and preserve offload/parallel boundaries. |
| [CFG](references/model-components.md#cfg) | For Wan-style unbatched DiTs, preserve serial conditional/unconditional execution and the scheduler's guidance semantics; add family-supported CFG parallelism with output parity. |

## Offload And Parallel Support

After the single-device serial path matches upstream, add the same execution modes that the base family supports. For Wan-derived models, this normally means block/model offload, sequence parallel and CFG parallel. Configure `parallel.seq_p_size`, `parallel.cfg_p_size` and, when applicable, `parallel.tensor_p_size`; `init_parallel()` derives the runtime `seq_parallel` / `cfg_parallel` flags.

Rules:

- Reuse the base family offload and distributed machinery before adding model-private code.
- For Wan-derived transformer infer, prefer inheriting or composing with existing Wan offload infer classes such as `WanOffloadTransformerInfer` when the phase order is compatible.
- Implement block offload for every custom weight/block phase introduced by the new model:
  - create matching CUDA block buffers such as `offload_block_cuda_buffers`
  - create lazy CPU-side buffers when the base weight class uses them
  - move non-block weights through the existing `non_block_weights_to_cuda/cpu` hooks
- Support model-level offload when the parent model already provides `to_cuda`/`to_cpu` style movement. Do not special-case devices in runner hot loops.
- Reject phase offload clearly if the new model has a custom phase order and phase offload has not been mapped yet. Do not silently reuse Wan phase offload for a different phase graph.
- For sequence parallel:
  - reuse existing process groups and helpers such as `seq_p_group`, sequence chunking, all-gather, and parallel attention ops
  - use the base self-attention parallel kernel when tensor layout matches
  - if a custom module needs the full token grid, gather full tokens for that module and chunk the result back to the local rank
  - for custom reference maps or multi-human masks, gather `q/k` before computing a full attention map when local chunks are insufficient
- For CFG parallel:
  - keep CFG branch orchestration in `model.infer`, not runner
  - for Wan-style unbatched DiTs, distribute branches across ranks instead of concatenating a batch dimension; preserve the supported tensor layout for other families
  - build an explicit branch list, for example `cond`, `drop_text`, `drop_audio`, `uncond`
  - support the actual branch count needed by the model, including 2-branch and 3-branch CFG
  - gather branch predictions and combine them with the same formula as serial CFG
  - make text/audio encoders produce every embedding needed by each rank; do not assume Wan's default rank0 prompt / rank1 negative split is enough for multi-branch CFG
- Keep serial, offload, SP, CFG parallel, and combined modes numerically aligned with the same seed/config whenever practical.

## Scheduler

Keep timestep, latent and solver state in the selected scheduler. LingBot-VA's optional `prepare_loop`/`bind_step_inputs` hooks are not `BaseScheduler` APIs; see [scheduler preparation and stepping](references/scheduler-and-cache.md#scheduler) when adapting a denoise loop.

## Runner

Runner should follow the closest existing family style.

Responsibilities:

- register with `@RUNNER_REGISTER("<model_cls>")` and add its import path to `runner_factory.RUNNER_MODULES`
- declare `supported_request_fields_by_task` using `request_fields.py` sets; add `input_info_cls_by_task` or `INPUT_INFO_TYPES` entries only when needed
- initialize/load modules
- initialize scheduler(s)
- bind `run_input_encoder`
- prepare per-run state in `init_run`
- run the denoise loop
- decode/save outputs
- clear caches and modules in `end_run`

Rules:

- Inherit `DefaultRunner` or nearest family runner.
- Do not put transformer math in runner.
- Do not put scheduler stepping inside model unless existing family requires it.
- Use `self.model.set_scheduler(self.scheduler)` before infer loop.
- For multi-stage models, it is fine to have multiple schedulers, for example video scheduler and action scheduler, but keep both using the same `step_pre -> infer -> step_post` pattern.
- Avoid fallback output paths like `train_out`; require or use `input_info.save_result_path`.
- Input paths such as `image_path` should come from `input_info`, not config defaults.
- `init_modules()` should follow the parent style: load modules, set scheduler on model, bind `run_input_encoder`, lock config, optionally compile.
- `init_run()` should prepare per-run state, KV cache manager, prompt embeds from `self.inputs`, action masks/stats, and AR dimensions.
- `run_segment()` or equivalent should only orchestrate scheduler loops and postprocess outputs.
- `end_run()` should clear request state and release transient/lazy-loaded modules through the existing cleanup hooks. Preserve reusable compiled graphs and apply the family's allocator policy; do not add unconditional `empty_cache()` to every eager request.

## KV Cache

Derive cache keys, reset boundaries and token ordering from the actual autoregressive task. Validate segment transitions and repeated requests rather than inferring cache correctness from one output. Use the [KV cache integration rules](references/scheduler-and-cache.md#kv-cache) when adding or adapting a cache scheme.

## Config And Launch

| Task and reference | Key constraints |
| --- | --- |
| [Configure startup and requests](references/config-and-launch.md#config-conventions) | Use current startup fields and request contracts: `size=[height,width]`, `fps` and parallel sizes. `build_startup_config` determines precedence; `prepare_request` rejects undeclared task fields. Avoid duplicate aliases and model-local fallback defaults. |
| [Add launch scripts](references/config-and-launch.md#bash-script-conventions) | Use the family's existing layout and `scripts/base/base.sh` contract. Keep machine paths editable and quoted, run from the repository root, and separate startup JSON from supported request arguments. |

## Registry And CLI Choices

When adding `model_cls`:

- register runner with `RUNNER_REGISTER`
- add the runner module path to `lightx2v/models/runners/runner_factory.py:RUNNER_MODULES`; `build_runner()` imports that module lazily
- `infer.py` uses `choices=RUNNER_MODULES`, so do not add eager runner imports or a second model choice list there
- update task choices only if a new `--task` string is truly needed; wire its request fields and `InputInfo` as described above
- `configs/model_pipeline.json` is not consumed by the current `build_runner()` registration path; do not add entries as a native CLI registration step

If the upstream config name says `i2av` but the actual infer mode is image-to-video-action, choose one LightX2V task name and keep it consistent. Prefer semantic names in code (`i2va`) and use config filenames only for compatibility if needed.

When an error says `argument --model_cls: invalid choice`, check the model's entry in `runner_factory.RUNNER_MODULES` first. Do not rename the model to an unrelated existing `model_cls` just to pass argument parsing.

## Parity And Porting Checks

| Check and reference | Key constraints |
| --- | --- |
| [Source parity](references/numerical-validation.md#source-parity-checklist) | Compare conditioning, timestep mapping, latent scaling, CFG, attention layout, cache and decoding with the upstream source before claiming native parity. |
| [Numerical alignment](references/numerical-validation.md#numerical-alignment-workflow) | Find the first divergent intermediate under matching inputs and precision. Record tolerances and evidence; a visually plausible output is insufficient. |
| [Wan-derived porting](references/numerical-validation.md#wan-derived-porting-checklist) | For Wan-derived models, follow the staged checklist through the family's offload and parallel modes; base inference alone is not complete delivery. |
| [Common pitfalls](references/numerical-validation.md#common-pitfalls) | Check upstream naming, key mapping, tensor layout, scheduler state and generated output before adding special cases. |

## Verification

Select checks for the files changed. Run syntax and reference checks for documentation-only maintenance; CLI import checks require the project environment:

```bash
cd <LightX2V repo root>
python3 -m py_compile <changed python files>
bash -n <changed scripts>
python3 -m json.tool <one changed config>
python -m lightx2v.infer --help
```

Run inference when weights and inputs are available. Confirm:

- output video/image exists
- action file exists if applicable
- output length matches `num_chunks`, chunk size, VAE stride, and FPS
- no unwanted batch dimension appears in debug shapes
- no unexpected diffusers/transformers runtime model dependency
- results are visually/numerically close to upstream with the same seed/config
- saved intermediate tensors show close parity at preprocessing, scheduler, `pre_infer`, block 0, and full DiT boundaries when upstream is available
- sparse-frame contact sheets or equivalent artifacts show every expected region, view, or modality is stable and in the correct location
- block/model offload mode completes and produces the same shape/duration as local mode
- CFG parallel returns the same branch-combined prediction shape and comparable output as serial CFG
- sequence parallel preserves token counts and reconstructs full-output shapes after all-gather/chunk operations

Report skipped checks clearly.
