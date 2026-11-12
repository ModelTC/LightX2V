# LightX2V Warmup Implementation Patterns

English | [简体中文](implementation-patterns.md)

Read this reference for a code skeleton, family-specific behavior, or performance diagnostics. Use `rg` to locate the current interfaces before adapting an example.

## Contents

- [Reference entry points](#reference-entry-points)
- [Execution mode snapshot](#execution-mode-snapshot)
- [Starting with a model that lacks warmup](#starting-with-a-model-that-lacks-warmup)
- [Runner skeleton](#runner-skeleton)
- [Wan modes](#wan-modes)
- [Qwen-Image modes](#qwen-image-modes)
- [MiniMax-H3-Image modes](#minimax-h3-image-modes)
- [LTX2 modes](#ltx2-modes)
- [Lingbot-Video modes](#lingbot-video-modes)
- [Cold-start boundaries and common pitfalls](#cold-start-boundaries-and-common-pitfalls)
- [Scheduler clear contract](#scheduler-clear-contract)
- [Allocator cache pitfalls](#allocator-cache-pitfalls)
- [Experiment template](#experiment-template)

## Reference entry points

| Purpose | Location |
|---|---|
| Shared warmup lifecycle and GC | `lightx2v/models/runners/base_runner.py` |
| Warmup toggle and rejection conditions | `lightx2v/models/runners/default_runner.py` |
| Wan task, MoE, and lazy modes | `lightx2v/models/runners/wan/wan_runner.py` |
| Wan state cleanup | `lightx2v/models/schedulers/wan/scheduler.py` |
| Qwen T2I and I2I modes | `lightx2v/models/runners/qwen_image/qwen_image_runner.py` |
| Qwen RoPE and diffusers state cleanup | `lightx2v/models/schedulers/qwen_image/scheduler.py` |
| MiniMax-H3-Image T2I and I2I prompts by round | `lightx2v/models/runners/minimax_h3_image/minimax_h3_image_runner.py` |
| LTX2 multistage and iterator paths | `lightx2v/models/runners/ltx2/ltx2_runner.py` |
| LTX2 latent and sigma cleanup | `lightx2v/models/schedulers/ltx2/scheduler.py` |
| Test discovery | `rg --files tests test_cases` |

## Execution mode snapshot

The table summarizes the current base implementations. Each cell reports production inference / warmup. Recheck the exact `model_cls`, task, and subclass when integrating a model.

| Model scope | Normal | CPU model | CPU block | CPU phase | Lazy block | Lazy phase |
|---|---|---|---|---|---|---|
| Generic Wan T2V, I2V, and FLF2V | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| Qwen-Image T2I and I2I, excluding layered | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| LTX2 T2AV and I2AV | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | — / — | — / — | — / — |
| Lingbot-Video T2I, T2V, and I2V | ✓ / ✓ | — / — | — / — | — / — | — / — | — / — |

Code evidence

- Wan offload inference dispatches model, block, and phase. Its weights provide block and phase CUDA staging and lazy CPU staging. The runner reuses production loading for lazy warmup and cleans up in `finally`. Check specialized subclasses separately.
- The Qwen-Image model moves the full model to and from the device. Its offload infer and weights implement block and phase and both kinds of lazy staging. The runner supports eager and lazy warmup for T2I and I2I and rejects layered.
- LTX2 offload infer accepts model and block and raises for phase. Its weights lack lazy block CPU staging, and its runner rejects lazy warmup explicitly.
- The Lingbot-Video model rejects CPU offload, and its runner rejects lazy load and unload. Warmup covers normal inference.

A dash marks an incomplete path. Keep integration within supported production offload and lazy-load modes.

## Starting with a model that lacks warmup

The shared hook runs once before the first formal request. Model-specific runners can rely on startup-only execution.

1. Complete the execution mode matrix from the main skill. Assess production inference and warmup separately.
2. Check whether the runner inherits `DefaultRunner` and identify the concrete `init_modules()` wrapped by the shared hook.
3. Trace production `run_pipeline()` through the encoder, scheduler prepare, step loop, stage transitions, decoder, and `end_run()`.
4. Add `run_warmup()` to the smallest runner class that shares the operator graph. Reuse the shared hook and `infer.py` entry point.
5. Require explicit opt-in from subclasses with different operator graphs.
6. Start with normal inference for one task, then reuse the verified loading and unloading methods for other modes.

Declare `_WARMUP_SHAPES` and `_WARMUP_PROMPTS` in the runner. Build warmup inputs in `_prepare_warmup_inputs()`. Split `run_warmup()`, `_run_warmup()`, and `clear_warmup_state()` according to loading and cleanup responsibilities. Prefer production cleanup for lazy load.

## Runner skeleton

Adapt the guard and loading methods to the target runner.

```python
@ProfilingContext4DebugL1("Warmup")
def run_warmup(self):
    if not self.supports_warmup():
        raise NotImplementedError(...)

    lazy_load = self.config.get("lazy_load", False)
    try:
        if lazy_load:
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self._run_warmup()
    finally:
        if lazy_load:
            self.clean_lazy_load_warmup()

    self._maybe_freeze_gc()
```

- `supports_warmup()` stands for the target task and subclass guard. Use the actual guard rather than adding a redundant helper.
- `_prepare_warmup_inputs()` is the runner convention for input preparation. Split other helpers by responsibility.
- Check version-specific features when a runner serves multiple versions. With explicit `warmup=true`, raise an error for unsupported models, versions, tasks, or modes.
- Keep startup warmup checks, locks, and one-time state in the startup hook.
- Run lazy cleanup in `finally`.
- Call `_maybe_freeze_gc()` after `_run_warmup()` returns. The shared method skips freezing in lazy mode.

This single-model T2I example keeps each prompt on one line. Fill in scheduler arguments and step selection for the actual model.

```python
class ExampleImageRunner(DefaultRunner):
    _WARMUP_SHAPES = ((480, 480), (832, 1248))
    _WARMUP_PROMPTS = (
        "A pear on a plate.",
        "A still life photograph of ripe pears on a ceramic plate beside a folded linen napkin. Soft afternoon light reveals the speckled fruit, the glazed rim, and the grain of the wooden table.",
    )

    def _prepare_warmup_inputs(self, warmup_index, height, width):
        self.input_info = self.create_input_info(
            {"task": "t2i", "prompt": self._WARMUP_PROMPTS[warmup_index], "size": [height, width], "seed": 0}
        )
        return self.run_input_encoder()

    def _run_warmup(self):
        scheduler = self.model.scheduler
        model_offload = self.config.get("cpu_offload", False) and self.config.get("offload_granularity") == "model"

        for warmup_index, (height, width) in enumerate(self._WARMUP_SHAPES):
            inputs = output = None
            try:
                scheduler.generator = None
                inputs = self._prepare_warmup_inputs(warmup_index, height, width)
                scheduler.prepare(...)

                # Reset only when the scheduler cannot reuse its state across a step gap.
                for step_index in self.get_warmup_step_indices(scheduler):
                    scheduler.step_pre(step_index)
                    self.model.infer(inputs)
                    scheduler.step_post()

                output = self.run_vae_decoder(...)
                consume_if_iterator(output)
                torch_device_module.synchronize()
            finally:
                inputs = output = None
                if model_offload:
                    self.model.to_cpu()
                self.clear_warmup_state()
                self.input_info = None
                self.__dict__.pop("inputs", None)
```

A single-round path selects one prompt through `_prepare_warmup_inputs(0, ...)`. Frame-only runners prepare frames or tensors in the helper and use `_WARMUP_PROMPTS = ()`.

Some paths call `RunnerLifecycleMixin.get_warmup_resolutions()` to append `config["size"]` and remove any identical preset. Index prompts using the final list. EdenLive and MatrixGame4 can run up to three rounds from two presets, so they provide a separate prompt for the third round. Some Wan autoregressive subclasses and SekoTalk AR have no presets and run one round at the configured size with multiple chunks inside that round.

For MoE or other multi-model branches, use the target runner's model set and offload every model that was actually loaded. Reuse an existing inner `finally` boundary if it handles offload.

Save only changed state that will need restoring, such as step count, sigmas, or guidance. Restore startup defaults for `input_info` and `inputs`. Leave the generator as `None` after `clear_warmup_state()` so the formal request can create one from its own seed. Use the scheduler's actual `prepare/reset` arguments and step-gap semantics. Reset only when solver history, branch state, or other internal state cannot carry across the gap. If the scheduler needs a reset and has no `reset(step_index=...)`, rerun its production `prepare()`.

Clear the generator before input preparation. An I2V VAE encoder may use it to sample conditioning before the scheduler continues to generate initial noise. Clearing it after encoding changes the production RNG consumption order.

## Wan modes

Use this pattern for T2V, I2V, and FLF2V, Wan2.2 dense and MoE, changing resolutions, and distillation schedulers.

- The current generic path warms `480×480` and `720×1280` for T2V, I2V, and FLF2V. Require explicit opt-in for subclasses.
- T2V needs only text input.
- I2V runs the image and VAE encoders at each warmup shape.
- FLF2V prepares first and last frames and passes both to the encoder.
- A single-model path typically uses Step 0.
- MoE uses the first effective step of the high-noise and low-noise branches.
- Reset Wan UniPC or MoE scheduler state when the two representative steps are nonconsecutive and would otherwise reuse the wrong solver or branch state.
- Wan MoE changes `sample_guide_scale` during branch switching. Save the initialization value and restore it in the outermost `finally`.
- Verify opt-in separately for specialized paths such as VACE, audio, animate, and self-forcing.

## Qwen-Image modes

Use this pattern for T2I, I2I, packed image latents, and position or RoPE caches.

- The current warmup uses `480×480` and `832×1248` for non-layered T2I and I2I.
- T2I uses prompts of different lengths and runs the text encoder in both rounds. Reuse encoded output and its `txt_seq_lens` only for matching text conditions.
- I2I reruns image-conditioned text encoding and VAE encoding at each shape.
- Execute one `step_pre → infer → step_post` directly to avoid the extra profiling from `run(total_steps=1)`.
- Step 0 output can go directly to the existing VAE decoder.
- Clear packed latents, image IDs, timesteps, RoPE, and diffusers `_step_index/_begin_index` in the scheduler.
- Clear the request-specific RoPE cache in model pre-infer.
- In each shape's `finally`, restore the startup baseline with `input_info = None` and remove `inputs`.
- For the first formal I2I step, compare actual image and text tokens, guard logs, and timing. Follow the requested sample count.

## MiniMax-H3-Image modes

- The current warmup shapes are `2048×2048` and `3552×4736`. T2I, I2I, single GPU, and SP4 index the same short and long single-line prompts by round.
- Verify the length difference with the actual tokenizer. The two current prompt texts are 8 and 79 tokens before any added conditioning. The final I2I conditioning sequence also includes reference-image tokens.
- For Dynamo diagnostics, check both text-related `aux_length` and image token shape. Use `TORCH_LOGS=recompiles` to locate the guard and each rank's `Warmup cost` to identify the phase.

## LTX2 modes

Use this pattern for T2AV and I2AV, Stage 1 with spatial upsampling and Stage 2, video decode iterators, and scheduler state for both video and audio latents.

- Single-stage warmup uses `480×480` and `512×768`. The upsampler path uses `480×480` and `1024×1536`.
- Align Stage 1 pixel dimensions to the VAE spatial factor.
- The upsampler warmup shape is the final target size. Stage 1 uses the aligned size after dividing by the upsample scale.
- Use Step 0 to cover the first formal DiT forward.
- Use the final step to unpatchify video and audio latents for Stage 2 or VAE decode.
- The current base Euler scheduler has no multistep history, so it can keep its state across the gap between the first and final steps.
- Use the existing upsampler and VAE encoder preparation path for Stage 2.
- Fully consume the iterator returned by `video_vae.decode()`.
- Scheduler cleanup releases video and audio latents, predictions, sigmas, and the generator.
- Stage 2 `reset_sigmas()` changes `infer_steps`. Save it as `stage1_infer_steps` and restore it before each warmup shape and in the outermost `finally`.

### LTX2 lazy-load and offload review

Current warmup supports CPU model and block offload and explicitly rejects phase and lazy load. Reassess warmup after the repository supports the corresponding production paths.

Check these files together.

```text
lightx2v/models/networks/ltx2/weights/transformer_weights.py
lightx2v/models/networks/ltx2/infer/offload/transformer_infer.py
lightx2v/common/offload/manager.py
```

Verify CPU and CUDA buffers, disk prefetch, CPU swap, and `reset_infer_states()` for each inference. Confirm loading succeeds before evaluating block warmup. If `load_transformer()` fails, the outer `Warmup cost` only measures the attempted path.

## Lingbot-Video modes

Use this pattern for normal T2I, T2V, and I2V.

- Current warmup uses `480×480` and `320×832`. These lower-area shapes cover different height and width regimes and a landscape aspect ratio while limiting live-tensor peaks during warmup.
- T2I and T2V encode the distinct prompts separately. I2V uses an in-memory image at each shape to exercise VAE and VLM conditioning.
- The I2V VAE encoder consumes the scheduler generator before noise generation. Clear the generator before preparing each shape, then let the scheduler continue generating noise.
- Step 0 covers the single transformer path. After decode, clear conditioning, generator, latents, timesteps, sigmas, and solver history.
- When the scheduler inherits Wan, reuse its full `clear()` and then clear Lingbot-specific state.

## Cold-start boundaries and common pitfalls

Warmup can preserve compiled graphs, selected kernels, the eager allocator cache, and reusable process or system file caches. Formal requests still perform the following work.

- Full-model CPU-to-GPU transfer in CPU model offload.
- Weight transfer, buffer swapping, and stream synchronization on each block or phase offload step.
- Rebuilding models, offload managers, staging buffers, and object state after lazy cleanup.
- Rebuilding request-specific RoPE or position caches cleared for correctness.
- Running encoders or branches specific to the formal input and handling its shape, dtype, or layout.

Compare warmup and no-warmup runs when evaluating warmup, then separate avoidable cold-start work from the costs of the formal request lifecycle. Use more evidence than Step 1 and Step 2 timing alone.

| Pitfall or symptom | What to verify |
|---|---|
| A script or config accepts an offload value | Locate inference dispatch, weight staging, manager initialization, and cleanup. |
| Calling full `run_pipeline()` during warmup | Reuse production stage methods to keep saving, full request profiling, and request cleanup out of warmup. |
| A dummy forward or a decoder iterator left unconsumed | Exercise the real scheduler and branches and fully consume the iterator. |
| Inferring the round count from `_WARMUP_SHAPES` | Check for an appended config size and deduplication. Index prompts by actual order. |
| Removing a missing LoRA or checkpoint from the original config | Label the alternative as a smoke test and record the blocked or unverified original config. |
| The production shape runs out of memory on the target device | Reproduce without warmup. Use smaller shapes for axis coverage and report the remaining exact graph coverage gap. |
| Warmup and production use the same H×W | Also match frames and tokens, dtype, strides and layout, CFG or MoE, and Dynamo leaf-op dispatch. |
| Setting the generator to `None` after I2V encoding | Reset before any encoder that may consume RNG to preserve conditioning and noise order. |
| Using only Step 0 for every model | Cover each branch. Include the final step when required by unpatchify or finalization. |
| Always resetting or always keeping state across step gaps | Inspect scheduler history and reset or prepare only where state cannot carry across the gap. |
| Keeping the generator or request-specific RoPE cache for speed | Clear request state to preserve seeds and shapes. Reuse only persistent caches defined as safe by the production implementation. |
| Unconditional `empty_cache()` after warmup | Check for allocator or workspace reallocation on formal Step 1. |
| Slow Step 1 under model offload | Measure the full-model transfer and then attribute the remaining overhead that warmup could eliminate. |
| Reinitializing the manager after lazy warmup | Preserve lazy cleanup semantics. The formal request rebuilds the released model and manager. |
| Lazy cleanup only sets `self.model = None` | Break infer and staging references from the scheduler and compiled closures. Verify release with `weakref`. |
| A new Triton compile appears in warmup logs | Check graph guards, shapes, branches, and leaf kernels before adding rounds. |
| VAE OOM, a 32-bit indexing error, or loading failure | Reproduce without warmup. Failure at the same point indicates a model, decoder, or offload issue. |
| A proposed fix changes shared weight or offload infrastructure | Keep warmup changes in the runner and scheduler unless the production path has a confirmed infrastructure defect. |

## Scheduler clear contract

Adapt the fields to the family.

| Category | Fields to clear |
|---|---|
| Random state | Generator |
| Latents | Latent, video, and audio state, masks, conditioning latents |
| Current outputs | Conditional, unconditional, and guided noise predictions |
| Solver | Model outputs, timestep history, last sample, order |
| Request indices | Step index, begin index, CFG or MoE branch |
| Shape cache | Request-specific RoPE, position, and `cu_seqlens` |

Preserve compiled graphs, kernel caches, reusable eager allocator cache, and resident model weights.

Trace `end_run()`. If the formal request also calls `clear()`, verify that changes preserve repeated-request behavior.

## Allocator cache pitfalls

Inspect the interval from warmup completion to the first formal DiT step.

```python
text_encoder_output = self.run_text_encoder(self.input_info)
torch_device_module.empty_cache()
gc.collect()
```

Typical sequence

```text
Warmup DiT and VAE decode complete
  → formal text, image, and VAE encoders
  → empty_cache()
  → formal DiT Step 1
```

Kernels and compile state remain warm. The allocator blocks or workspace have been released. Typical signs include the following.

- Warmup logs show the encoders, DiT, and decoder completed.
- Formal Step 1 is consistently slower than Steps 2–5 of the same stage.
- A GPU idle gap before Step 1 is followed by a rise in power or utilization.
- Scheduler behavior, selected steps, and the formal shape are verified.

In eager mode, remove unconditional cleanup or use `self.maybe_empty_cache()`. For lazy load or unload, retain required cleanup after releasing temporary references. Check T2*, I2*, and decode branches separately. Compare these conditions.

```text
warmup + unconditional empty_cache
warmup + pressure-aware/no empty_cache
no-warmup
```

If the second group reliably eliminates the Step 1 gap, attribute the gap to allocator or workspace cold start. Keep the step selection and scheduler behavior focused on the original coverage requirements.

### Two cleanup boundaries in model offload

Test cleanup after encoding and before transformer transfer independently of cleanup after transformer offload.

- Unconditional `empty_cache()` after encoding can release allocator blocks or workspaces preserved by warmup. If pressure-aware cleanup reliably lowers latency at the same peak memory, limit the change to this boundary.
- After model offload, large cached allocations can keep idle GPU memory near capacity even though the weights are on CPU. The next request may trigger reclamation or retries during allocation, slowing encoding, model transfer, and Step 1. Explicit release is often still needed here. Validate with consecutive requests and idle memory measurements.

Run at least three consecutive identical requests for this diagnostic. Record the encoder, Prepare DiT/H2D, Step 1, D2H, VAE, end-to-end latency, and idle memory between requests. Require stable results across requests and acceptable idle memory.

## Experiment template

Resolve the effective config and check checkpoint, LoRA or adapter, and media paths before using a GPU. Keep the user script's model, task, config, prompt, and inputs. Change only what the experiment requires.

```text
Startup JSON: "warmup": true / false
--return_result_tensor
CUDA_VISIBLE_DEVICES=<free GPU>
```

Complete built-in warmup and a formal request at least once per target mode. Follow the requested run count. Add samples when estimating stability. Label alternative configs or shapes as smoke tests when the original config cannot run. For Dynamo diagnostics, set `TORCH_LOGS=recompiles` on the original inference command and split events at each rank's `Warmup cost`.

```bash
rg "Warmup|Recompiling function|Run Text Encoder|Run Image Encoder|Run VAE Encoder|Run VAE Decoder|Run Dit every step|Run DiT|infer_main cost|Save Image|RUN pipeline cost|Traceback|ERROR" <log>
```

Group multistage logs by stage and compute timings separately for Stage 1 and Stage 2.

Check GPU occupancy before and during runs. For a single-run benchmark, record warmup, formal stage timings, and end-to-end latency. Record the sampled rank for four-GPU runs. When comparing warmup and no-warmup, also report first-request and total startup latency for both groups, and check frequency, temperature, and save or I/O timing boundaries. Use output hashes, tensors, or reasonable tolerance with the same seed to verify isolation. If lazy runs with and without warmup fail at the same `load_transformer/offload_manager` location, classify the failure as a base lazy-load or offload issue.
