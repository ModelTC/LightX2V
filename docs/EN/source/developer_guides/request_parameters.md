# Adding request parameters

Make the model use the new value first, then connect the required entry points. This guide uses the existing Qwen-Image-2.1 `resolution` implementation to show where changes belong. Follow the same steps for a new parameter.

`InputInfo` holds parameters for one generation; the runner uses them to execute model inference.

## 1. Define the parameter in its Spec

Declare an `Argument` for the intended tasks in the model's own `request_spec.py`, keeping its existing fields. Qwen-Image-2.1 declares these in `lightx2v/models/runners/qwen_image_21/request_spec.py`:

```python
from lightx2v.models.runners.request_args import IMAGE
from lightx2v.utils.request_spec import Argument

SPECS = {
    "qwen_image_21": QwenImage21Spec(
        {
            "t2i": dict(IMAGE, resolution=Argument(type=int, gt=0, help="Output resolution.")),
            "i2i": dict(
                IMAGE,
                image_path=Argument(type=list[str], input_info_default=[], help="Input image paths."),
                resolution=Argument(type=int, gt=0, help="Output resolution."),
            ),
        }
    ),
}
```

Specs contain tasks, input arguments and context defaults, without output-type declarations. `POST /v1/tasks/` accepts requests for the deployed model; `/docs` is generated from its tasks and arguments. Legacy image/video routes and the OpenAI image adapter reuse these definitions and check the actual generated artifacts.

The CLI and HTTP API generate their input models, validation and documentation from this Spec. `BaseRunner` uses the same Spec to validate tasks and request fields. Define model-specific arguments locally without changing a global argument table or repeating declarations in the runner or server schema. The tasks and fields accepted by an initialized instance still depend on its startup task and model configuration. For parameters involved in model initialization, first confirm that the model supports changing them at runtime.

Declare multi-input fields as `list[T]`, for example `Argument(type=list[str])`. CLI values are space-separated; JSON requests and startup configs use arrays; Python calls use lists. A single item still uses a list. Runners consume lists directly without splitting comma-separated strings.

| Parameter | CLI | JSON / Python |
|---|---|---|
| Reference images | `--image_path a.png b.png` | `"image_path": ["a.png", "b.png"]` |
| Conditioning strengths | `--image_strength 1.0 0.8` | `"image_strength": [1.0, 0.8]` |
| Frame indices | `--image_frame_indices 0 32` | `"image_frame_indices": [0, 32]` |

Only multi-input fields use lists; single-image fields remain `str`. Nested structures such as `layout_bboxes` and `motions` use JSON text in the CLI and native arrays or objects over HTTP. `layout_bboxes` also accepts a JSON file path.

Form lists and objects use JSON text. Upload multiple files by repeating the same upload field; OpenAPI shows an array of files for list-valued inputs. Saved paths retain their list type and order.

`Argument.type` describes the complete value, such as `int`, `list[int]`, or `str | dict`. The CLI turns flat lists into space-separated options and nested lists or objects into JSON inputs; HTTP accepts the corresponding JSON values. Use `gt`, `ge`, `lt` and `le` for shared numeric bounds. Set `cli=False` or `server=False` for inputs specific to an entry point; callbacks and in-memory tensors should disable both.

Declare only public tasks in the Spec. For example, HunyuanImage3 generates text internally to enhance image prompts; its public tasks are `t2i`, `ti2i` and `i2i`.

Use `MultiTaskSpec` when one loaded checkpoint supports multiple request tasks. The runner and scheduler must read `input_info.task` during inference. Load the components needed by those tasks at initialization and reset request state between runs; changing `config["task"]` per request would mix deployment settings with request state.

Keep declared arguments in the Spec when a startup setting makes them unusable. Validate incompatible combinations in the runner's request preparation or inference path, with an error explaining the conflicting setting and how to resolve it. Do not conditionally remove these fields from the Spec.

For a supported `negative_prompt` supplied while CFG is disabled, log a warning and ignore the request value instead of raising an error.

## 2. Create InputInfo from the Spec

`Argument` is the single definition of each request field. Its `type` supplies the annotation; `input_info_default` supplies the initial value when creating InputInfo. Write values directly, such as `input_info_default=""`, `input_info_default=[]` or `input_info_default=[1.0]`. InputInfo creation deep-copies each default, so requests have independent lists and dictionaries, including nested containers.

`BaseRunner.get_input_info_type(task)` combines the Spec fields with the runner's `input_info_state` into a dataclass and caches the type on that runner. Each request creates a fresh instance, applies startup settings, then applies request values. Adding a request parameter does not require editing an InputInfo class or a global task-to-type registry.

Omitted CLI/server fields still use `None` to mean no override. They do not use `Argument.input_info_default` as a request override.

Shared `ShapeState`, `ImageState`, `AudioVideoState` and `SRState` classes live in `lightx2v/utils/input_info.py`. Put model-specific runtime state in its runner directory:

```python
@dataclass
class ModelState(ShapeState):
    encoded_condition: Any = field(default=None, repr=False)

class ModelRunner(DefaultRunner):
    input_info_state = ModelState
```

State classes contain runtime state, without redeclaring request fields. Use `RequestSpec(..., context={...})` for `Argument` definitions initialized from startup settings but unavailable as request overrides; reuse existing common or model definitions where possible. Contexts also retain fields from the model's other tasks for shared execution paths. The current task's `get_arguments()` still determines which request fields are accepted.

Warmup and internal code can construct a context through `runner.get_input_info_type(task)(...)`. Normal requests go through `prepare_request()`.

## 3. Use the parameter during inference

Read the request value where computation uses it:

```python
resolution = self.input_info.resolution
```

Reading `self.config["resolution"]` would use the startup value and miss the request override. Qwen-Image-2.1 reads and validates the value in `run_input_encoder()` for reference-image resizing and automatic output dimensions.

`BaseRunner` applies startup configuration, then non-`None` request values. For a startup configuration containing `resolution=1024`:

| Request | Value seen by the model |
|---|---|
| Omitted, `None`, or JSON `null` | `1024` |
| `768` | `768` |

If configuration also omits the field, the runtime default declared in the Spec remains; model-specific request preparation or preprocessing may resolve it further. Put necessary type, range and parameter-combination checks in those stages so all entry points pass through them.

The runner preserves `0`, `False`, `""` and `[]` when merging parameters. Field rules determine whether they are valid; do not treat them as omitted by filtering with `if value`.

## 4. Connect entry points

| Entry point | Where to change |
|---|---|
| Python | Usually no change: `generate(resolution=768)` forwards new fields through `**task_inputs`. |
| CLI | The `Argument` declared in step 1 generates the option automatically; no change to `infer.py` is needed. |
| Native HTTP JSON / forms | Generated from the same Spec at service startup; no additional field declaration. |
| OpenAI image API | Inference fields are generated from the Spec; protocol fields remain in the OpenAI adapter. |

CLI and HTTP request overrides default to `None`, leaving startup and `InputInfo` defaults in control. HTTP forwards only explicitly submitted fields through `model_fields_set`.

The service builds request models from the deployed model’s tasks. Multi-task requests select the task through the required `task` field. Single-task request models do not contain `task`; submitting it is rejected with HTTP 422, and the runner uses its startup task. `/docs` shows the selected model's fields rather than a global image/video parameter collection.

OpenAI image endpoints reuse the Spec's inference fields while retaining protocol conventions such as required `prompt`, `n`, `response_format` and `size="WIDTHxHEIGHT"`. Format conversions and new protocol controls still belong in that adapter.

Submit lists or objects as JSON text in forms, such as `size=[512,768]`. A new upload also needs an endpoint `File(...)` argument, file-saving code, and upload-field documentation in `form_request_openapi()`.

## 5. Verify the effect

- **Defaults and isolation:** check omission, `None`, explicit values, and whether a subsequent request without the field returns to the expected default.
- **Support:** exposed fields work for the intended model and task; unsupported combinations fail clearly.
- **Entry-point consistency:** submit through each intended interface and inspect the value received by the runner.
- **Actual effect:** check that computation uses the value, including its precedence relative to other parameters.

For example, a nonempty `size` takes precedence for Qwen-Image-2.1 output dimensions, even when inherited from startup configuration. When verifying how `resolution` affects automatic dimensions, also check `size`; interface acceptance alone is insufficient.

## Register generated artifacts

Single-file image/video runners can reuse `BaseRunner.get_outputs()`, which collects the returned `image`, `images` or `video` field and its actual saved path. For multiple files, joint outputs, or rewritten filenames, register the real paths after saving:

```python
self.record_output("video", video_path)
self.record_output("action", action_path)
```

This preserves `run_pipeline()` return values and `InputInfo`. `/v1/tasks/{task_id}/results` lists the actual artifacts with individual download URLs. Action-only models use the unified submission endpoint too. A legacy image/video request fails after execution if the required artifact was not produced; Specs no longer filter output capabilities before queueing.
