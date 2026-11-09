# Adding request parameters

Make the model use the new value first, then connect the required entry points. This guide uses the existing Qwen-Image-2.1 `resolution` implementation to show where changes belong. Follow the same steps for a new parameter.

`InputInfo` holds parameters for one generation; the runner uses them to execute model inference.

## 1. Add the field

Check the runner's `input_info_cls_by_task` first. If it has no entry for the task, look up the class in `INPUT_INFO_TYPES` in `lightx2v/utils/input_info.py`. Declare the parameter as a dataclass field in that class:

```python
resolution: Optional[int] = None
```

Qwen-Image-2.1 uses `T2IInputInfo` for `t2i` and `I2IInputInfo` for `i2i`; both declare this field. A model-specific field can also live in a dedicated subclass selected through `input_info_cls_by_task`.

## 2. Declare support

Declare an `Argument` for the intended tasks in the model's own `request_spec.py`, keeping its existing fields. Qwen-Image-2.1 declares these in `lightx2v/models/runners/qwen_image_21/request_spec.py`:

```python
from lightx2v.models.runners.request_args import IMAGE
from lightx2v.utils.request_spec import Argument, TaskSpec

SPECS = {
    "qwen_image_21": QwenImage21Spec(
        {
            "t2i": TaskSpec(
                dict(IMAGE, resolution=Argument(type=int, gt=0, help="Output resolution.")),
                outputs=("image",),
            ),
            "i2i": TaskSpec(
                dict(
                    IMAGE,
                    image_path=Argument(help="Input image path."),
                    resolution=Argument(type=int, gt=0, help="Output resolution."),
                ),
                outputs=("image",),
            ),
        }
    ),
}
```

`TaskSpec` groups a task's arguments and output types. An `i2v` task declares `outputs=("video",)` even though it accepts an image. The server registers image/video routes only for compatible tasks. `ImageVideoTaskSpec` in `request_args.py` declares both outputs and resolves the actual type from the merged startup/request paths; mismatched requests fail with HTTP 422 before queueing. Other output types require a service adapter before they can be served.

The CLI and HTTP API generate their input models, validation and documentation from this Spec. `BaseRunner` uses the same Spec to validate tasks and request fields. Define model-specific arguments locally without changing a global argument table or repeating declarations in the runner or server schema. The tasks and fields accepted by an initialized instance still depend on its startup task and model configuration. For parameters involved in model initialization, first confirm that the model supports changing them at runtime.

`Argument.type` describes the complete value, such as `int`, `list[int]`, or `str | dict`. The CLI turns lists into space-separated options and structured values into JSON inputs; HTTP accepts the corresponding JSON values. Use `gt`, `ge`, `lt` and `le` for shared numeric bounds. Set `cli=False` or `server=False` for inputs specific to an entry point; callbacks and in-memory tensors should disable both.

Declare only public tasks in the Spec. For example, HunyuanImage3 generates text internally to enhance image prompts; its public tasks are `t2i`, `ti2i` and `i2i`.

Use `MultiTaskSpec` when one loaded checkpoint supports multiple request tasks. The runner and scheduler must read `input_info.task` during inference. Load the components needed by those tasks at initialization and reset request state between runs; changing `config["task"]` per request would mix deployment settings with request state.

Keep declared arguments in the Spec when a startup setting makes them unusable. Validate incompatible combinations in the runner's request preparation or inference path, with an error explaining the conflicting setting and how to resolve it. Do not conditionally remove these fields from the Spec.

For a supported `negative_prompt` supplied while CFG is disabled, log a warning and ignore the request value instead of raising an error.

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

If configuration also omits the field, the `InputInfo` default remains; model-specific request preparation or preprocessing may resolve it further. Put necessary type, range and parameter-combination checks in those stages so all entry points pass through them.

The runner preserves `0`, `False`, `""` and `[]` when merging parameters. Field rules determine whether they are valid; do not treat them as omitted by filtering with `if value`.

## 4. Connect entry points

| Entry point | Where to change |
|---|---|
| Python | Usually no change: `generate(resolution=768)` forwards new fields through `**task_inputs`. |
| CLI | The `Argument` declared in step 2 generates the option automatically; no change to `infer.py` is needed. |
| Native HTTP JSON / forms | Generated from the same Spec at service startup; no additional field declaration. |
| OpenAI image API | Inference fields are generated from the Spec; protocol fields remain in the OpenAI adapter. |

CLI and HTTP request overrides default to `None`, leaving startup and `InputInfo` defaults in control. HTTP forwards only explicitly submitted fields through `model_fields_set`.

The service builds request models from each output API's compatible tasks. The single/multi-task rule uses the entire service's task set, even if an API contains only one of those tasks. Multi-task requests select the model through the required `task` field. Single-task request models do not contain `task`; submitting it is rejected with HTTP 422, and the runner uses its startup task. `/docs` shows the selected model's fields rather than a global image/video parameter collection.

OpenAI image endpoints reuse the Spec's inference fields while retaining protocol conventions such as required `prompt`, `n`, `response_format` and `size="WIDTHxHEIGHT"`. Format conversions and new protocol controls still belong in that adapter.

Submit lists or objects as JSON text in forms, such as `size=[512,768]`. A new upload also needs an endpoint `File(...)` argument, file-saving code, and upload-field documentation in `form_request_openapi()`.

## 5. Verify the effect

- **Defaults and isolation:** check omission, `None`, explicit values, and whether a subsequent request without the field returns to the expected default.
- **Support:** exposed fields work for the intended model and task; unsupported combinations fail clearly.
- **Entry-point consistency:** submit through each intended interface and inspect the value received by the runner.
- **Actual effect:** check that computation uses the value, including its precedence relative to other parameters.

For example, a nonempty `size` takes precedence for Qwen-Image-2.1 output dimensions, even when inherited from startup configuration. When verifying how `resolution` affects automatic dimensions, also check `size`; interface acceptance alone is insufficient.
