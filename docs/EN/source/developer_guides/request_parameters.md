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

Add the field to the intended tasks in the runner's `supported_request_fields_by_task`, keeping its existing fields. Qwen-Image-2.1 declares these in `lightx2v/models/runners/qwen_image_21/qwen_image_21_runner.py`:

```python
supported_request_fields_by_task = {
    "t2i": IMAGE_REQUEST_FIELDS | {"resolution"},
    "i2i": IMAGE_REQUEST_FIELDS | {"image_path", "resolution"},
}
```

This permits the corresponding tasks to override `resolution`. The tasks and fields accepted by an initialized instance still depend on its startup task and model configuration. For parameters involved in model initialization, first confirm that the model supports changing them at runtime.

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
| CLI | Declare an option in `build_parser()` in `lightx2v/infer.py`. |
| Native HTTP JSON / forms | Declare a field in `ImageTaskRequest` or `VideoTaskRequest` in `lightx2v/server/schema.py`. |
| OpenAI image API | Declare a field in a request class in `lightx2v/server/api/openai_images.py`, and ensure `ImageTaskRequest` also accepts it. |

Use `default=None` for CLI options so omitted parameters retain startup values:

```python
parser.add_argument("--resolution", type=int, default=None)
```

Example field declaration in an HTTP request class:

```python
resolution: Optional[int] = Field(None, gt=0, description="Qwen-Image-2.1 resolution")
```

JSON and forms share this declaration for validation and documentation. Ordinary new fields need no additional `Form(...)` arguments or parsing whitelist. HTTP forwards submitted inference fields using `model_fields_set`, so request-class construction defaults do not automatically override startup configuration.

Put fields shared by both OpenAI image endpoints in `OpenAIImageRequest`, and endpoint-specific fields in `OpenAIImageGenerationRequest` or `OpenAIImageEditRequest`. Ordinary inference fields with matching names pass through automatically. Format conversions, such as `size`, and protocol controls still require explicit adapter changes.

Submit lists or objects as JSON text in forms, such as `size=[512,768]`. A new upload also needs an endpoint `File(...)` argument, file-saving code, and upload-field documentation in `form_request_openapi()`.

## 5. Verify the effect

- **Defaults and isolation:** check omission, `None`, explicit values, and whether a subsequent request without the field returns to the expected default.
- **Support:** exposed fields work for the intended model and task; unsupported combinations fail clearly.
- **Entry-point consistency:** submit through each intended interface and inspect the value received by the runner.
- **Actual effect:** check that computation uses the value, including its precedence relative to other parameters.

For example, a nonempty `size` takes precedence for Qwen-Image-2.1 output dimensions, even when inherited from startup configuration. When verifying how `resolution` affects automatic dimensions, also check `size`; interface acceptance alone is insufficient.
