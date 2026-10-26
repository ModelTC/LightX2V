# 新增请求参数

新增请求参数时，先让模型使用这个值，再接通需要的调用入口。下面以已实现的 Qwen-Image-2.1 `resolution` 为例；代码片段用于定位，新参数按相同步骤添加。

`InputInfo` 保存单次生成的参数，runner 负责使用这些参数执行模型推理。

## 1. 添加字段

先看 runner 的 `input_info_cls_by_task`；未指定时，在 `lightx2v/utils/input_info.py` 的 `INPUT_INFO_TYPES` 中查找任务对应的类。把新参数声明为该类的 dataclass 字段：

```python
resolution: Optional[int] = None
```

Qwen-Image-2.1 的 `t2i`、`i2i` 分别使用 `T2IInputInfo`、`I2IInputInfo`，两者都有此字段。模型专属字段也可以放在专用子类中，由 `input_info_cls_by_task` 选择。

## 2. 声明支持范围

在目标 runner 的 `supported_request_fields_by_task` 中，为需要支持的任务加入字段，保留原有字段。Qwen-Image-2.1 的声明位于 `lightx2v/models/runners/qwen_image_21/qwen_image_21_runner.py`：

```python
supported_request_fields_by_task = {
    "t2i": IMAGE_REQUEST_FIELDS | {"resolution"},
    "i2i": IMAGE_REQUEST_FIELDS | {"image_path", "resolution"},
}
```

该声明允许对应任务覆盖 `resolution`；当前实例实际接受的任务和字段还受启动任务及模型配置限制。涉及模型初始化的参数，需要先确认模型支持在运行时修改。

## 3. 在推理中使用参数

在实际计算位置读取本次请求的值：

```python
resolution = self.input_info.resolution
```

继续读取 `self.config["resolution"]` 会拿到启动值，无法应用请求覆盖。Qwen-Image-2.1 在 `run_input_encoder()` 中读取并校验这个值，用于参考图缩放和自动输出尺寸。

`BaseRunner` 先应用启动配置，再应用非 `None` 的请求值。例如启动配置为 `resolution=1024`：

| 本次请求 | 模型读到的值 |
|---|---|
| 不传、传 `None` 或 JSON `null` | `1024` |
| 传 `768` | `768` |

配置也未提供时，保留 `InputInfo` 默认值；模型可在请求准备或预处理中进一步处理。必要的类型、范围和参数组合校验也应放在这些位置，让所有入口都经过检查。

runner 合并参数时保留 `0`、`False`、`""`、`[]`。它们是否有效由字段规则决定，不能用 `if value` 把它们当作未传值。

## 4. 接通调用入口

| 入口 | 修改位置 |
|---|---|
| Python | 通常无需修改；`generate(resolution=768)` 会通过 `**task_inputs` 转发新字段。 |
| CLI | 在 `lightx2v/infer.py` 的 `build_parser()` 中声明选项。 |
| 原生 HTTP JSON / 表单 | 在 `lightx2v/server/schema.py` 的 `ImageTaskRequest` 或 `VideoTaskRequest` 中声明字段。 |
| OpenAI 图片接口 | 在 `lightx2v/server/api/openai_images.py` 的请求类中声明字段，并确保 `ImageTaskRequest` 也接受它。 |

CLI 使用 `default=None`，让未传参数保留启动值：

```python
parser.add_argument("--resolution", type=int, default=None)
```

HTTP 请求类的字段声明示例：

```python
resolution: Optional[int] = Field(None, gt=0, description="Qwen-Image-2.1 resolution")
```

JSON 和表单共用此声明进行校验并生成文档。普通新字段无需再添加 `Form(...)` 形参或解析白名单。HTTP 按 `model_fields_set` 转发用户提交的普通推理字段，因此请求类的构造默认值不会自动覆盖启动配置。

OpenAI 两个图片接口共用的字段放在 `OpenAIImageRequest`，接口独有字段放在 `OpenAIImageGenerationRequest` 或 `OpenAIImageEditRequest`。普通同名推理字段自动转发；需要格式转换的字段（如 `size`）或协议控制字段，仍需显式调整适配器。

列表或对象在表单中使用 JSON 文本，例如 `size=[512,768]`。新增文件上传还需声明端点的 `File(...)` 参数、保存文件，并在 `form_request_openapi()` 中补充上传字段文档。

## 5. 验证效果

- **默认值与隔离**：检查未传、`None`、显式值，以及下一次不传时是否恢复预期默认值。
- **支持范围**：已公开的字段在目标模型、任务中生效；不支持的组合明确报错。
- **入口一致性**：从需要支持的每个入口传入，检查 runner 收到的值。
- **实际效果**：检查模型计算是否使用新值，以及与其他参数的优先关系。

例如，非空的 `size` 会优先决定 Qwen-Image-2.1 的输出尺寸，即使它来自启动配置。验证 `resolution` 对自动输出尺寸的影响时，应同时检查 `size`，不能只确认接口是否接受参数。
