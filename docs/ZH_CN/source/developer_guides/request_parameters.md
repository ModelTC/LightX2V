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

在模型自己的 `request_spec.py` 中，为需要支持的任务声明 `Argument`，保留原有字段。Qwen-Image-2.1 的声明位于 `lightx2v/models/runners/qwen_image_21/request_spec.py`：

```python
from lightx2v.models.runners.request_args import IMAGE
from lightx2v.utils.request_spec import Argument

SPECS = {
    "qwen_image_21": QwenImage21Spec(
        {
            "t2i": dict(
                IMAGE,
                resolution=Argument(type=int, gt=0, help="Output resolution."),
            ),
            "i2i": dict(
                IMAGE,
                image_path=Argument(help="Input image path."),
                resolution=Argument(type=int, gt=0, help="Output resolution."),
            ),
        }
    ),
}
```

CLI 和 HTTP 都根据这份 Spec 生成请求参数、校验规则和文档，`BaseRunner` 也从同一份 Spec 校验任务与请求字段。模型专属参数直接定义在自己的 `request_spec.py` 中，无需修改公共参数表，或在 runner、服务端 Schema 中重复声明。当前实例实际接受的任务和字段还受启动任务及模型配置限制。涉及模型初始化的参数，需要先确认模型支持在运行时修改。

参数与某个启动设置不兼容时，仍在 Spec 中保留声明，在 runner 的请求准备或推理路径中检查，并报出冲突设置及解决方法。不要根据配置从 Spec 中移除这些字段。

已声明支持的 `negative_prompt` 是一个例外：关闭 CFG 时，显式传入该字段只记录警告并忽略请求值，不报错。

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
| Python | 通常无需修改；`generate(resolution=768)` 通过 `**task_inputs` 转发新字段。 |
| CLI | 第 2 步中的 `Argument` 自动生成选项和帮助，无需修改 `infer.py`。 |
| 原生 HTTP JSON / 表单 | 服务启动时从同一份 Spec 生成请求模型，无需额外声明字段。 |
| OpenAI 图片接口 | 推理字段从 Spec 生成；协议字段继续放在 OpenAI 适配层。 |

`Argument.type` 表示完整的参数类型，例如 `int`、`list[int]`、`str | dict`。CLI 将列表转换为空格分隔的选项，将对象转换为 JSON 输入；HTTP 接收对应的 JSON 值。`gt`、`ge`、`lt`、`le` 定义两端共用的数值范围。

使用 `cli=False` 或 `server=False` 声明入口差异；Python 回调、内存张量等内部输入应同时关闭两者。NeoPP 的 CLI 初始化参数仍通过自己的 `get_cli_arguments()` 声明。

CLI 和 HTTP 的请求覆盖值默认都为 `None`，实际默认值继续由启动配置和 `InputInfo` 决定。HTTP 按 `model_fields_set` 转发显式提交的字段，不会用请求模型的构造默认值覆盖启动配置。

服务为每个支持的任务生成请求模型，多任务请求通过 `task` 选择，单任务请求可以省略它。`/docs` 展示当前模型的字段，而不是所有模型的参数集合。

OpenAI 图片接口共用 Spec 中的推理字段，同时保留必填 `prompt`、`n`、`response_format` 和 `size="WIDTHxHEIGHT"` 等协议约定。格式转换和新增协议控制仍需调整适配器。

列表或对象在表单中使用 JSON 文本，例如 `size=[512,768]`。新增文件上传还需声明端点的 `File(...)` 参数、保存文件，并在 `form_request_openapi()` 中补充上传字段文档；普通服务端文件路径无需这些处理。

## 5. 验证效果

- **默认值与隔离**：检查未传、`None`、显式值，以及下一次不传时是否恢复预期默认值。
- **支持范围**：已公开的字段在目标模型、任务中生效；不支持的组合明确报错。
- **入口一致性**：从需要支持的每个入口传入，检查 runner 收到的值。
- **实际效果**：检查模型计算是否使用新值，以及与其他参数的优先关系。

例如，非空的 `size` 会优先决定 Qwen-Image-2.1 的输出尺寸，即使它来自启动配置。验证 `resolution` 对自动输出尺寸的影响时，应同时检查 `size`，不能只确认接口是否接受参数。
