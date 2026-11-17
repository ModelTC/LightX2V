# 新增请求参数

新增请求参数时，先让模型使用这个值，再接通需要的调用入口。下面以已实现的 Qwen-Image-2.1 `resolution` 为例；代码片段用于定位，新参数按相同步骤添加。

`InputInfo` 保存单次生成的参数，runner 负责使用这些参数执行模型推理。

## 1. 在 Spec 中定义参数

在模型自己的 `request_spec.py` 中，为需要支持的任务声明 `Argument`，保留原有字段。Qwen-Image-2.1 的声明位于 `lightx2v/models/runners/qwen_image_21/request_spec.py`：

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

Spec 维护任务、输入参数和上下文默认值，不声明输出类型。统一接口 `POST /v1/tasks/` 接受当前部署模型的请求；`/docs` 根据任务和参数生成。旧图片、视频接口与 OpenAI 图片接口共用请求定义，通过实际生成的产物检查输出类型。

CLI 和 HTTP 都根据这份 Spec 生成请求参数、校验规则和文档，`BaseRunner` 也从同一份 Spec 校验任务与请求字段。模型专属参数直接定义在自己的 `request_spec.py` 中，无需修改公共参数表，或在 runner、服务端 Schema 中重复声明。当前实例实际接受的任务和字段还受启动任务及模型配置限制。涉及模型初始化的参数，需要先确认模型支持在运行时修改。

Spec 只声明对外提供的任务。例如 HunyuanImage3 的文本生成用于图像流程内部的 prompt 增强，对外只提供 `t2i`、`ti2i`、`i2i`。

同一套已加载权重支持多个请求任务时，使用 `MultiTaskSpec`。runner 和 scheduler 在推理时读取 `input_info.task`，初始化时加载各任务需要的组件，并在请求之间清理状态。不要通过每次修改 `config["task"]` 来切换任务，以免将部署设置与请求状态混在一起。

参数与某个启动设置不兼容时，仍在 Spec 中保留声明，在 runner 的请求准备或推理路径中检查，并报出冲突设置及解决方法。不要根据配置从 Spec 中移除这些字段。

已声明支持的 `negative_prompt` 是一个例外：关闭 CFG 时，显式传入该字段只记录警告并忽略请求值，不报错。

## 2. 自动创建 InputInfo

`Argument` 是请求字段的唯一来源。`type` 定义类型，`input_info_default` 定义创建 InputInfo 时的初始值，例如 `input_info_default=""`、`input_info_default=[]` 或 `input_info_default=[1.0]`。创建 InputInfo 时会深拷贝每个默认值，保证不同请求的列表和字典独立，包括嵌套容器。

`BaseRunner.get_input_info_type(task)` 从 Spec 读取字段，与 runner 的 `input_info_state` 合成 dataclass，并在该 runner 中缓存类型。每次请求创建独立实例，先应用启动配置，再应用本次请求。新增参数无需修改 InputInfo 类或全局任务类型表。

CLI/server 未提交的字段仍为 `None`，表示不覆盖启动配置；不会直接使用 `Argument.input_info_default` 作为请求覆盖值。

在启动 JSON 的 `request_defaults` 子字典中设置请求默认值。例如 Qwen-Image-2.1：

```json
{
    "infer_steps": 40,
    "request_defaults": {
        "resolution": 1024,
        "seed": 42
    }
}
```

这里只放当前模型的 CLI 和 HTTP 共同支持的可序列化请求字段。模型名称、`task`、采样步数、CFG、算子、量化、offload 和并行配置保留顶层。受启动模式限制的固定字段也保留顶层，例如固定帧数 attention 下的 Wan `num_frames`。同一个字段不能同时出现在顶层和 `request_defaults` 中。

模型配置加载器校验字段名称和类型，始终将这些值保留在 `request_defaults` 中。checkpoint 元数据中的同类请求默认值也归入这个子字典。`runner.config`、`pipe.config` 保持相同的结构：固定参数在顶层，请求默认值在子字典；初始化和预热显式读取子字典。这些默认值不会被加入显式请求字典，因此保留 LTX-2.5 自动时长和 Cosmos3 动作文件的优先级。配置中显式的 `null` 保留原值，包括 EdenLive 的自动帧数标记。

CLI 和 POST 请求格式保持不变，例如 `--resolution 768` 和 `{"resolution": 768}`，POST 不包裹 `request_defaults`。显式请求值覆盖配置默认值；请求省略字段或传入 `None` 时保留默认值。创建每次请求时会复制列表和字典。

公共的 `ShapeState`、`ImageState`、`AudioVideoState` 和 `SRState` 放在 `lightx2v/utils/input_info.py`。模型专属运行状态放在自己的 runner 目录，例如：

```python
@dataclass
class ModelState(ShapeState):
    encoded_condition: Any = field(default=None, repr=False)

class ModelRunner(DefaultRunner):
    input_info_state = ModelState
```

状态类只声明运行状态，不再重复声明 Spec 中的请求字段。需要从启动配置保留、但不允许请求覆盖的字段，可以通过 `RequestSpec(..., context={...})` 提供 `Argument` 定义；优先复用公共或模型已有的定义。生成的上下文也保留同一模型其他任务的字段，便于共享执行路径使用；对外允许哪些字段仍由当前任务的 `get_arguments()` 决定。

预热或内部流程需要直接创建上下文时，使用 `runner.get_input_info_type(task)(...)`；正常请求使用 `prepare_request()`。

## 3. 在推理中使用参数

在实际计算位置读取本次请求的值：

```python
resolution = self.input_info.resolution
```

配置默认值位于 `self.config["request_defaults"]["resolution"]`，推理时直接读取它会忽略请求覆盖。实际计算应读取 `self.input_info.resolution`。Qwen-Image-2.1 在 `run_input_encoder()` 中读取并校验这个有效值，用于参考图缩放和自动输出尺寸。

`BaseRunner` 先初始化固定上下文字段，再应用 `request_defaults`，最后应用非 `None` 的请求值。例如启动配置包含 `"request_defaults": {"resolution": 1024}`：

| 本次请求 | 模型读到的值 |
|---|---|
| 不传、传 `None` 或 JSON `null` | `1024` |
| 传 `768` | `768` |

配置也未提供时，保留 Spec 中声明的运行默认值；模型可在请求准备或预处理中进一步处理。必要的类型、范围和参数组合校验也应放在这些位置，让所有入口都经过检查。

runner 合并参数时保留 `0`、`False`、`""`、`[]`。它们是否有效由字段规则决定，不能用 `if value` 把它们当作未传值。

## 4. 接通调用入口

| 入口 | 修改位置 |
|---|---|
| Python | 通常无需修改；`generate(resolution=768)` 通过 `**task_inputs` 转发新字段。 |
| CLI | 第 1 步中的 `Argument` 自动生成选项和帮助，无需修改 `infer.py`。 |
| 原生 HTTP JSON / 表单 | 服务启动时从同一份 Spec 生成请求模型，无需额外声明字段。 |
| OpenAI 图片接口 | 推理字段从 Spec 生成；协议字段继续放在 OpenAI 适配层。 |

多输入参数统一声明为 `list[T]`，例如 `Argument(type=list[str])`。CLI 使用空格分隔，JSON 请求与启动配置使用数组，Python 调用传列表；即使只有一项，也保留列表。runner 直接使用列表，不再拆分逗号字符串。

| 参数 | CLI | JSON / Python |
|---|---|---|
| 多张参考图 | `--image_path a.png b.png` | `"image_path": ["a.png", "b.png"]` |
| 条件强度 | `--image_strength 1.0 0.8` | `"image_strength": [1.0, 0.8]` |
| 帧索引 | `--image_frame_indices 0 32` | `"image_frame_indices": [0, 32]` |

只有支持多输入的字段声明为列表；单图字段仍使用 `str`。`layout_bboxes`、`motions` 等嵌套结构在 CLI 中使用 JSON 文本，HTTP 直接传数组或对象。`layout_bboxes` 也可以传 JSON 文件路径。

表单中的列表和对象使用 JSON 文本；文件上传对列表字段重复提交同名文件字段，OpenAPI 会显示为文件数组。上传后保持列表类型及顺序。

`Argument.type` 表示完整的参数类型，例如 `int`、`list[int]`、`str | dict`。CLI 将简单列表转换为空格分隔的选项，将嵌套列表和对象转换为 JSON 输入；HTTP 接收对应的 JSON 值。`gt`、`ge`、`lt`、`le` 定义两端共用的数值范围。

使用 `cli=False` 或 `server=False` 声明入口差异；Python 回调、内存张量等内部输入应同时关闭两者。NeoPP 的 CLI 初始化参数仍通过自己的 `get_cli_arguments()` 声明。

CLI 和 HTTP 的请求覆盖值默认都为 `None`，实际默认值继续由启动配置和 `InputInfo` 决定。HTTP 按 `model_fields_set` 转发显式提交的字段，不会用请求模型的构造默认值覆盖启动配置。

服务按部署模型支持的任务生成请求模型，多任务请求通过必填的 `task` 选择。单任务请求模型不包含 `task`，传入会返回 HTTP 422，由 runner 使用启动时绑定的任务。`/docs` 展示当前模型的字段，而不是所有模型的参数集合。

OpenAI 图片接口共用 Spec 中的推理字段，同时保留必填 `prompt`、`n`、`response_format` 和 `size="WIDTHxHEIGHT"` 等协议约定。格式转换和新增协议控制仍需调整适配器。

列表或对象在表单中使用 JSON 文本，例如 `size=[512,768]`。新增文件上传还需声明端点的 `File(...)` 参数、保存文件，并在 `form_request_openapi()` 中补充上传字段文档；普通服务端文件路径无需这些处理。

## 5. 验证效果

- **默认值与隔离**：检查未传、`None`、显式值，以及下一次不传时是否恢复预期默认值。
- **支持范围**：已公开的字段在目标模型、任务中生效；不支持的组合明确报错。
- **入口一致性**：从需要支持的每个入口传入，检查 runner 收到的值。
- **实际效果**：检查模型计算是否使用新值，以及与其他参数的优先关系。

例如，非空的 `size` 会优先决定 Qwen-Image-2.1 的输出尺寸，即使它来自启动配置。验证 `resolution` 对自动输出尺寸的影响时，应同时检查 `size`，不能只确认接口是否接受参数。

## 登记生成结果

普通单文件图片、视频 runner 可复用 `BaseRunner.get_outputs()`，从返回的 `image`、`images` 或 `video` 字段及实际保存路径收集结果。多文件、联合输出或保存路径发生变化时，在保存成功后登记真实路径：

```python
self.record_output("video", video_path)
self.record_output("action", action_path)
```

这不会改变 `run_pipeline()` 的返回结构或 `InputInfo`。服务通过 `/v1/tasks/{task_id}/results` 展示实际结果，每项有独立下载地址。仅生成 action 的模型也使用统一入口。旧图片、视频接口缺少所需产物时，任务会在执行后失败；不再通过静态 Spec 在入队前筛选。
