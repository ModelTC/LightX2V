# 服务部署

LightX2V 通过 FastAPI 提供带异步任务队列的推理服务，启动入口为 `python -m lightx2v.server`。启动时选择的模型和任务决定请求支持哪些字段，详见[新增请求参数](../developer_guides/request_parameters.md)。

## 启动服务

在仓库根目录下，先修改 `scripts/server/start_server.sh` 中的 `lightx2v_path` 和 `model_path`，再运行：

```bash
bash scripts/server/start_server.sh
```

示例使用 Wan2.1 T2V、`configs/wan/wan_t2v.json`，监听 8000 端口。切换模型时，需要配套修改 `--model_cls`、`--task` 和 `--config_json`。`--max_queue_size` 默认为 10，统计等待中和执行中的任务总数。

单任务服务在启动时绑定任务，JSON 和表单请求都不能包含 `task`。MiniMax-H3 等多任务服务则需要每次在原生请求中传入 `task`。运行中服务的 `/docs` 会展示对应的请求结构。

统一入口 `POST /v1/tasks/` 支持图片、视频和 action，参数与文档只由当前模型的 Spec 决定。`POST /v1/tasks/form` 支持上传文件；上传字段按任务支持的输入生成。原有图片、视频和 OpenAI 图片接口继续兼容，共用同一个任务队列与推理服务。

BAGEL、Flux2、HiDream-O1-Image、Z-Image、LingBot-Video、HunyuanImage3、LTX2/LTX2.5、Wan2.2 TI2V-5B 和 Cosmos3 Omni 支持请求级任务：启动时省略 `--task`，每次原生请求指定 `task`。HunyuanImage3 只开放图像任务；Cosmos3 在启动 JSON 中通过 `model_variant` 区分模型版本。

## 提交视频任务

服务就绪后，在另一个终端提交请求。以下示例适用于默认 T2V 服务：

```bash
curl -X POST http://localhost:8000/v1/tasks/ \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Two cats boxing on a stage.","seed":42,"save_result_path":"cat_boxing_seed42.mp4"}'
```

响应包含 `task_id`、`task_status: "pending"` 和 `save_result_path`。请保存任务 ID 用于轮询。I2V 需使用匹配的模型与配置启动服务，并提供 `image_path`，或通过 multipart 表单上传图片。本地输入路径指服务端文件系统中的路径。

可选生成字段省略或传入 `null` 时继承启动默认值，同时仍需符合当前模型/任务支持的范围。关闭 CFG 时，显式传入的 `negative_prompt`（包括空字符串）会被忽略，并记录警告。JSON 中的 `size` 按 `[高, 宽]` 表示。

## 查询、下载与取消

将下面的 `TASK_ID` 替换为返回的任务 ID：

```bash
curl http://localhost:8000/v1/tasks/TASK_ID/status
curl http://localhost:8000/v1/tasks/queue/status
```

也可使用 `scripts/server/check_status.py` 查询服务状态、任务列表和单个任务。运行前修改其中的服务地址，并将 `test_task_001` 替换为实际任务 ID：

```bash
python scripts/server/check_status.py
```

任务状态包括 `pending`、`processing`、`completed`、`failed`、`cancelled`。失败时应检查响应中的错误信息。任务完成且结果已保存为文件后，可下载结果：

```bash
curl -f http://localhost:8000/v1/tasks/TASK_ID/result -o output.mp4
```

旧 `/result` 接口继续下载单个主文件，并限制在服务输出目录内。完整结果列表与各产物的下载地址通过 `/results` 获取：

```bash
curl -f http://localhost:8000/v1/tasks/TASK_ID/results
# 根据列表中返回的 url 下载具体产物，例如：
curl -f http://localhost:8000/v1/tasks/TASK_ID/results/actions -o actions.npy
```

每项包含 `name`、`type`、`media_type`、`url`。联合输出会同时列出视频和 action，下载不会重新执行推理。未指定保存路径时，异步服务自动分配任务输出路径；`save_result_path` 继续表示主结果路径。内存图片保留现有缓存数量限制，已被回收的结果下载返回 HTTP 410。

`GET /v1/service/status` 的 `busy` / `idle` 表示执行状态。只要队列还有容量，忙碌的服务也能接受新请求。`GET /v1/tasks/queue/status` 提供 `pending_count`、`active_count`（等待中与执行中之和）、`queue_size`、`queue_available`；队列满时提交请求返回 HTTP 503。

```bash
curl -X DELETE http://localhost:8000/v1/tasks/TASK_ID
```

取消单个任务不会清空其他等待任务。`scripts/server/stop_running_task.py` 取消当前任务；`DELETE /v1/tasks/all/running` 取消所有等待中和执行中的任务。取消后服务仍可继续接收请求。

## 图像请求

图像任务可以直接提交到统一入口。也可以继续使用 `/v1/tasks/image/`（JSON）或 `/v1/tasks/image/form`（multipart）。`/v1/tasks/image/sync` 等待完成，通常直接返回 PNG 字节；设置 `presigned_url` 时会上传结果并返回 JSON。该接口的查询参数 `timeout_seconds` 默认为 600。

另提供 OpenAI 兼容接口 `/v1/images/generations` 和 `/v1/images/edits`。可在运行中服务的 `/docs` 查看请求结构，在 `/v1/service/metadata` 查看模型/任务能力。

## 多 GPU 与多服务

分布式推理服务使用 `torchrun --nproc_per_node N -m lightx2v.server`，并配套设置并行配置；各 rank 协作处理同一个服务的任务，详见[并行推理](../method_tutorials/parallel.md)。

指标服务默认启动，分布式服务由 rank 0 启动指标监听。未指定 `--metric_port` 时，使用环境变量 `LIGHTX2V_METRIC_PORT`，其默认值为 `8001`；省略该参数不会禁用指标服务。

运行多个独立服务时，分别启动进程并配置模型和路径，设置不同的 `CUDA_VISIBLE_DEVICES`。在同一主机网络或网络命名空间内，还需为各服务分配互不冲突的 API 和指标端口，例如第一个服务使用 `--port 8000 --metric_port 18000`，第二个使用 `--port 8002 --metric_port 18002`。只修改 API 端口会造成默认指标端口冲突。

当前 `start_multi_servers.sh` 是带占位配置的特定模型分布式示例，并不是由 `num_gpus` 控制的通用多服务启动器。

### 客户端分发多个请求

`scripts/server/post_multi_servers.py` 保留了多服务客户端工具函数。它查询服务状态，将请求交给空闲服务，再轮询任务结果。等待空闲是该客户端的调度策略，服务本身仍可在队列未满时接收请求。

该文件没有直接执行的入口；从仓库根目录启动 Python，调用其中的函数。将地址替换为已启动的同类视频服务，按模型支持的字段构造请求：

```python
from scripts.server.post_multi_servers import get_available_urls, process_tasks_async

urls = ["http://localhost:8000", "http://localhost:8002"]
messages = [
    {"prompt": "A cat walking on a beach.", "seed": 42, "save_result_path": "cat.mp4"},
    {"prompt": "A dog running in a park.", "seed": 43, "save_result_path": "dog.mp4"},
]
available_urls = get_available_urls(urls)
if not available_urls:
    raise RuntimeError("No reachable inference services")
all_succeeded = process_tasks_async(messages, available_urls)
```

该工具向统一入口 `POST /v1/tasks/` 提交请求。切换模型时仍需按服务的任务选择规则和参数 Schema 构造请求。

## 接口索引

| 接口 | 方法 | 用途 |
|------|------|------|
| `/v1/tasks/` | POST | 统一提交图片、视频和 action 任务 |
| `/v1/tasks/form` | POST | 上传文件并统一提交任务 |
| `/v1/tasks/video/` | POST | 通过 JSON 提交视频任务 |
| `/v1/tasks/video/form` | POST | 通过 multipart 上传并提交视频任务 |
| `/v1/tasks/image/` | POST | 通过 JSON 提交图像任务 |
| `/v1/tasks/image/form` | POST | 通过 multipart 上传并提交图像任务 |
| `/v1/tasks/image/sync` | POST | 等待图像生成结果 |
| `/v1/tasks/` | GET | 列出任务 |
| `/v1/tasks/queue/status` | GET | 查询队列容量 |
| `/v1/tasks/{task_id}/status` | GET | 查询单个任务 |
| `/v1/tasks/{task_id}/results` | GET | 获取完整产物列表 |
| `/v1/tasks/{task_id}/results/{name}` | GET | 下载指定产物 |
| `/v1/tasks/{task_id}/result` | GET | 下载已完成且保存为文件的结果 |
| `/v1/tasks/{task_id}` | DELETE | 取消单个任务 |
| `/v1/tasks/all/running` | DELETE | 取消所有等待中/执行中的任务 |
| `/v1/service/status` | GET | 查询服务执行状态 |
| `/v1/service/metadata` | GET | 查询模型/任务能力 |
| `/v1/files/download/{file_path}` | GET | 下载允许访问的输出文件 |
| `/v1/images/generations` | POST | OpenAI 兼容图像生成 |
| `/v1/images/edits` | POST | OpenAI 兼容图像编辑 |

兼容入口保持现有请求和响应格式。`/image/` 与 `/video/` 要求实际产物包含对应类型，类型不匹配时在执行后将任务标记为失败；同步图片与 OpenAI 图片接口直接返回错误。统一入口不按静态输出声明筛选任务，也不要求传入输出类型。
