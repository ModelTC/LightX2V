# 服务部署

LightX2V 通过 FastAPI 提供带异步任务队列的推理服务，启动入口为 `python -m lightx2v.server`。启动时选择的模型和任务决定请求支持哪些字段，详见[新增请求参数](../developer_guides/request_parameters.md)。

## 启动服务

在仓库根目录下，先修改 `scripts/server/start_server.sh` 中的 `lightx2v_path` 和 `model_path`，再运行：

```bash
bash scripts/server/start_server.sh
```

示例使用 Wan2.1 T2V、`configs/wan/wan_t2v.json`，监听 8000 端口。切换模型时，需要配套修改 `--model_cls`、`--task` 和 `--config_json`。`--max_queue_size` 默认为 10，统计等待中和执行中的任务总数。

## 提交视频任务

服务就绪后，在另一个终端提交请求。以下示例适用于默认 T2V 服务：

```bash
curl -X POST http://localhost:8000/v1/tasks/video/ \
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

结果接口仅提供服务配置的输出目录内的文件。使用上例中的相对输出文件名，可将结果保存在该目录中。

`GET /v1/service/status` 的 `busy` / `idle` 表示执行状态。只要队列还有容量，忙碌的服务也能接受新请求。`GET /v1/tasks/queue/status` 提供 `pending_count`、`active_count`（等待中与执行中之和）、`queue_size`、`queue_available`；队列满时提交请求返回 HTTP 503。

```bash
curl -X DELETE http://localhost:8000/v1/tasks/TASK_ID
```

取消单个任务不会清空其他等待任务。`scripts/server/stop_running_task.py` 取消当前任务；`DELETE /v1/tasks/all/running` 取消所有等待中和执行中的任务。取消后服务仍可继续接收请求。

## 图像请求

图像模型以 `--task t2i` 或 `--task i2i` 启动后，使用 `/v1/tasks/image/`（JSON）或 `/v1/tasks/image/form`（multipart）。`/v1/tasks/image/sync` 等待完成，通常直接返回 PNG 字节；设置 `presigned_url` 时会上传结果并返回 JSON。该接口的查询参数 `timeout_seconds` 默认为 600。

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

该工具目前向视频任务的兼容路由 `POST /v1/tasks/` 提交请求，不能直接作为图像任务客户端使用。

## 接口索引

| 接口 | 方法 | 用途 |
|------|------|------|
| `/v1/tasks/video/` | POST | 通过 JSON 提交视频任务 |
| `/v1/tasks/video/form` | POST | 通过 multipart 上传并提交视频任务 |
| `/v1/tasks/image/` | POST | 通过 JSON 提交图像任务 |
| `/v1/tasks/image/form` | POST | 通过 multipart 上传并提交图像任务 |
| `/v1/tasks/image/sync` | POST | 等待图像生成结果 |
| `/v1/tasks/` | GET | 列出任务 |
| `/v1/tasks/queue/status` | GET | 查询队列容量 |
| `/v1/tasks/{task_id}/status` | GET | 查询单个任务 |
| `/v1/tasks/{task_id}/result` | GET | 下载已完成且保存为文件的结果 |
| `/v1/tasks/{task_id}` | DELETE | 取消单个任务 |
| `/v1/tasks/all/running` | DELETE | 取消所有等待中/执行中的任务 |
| `/v1/service/status` | GET | 查询服务执行状态 |
| `/v1/service/metadata` | GET | 查询模型/任务能力 |
| `/v1/files/download/{file_path}` | GET | 下载允许访问的输出文件 |
| `/v1/images/generations` | POST | OpenAI 兼容图像生成 |
| `/v1/images/edits` | POST | OpenAI 兼容图像编辑 |

`POST /v1/tasks/` 仍保留为视频提交的弃用别名。新客户端应选择明确的视频或图像路由；`/v1/tasks/form` 未注册。
