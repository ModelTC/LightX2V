# Service Deployment

LightX2V provides a FastAPI service with asynchronous task queues. Start it with `python -m lightx2v.server`. The startup model and task determine which request fields are supported; see [Adding Request Parameters](../developer_guides/request_parameters.md).

## Start the service

From the repository root, set `lightx2v_path` and `model_path` in `scripts/server/start_server.sh`, then run:

```bash
bash scripts/server/start_server.sh
```

The example loads Wan2.1 T2V with `configs/wan/wan_t2v.json` and listens on port 8000. Change `--model_cls`, `--task`, and `--config_json` together when selecting another model. `--max_queue_size` defaults to 10 and counts both pending and processing tasks.

Single-task services bind their task at startup; JSON and form requests must not include `task`. Multi-task services such as MiniMax-H3 require `task` in each native request. `/docs` shows the request schema for the running service.

`POST /v1/tasks/` submits image, video and action tasks. The deployed model's Spec determines request fields and documentation, without static output declarations. `POST /v1/tasks/form` adds uploads documented from each task's supported inputs. Legacy image/video routes and OpenAI image endpoints share the same queue and inference service.

BAGEL, Flux2, HiDream-O1-Image, Z-Image, LingBot-Video, HunyuanImage3, LTX2/LTX2.5, Wan2.2 TI2V-5B and Cosmos3 Omni support per-request tasks. Omit startup `--task` and select a task in each native request. HunyuanImage3 exposes image tasks only; Cosmos3 uses an explicit `model_variant` in the startup JSON.

## Submit a video task

Once the service is ready, submit a request from another terminal. This example uses the default T2V service:

```bash
curl -X POST http://localhost:8000/v1/tasks/ \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Two cats boxing on a stage.","seed":42,"save_result_path":"cat_boxing_seed42.mp4"}'
```

The response includes `task_id`, `task_status: "pending"`, and `save_result_path`. Keep the returned task ID for polling. For I2V, start the service with a matching I2V model/config and provide `image_path` or use the multipart form endpoint. Local input paths refer to the server filesystem.

Optional generation fields inherit the startup defaults when omitted or set to `null`; fields must still be supported by the selected model/task. When CFG is disabled, an explicitly supplied `negative_prompt`, including an empty string, is ignored with a warning. A JSON `size` uses `[height, width]`.

## Poll, download, and cancel

Replace `TASK_ID` below with the returned ID:

```bash
curl http://localhost:8000/v1/tasks/TASK_ID/status
curl http://localhost:8000/v1/tasks/queue/status
```

The repository also provides `scripts/server/check_status.py` to inspect service status, the task list, and an individual task. Set the service URLs and replace its `test_task_001` placeholder with a real task ID before running:

```bash
python scripts/server/check_status.py
```

Task states are `pending`, `processing`, `completed`, `failed`, and `cancelled`. Inspect the error information if a task fails. After a file-backed task completes, download its result:

```bash
curl -f http://localhost:8000/v1/tasks/TASK_ID/result -o output.mp4
```

The legacy `/result` endpoint downloads the primary file under the configured output directory. Query `/results` for all artifacts and their download URLs:

```bash
curl -f http://localhost:8000/v1/tasks/TASK_ID/results
# Download a specific artifact using its returned URL, for example:
curl -f http://localhost:8000/v1/tasks/TASK_ID/results/actions -o actions.npy
```

Each entry contains `name`, `type`, `media_type` and `url`. Joint generation lists both video and action artifacts; downloading does not rerun inference. Asynchronous requests without a save path receive a task-specific output path automatically. `save_result_path` continues to identify the primary result. In-memory images retain the existing cache count limit; downloads return HTTP 410 after eviction.

`GET /v1/service/status` reports execution state (`busy` or `idle`). A busy server can still accept requests while queue capacity remains. `GET /v1/tasks/queue/status` exposes `pending_count`, `active_count` (pending + processing), `queue_size`, and `queue_available`. A full queue rejects submissions with HTTP 503.

```bash
curl -X DELETE http://localhost:8000/v1/tasks/TASK_ID
```

Cancelling one task leaves other pending tasks in the queue. `scripts/server/stop_running_task.py` cancels the current task; `DELETE /v1/tasks/all/running` cancels all pending and processing tasks. The server remains available afterward.

## Image requests

For an image model started with `--task t2i` or `--task i2i`, use `/v1/tasks/image/` (JSON) or `/v1/tasks/image/form` (multipart). `/v1/tasks/image/sync` waits for completion and normally returns PNG bytes; when `presigned_url` is supplied it uploads the result and returns JSON. It accepts `timeout_seconds` as a query parameter (default 600).

OpenAI-compatible image endpoints are also available at `/v1/images/generations` and `/v1/images/edits`. Check the running service's `/docs` for request schemas and `/v1/service/metadata` for model/task capabilities.

## Multiple GPUs and multiple services

A distributed inference service uses `torchrun --nproc_per_node N -m lightx2v.server` with a matching parallel configuration; all ranks cooperate on one service. See [Parallel Inference](../method_tutorials/parallel.md).

The metrics server starts by default; rank 0 starts it for a distributed service. If `--metric_port` is omitted, the server uses `LIGHTX2V_METRIC_PORT`, which defaults to `8001`. Omitting the option does not disable metrics.

To run independent services, launch separate processes, configure each model and its paths, and set different `CUDA_VISIBLE_DEVICES` values. Within the same host network or network namespace, assign nonconflicting API and metrics ports to each service. For example, use `--port 8000 --metric_port 18000` for the first service and `--port 8002 --metric_port 18002` for the second. Changing only the API port leaves a conflict on the default metrics port.

The current `start_multi_servers.sh` is a model-specific distributed example with a placeholder config, not a general launcher controlled by `num_gpus`.

### Distribute requests from a client

`scripts/server/post_multi_servers.py` provides client helpers for multiple services. It checks service status, sends each request to an idle service, and polls for completion. Waiting for an idle service is this client's scheduling policy; a service itself can still accept requests while its queue has capacity.

The file has no executable entry point. Start Python from the repository root and call its functions. Replace the URLs with running video services of the same kind and use request fields supported by their model:

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

The helper currently submits to the compatibility route `POST /v1/tasks/` for video tasks. It cannot be used directly as an image-task client.

## API reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/tasks/` | POST | Submit image, video or action tasks |
| `/v1/tasks/form` | POST | Upload files and submit tasks |
| `/v1/tasks/video/` | POST | Submit a video task as JSON |
| `/v1/tasks/video/form` | POST | Submit a video task with multipart uploads |
| `/v1/tasks/image/` | POST | Submit an image task as JSON |
| `/v1/tasks/image/form` | POST | Submit an image task with multipart uploads |
| `/v1/tasks/image/sync` | POST | Wait for an image result |
| `/v1/tasks/` | GET | List tasks |
| `/v1/tasks/queue/status` | GET | Inspect queue capacity |
| `/v1/tasks/{task_id}/status` | GET | Inspect a task |
| `/v1/tasks/{task_id}/results` | GET | List all generated artifacts |
| `/v1/tasks/{task_id}/results/{name}` | GET | Download a specific artifact |
| `/v1/tasks/{task_id}/result` | GET | Download a completed file-backed result |
| `/v1/tasks/{task_id}` | DELETE | Cancel one task |
| `/v1/tasks/all/running` | DELETE | Cancel all pending/processing tasks |
| `/v1/service/status` | GET | Inspect service execution state |
| `/v1/service/metadata` | GET | Inspect model/task capabilities |
| `/v1/files/download/{file_path}` | GET | Download an allowed output file |
| `/v1/images/generations` | POST | OpenAI-compatible image generation |
| `/v1/images/edits` | POST | OpenAI-compatible image editing |

Compatibility endpoints retain their request and response formats. Image/video endpoints require an artifact of the corresponding type; a mismatch fails the task after execution. Synchronous image and OpenAI endpoints return an error directly. The unified endpoint uses no static output declarations or output-type request selector.
