# LightX2V `sync` 接口调用方法

本文档说明如何调用 `POST /v1/tasks/image/sync` 接口。

## 1. 接口说明

- **接口**：`POST /v1/tasks/image/sync`
- **用途**：同步生成图片（服务端处理完成后直接返回结果）
- **Query 参数**：
  - `timeout_seconds`（可选，默认 `600`）
  - `poll_interval_seconds`（可选，默认 `0.5`）

---

## 2. 场景A：不传 `presigned_url`（直接返回 PNG 二进制流）

### curl

```bash
curl -X POST "http://127.0.0.1:8000/v1/tasks/image/sync?timeout_seconds=600&poll_interval_seconds=0.5" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute cat, studio light",
    "negative_prompt": "",
    "infer_steps": 30,
    "seed": 42,
    "aspect_ratio": "16:9"
  }' \
  --output result.png
```

### Python

```python
import requests

url = "http://127.0.0.1:8000/v1/tasks/image/sync"
params = {"timeout_seconds": 600, "poll_interval_seconds": 0.5}
payload = {
    "prompt": "a cute cat, studio light",
    "negative_prompt": "",
    "infer_steps": 30,
    "seed": 42,
    "aspect_ratio": "16:9",
}

resp = requests.post(url, params=params, json=payload, timeout=630)
resp.raise_for_status()

with open("result.png", "wb") as f:
    f.write(resp.content)
print("saved: result.png")
```

---

## 3. 场景B：传 `presigned_url`（服务端上传结果，接口返回 JSON）

### curl

```bash
curl -X POST "http://127.0.0.1:8000/v1/tasks/image/sync?timeout_seconds=600&poll_interval_seconds=0.5" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute cat, studio light",
    "negative_prompt": "",
    "infer_steps": 30,
    "seed": 42,
    "aspect_ratio": "16:9",
    "presigned_url": "https://your-presigned-put-url"
  }'
```

预期返回（示例）：

```json
{
  "task_id": "xxxx",
  "task_status": "completed",
  "uploaded_to_presigned_url": true,
  "presigned_url": "https://your-presigned-put-url"
}
```

### Python

```python
import requests

url = "http://127.0.0.1:8000/v1/tasks/image/sync"
params = {"timeout_seconds": 600, "poll_interval_seconds": 0.5}
payload = {
    "prompt": "a cute cat, studio light",
    "negative_prompt": "",
    "infer_steps": 30,
    "seed": 42,
    "aspect_ratio": "16:9",
    "presigned_url": "https://your-presigned-put-url",
}

resp = requests.post(url, params=params, json=payload, timeout=630)
resp.raise_for_status()
print(resp.json())
```

---

## 4. 常见问题

- `403`：通常是 presigned URL 签名与请求不匹配（`region`、`endpoint`、`addressing_style`、过期时间等）。
- 返回 5xx：检查服务端日志，确认模型推理流程是否正常。
- 下载结果失败：确认对象存储权限，以及是否使用了可读 URL（例如 GET presigned URL）。
