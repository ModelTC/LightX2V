# LightX2V `sync` 接口调用方法

本文档说明如何调用 `POST /v1/tasks/image/sync` 接口。

文本、输入媒体和 seed 由请求提供，seed 省略时使用 42。尺寸、帧数、宽高比等规格可继承部署 JSON 的默认值，并由请求覆盖。`negative_prompt` 省略时不发送该字段；显式值（包括空字符串）会按当前模型的能力校验，开启 CFG 的模型仍可能对空字符串应用默认模板。省略 `save_result_path` 或传 null 时不保存文件，同步图片接口继续从内存返回图片。

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
