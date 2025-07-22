# 服务化部署

lightx2v 提供异步服务功能。代码入口点在 [这里](https://github.com/ModelTC/lightx2v/blob/main/lightx2v/api_server.py)

### 启动服务

```shell
# 修改脚本中的路径
bash scripts/start_server.sh
```

`--port 8000` 选项表示服务将绑定到本地机器的 `8000` 端口。您可以根据需要更改此端口。

### 客户端发送请求

```shell
python scripts/post.py
```

服务端点：`/v1/tasks/`

`scripts/post.py` 中的 `message` 参数如下：

```python
message = {
    "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "image_path": ""
}
```

1. `prompt`、`negative_prompt` 和 `image_path` 是视频生成的基本输入。`image_path` 可以是空字符串，表示不需要图像输入。


### 客户端检查服务器状态

```shell
python scripts/check_status.py
```

服务端点包括：

1. `/v1/service/status` 用于检查服务状态。返回服务是 `busy` 还是 `idle`。服务只有在 `idle` 时才接受新请求。

2. `/v1/tasks/` 用于获取服务器接收和完成的所有任务。

3. `/v1/tasks/{task_id}/status` 用于获取指定 `task_id` 的任务状态。返回任务是 `processing` 还是 `completed`。

### 客户端随时停止服务器上的当前任务

```shell
python scripts/stop_running_task.py
```

服务端点：`/v1/tasks/running`

终止任务后，服务器不会退出，而是返回等待新请求的状态。

### 在单个节点上启动多个服务

在单个节点上，您可以使用 `scripts/start_server.sh` 启动多个服务（注意同一 IP 下的端口号必须不同），或者可以使用 `scripts/start_multi_servers.sh` 同时启动多个服务：

```shell
num_gpus=8 bash scripts/start_multi_servers.sh
```

其中 `num_gpus` 表示要启动的服务数量；服务将从 `--start_port` 开始在连续端口上运行。

### 多个服务之间的调度

```shell
python scripts/post_multi_servers.py
```

`post_multi_servers.py` 将根据服务的空闲状态调度多个客户端请求。

### API 端点总结

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/tasks/` | POST | 创建视频生成任务 |
| `/v1/tasks/form` | POST | 通过表单创建视频生成任务 |
| `/v1/tasks/` | GET | 获取所有任务列表 |
| `/v1/tasks/{task_id}/status` | GET | 获取指定任务状态 |
| `/v1/tasks/{task_id}/result` | GET | 获取指定任务的结果视频文件 |
| `/v1/tasks/running` | DELETE | 停止当前运行的任务 |
| `/v1/files/download/{file_path}` | GET | 下载文件 |
| `/v1/service/status` | GET | 获取服务状态 |
