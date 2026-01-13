# Service Deployment

lightx2v provides asynchronous service functionality. The code entry point is [here](https://github.com/ModelTC/LightX2V/blob/main/lightx2v/server/main.py)

### Start the Service

```shell
# Modify the paths in the script
bash scripts/server/start_server.sh
```

The `--port 8000` option means the service will bind to port `8000` on the local machine. You can change this as needed.

### Client Sends Request

```shell
python scripts/server/post.py
```

The service endpoint is: `/v1/tasks/`

The `message` parameter in `scripts/server/post.py` is as follows:

```python
message = {
    "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "image_path": "",
    "target_shape": [720, 720],
}
```

1. `prompt`, `negative_prompt`, and `image_path` are basic inputs for video generation. `image_path` can be an empty string, indicating no image input is needed. `target_shape` optional output video resolution, defaults to the configured resolution.


### Client Checks Server Status

```shell
python scripts/server/check_status.py
```

The service endpoints include:

1. `/v1/service/status` is used to check the status of the service. It returns whether the service is `busy` or `idle`. The service only accepts new requests when it is `idle`.

2. `/v1/tasks/` is used to get all tasks received and completed by the server.

3. `/v1/tasks/{task_id}/status` is used to get the status of a specified `task_id`. It returns whether the task is `processing` or `completed`.

### Client Stops the Current Task on the Server at Any Time

```shell
python scripts/server/stop_running_task.py
```

The service endpoint is: `/v1/tasks/running`

After terminating the task, the server will not exit but will return to waiting for new requests.

### Starting Multiple Services on a Single Node

On a single node, you can start multiple services using `scripts/server/start_server.sh` (Note that the port numbers under the same IP must be different for each service), or you can start multiple services at once using `scripts/server/start_multi_servers.sh`:

```shell
num_gpus=8 bash scripts/server/start_multi_servers.sh
```

Where `num_gpus` indicates the number of services to start; the services will run on consecutive ports starting from `--start_port`.

### Scheduling Between Multiple Services

```shell
python scripts/server/post_multi_servers.py
```

`post_multi_servers.py` will schedule multiple client requests based on the idle status of the services.

### API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tasks/` | POST | Create video generation task |
| `/v1/tasks/form` | POST | Create video generation task via form |
| `/v1/tasks/` | GET | Get all task list |
| `/v1/tasks/{task_id}/status` | GET | Get status of specified task |
| `/v1/tasks/{task_id}/result` | GET | Get result video file of specified task |
| `/v1/tasks/running` | DELETE | Stop currently running task |
| `/v1/files/download/{file_path}` | GET | Download file |
| `/v1/service/status` | GET | Get service status |
