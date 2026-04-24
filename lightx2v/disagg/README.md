# disagg / `run_dynamic.sh` 使用说明

`scripts/disagg/run_dynamic.sh` 是 LightX2V 的动态多机/单机离线调度启动脚本。它会自动完成以下工作：

1. 激活 `lightx2v` conda 环境，除非显式关闭。
2. 先执行 `scripts/disagg/kill_service.sh` 清理残留进程和端口。
3. 读取 controller 配置，准备 `multi_node` 或 `single_node` 启动参数。
4. 为 Mooncake / RDMA / ZMQ / 日志收集设置默认环境变量。
5. 启动 controller，并按配置拉起 encoder / transformer / decoder。

## 基本用法

最常见的方式是直接运行脚本：

```bash
bash scripts/disagg/run_dynamic.sh
```

如果要切换拓扑或覆盖默认配置，可以在命令前追加环境变量：

```bash
DISAGG_TOPOLOGY=multi_node \
DISAGG_CONTROLLER_CFG=/root/zht/LightX2V/configs/disagg/multi_node/wan22_i2v_distill_controller.json \
bash scripts/disagg/run_dynamic.sh
```

单机调试可以改成：

```bash
DISAGG_TOPOLOGY=single_node \
bash scripts/disagg/run_dynamic.sh
```

## 脚本会自动处理的事情

脚本会自动：

1. 如果当前没有激活到 `DISAGG_CONDA_ENV`，就尝试 `conda activate`。
2. 设置编译器和 `NVCC_PREPEND_FLAGS`，便于本地编译或运行扩展。
3. 默认将 `RDMA_IFACE` 设为 `erdma_0`，将 `MOONCAKE_DEVICE_NAME` 设为 `eth0`。
4. 如果没有显式设置 `MOONCAKE_LOCAL_HOSTNAME`，就从 `MOONCAKE_DEVICE_NAME` 对应网卡自动解析本机 IPv4。
5. 根据 controller 配置里的 `bootstrap_addr` 自动推导 `DISAGG_CONTROLLER_HOST`。
6. 先执行 `kill_service.sh` 清理旧服务，避免端口冲突。

## 环境变量说明

下面按功能分组说明常用变量。未特别说明时，都是脚本默认值。

### 运行模式与配置

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `DISAGG_TOPOLOGY` | 运行拓扑，`multi_node` 表示多机，`single_node` 表示单机。 | `multi_node` |
| `DISAGG_CONTROLLER_CFG` | controller 配置文件路径。脚本会根据拓扑自动选择默认配置。 | `configs/disagg/multi_node/wan22_i2v_distill_controller.json` 或 single_node 对应文件 |
| `DISAGG_CONDA_ENV` | 启动时要激活的 conda 环境名。 | `lightx2v` |
| `DISAGG_SKIP_CONDA_ACTIVATE` | 设为 `1` 时跳过 conda 激活。 | `0` |
| `DISAGG_CONTROLLER_HOST` | controller 对外使用的主机地址。若未设置，脚本会尝试从配置文件 `bootstrap_addr` 推导。 | 配置里的 `bootstrap_addr`，否则 `127.0.0.1` |

### RDMA / Mooncake

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `RDMA_IFACE` | 本机 RDMA / eRDMA 网卡名。 | `erdma_0` |
| `MOONCAKE_DEVICE_NAME` | Mooncake 用来解析本机 IPv4 的网卡名。 | `eth0` |
| `MOONCAKE_LOCAL_HOSTNAME` | Mooncake 认为的本机地址。若未设置，脚本会自动从 `MOONCAKE_DEVICE_NAME` 对应网卡提取 IPv4。 | 自动推导 |
| `RDMA_PREFERRED_IPV4` | 优先选择的 RDMA 数据平面 IPv4，通常用于多网卡环境下稳定选择 gid_index。 | 自动推导为 `DISAGG_CONTROLLER_HOST`（当其是 IPv4 且不是 `127.0.0.1`） |

### 控制面端口与启动等待

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `DISAGG_CONTROLLER_REQUEST_PORT` | controller 请求入口端口。 | `12786` |
| `DISAGG_INSTANCE_START_TIMEOUT_SECONDS` | 等待实例启动完成的超时时间。 | `single_node: 90`，`multi_node: 300` |
| `DISAGG_REMOTE_PROXY_START_TIMEOUT_SECONDS` | 等待远端 proxy 启动的超时时间。 | `120` |
| `DISAGG_SIDECAR_START_TIMEOUT_SECONDS` | 等待 sidecar 启动的超时时间。 | `60` |
| `CONTROLLER_WAIT_TIMEOUT_S` | 等待 controller 完成整轮任务的超时时间。 | `single_node: 3000`，`multi_node: 7200` |
| `CONTROLLER_POLL_INTERVAL_S` | controller 状态轮询间隔。 | `5` |

### 请求数量、调试与通信方式

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `LOAD_FROM_USER` | 设为非 `0` 时，由 user 侧持续发请求，直到阶段结束。 | `0` |
| `DISAGG_AUTO_REQUEST_COUNT` | 自动请求的默认数量。`LOAD_FROM_USER=0` 时会使用这个值。 | `30` |
| `USER_MAX_REQUESTS` | 手动限制 user 进程最多发多少个请求，优先级高于 `DISAGG_AUTO_REQUEST_COUNT`。 | 未设置 |
| `USER_START_DELAY_S` | user 进程启动后的延迟时间。 | `0` |
| `SYNC_COMM` | 是否启用同步通信模式。 | `0` |

### Nsight 采集

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `DISAGG_ENABLE_NSYS` | 是否启用 `nsys profile` 包裹实例进程。 | `0` |
| `DISAGG_NSYS_BIN` | `nsys` 可执行文件路径或命令名。 | `nsys` |
| `DISAGG_NSYS_OUTPUT_DIR` | nsys trace 输出目录。 | `save_results/nsys` |
| `DISAGG_NSYS_TRACE` | `nsys profile` 的 trace 类型。 | `cuda,nvtx,osrt` |
| `DISAGG_NSYS_EXTRA_ARGS` | 额外传给 `nsys profile` 的参数。 | 空 |

### 日志与清理

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `REMOTE_LOG_COLLECT` | 是否在结束后拉取远端日志。 | `1` |
| `REMOTE_LOG_COLLECT_DIR` | 远端日志收集到本地的目录。 | `save_results/remote_logs` |
| `DISAGG_REMOTE_PRE_CLEAN` | 是否在启动前先远端执行 `kill_service.sh`。 | `1` |
| `SEED` | 随机种子。 | `42` |
| `PROMPT` | 文本提示词。 | 脚本内置示例 prompt |
| `NEGATIVE_PROMPT` | 负向提示词。 | 脚本内置示例 negative prompt |
| `SAVE_RESULT_PATH` | 最终视频保存路径。 | `save_results/wan22_i2v_dynamic.mp4` |

## 推荐的常见组合

### 本地单机调试

```bash
DISAGG_TOPOLOGY=single_node \
LOAD_FROM_USER=0 \
DISAGG_AUTO_REQUEST_COUNT=1 \
bash scripts/disagg/run_dynamic.sh
```

### 多机标准跑法

```bash
DISAGG_TOPOLOGY=multi_node \
DISAGG_CONTROLLER_CFG=/root/zht/LightX2V/configs/disagg/multi_node/wan22_i2v_distill_controller.json \
DISAGG_AUTO_REQUEST_COUNT=30 \
bash scripts/disagg/run_dynamic.sh
```

### 开启 Nsight

```bash
DISAGG_ENABLE_NSYS=1 \
DISAGG_NSYS_TRACE=cuda,nvtx,osrt \
bash scripts/disagg/run_dynamic.sh
```

## 备注

1. 多机运行时，`DISAGG_CONTROLLER_CFG` 里的 `bootstrap_addr`、`static_instance_slots` 和各 slot 的 `env` 会直接影响远端实例如何绑定网络与 Mooncake 地址。
2. 如果遇到端口占用，优先检查 `scripts/disagg/kill_service.sh` 是否已经把旧实例和 proxy 清理干净。
3. 如果需要了解 controller 配置文件本身的字段含义，可以继续查看 `configs/disagg/` 下对应 JSON。