# YuE2 端到端音频推理

环境默认已经配置好，脚本直接使用当前环境的 `python` / `torchrun`。13 个功能共用一个数据目录和一个 `run_yue2.sh`，第一个参数指定功能序号。

以下命令均在 LightX2V 仓库根目录执行。运行前，按实际位置设置启动脚本顶部的 `lightx2v_path` 和 `model_path`；样例数据默认读取仓库内的 `assets/inputs/yue2`。

```text
bash scripts/yue2/run_yue2.sh [功能序号 1–13]
```

- 功能序号默认 1，对应下表；功能编号只在示例脚本中选择输入，后端任务统一为 `yue2 / t2a`。
- 输出默认 `save_results/yue2/function_<序号>.flac`；需要其他 `.flac` 或 `.wav` 路径时，直接修改脚本中的 `--save_result_path`。
- 部署配置默认 `configs/yue2/yue2.json`，13 个功能共用这一份配置；需要切换时直接修改脚本中的 `--config_json`。
- `lightx2v_path`、`model_path` 在脚本顶部直接填写，`data_dir` 默认由仓库路径推导。单卡脚本默认 GPU 0，双卡脚本默认 GPU 0、1；需要换卡时，直接修改脚本顶部的 `CUDA_VISIBLE_DEVICES`。

## 共享数据

样例数据随仓库提供，目录为 `assets/inputs/yue2`，共 11 个文件，供 13 个样例共用。CLI 和 HTTP 客户端默认读取此目录，无需单独准备外部数据目录。

使用自定义样例时，修改 CLI 脚本顶部的 `data_dir`，或向 HTTP 客户端传入 `--data-dir`。

```text
assets/inputs/yue2/
├── presets.json
├── lyrics_en.txt
├── lyrics_zh.txt
├── lyrics_cover_zh.txt
├── lyrics_full_song_zh.txt
├── melody_en.abc
├── melody_cover_zh.abc
├── score_en.abc
├── score_en_edited.abc
├── score_instrumental.abc
└── melody_full_song_zh.abc
```

`presets.json` 保存公共参数、5 条风格描述和 13 个编号的输入组合。`functions` 中的 `style` 引用 `styles` 中的风格文本，`lyrics_file` / `abc_file` 相对于共享目录读取。`lyrics_file: null` 明确表示空歌词；`abc_file: null` 表示没有外部 ABC，随后由 `cot` 决定自动规划或跳过。`id` 保留原始样本标识，`name` 用于识别样本，两者都不用于选择推理分支。

歌词和乐谱仅在文件中保存一份，不再重复嵌入请求 JSON。修改共享文件会影响所有引用它的功能；运行时直接读取这些文件。参考音频和来源资料保留在原数据目录，推理不读取它们。

## 功能编号与输入区别

所有功能均输出 48 kHz 双声道音频。风格对应 `presets.json` 中的键；自动规划的具体方式由 `cot` 决定。

| 编号 | 功能 | 歌词文件 | ABC 文件 / 来源 | 风格键 | cot |
|---:|---|---|---|---|---|
| 1 | 指定旋律生成英文歌 | `lyrics_en.txt` | `melody_en.abc` | `en_pop` | `melody` |
| 2 | 英文歌词：完整规划 | `lyrics_en.txt` | 自动生成 | `en_pop` | `full` |
| 3 | 英文歌词：旋律规划 | `lyrics_en.txt` | 自动生成 | `en_pop` | `melody` |
| 4 | 英文歌词：跳过乐谱规划 | `lyrics_en.txt` | 不使用 | `en_pop` | `off` |
| 5 | 中文歌词：完整规划 | `lyrics_zh.txt` | 自动生成 | `zh_pop` | `full` |
| 6 | 中文歌词：旋律规划 | `lyrics_zh.txt` | 自动生成 | `zh_pop` | `melody` |
| 7 | 中文歌词：跳过乐谱规划 | `lyrics_zh.txt` | 不使用 | `zh_pop` | `off` |
| 8 | 换曲风：流行摇滚 | `lyrics_cover_zh.txt` | `melody_cover_zh.abc` | `pop_rock` | `melody` |
| 9 | 乐谱编辑前 | `lyrics_en.txt` | `score_en.abc` | `en_pop` | `full` |
| 10 | 乐谱编辑后 | `lyrics_en.txt` | `score_en_edited.abc` | `en_pop` | `full` |
| 11 | 器乐生成 | 空字符串 | `score_instrumental.abc` | `instrumental` | `full` |
| 12 | 多段中文整曲生成 | `lyrics_full_song_zh.txt` | `melody_full_song_zh.abc` | `pop_rock` | `melody` |
| 13 | 编曲：原声配器 | `lyrics_en.txt` | `score_en.abc` | `acoustic` | `full` |


共同参数为 `seed=831001`、`cfg_scale=null`。后者使用模型默认值：full/melody 为 1.0，off 为 1.01。脚本保留空歌词、多行文本和内部换行，歌词文件末尾的一个换行符会被移除，`null` 字段不作为显式 CLI 参数传入。

8 保留流行摇滚换曲风样例；9/10 只改变 ABC；13 保留原声配器编曲样例。器乐使用空歌词和器乐风格，多段整曲使用更长的歌词与乐谱。

本接入对应 YuE2-3B AR/NAR MoT，不适用于旧版 stage1/stage2 权重。完整链路包括可选 ABC 规划、语义 token 生成、NAR 声学生成和 VAE 解码。音频转谱、分轨等前置数据准备仍在应用仓库完成。换曲风、乐谱编辑和配器样本都是条件变化后重新生成音频，音乐效果需要试听比较。

## 13 条可直接执行的命令

每条命令完整独立，不依赖公共 Shell 变量。输入统一来自上述共享目录。命令只传功能序号，输出路径列在每条命令下方；配置和输出路径直接写在脚本内，不再通过第二、第三个位置参数覆盖。

### 1. 指定旋律生成英文歌

```bash
bash scripts/yue2/run_yue2.sh 1
```

输出：`save_results/yue2/function_1.flac`

### 2. 英文歌词：完整规划

```bash
bash scripts/yue2/run_yue2.sh 2
```

输出：`save_results/yue2/function_2.flac`

### 3. 英文歌词：旋律规划

```bash
bash scripts/yue2/run_yue2.sh 3
```

输出：`save_results/yue2/function_3.flac`

### 4. 英文歌词：跳过乐谱规划

```bash
bash scripts/yue2/run_yue2.sh 4
```

输出：`save_results/yue2/function_4.flac`

### 5. 中文歌词：完整规划

```bash
bash scripts/yue2/run_yue2.sh 5
```

输出：`save_results/yue2/function_5.flac`

### 6. 中文歌词：旋律规划

```bash
bash scripts/yue2/run_yue2.sh 6
```

输出：`save_results/yue2/function_6.flac`

### 7. 中文歌词：跳过乐谱规划

```bash
bash scripts/yue2/run_yue2.sh 7
```

输出：`save_results/yue2/function_7.flac`

### 8. 换曲风：流行摇滚

```bash
bash scripts/yue2/run_yue2.sh 8
```

输出：`save_results/yue2/function_8.flac`

### 9. 乐谱编辑前

```bash
bash scripts/yue2/run_yue2.sh 9
```

输出：`save_results/yue2/function_9.flac`

### 10. 乐谱编辑后

```bash
bash scripts/yue2/run_yue2.sh 10
```

输出：`save_results/yue2/function_10.flac`

### 11. 器乐生成

```bash
bash scripts/yue2/run_yue2.sh 11
```

输出：`save_results/yue2/function_11.flac`

### 12. 多段中文整曲生成

```bash
bash scripts/yue2/run_yue2.sh 12
```

输出：`save_results/yue2/function_12.flac`

### 13. 编曲：原声配器

```bash
bash scripts/yue2/run_yue2.sh 13
```

输出：`save_results/yue2/function_13.flac`

## 部署方式：与功能选择相互独立

13 个功能都可以继续通过功能序号切换。普通推理只需要 `run_yue2.sh`；`run_yue2_tp2.sh` 是双卡部署示例，直接调用 `torchrun`，不代表另一项音乐功能。

| 配置 | 用途 |
|---|---|
| `yue2.json` | 默认单卡 eager，关闭预热 |
| `yue2_offload.json` | AR/NAR block 权重放 CPU，每分支两个 GPU 复用缓冲区；embedding/head/VAE 留在 GPU，以传输开销换显存 |
| `yue2_tp2.json` | NAR 两卡张量并行；AR 保持副本，rank 0 采样和 VAE 解码 |
| `yue2_compile_warmup.json` | 编译 NAR block，并开启框架启动预热；AR 保持 eager |

单卡默认使用 `yue2.json`。CPU offload 或编译模式直接修改 `run_yue2.sh` 内的配置行：

```text
CPU offload: --config_json ${lightx2v_path}/configs/yue2/yue2_offload.json
编译与预热: --config_json ${lightx2v_path}/configs/yue2/yue2_compile_warmup.json
恢复默认: --config_json ${lightx2v_path}/configs/yue2/yue2.json
```

修改后仍按功能序号运行，例如：

```bash
bash scripts/yue2/run_yue2.sh 1
```

双卡脚本已直接填写 `yue2_tp2.json`（默认 GPU 0、1）：

```bash
bash scripts/yue2/run_yue2_tp2.sh 1
```

输出由各脚本的 `--save_result_path` 决定，当前均为 `save_results/yue2/function_<序号>.flac`。需要保留不同部署方式的结果时，直接修改这一行的文件名。

`use_compile` 和 `warmup` 是部署 JSON 中相互独立的开关，均默认 false。脚本直接读取原始配置，不再生成临时 JSON，也不再自动把当前样本注入预热配置。CLI 与服务端统一由 runner 执行启动预热：设置 `warmup_requests` 时使用指定请求，否则使用内置的两组短器乐请求。预热隔离 RNG 并清理请求状态；实际请求签名未被覆盖时仍可能触发首次编译。

编译仅覆盖 NAR block，保留 `emulate_precision_casts=true`。四份配置分别对应已接入的部署能力，组合优化不属于这四份配置的验证范围。

## HTTP 服务

单卡启动（第一个可选参数为部署配置，第二个为端口，默认 8000）：

```bash
bash scripts/yue2/server/start_server.sh
```

双卡启动时改用：

```bash
bash scripts/yue2/server/start_server_tp2.sh
```

另开终端，按编号提交请求并下载最终音频；修改 `--function-id` 即可切换上表的 13 个功能：

```bash
python scripts/yue2/server/post_t2a.py \
  --function-id 11 \
  --url http://127.0.0.1:8000 \
  --output save_results/yue2/11_instrumental_http.flac
```

服务端使用 `python -m lightx2v.server`（双卡为 `torchrun`）。客户端使用相同的共享数据和编号组装请求，提交 `/v1/tasks/`，轮询后下载 `audio` 结果。客户端等待超时不会取消服务端任务，可用打印的 task_id 继续查询。

## 模型权重

核心模型使用官方 [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)，权重格式为 Safetensors。可通过 Hugging Face 下载到自选目录，例如：

```bash
hf download m-a-p/YuE2-3B --local-dir models/YuE2-3B
```

核心模型目录的主要文件如下；完整文件列表见 [Hugging Face Files](https://huggingface.co/m-a-p/YuE2-3B/tree/main)：

```text
models/
└── YuE2-3B/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── qwen.tiktoken
    └── yue2_generation_config.json
```

`model.safetensors` 为当前官方发布的单文件核心权重，`qwen.tiktoken` 为分词器文件。下载目录还会包含上游说明、许可和其他辅助文件。将启动脚本中的 `model_path` 设置为下载后的 `YuE2-3B` 目录的绝对路径。
