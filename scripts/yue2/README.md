# YuE2 end-to-end audio inference

The environment is assumed to be ready. Scripts use the active `python` / `torchrun`. All 13 functions share one data directory and one `run_yue2.sh`; select the function with the first argument.

Run the commands below from the LightX2V repository root. Before running, set `lightx2v_path` and `model_path` at the top of the launch script to your actual locations. Sample data defaults to `assets/inputs/yue2` inside the repository.

```text
bash scripts/yue2/run_yue2.sh [function number 1–13]
```

- Function number defaults to 1; see the table below. Numbers select inputs in the example script; the backend always uses `yue2 / t2a`.
- Output defaults to `save_results/yue2/function_<number>.flac`; edit `--save_result_path` in the script to choose another `.flac` or `.wav` path.
- Deployment configuration defaults to `configs/yue2/yue2.json`, shared by all 13 functions; edit `--config_json` in the script to change it.
- Edit `lightx2v_path` and `model_path` at the top of the script; `data_dir` is derived from the repository path by default. The single-device script uses GPU 0; the two-device script uses GPUs 0 and 1. Edit `CUDA_VISIBLE_DEVICES` at the top of the launch script to select other devices.

## Shared data

Sample data is included in the repository at `assets/inputs/yue2`: 11 files shared by all 13 examples. Both CLI and HTTP clients use this directory by default; no separate external data directory is needed.

For custom samples, change `data_dir` at the top of the CLI script or pass `--data-dir` to the HTTP client.

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

`presets.json` contains shared defaults, five style descriptions, and 13 input combinations. A function's `style` selects text from `styles`; `lyrics_file` / `abc_file` are relative to the shared directory. `lyrics_file: null` explicitly means empty lyrics. `abc_file: null` means no external ABC; `cot` then selects automatic planning or bypassing. `id` preserves the original sample identifier and `name` labels the sample; neither selects an inference branch.

Lyrics and scores are stored once in their files, without duplicating their text in request JSON. Editing a shared file affects every function that references it; files are read directly at runtime. Reference audio and provenance remain in the original dataset and are not read by inference.

## Function numbers and input differences

Every function outputs 48 kHz stereo audio. Style keys refer to `presets.json`; `cot` determines the type of automatic score planning.

| Number | Function | Lyrics file | ABC file / source | Style key | cot |
|---:|---|---|---|---|---|
| 1 | English song from a prepared melody | `lyrics_en.txt` | `melody_en.abc` | `en_pop` | `melody` |
| 2 | English lyrics: full planning | `lyrics_en.txt` | Generated | `en_pop` | `full` |
| 3 | English lyrics: melody planning | `lyrics_en.txt` | Generated | `en_pop` | `melody` |
| 4 | English lyrics: no score planning | `lyrics_en.txt` | Unused | `en_pop` | `off` |
| 5 | Chinese lyrics: full planning | `lyrics_zh.txt` | Generated | `zh_pop` | `full` |
| 6 | Chinese lyrics: melody planning | `lyrics_zh.txt` | Generated | `zh_pop` | `melody` |
| 7 | Chinese lyrics: no score planning | `lyrics_zh.txt` | Unused | `zh_pop` | `off` |
| 8 | Style transfer: pop rock | `lyrics_cover_zh.txt` | `melody_cover_zh.abc` | `pop_rock` | `melody` |
| 9 | Score editing: before | `lyrics_en.txt` | `score_en.abc` | `en_pop` | `full` |
| 10 | Score editing: after | `lyrics_en.txt` | `score_en_edited.abc` | `en_pop` | `full` |
| 11 | Instrumental generation | Empty string | `score_instrumental.abc` | `instrumental` | `full` |
| 12 | Chinese song with multiple sections | `lyrics_full_song_zh.txt` | `melody_full_song_zh.abc` | `pop_rock` | `melody` |
| 13 | Arrangement: acoustic | `lyrics_en.txt` | `score_en.abc` | `acoustic` | `full` |


Shared defaults are `seed=831001` and `cfg_scale=null`. The latter uses model defaults: 1.0 for full/melody and 1.01 for off. The script preserves empty lyrics, multiline text, and internal line breaks; one file-ending newline is omitted from lyrics; null fields are omitted from explicit CLI arguments.

8 retains the pop-rock style-transfer example. 9/10 change only ABC. 13 retains the acoustic arrangement example. Instrumental generation uses empty lyrics and an instrumental style; the multi-section song uses longer lyrics and a longer score.

This integration targets YuE2-3B AR/NAR MoT, not legacy stage1/stage2 weights. The pipeline consists of optional ABC planning, semantic token generation, NAR acoustic generation, and VAE decoding. Transcription, separation, and other preprocessing remain in the application repository. Style, score editing, and arrangement examples regenerate audio from changed conditions; assess their musical effects by listening.

## 13 independently runnable commands

Each command is complete and needs no shared shell variables. Inputs come from the shared directory above. Commands take only the function number; each output path is listed below its command. Configuration and output paths are written directly in the script, without second/third positional overrides.

### 1. English song from a prepared melody

```bash
bash scripts/yue2/run_yue2.sh 1
```

Output: `save_results/yue2/function_1.flac`

### 2. English lyrics: full planning

```bash
bash scripts/yue2/run_yue2.sh 2
```

Output: `save_results/yue2/function_2.flac`

### 3. English lyrics: melody planning

```bash
bash scripts/yue2/run_yue2.sh 3
```

Output: `save_results/yue2/function_3.flac`

### 4. English lyrics: no score planning

```bash
bash scripts/yue2/run_yue2.sh 4
```

Output: `save_results/yue2/function_4.flac`

### 5. Chinese lyrics: full planning

```bash
bash scripts/yue2/run_yue2.sh 5
```

Output: `save_results/yue2/function_5.flac`

### 6. Chinese lyrics: melody planning

```bash
bash scripts/yue2/run_yue2.sh 6
```

Output: `save_results/yue2/function_6.flac`

### 7. Chinese lyrics: no score planning

```bash
bash scripts/yue2/run_yue2.sh 7
```

Output: `save_results/yue2/function_7.flac`

### 8. Style transfer: pop rock

```bash
bash scripts/yue2/run_yue2.sh 8
```

Output: `save_results/yue2/function_8.flac`

### 9. Score editing: before

```bash
bash scripts/yue2/run_yue2.sh 9
```

Output: `save_results/yue2/function_9.flac`

### 10. Score editing: after

```bash
bash scripts/yue2/run_yue2.sh 10
```

Output: `save_results/yue2/function_10.flac`

### 11. Instrumental generation

```bash
bash scripts/yue2/run_yue2.sh 11
```

Output: `save_results/yue2/function_11.flac`

### 12. Chinese song with multiple sections

```bash
bash scripts/yue2/run_yue2.sh 12
```

Output: `save_results/yue2/function_12.flac`

### 13. Arrangement: acoustic

```bash
bash scripts/yue2/run_yue2.sh 13
```

Output: `save_results/yue2/function_13.flac`

## Deployment is independent of musical function

Continue selecting the same 13 inputs by function number. Normal inference only needs `run_yue2.sh`. `run_yue2_tp2.sh` demonstrates two-device deployment with a direct `torchrun` command; it is not a separate musical function.

| Configuration | Purpose |
|---|---|
| `yue2.json` | Single-device eager baseline, no warmup |
| `yue2_offload.json` | AR/NAR block weights on CPU, two reusable GPU buffers per branch; embedding/head/VAE stay on GPU; trades transfers for lower memory use |
| `yue2_tp2.json` | Two-device NAR tensor parallelism; AR is replicated, rank 0 samples and decodes with the VAE |
| `yue2_compile_warmup.json` | Compile NAR blocks and enable framework startup warmup; AR stays eager |

Single-device inference uses `yue2.json` by default. To select CPU offload or compilation, edit the configuration line inside `run_yue2.sh`:

```text
CPU offload: --config_json ${lightx2v_path}/configs/yue2/yue2_offload.json
Compile and warmup: --config_json ${lightx2v_path}/configs/yue2/yue2_compile_warmup.json
Restore default: --config_json ${lightx2v_path}/configs/yue2/yue2.json
```

Then run by function number as usual, for example:

```bash
bash scripts/yue2/run_yue2.sh 1
```

The two-device script directly selects `yue2_tp2.json` (GPUs 0 and 1 by default):

```bash
bash scripts/yue2/run_yue2_tp2.sh 1
```

Each script's `--save_result_path` determines the output; both currently use `save_results/yue2/function_<number>.flac`. Edit this filename to keep results from different deployment modes separately.

`use_compile` and `warmup` are independent switches in deployment JSON and both default to false. Scripts read the original configuration directly: no temporary JSON is generated, and the selected sample is no longer automatically inserted into warmup settings. CLI and server use the same runner startup hook. Set `warmup_requests` for representative requests; otherwise the runner uses two short built-in instrumental requests. Warmup isolates RNG and clears request state; uncovered input signatures may still compile on first use.

Compilation covers NAR blocks only and retains `emulate_precision_casts=true`. The four configurations demonstrate separate deployment capabilities; combined optimizations are outside these configurations' validation scope.

## HTTP server

Single-device startup (optional first argument: deployment config; second: port, default 8000):

```bash
bash scripts/yue2/server/start_server.sh
```

For two devices, use:

```bash
bash scripts/yue2/server/start_server_tp2.sh
```

In another terminal, submit a numbered request and download the final audio. Change `--function-id` to select any of the 13 functions above:

```bash
python scripts/yue2/server/post_t2a.py \
  --function-id 11 \
  --url http://127.0.0.1:8000 \
  --output save_results/yue2/11_instrumental_http.flac
```

The server uses `python -m lightx2v.server` (`torchrun` for two devices). The client assembles the request from the same shared data and function number, posts to `/v1/tasks/`, polls, and downloads the `audio` result. A client timeout does not cancel the server task; use the printed task_id to check it.

## Model weights

Use the official [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) core model in Safetensors format. Download it from Hugging Face to a directory of your choice, for example:

```bash
hf download m-a-p/YuE2-3B --local-dir models/YuE2-3B
```

The main files in the core model directory are shown below. See [Hugging Face Files](https://huggingface.co/m-a-p/YuE2-3B/tree/main) for the complete listing:

```text
models/
└── YuE2-3B/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── qwen.tiktoken
    └── yue2_generation_config.json
```

`model.safetensors` is the single core weight file in the current official release, and `qwen.tiktoken` is the tokenizer file. The downloaded repository also contains upstream documentation, licenses, and other supporting files. Set `model_path` in the launch script to the absolute path of the downloaded `YuE2-3B` directory.
