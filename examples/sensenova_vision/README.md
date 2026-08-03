# SenseNova-Vision in LightX2V

This integration reuses the LightX2V BAGEL model, scheduler, KV-cache and
diffusion inference path while preserving SenseNova-Vision's official task
profiles, prompts, image transforms and multi-view reconstruction protocol.

## Run the official example suite

```bash
cd /data/nvme0/lhd_codes/LightX2V
CUDA_VISIBLE_DEVICES=0 bash scripts/sensenova_vision/run_sensenova_vision.sh example
```

The model is loaded once and all 14 public subtasks are executed. Examples
01-11 follow upstream `example_visualize.py`; examples 12-14 use the upstream
point detection, keypoint and OCR validation assets. Results are written to
`save_results/sensenova_vision_example` by default. Override this with
`OUTPUT_DIR=/path/to/output`.

Each official example can also be run independently. Every script below uses
the same input, prompt, seed and output filename as the full suite:

| Example | Task | Script |
| --- | --- | --- |
| 01 | General understanding | `run_sensenova_vision_example_01_understanding.sh` |
| 02 | Binary segmentation | `run_sensenova_vision_example_02_binary_segmentation.sh` |
| 03 | Depth estimation | `run_sensenova_vision_example_03_depth.sh` |
| 04 | Surface normal estimation | `run_sensenova_vision_example_04_normal.sh` |
| 05 | Grounded caption and segmentation | `run_sensenova_vision_example_05_gcg_segmentation.sh` |
| 06 | Object detection | `run_sensenova_vision_example_06_object_detection.sh` |
| 07 | Multi-view 3D reconstruction | `run_sensenova_vision_example_07_recon3d.sh` |
| 08 | Panoptic segmentation | `run_sensenova_vision_example_08_panoptic_segmentation.sh` |
| 09 | Interactive segmentation | `run_sensenova_vision_example_09_interactive_segmentation.sh` |
| 10 | Visual-grounded segmentation | `run_sensenova_vision_example_10_vgd_segmentation.sh` |
| 11 | Relative camera pose | `run_sensenova_vision_example_11_camera_pose.sh` |
| 12 | Point detection | `run_sensenova_vision_example_12_point_detection.sh` |
| 13 | Human keypoint detection | `run_sensenova_vision_example_13_keypoint.sh` |
| 14 | Word-level OCR | `run_sensenova_vision_example_14_ocr.sh` |

For example:

```bash
bash scripts/sensenova_vision/run_sensenova_vision_example_03_depth.sh --gpus 0
```

The default output directory remains `save_results/sensenova_vision_example`.
Use `OUTPUT_DIR=/path/to/output` to change it. The generic example launcher also
accepts a number directly, for example
`bash scripts/sensenova_vision/run_sensenova_vision_example.sh 03 --gpus 0`.

`--gpus` accepts one index or a comma-separated visible-device list:

```bash
# Run on physical GPU 3.
bash scripts/sensenova_vision/run_sensenova_vision_example_03_depth.sh --gpus 3

# Make physical GPUs 2 and 3 visible to the process.
bash scripts/sensenova_vision/run_sensenova_vision_example_03_depth.sh --gpus 2,3
```

The current SenseNova-Vision example runner is single-process/single-device, so
when multiple GPUs are visible it executes on the first visible GPU. A list is
mainly useful for compatibility with launch environments; it does not enable
model parallelism by itself. The existing `CUDA_VISIBLE_DEVICES=...` form is
still supported.

## Run one task

```bash
IMAGE_PATH=/path/to/input.jpg \
SAVE_PATH=/path/to/depth.png \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/sensenova_vision/run_sensenova_vision.sh depth
```

The script starts LightX2V with `task=omni_vision_task` and passes its first
argument as `omni_vision_subtask`. Public subtask names are `understanding`,
`binary_segmentation`, `depth`, `normal`, `gcg_segmentation`,
`object_detection`, `point_detection`, `keypoint`, `ocr`, `recon3d`,
`panoptic_segmentation`, `interactive_segmentation`, `vgd_segmentation`, and
`camera_pose`. Use `PROMPT` when a task requires a category, question, or instruction. Multiple input images are
passed as a comma-separated `IMAGE_PATH`.

For direct offline 3D reconstruction, the raw point map is derived from
`SAVE_PATH` and saved as NPY:

```bash
IMAGE_PATH=/path/view1.png,/path/view2.png \
SAVE_PATH=/path/result.npy \
bash scripts/sensenova_vision/run_sensenova_vision.sh recon3d
```

Use the resident server's `--postprocess-3d` request option or official example 07 when a GLB artifact is required.

The checkpoint and upstream source paths can be overridden through
`model_path` and `SENSENOVA_SOURCE_PATH`, respectively.

## Run one resident multi-task server

The server loads the complete SenseNova-Vision checkpoint once. Requests are
serialized through the resident runner and select their own public `task`; the
service maps that task to SenseNova's internal task and inference mode without
reloading weights.

```bash
cd /data/nvme0/lhd_codes/LightX2V
bash scripts/sensenova_vision/start_sensenova_vision_server.sh --gpus 0 --port 8000
```

Useful environment overrides are:

```bash
model_path=/path/to/SenseNova-Vision-7B-MoT \
SENSENOVA_SOURCE_PATH=/path/to/sensenova-vision-v2 \
LIGHTX2V_CACHE_DIR=/path/to/server-cache \
bash scripts/sensenova_vision/start_sensenova_vision_server.sh --gpus 2
```

Server results are stored under
`$LIGHTX2V_CACHE_DIR/outputs/<task_id>/`. The default root is
`save_results/sensenova_vision_server_cache`. The current BAGEL/SenseNova
runner is one-process/one-device, so `--gpus 0` is recommended. Passing more
visible GPUs does not shard one model; use separate server replicas on
different ports for throughput scaling.

### Submit requests

The client accepts one or more `--image` arguments. Existing client-local
files are Base64 encoded automatically; HTTP(S) URLs, raw Base64 and paths
visible to the server are also accepted.

General understanding (text result):

```bash
python scripts/server/sensenova/post_sensenova_vision.py \
  --task understanding \
  --image /data/nvme0/lhd_codes/SenseNova-Vision/examples/images/1.jpg \
  --prompt "What are the main objects in this scene and their relationships?"
```

Depth (raw prediction image):

```bash
python scripts/server/sensenova/post_sensenova_vision.py \
  --task depth \
  --image /data/nvme0/lhd_codes/SenseNova-Vision/examples/images/3.jpg
```

Binary segmentation (raw mask plus official-style visualization):

```bash
python scripts/server/sensenova/post_sensenova_vision.py \
  --task binary_segmentation \
  --image /data/nvme0/lhd_codes/SenseNova-Vision/examples/images/2.jpg \
  --prompt "person furthest to the right"
```

Multi-view reconstruction (NPY plus optional GLB):

```bash
python scripts/server/sensenova/post_sensenova_vision.py \
  --task recon3d \
  --image /path/to/view1.png \
  --image /path/to/view2.png \
  --postprocess-3d
```

Interactive segmentation uses exactly two images: the source image followed
by the visual-prompt mask.

```bash
python scripts/server/sensenova/post_sensenova_vision.py \
  --task interactive_segmentation \
  --image /path/to/source.jpg \
  --image /path/to/visual_prompt.png \
  --prompt "Segment the object indicated by the visual prompt."
```

The client uses asynchronous submit-and-poll by default. Add `--sync` to use
the blocking endpoint. Artifacts and `result.json` are downloaded to
`save_results/sensenova_vision_client/<task_id>/` unless `--output-dir` is
specified.

Public task names are `understanding`, `binary_segmentation`, `depth`,
`normal`, `gcg_segmentation`, `object_detection`, `point_detection`,
`keypoint`, `ocr`, `recon3d`, `panoptic_segmentation`,
`interactive_segmentation`, `vgd_segmentation`, and `camera_pose`.

The API endpoints are:

- `POST /v1/tasks/sensenova-vision/`: asynchronous submission.
- `GET /v1/tasks/{task_id}/status`: queue/inference status.
- `GET /v1/tasks/sensenova-vision/{task_id}/result`: structured result.
- `POST /v1/tasks/sensenova-vision/sync`: blocking submission.
- `GET /v1/files/download/{path}`: artifact download.

For a direct JSON request, the key fields are:

```json
{
  "task": "depth",
  "images": ["<base64, URL, or server-local path>"],
  "prompt": "",
  "seed": 42,
  "target_shape": [],
  "visualize": true,
  "postprocess_3d": false
}
```

The result JSON contains task/mode metadata, optional `text`, warnings, and an
`artifacts` list. Each artifact includes its kind, media type, size, server
filename and download URL, so image, text, NPY and GLB tasks share one stable
response format.
