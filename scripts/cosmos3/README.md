# Cosmos3

Set `model_path` in the launcher to the matching checkpoint. Each supplied startup JSON now declares `model_variant` explicitly:

| Checkpoint | `model_variant` | Tasks |
| --- | --- | --- |
| Cosmos3-Nano / Cosmos3-Super | `omni` | `t2i`, `t2v`, `i2v`, `t2av`, `i2av`, `i2va`, `v2av` |
| Cosmos3-Super-Text2Image | `t2i` | `t2i` |
| Cosmos3-Super-Image2Video | `i2v` | `i2v` |
| Cosmos3-Nano-Policy-DROID | `policy` | `i2va`, using the policy settings |

The variant identifies the checkpoint's intended task scope. `sound_gen` and `action_gen` from `transformer/config.json` separately control the audio and action modules; the runner checks these capabilities for each request. They do not define the complete task list.

CLI inference selects a task with `--task`. An Omni service can start without it:

```bash
python -m lightx2v.server \
  --model_cls cosmos3 \
  --model_path /path/to/Cosmos3-Nano \
  --config_json configs/cosmos3/cosmos3_nano_omni.json
```

Every native request to this service supplies `task`. Send `t2i` to `/v1/tasks/image/`; send video and audio-video tasks to `/v1/tasks/video/`. Changing tasks reuses the loaded weights.

Specialized variants remain single-task services; their native requests omit `task`. Policy-DROID with `decode_video=false` returns actions and uses the Python `Cosmos3Policy` interface; it has no image/video HTTP output adapter. Enable `decode_video` only when video output is needed.
