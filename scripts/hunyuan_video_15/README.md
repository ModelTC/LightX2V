# HunyuanVideo1.5

## Quick Start

1. Prepare docker environment:

```bash
docker pull lightx2v/lightx2v:25111101-cu128
```

2. Run the container:
```bash
docker run --gpus all -itd --ipc=host --name [container_name] -v [mount_settings] --entrypoint /bin/bash [image_id]
```

3. Prepare the models

Please follow the instructions in [HunyuanVideo1.5 Github](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/checkpoints-download.md) to download and place the model files.

4. Running

Running using bash script
```bash
# enter the docker container

git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V/scripts/hunyuan_video_15

# set LightX2V path and model path in the script
bash run_hy15_t2v_480p.sh
```

Running using Python code: edit the paths in the selected file and run it from
the repository root after installation.

- [Text to video with LightTAE](inference/hunyuan_t2v.py)
- [Image to video with quantization](inference/hunyuan_i2v.py)
- [Distilled text to video](inference/hunyuan_t2v_distill.py)

See the [Python API guide](../../docs/EN/source/getting_started/python_api.md)
for configuration and generator lifecycle.

5. Check results

You can find the generated video files in the `save_results` folder.

6. Modify detailed configurations

You can refer to the config file pointed to by `--config_json` in the script and modify its parameters as needed.
