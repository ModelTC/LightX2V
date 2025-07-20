# Quick Start

## Prepare Environment

We recommend using a docker environment. Here is the [dockerhub](https://hub.docker.com/r/lightx2v/lightx2v/tags) for lightx2v. Please select the tag with the latest date, for example, 25061301.

```shell
docker pull lightx2v/lightx2v:25061301
docker run --gpus all -itd --ipc=host --name [container_name] -v [mount_settings]  --entrypoint /bin/bash [image_id]
```

If you want to set up the environment yourself using conda, you can refer to the following steps:

```shell
# clone repo and submodules
git clone https://github.com/ModelTC/lightx2v.git lightx2v && cd lightx2v

conda create -n lightx2v python=3.11 && conda activate lightx2v
pip install -r requirements.txt

# The Hunyuan model needs to run under this version of transformers. If you do not need to run the Hunyuan model, you can ignore this step.
# pip install transformers==4.45.2

# install flash-attention 2
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
cd flash-attention && python setup.py install

# install flash-attention 3, only if hopper
cd flash-attention/hopper && python setup.py install
```

## Infer

```shell
# Modify the path in the script
bash scripts/wan/run_wan_t2v.sh
```

In addition to the existing input arguments in the script, there are also some necessary parameters in the `wan_t2v.json` file specified by `--config_json`. You can modify them as needed.
