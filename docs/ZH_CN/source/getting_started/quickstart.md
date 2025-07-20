# 快速入门

## 准备环境

我们推荐使用docker环境，这是lightx2v的[dockerhub](https://hub.docker.com/r/lightx2v/lightx2v/tags)，请选择一个最新日期的tag，比如25061301

```shell
docker pull lightx2v/lightx2v:25061301
docker run --gpus all -itd --ipc=host --name [容器名] -v [挂载设置]  --entrypoint /bin/bash [镜像id]
```

对于中国大陆地区，若拉取镜像的时候，网络不稳定，可以从[渡渡鸟](https://docker.aityp.com/r/docker.io/lightx2v/lightx2v)上拉取

```shell
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lightx2v/lightx2v:25061301
```


如果你想使用conda自己搭建环境，可以参考如下步骤：

```shell
# 下载github代码
git clone https://github.com/ModelTC/lightx2v.git lightx2v && cd lightx2v

conda create -n lightx2v python=3.11 && conda activate lightx2v
pip install -r requirements.txt

# 混元模型需要在4.45.2版本的transformers下运行，如果不需要跑混元模型，可以忽略
# pip install transformers==4.45.2

# 安装 flash-attention 2
git clone https://github.com/Dao-AILab/flash-attention.git --recursive
cd flash-attention && python setup.py install

# 安装 flash-attention 3, 用于 hopper 显卡
cd flash-attention/hopper && python setup.py install
```

## 推理

```shell
# 修改脚本中的路径
bash scripts/wan/run_wan_t2v.sh
```

除了脚本中已有的输入参数，`--config_json`指向的`wan_t2v.json`中也会存在一些必要的参数，可以根据需要，自行修改。
