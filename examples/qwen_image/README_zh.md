# Qwen Image 示例

本目录包含 Qwen Image 和 Qwen Image Edit 模型的使用示例。

## 测速结果

Dit部分的推理耗时对比(不包含预热时间，数据更新于2025.12.23):

<div align="center">
  <img src="../../assets/figs/qwen/qwen-image-edit-2511.png" alt="Qwen-Image-Edit-2511" width="60%">
</div>


## 模型下载

在使用示例脚本之前，需要先下载相应的模型。所有模型都可以从以下地址下载：

文生图模型(2512是最新的模型)
- [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512)
- [Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning)

图像编辑模型(2511是最新的模型)
- [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [LightX2V-Qwen-Image-Edit-2511](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)


## 使用方式一：使用bash脚本(强烈推荐)

环境安装推荐使用我们的docker镜像，可以参考[quickstart](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/getting_started/quickstart.html)

```
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V/scripts/qwen_image

# 运行下面的脚本之前，都需要将脚本中的lightx2v_path和model_path替换为实际路径
# 例如：export lightx2v_path=/home/user/LightX2V
# 例如：export model_path=/home/user/models/Qwen-Image-Edit-2511
```

文生图模型
```
# 推理2512文生图原始模型，默认是50步
bash qwen_image_t2i_2512.sh

# 推理2512文生图步数蒸馏模型，默认是4步，需要下载LoRA模型，然后修改config_json文件中的lora_configs的路径
bash qwen_image_t2i_2512_lora.sh

# 推理2512文生图步数蒸馏+FP8量化模型，默认是4步，需要下载FP8量化模型，然后修改config_json文件中的dit_quantized_ckpt的路径
bash qwen_image_t2i_2512_distill_fp8.sh
```

图像编辑模型
```
# 推理2511图像编辑原始模型，默认是40步
bash qwen_image_i2i_2511.sh

# 推理2511图像编辑步数蒸馏模型，默认是4步，需要下载LoRA模型，然后修改config_json文件中的lora_configs的路径
bash qwen_image_i2i_2511_lora.sh

# 推理2511图像编辑步数蒸馏+FP8量化模型，默认是4步，需要下载FP8量化模型，然后修改config_json文件中的dit_quantized_ckpt的路径
bash qwen_image_i2i_2511_distill_fp8.sh
```

## 使用方式二：安装并使用python脚本

环境安装推荐使用我们的docker镜像，可以参考[quickstart](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/getting_started/quickstart.html)

首先克隆仓库并安装依赖：

```bash
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
pip install -v -e .
```

运行步数蒸馏 + FP8 量化模型

运行 `qwen_2511_fp8.py` 脚本，该脚本使用步数蒸馏和 FP8 量化优化的模型：

```bash
cd examples/qwen_image/
python qwen_2511_fp8.py
```

该方式通过步数蒸馏技术减少推理步数，同时使用 FP8 量化降低模型大小和内存占用，实现更快的推理速度。

运行 Qwen-Image-Edit-2511 模型 + 蒸馏 LoRA

运行 `qwen_2511_with_distill_lora.py` 脚本，该脚本使用 Qwen-Image-Edit-2511 基础模型配合蒸馏 LoRA：

```bash
cd examples/qwen_image/
python qwen_2511_with_distill_lora.py
```

该方式使用完整的 Qwen-Image-Edit-2511 模型，并通过蒸馏 LoRA 进行模型优化，在保持模型性能的同时提升推理效率。
