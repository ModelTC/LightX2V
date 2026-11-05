# TRELLIS.2-4B BF16 推理

[官方项目](https://github.com/microsoft/TRELLIS.2)。本入口使用 512 流水线，输入为单张图像。

模型目录应包含 `pipeline.json`、`ckpts/`、`base/`、`dinov3/`、`rembg/`。沿用原提交的公开模型版本：

| 目录 | 仓库 | Revision |
|---|---|---|
| 根目录 | microsoft/TRELLIS.2-4B | af44b45f2e35a493886929c6d786e563ec68364d |
| base | microsoft/TRELLIS-image-large | 459ee480d7545f873a0768364bc5f5443eb1ff5e |
| dinov3 | camenduru/dinov3-vitl16-pretrain-lvd1689m | 3c276edd87d6f6e569ff0c4400e086807d0f3881 |
| rembg | ZhengPeng7/BiRefNet | e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 |

BiRefNet 目录需包含其公开 Python 模型文件，按现有逻辑在本地加载。`base_model_path`、`dino_model_path` 和 `rembg_model_path` 可在 JSON 中指定。

```bash
python -m pip install -r scripts/trellis2/requirements-mlu.txt
MODEL_PATH=/path/to/TRELLIS.2 INPUT_PATH=/path/to/image.png bash scripts/trellis2/run_trellis2.sh
```

通过 `PLATFORM` 和对应平台的设备可见性变量选择设备。脚本使用 BF16 和 seed 42，可通过 `SEED`、`SAVE_RESULT_PATH` 或末尾的 CLI 参数覆盖。预处理、结构/形状/纹理三阶段 Euler/CFG 采样和解码沿用已有实现。

保存结果为几何 GLB，Runner 同时返回纹理体素。本集成限于 512，未包含纹理烘焙和 cascade/1024/1536。模型权重声明与加载位于 `weights/`，计算位于 `infer/`；公共稀疏算子仍位于 `common/ops/sparse/`。
