# Lightx2v-ComfyUI便携环境（Windows）

此说明将指导您如何下载与使用便携版的Lightx2v-ComfyUI环境，如此可以免去手动配置环境的步骤，适用于想要在Windows系统下快速开始体验使用Lightx2v加速视频生成的用户。



## 下载Windows便携环境：

- [百度网盘下载](https://pan.baidu.com/s/1FVlicTXjmXJA1tAVvNCrBw?pwd=wfid)，提取码：wfid

便携环境中已经打包了所有Python运行相关的依赖，也包括ComfyUI和LightX2V的代码及其相关依赖，下载后解压即可使用。

解压后对应的文件目录说明如下：

```shell
lightx2v_env
├──📂 ComfyUI                    # ComfyUI代码
├──📂 portable_python312_embed   # 独立的Python环境
└── run_nvidia_gpu.bat            # Windows启动脚本（双击启动）
```

## 启动ComfyUI

直接双击run_nvidia_gpu.bat文件，系统会打开一个Command Prompt窗口并运行程序，一般第一次启动时间会比较久，请耐心等待，启动完成后会自动打开浏览器并出现ComfyUI的前端界面。

![i2v示例工作流](../../../../assets/figs/portabl_windows/pic1.png)

LightX2V-ComfyUI的插件使用的是，[ComfyUI-Lightx2vWrapper](https://github.com/ModelTC/ComfyUI-Lightx2vWrapper)，示例工作流可以从此项目中获取。

## 已测试显卡（offload模式）

- 测试模型`Wan2.1-I2V-14B-480P`

| 显卡型号   | 任务类型    | 显存容量   | 实际最大显存占用 | 实际最大内存占用  |
|:----------|:-----------|:-----------|:--------       |:----------   |
| 3090Ti    | I2V        | 24G        | 6.1G              | 7.1G         |
| 3080Ti    | I2V        | 12G        | 6.1G             | 7.1G         |
| 3060Ti    | I2V        | 8G         | 6.1G             | 7.1G         |


### 环境打包和使用参考
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Portable-Windows-ComfyUI-Docs](https://docs.comfy.org/zh-CN/installation/comfyui_portable_windows#portable-%E5%8F%8A%E8%87%AA%E9%83%A8%E7%BD%B2)