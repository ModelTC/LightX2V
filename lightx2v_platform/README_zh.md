# lightx2v_platform

**\[ [English](README.md) | 中文 \]**

`lightx2v_platform`是独立于`lightx2v`的一个功能平台，用于对齐非nvidia的芯片后端的推理接口。支持新的芯片后端，仅需关注`lightx2v_platform`里面的实现即可。

目前已经支持的后端有：

- 寒武纪MLU590
- 沐曦C500
- 海光DCU
- 华为Ascend 910B
- AMD ROCm

相关的docker环境可以参考：https://github.com/ModelTC/LightX2V/dockerfiles/platforms

相关的使用脚本可以参考：https://github.com/ModelTC/LightX2V/scripts/platforms
