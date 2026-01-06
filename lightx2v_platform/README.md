# lightx2v_platform

**\[ English | [中文](README_zh.md) \]**

`lightx2v_platform` is a functional platform that is independent of `lightx2v`. It is used to align inference interfaces for non-NVIDIA chip backends. To support a new chip backend, you only need to focus on the implementation inside `lightx2v_platform`.

Currently supported backends include:
- Cambricon MLU590
- MetaX C500
- Hygon DCU
- Ascend 910B
- AMD ROCm
- MThreads MUSA
- Enflame S60 (GCU)

For the corresponding Docker environments, see: https://github.com/ModelTC/LightX2V/tree/main/dockerfiles/platforms

For the corresponding usage scripts, see: https://github.com/ModelTC/LightX2V/tree/main/scripts/platforms
