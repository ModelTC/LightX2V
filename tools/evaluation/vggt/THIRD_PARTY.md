# 参考实现来源

源码准备入口下载`configs/mlu/source-lock.json`固定的官方VGGT源码，并应用`reference_patches/mlu.json`记录的现有MLU兼容修改。补丁是逐行编辑，含修改前后SHA256，重建结果必须与已评测文件逐字节一致。

官方仓库：https://github.com/facebookresearch/vggt。
原许可随兼容修改保存在`reference_patches/LICENSE.txt`，下载后的源码也保留其原许可。本项目其余原创评测代码遵循LightX2V仓库许可，参考材料保持原许可。

LightX2V运行依赖固定为模型PR#1675的提交，通过公开fork下载到被Git忽略的sources目录；并不把模型PR代码重复提交到评测PR。

Ascend 910B 配置、依赖镜像与参考兼容补丁已提供；当前仅完成设备 API/配置检查，模型 smoke 和正式评测待执行。详见 [Ascend 配置](configs/ascend_npu/README.md)。
