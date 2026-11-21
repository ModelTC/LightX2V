---
name: contribute-code
description: 新增或修改 LightX2V 代码时遵循的贡献指南，涵盖新模型接入、非 NVIDIA 平台推理支持、代码可读性和 pre-commit 检查。
---

# 贡献 LightX2V 代码

[English](SKILL.md) | 简体中文

- 新增模型支持时，先查看仓库里同类模型的接入方式，按需参考 [support-new-model](../support-new-model/SKILL.md)。
- 适配非 NVIDIA 平台的推理时，优先将专用算子、后端依赖和设备能力检查放在 `lightx2v_platform`，通过已有的后端注册和算子接口接入。`lightx2v` 中保留通用接口和共享推理逻辑。
- 复用已有实现，让正常路径清晰易读。按实际调用约定处理输入，去掉重复校验和没有实际触发场景的兜底，保留必要的边界检查。参考 [remove-ai-code-smell](../remove-ai-code-smell/SKILL.md)。

## 交付前检查

删除本次任务中 AI 为中间验证编写的所有临时测试文件。

交付前必须在仓库根目录运行 `pre-commit run --all-files` 做全量检查。环境缺少工具时，运行 `pip install ruff pre-commit`。

检查钩子的自动修正，处理报错后重跑全量检查，确认检查通过。
