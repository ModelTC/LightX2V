# 新平台适配与结果交付

公共GSO清单、指标和计时边界保持不变。平台差异通过`scripts/platform/<platform>/`启动脚本及`configs/<platform>/`配置接入；推理统一使用LightX2V。MLU配置已用于本地评测，NVIDIA配置需要在目标设备完成验证后才能标记为已验证。

新增A100等平台时：

1. 记录卡型、UUID、驱动/运行时、PyTorch、依赖版本和容器镜像ID。
2. 新增runtime、model、paths、source-lock及evaluation-origin配置；source-lock固定LightX2V提交与文件哈希。
3. 新增运行脚本和设备监控适配，提供设备可见性、同步、计时事件、峰值显存及资源占用快照。CPU后处理容器不得映射加速设备。
4. 使用相同冻结GSO输入和权重运行smoke，再运行完整1023对象、每对象2次。结果标明平台和运行配置哈希；不可把其他平台时延复制过来。
5. 提交汇总指标、逐对象指标、失败清单、环境和配置快照；原始预测及大型产物放在独立存储。

详细步骤见[PLATFORM_ADAPTATION.md](PLATFORM_ADAPTATION.md)。
Ascend 910B 配置已加入，模型 smoke 和正式评测仍待验证。
