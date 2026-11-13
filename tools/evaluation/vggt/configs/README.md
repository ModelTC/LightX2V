# 配置归属

| 目录/文件 | 内容 |
|---|---|
| common/suite.json | 完整1023对象GSO输入清单与哈希、seed、尺寸/视图、重复与精度协议 |
| common/weight-lock.json | 同一批模型/权重文件字节数与SHA256 |
| `<platform>/runtime.json` | 平台、PyTorch设备API、镜像/Python、环境、设备映射与监控 |
| `<platform>/model.json` | 该平台的LightX2V模型后端预设 |
| `<platform>/paths.json` | 数据根与权重相对路径；默认项目data/，CLI可覆盖 |
| `<platform>/source-lock.json` | 公开源码URL、固定提交、LightX2V源码提交及文件哈希 |
| `<platform>/evaluation-origin.json` | 该平台已核对的历史来源；新平台初始为空 |

当前目录含`mlu/`已验证配置和`nvidia/`配置模板。启动时用`--platform <目录名>`选择；新增平台按项目根目录的`PLATFORM_ADAPTATION.md`接线。

公共协议不随硬件改变；平台运行、适配源码及来源独立记录。新增平台要完成设备API、监控脚本、模型后端和真机验证。新运行保存common/及所选平台配置快照和哈希。

Ascend 910B 配置与依赖镜像已提供；当前仅完成设备 API/配置检查，模型 smoke 和正式评测待执行。详见 [Ascend 配置](ascend_npu/README.md)。
