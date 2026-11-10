# 配置归属

| 目录/文件 | 内容 |
|---|---|
| common/suite.json | 40对象GSO输入、seed、尺寸/视图、重复与精度协议 |
| common/weight-lock.json | 同一批模型/权重文件字节数与SHA256 |
| mlu/runtime.json | 镜像、容器Python、环境、设备映射与监控 |
| mlu/model.json | LightX2V MLU模型后端预设 |
| mlu/paths.json | 数据根与权重相对路径；默认项目data/，CLI可覆盖 |
| mlu/source-lock.json | 公开源码URL、固定提交、参考兼容修改及文件哈希 |
| mlu/evaluation-origin.json | 已证明不改输入/计算的历史来源与配置迁移对应 |

公共协议不随硬件改变；平台运行、适配源码及来源独立记录。当前只有MLU后端，A100等新增平台需要按[PLATFORMS.md](../PLATFORMS.md)完成代码接线和真实测试。新运行保存common/及所选平台配置快照和哈希。
