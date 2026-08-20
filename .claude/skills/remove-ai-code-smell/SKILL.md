---
name: remove-ai-code-smell
description: 审查并简化 AI 生成感较重的代码，重点移除内部配置路径中的过度防御、输入格式穷举、无依据的能力拦截、冗余异常和模板化注释，同时保留防止静默错误的算法与领域边界。用于用户要求“去 AI 味”、简化代码、删除过度防御、按项目约定而非不可信输入编程，或逐处审查可疑 guard、validation 和 comment 时。
---

# 去 AI 味代码审查

英文版见 [SKILL_EN.md](SKILL_EN.md)。

## 核心原则

相信已经确认的调用契约。不要把项目内部配置当成不可信的公共 API，也不要为调用方不会传入的格式穷举分支。

先区分两类检查：

- **格式防御**：兼容 `None`、大小写变体、`bool`、任意错误类型或项目不会产生的配置组合。删除这类检查；必要时用一句话写明输入约定。
- **真实边界**：防止除零、越界、重复索引、静默选择错误策略、数据损坏或资源安全问题。保留这类检查。

不要以“更安全”为由保留所有 guard。判断它失败时会发生什么：如果只是让内部错误更早并更啰嗦地报出，通常不值得；如果会继续运行并产生悄无声息的错误结果，就应保留。

## 审查步骤

1. 查看调用点、配置文件和默认值，确认输入由谁控制。
2. 找出 `isinstance` 链、`None` 回退、大小写归一化、重复 `hasattr`、能力组合拦截和长异常信息。
3. 删除不属于实际输入空间的分支，让代码直接表达正常路径。
4. 对不明显但稳定的约定，只留一句简短注释；不要用注释复述代码。
5. 保留会阻止静默错误的最小边界检查。
6. 保持改动局部，运行对应的语法、lint 和聚焦测试。

## 常见简化

将内部配置的格式穷举：

```python
if value is None:
    value = 0
if isinstance(value, str):
    if value.lower() != "all":
        raise ValueError(...)
    count = num_blocks
elif isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(...)
else:
    count = value
```

改成契约加正常路径：

```python
# Resident block counts are integers or "all".
count = num_blocks if value == "all" else value
```

删除只为未使用组合设置的提前拦截：

```python
if resident_blocks and not event_offload:
    raise ValueError(...)
```

如果维护者确实需要知道限制，在相关路径旁留一句备注即可：

```python
# Event block offload does not support lazy_load.
```

## 应当保留的检查

保留由算法本身要求的边界。例如 resident block 数量超过总层数会产生重复索引或错误布局，因此范围检查不是过度防御：

```python
if not 0 <= count <= num_blocks:
    raise ValueError(...)
```

同样保留以下检查：

- 公共 API、用户输入、网络数据和不可信文件的校验。
- 分布式 collective、显存容量、文件覆盖和数据持久化等安全边界。
- 错误输入会被继续接受并产生错误结果的策略或枚举检查。

不要机械删除防御。先证明调用契约，再精简代码。
