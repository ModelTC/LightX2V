# 少写接线，聚焦未决判断

首次接线或复核成本偏高时阅读。先复用当前工具的生产者、原始记录和已有任务adapter；只补真正缺少的输入/状态、执行监督和完成边界，不为每个模型复制worker、计时器或新建公共监督schema。

## 选择最短已有路径

| 当前问题 | 复用入口 | 任务仍负责 |
|---|---|---|
| 算子/backend局部比较 | [算子benchmark](../../../../tools/benchmarks/README.md)，raw交给 `operator_comparison` | 真实输入/replay、实际后端调用及运行监督 |
| 真实block、模块或阶段 | [PreparedModule与benchmark](../../../../tools/benchmarks/README.md#完整模型层tpsp-比较)，report/raw交给 `module_comparison` | 生产对象/权重、call、reset、同步及完整输出参考；不另写计时循环 |
| 完整请求或输出流水 | 正式请求入口、现有任务collector和原始记录 | 完成边界、每个正常样本的worker/输入/输出归属、退出及收尾；不拿局部report冒充请求结果 |
| 交付/复验 | [交付验收](../../../../tools/EXPLORATION_REPORT_CHECKS.md)，已有工作包/报告和固定消费者 | 范围及执行依赖，只适配确实不同的任务格式 |

同一真实call能由PreparedModule完整保留边界时，复用其预热、计时、rank聚合、显存和输出预检。接口能接call不证明已覆盖落盘、后台解码、所有stream或跨请求状态；沿实际完成点确认，不相容的请求保留自身生产者和collector。先在最小充分范围证明接线，不为学接口加载整网。

## 从真实CPU生产者到读取器

以下是合成输入烟测，不是模型候选或优化收益实验。保存为 `task_probe.py`，在有PyTorch的项目环境从仓库根运行 `python -B task_probe.py /new/output`。目录须不存在；脚本只用CPU，不安装依赖。

```python
import hashlib
import json
import sys
from pathlib import Path

import torch
from tools.benchmarks.module_bench import PreparedModule, benchmark
from tools.exploration_acceptance import Inputs
from tools.exploration_evidence import module_comparison

output = Path(sys.argv[1]).resolve()
output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
weight = torch.eye(8)
original = x.clone()
torch.save({"input": x, "weight": weight}, output / "inputs.pt")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

def check(value):
    if not torch.equal(value, original) or not torch.equal(x, original):
        raise ValueError("CPU smoke output/input changed")
    return {"status": "strict_equal", "scope": "synthetic 8x8 CPU output", "input_unchanged": True}

prepared = PreparedModule(
    call=lambda: x @ weight, reset=lambda: None, synchronize=lambda: None,
    check=check, device="cpu",
    identity={
        "target": {"completion_boundary": "complete CPU matrix multiply"},
        "input": {"kind": "synthetic", "sha256": sha(output / "inputs.pt")},
        "source": {"script_sha256": sha(Path(__file__)), "torch": str(torch.__version__)},
    },
)
pairs = []
for index in range(2):
    report, raw = benchmark(prepared, warmup=2, iterations=3, repeats=2)
    report["factory"] = {"entry": str(Path(__file__).resolve()), "config": {"implementation": "torch.mm"}}
    report_path, raw_path = output / f"report{index}.json", output / f"raw{index}.jsonl"
    report_path.write_text(json.dumps(report, allow_nan=False) + "\n")
    raw_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in raw))
    pairs.append((str(report_path), str(raw_path)))
fact = module_comparison(Inputs(), [pairs[0]], [pairs[1]])
(output / "FACT.json").write_text(json.dumps(fact, indent=2, allow_nan=False) + "\n")
print(json.dumps({"kind": fact["kind"], "scope": fact["subject"]["scope"], "run_validated": fact["valid"]}))
```

两组为同一实现，只验证生产者、原生文件和读取器串接，不评价优化。读数仍为 `valid=False`：命令成功、数值预检和样本一致不等于运行监督已核。真实任务替换输入与生产call/状态/输出参考，保留原生格式；collector核真实命令、输入/配置、输出及监督/退出/收尾原件后才能认可运行有效性。无PyTorch时保留未运行，不伪造raw。

## 语义复核由Agent先完成

语义判断不能由hash或非空理由替代，但不等于默认让开发者逐项审核。执行Agent复算事实、追producer/consumer及补缺；新上下文复核者从原题、真实入口、方法库和冻结证据核必要子项、适用/负结果范围、优先级、粒度与停止依据。用现有工作包/报告组织聚焦未决判断的视图，不另维护活动状态：

| 要判断什么 | 最小依据 | 实际复核动作 |
|---|---|---|
| 事实是否成立 | 最终参数合并/输入变换、调用/consumer、正常输出和命令入口 | 做可改变判断的最小追踪或复算，核生产者而非只核声明 |
| 取舍是否有据 | 问题、决定、成本来源、最小充分范围、未知和重开条件 | 主动找会推翻结论的条件或反例；不足则补证或保留未知 |
| 谁需决定 | 已授权目标/资源/质量边界与具体取舍 | 技术核查和修复由Agent推进；新增授权、主观画面接受及目标取舍才交开发者 |

复用review的topics、逐项决定及observed_inputs，必修发现绑定输入与整改，不用“认可”文字代替行动。先冻结初审再交换作者解释，增量只核受影响判断。完整发现复核仍须看原题、真实代码和方法库，不能只给已登记项让其确认。此视图减少找材料的成本，不自动证明语义真实或发现充分。
