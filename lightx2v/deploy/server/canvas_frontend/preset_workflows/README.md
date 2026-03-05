# 预设工作流

本目录下的 `.ts` 文件会被 **自动扫描** 并汇总为 `PRESET_WORKFLOWS`，无需在 `index.ts` 里手写 import。

## 新增预设

1. 在本目录新建一个 `.ts` 文件（例如 `MyNewPreset.ts`）。
2. 按下面格式编写并 **默认导出** 一个 `WorkflowState`：

```ts
import { WorkflowState, NodeStatus } from '../types';

const workflow: WorkflowState = {
  id: 'preset-my-new',
  name: '我的新预设',
  updatedAt: Date.now(),
  isDirty: false,
  isRunning: false,
  globalInputs: {},
  connections: [ /* ... */ ],
  nodes: [ /* ... */ ],
};

export default workflow;
```

3. 保存即可，`index.ts` 会自动识别该文件并加入列表。

**注意**：不要修改 `index.ts`，也不要让预设文件名与 `index.ts` 重名。
