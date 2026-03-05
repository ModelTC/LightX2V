import type { WorkflowState } from '../types';

/**
 * 自动扫描本目录下除 index.ts 外的所有 .ts 文件，每个文件需 default 导出一个 WorkflowState。
 * 新增预设：在本目录新建 .ts 文件，导出 default 即可，无需修改本文件。
 */
const presetModules = import.meta.glob<{ default: WorkflowState }>('./*.ts', { eager: true });

export const PRESET_WORKFLOWS: WorkflowState[] = Object.entries(presetModules)
  .filter(([path]) => !path.endsWith('index.ts'))
  .map(([, mod]) => mod.default)
  .filter((w): w is WorkflowState => w != null && typeof w === 'object' && 'id' in w);
