/**
 * output_value 统一为以 port_id 为键的字典：{ [portID]: entry | list[entry] }
 * - 输入类 image-input 的 out-image 可为多图：list[entry]
 * - 其他端口统一为单条 entry
 * entry 必须为 { kind: "file", ... } 或 { kind: "task", ... }，不得将 entry 存成 url 字符串。
 * - 图像: output_value["out-image"] = entry | entry[]  (image-input 可为数组)
 * - 音频: output_value["out-audio"] = entry
 * - 视频: output_value["out-video"] = entry
 * - 文本(文件): output_value["out-text"] = entry
 * - LightX2V 任务: output_value["out-image"] = { kind: "task", task_id, output_name, ... }
 * 兼容旧版：单值、数组或非 port-keyed 对象。
 */

import { TOOLS } from '../../constants';

export const INPUT_PORT_IDS: Record<string, string> = {
  'image-input': 'out-image',
  'audio-input': 'out-audio',
  'video-input': 'out-video',
  'text-input': 'out-text'
};

/** 判断是否为 port-keyed 结构（至少有一个键为 out-*，避免 task ref { kind, task_id, output_name } 被误判） */
export function isPortKeyedOutputValue(ov: any): ov is Record<string, any> {
  if (ov == null || typeof ov !== 'object' || Array.isArray(ov)) return false;
  const keys = Object.keys(ov);
  return keys.some(k => typeof k === 'string' && k.startsWith('out-'));
}

/** 将旧版 output_value（单值/数组）规范为 port-keyed。不修改原对象。 */
export function normalizeOutputValueToPortKeyed(
  outputValue: any,
  toolId: string
): Record<string, any> {
  const portId = INPUT_PORT_IDS[toolId];
  if (!portId) {
    if (outputValue != null && typeof outputValue === 'object' && !Array.isArray(outputValue)) return { ...outputValue };
    return {};
  }
  if (isPortKeyedOutputValue(outputValue)) return { ...outputValue };
  if (outputValue == null) return {};
  const isArray = Array.isArray(outputValue);
  const arr = isArray ? outputValue : [outputValue].filter(Boolean);
  const isImage = portId === 'out-image';
  return { [portId]: isImage ? arr : arr[0] ?? null };
}

/** 从节点取某 port 的当前值（兼容旧版单值/数组；兼容曾以 __json__ 整存时的单 port 展开） */
export function getOutputValueByPort(node: { output_value?: any; data?: { value?: any }; tool_id: string }, portId: string): any {
  const ov = node.output_value ?? node.data?.value;
  if (ov == null) return undefined;
  if (isPortKeyedOutputValue(ov) && portId in ov) return ov[portId];
  // 曾按 __json__ 保存时：后端写入了 output_value["__json__"] = { "out-image": ref }，按 port 取不到，从 __json__ 里取
  const jsonVal = ov['__json__'];
  if (jsonVal != null && typeof jsonVal === 'object' && portId in jsonVal) return jsonVal[portId];
  const toolId = node.tool_id;
  const singlePortId = INPUT_PORT_IDS[toolId];
  if (singlePortId === portId) return ov;
  return undefined;
}

/** 判断是否为合法的 entry（仅允许 kind === 'file' | 'task'，禁止 url 字符串或 kind === 'url'） */
export function isValidOutputEntry(val: any, toolId: string): boolean {
  if (val == null) return false;
  if (typeof val === 'string') {
    if (val.startsWith('data:')) return true;
    // 图像输入不允许 url 字符串
    // 因为out-image为列表，需要通过改变output_value列表删除某张图
    // 解析成 url 之后，就无法转回 file_entry
    if (toolId == 'image-input') return false;
    else return true;
  }
  if (typeof val !== 'object' || Array.isArray(val)) return false;
  const k = (val as { kind?: string }).kind;
  return k === 'file' || k === 'task';
}

/** 设置节点 output_value 中某 port 的值，返回新的 port-keyed output_value（不修改原节点）。value 应为 entry(ref) 或 entry[]，不得为 url 字符串。 */
export function setOutputValueByPort(
  currentOutputValue: any,
  toolId: string,
  portId: string,
  value: any
): Record<string, any> {
  const base = isPortKeyedOutputValue(currentOutputValue) ? { ...currentOutputValue } : {};
  if (value === undefined || value === null) {
    delete base[portId];
    return base;
  }
  base[portId] = value;
  return base;
}

/** 从 save API 返回的 ref 中提取仅用于写入 output_value 的 entry（不含 file_url，保证 kind 为 file）。 */
export function toFileRefForOutputValue(
  saveRef: { kind?: string; file_id: string; mime_type?: string; ext?: string; run_id?: string; file_url?: string; user_id?: string; workflow_id?: string } | null): { kind: 'file'; file_id: string; mime_type?: string; ext?: string; run_id?: string; user_id?: string; workflow_id?: string } | null {
  if (!saveRef || !saveRef.file_id) return null;
  return {
    kind: 'file',
    file_id: saveRef.file_id,
    user_id: saveRef.user_id,
    workflow_id: saveRef.workflow_id,
    run_id: saveRef.run_id,
    ext: saveRef.ext,
    mime_type: saveRef.mime_type,
  };
}

/** 规范 ref 为 kind（写入用）。兼容读 type。 */
export function normalizeRefKind(ref: any): any {
  if (ref == null || typeof ref !== 'object') return ref;
  const next = { ...ref };
  if (next.type === 'file' && next.kind === undefined) next.kind = 'file';
  if ((next.type === 'task' || next.__type === 'lightx2v_result') && next.kind === undefined) next.kind = 'task';
  if (next.kind === 'task') {
    delete (next as any).type;
    delete (next as any).__type;
  }
  return next;
}
