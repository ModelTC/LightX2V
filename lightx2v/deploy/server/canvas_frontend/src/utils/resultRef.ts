import { apiRequest, getSharedStore } from './apiClient';

function getAccessToken(): string | null {
  const sharedStore = getSharedStore();
  return sharedStore ? sharedStore.getState('token') : (typeof localStorage !== 'undefined' ? localStorage.getItem('accessToken') : null);
}

/** 给 URL 追加 ?token=accessToken 或 &token=...，便于 <img> 等直接访问时带鉴权 */
function appendTokenToUrl(url: string): string {
  if (!url.startsWith('./assets/')) return url;
  const token = getAccessToken();
  if (!token) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}

/** 任务结果引用：用 kind（统一，兼容旧 type / __type） */
export type LightX2VResultRef = {
  user_id?: string;
  workflow_id?: string;
  node_id?: string;
  port_id?: string;
  kind: 'task';
  task_id: string;
  output_name: string;
  is_cloud: boolean;
};

export function isLightX2VResultRef(val: any): val is LightX2VResultRef {
  return val != null && typeof val === 'object' && !Array.isArray(val) &&
    ((val as any).kind === 'task' || (val as any).type === 'task' || (val as any).__type === 'lightx2v_result') &&
    typeof (val as any).task_id === 'string' &&
    typeof (val as any).output_name === 'string';
}

export function toLightX2VResultRef(
  workflow_id: string,
  node_id: string,
  port_id: string,
  task_id: string,
  output_name: string,
  is_cloud: boolean,
): LightX2VResultRef {
  return { workflow_id, node_id, port_id, kind: 'task', task_id, output_name, is_cloud };
}

/** Collect all LightX2V result refs from a value (handles nested objects/arrays) */
export function collectLightX2VResultRefs(val: any): LightX2VResultRef[] {
  if (val == null) return [];
  if (isLightX2VResultRef(val)) return [val];
  if (Array.isArray(val)) return val.flatMap(collectLightX2VResultRefs);
  if (typeof val === 'object') return Object.values(val).flatMap(collectLightX2VResultRefs);
  return [];
}

/** 缓存 result_url 解析结果，避免同一 ref 反复请求（presigned URL 有效期较长，缓存 1 小时） */
const LIGHTX2V_RESULT_URL_CACHE_TTL_MS = 60 * 60 * 1000;
const lightX2VResultUrlCache = new Map<string, { url: string; ts: number }>();

function getCachedResultUrl(cacheKey: string): string | null {
  const entry = lightX2VResultUrlCache.get(cacheKey);
  if (!entry || Date.now() - entry.ts > LIGHTX2V_RESULT_URL_CACHE_TTL_MS) return null;
  return entry.url;
}

function setCachedResultUrl(cacheKey: string, url: string): void {
  lightX2VResultUrlCache.set(cacheKey, { url, ts: Date.now() });
}

/**
 * Fetch LightX2V result URL via canvas proxy API (task_id + output_name).
 * 使用 apiRequest 自动携带 accessToken，否则接口无权限。
 */
export async function fetchLightX2VResultUrl(
  workflow_id: string,
  node_id: string,
  port_id: string,
  task_id: string,
): Promise<string> {
  const url = `/api/v1/workflow/${encodeURIComponent(workflow_id)}/node/${encodeURIComponent(node_id)}/output/${encodeURIComponent(port_id)}/url?task_id=${encodeURIComponent(task_id)}`;
  const res = await apiRequest(url, { method: 'GET' });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`result_url failed: ${res.status} ${err}`);
  }
  const data = (await res.json().catch(() => ({}))) as { url?: string };
  if (!data.url) throw new Error('result_url missing url');
  return appendTokenToUrl(data.url);
}

/**
 * Resolve LightX2V result ref to a displayable URL.
 * 统一调用 /api/v1/workflow/{workflow_id}/node/{node_id}/output/{port_id}/url 获取 url（输入节点与 task 节点一致），不修改 output_value。
 * 加载 workflow 时 ref 可能无 workflow_id/node_id/port_id，可传入 context 补全。
 */
export async function resolveLightX2VResultRef(
  ref: LightX2VResultRef,
  context?: { workflow_id?: string; node_id?: string; port_id?: string }
): Promise<string> {
  const wfId = ref.workflow_id ?? context?.workflow_id;
  const nodeId = ref.node_id ?? context?.node_id;
  const portId = ref.port_id ?? context?.port_id;
  const cacheKey = `${wfId ?? ''}:${nodeId ?? ''}:${portId ?? ''}:${ref.task_id}:${ref.output_name}`;
  const taskId = ref.task_id;
  const cached = getCachedResultUrl(cacheKey);
  if (cached != null) return cached;

  if (wfId && nodeId && portId) {
    const url = await fetchLightX2VResultUrl(wfId, nodeId, portId, taskId);
    setCachedResultUrl(cacheKey, url);
    return url;
  }
  throw new Error(`resolveLightX2VResultRef: missing workflow_id/node_id/port_id for ref ${JSON.stringify(ref)}`);
}

/**
 * 通过 resolveLightX2VResultRef 获取 task 结果预览 URL（不再直连拼 /assets/task/result）。
 * 调用方需传入 resolver 与 context，异步返回与 resolveLightX2VResultRef 一致。
 */
export async function getResultRefPreviewUrl(
  ref: LightX2VResultRef,
  resolveFn: (r: LightX2VResultRef, ctx?: { workflow_id?: string; node_id?: string; port_id?: string }) => Promise<string>,
  context?: { workflow_id?: string; node_id?: string; port_id?: string }
): Promise<string> {
  return resolveFn(ref, context);
}
