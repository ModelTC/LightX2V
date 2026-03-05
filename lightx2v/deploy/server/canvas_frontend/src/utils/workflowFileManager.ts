/**
 * 工作流文件管理工具
 * 使用新的统一接口（基于 data_store.outputs）
 *
 * 仅前端（standalone）时 IndexedDB 用法：
 * - 写入：uploadNodeInputFile(文件上传) 或 persistDataUrlToLocal(data: URL) → 存 Blob，得到 local://key
 * - 读取：getLocalFileDataUrl(local://key) → 得到 data URL，用于 <img>/预览/执行入参
 * - 列表缩略图：WorkflowCard 里 previewImage 为 local:// 时用 getLocalFileDataUrl 异步解析后显示
 */

import { apiRequest } from './apiClient';
import { isStandalone } from '../config/runtimeMode';
import { getEntryPortKeyedValue } from './historyEntry';
import { getAssetPath } from './assetPath';
import { isLightX2VResultRef, resolveLightX2VResultRef } from './resultRef';

const EXT_TO_MIME: Record<string, string> = {
  '.txt': 'text/plain', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.gif': 'image/gif', '.webp': 'image/webp', '.mp4': 'video/mp4', '.webm': 'video/webm',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg', '.json': 'application/json',
};
function extToMime(ext: string): string {
  const norm = ext?.startsWith('.') ? ext : (ext ? `.${ext}` : '');
  return EXT_TO_MIME[norm] ?? 'application/octet-stream';
}

const LOCAL_FILES_DB = 'canvas_local_files';
const LOCAL_FILES_STORE = 'files';

function openLocalDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(LOCAL_FILES_DB, 1);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(LOCAL_FILES_STORE)) {
        db.createObjectStore(LOCAL_FILES_STORE);
      }
    };
  });
}

/**
 * 纯前端模式：从 IndexedDB 读取 local://key 对应的文件，返回 data URL
 */
export async function getLocalFileDataUrl(key: string): Promise<string | null> {
  if (!key) return null;
  const rawKey = key.startsWith('local://') ? key.slice(8) : key;
  try {
    const db = await openLocalDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(LOCAL_FILES_STORE, 'readonly');
      const store = tx.objectStore(LOCAL_FILES_STORE);
      const req = store.get(rawKey);
      req.onerror = () => { db.close(); reject(req.error); };
      req.onsuccess = () => {
        db.close();
        const blob = req.result as Blob | undefined;
        if (!blob) {
          resolve(null);
          return;
        }
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(blob);
      };
    });
  } catch (e) {
    console.warn('[WorkflowFileManager] getLocalFileDataUrl failed:', rawKey, e);
    return null;
  }
}

/**
 * 纯前端模式：将 Blob 存入 IndexedDB，返回 local://key
 */
export async function saveLocalFile(key: string, blob: Blob): Promise<void> {
  try {
    const db = await openLocalDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(LOCAL_FILES_STORE, 'readwrite');
      const store = tx.objectStore(LOCAL_FILES_STORE);
      store.put(blob, key);
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  } catch (e) {
    console.error('[WorkflowFileManager] saveLocalFile failed:', key, e);
    throw e;
  }
}

/**
 * 从 base64 data URL 提取 MIME 类型和扩展名
 */
function getMimeTypeAndExt(dataUrl: string): { mimeType: string; ext: string } {
  const match = dataUrl.match(/data:([^;]+);base64,/);
  if (match) {
    const mimeType = match[1];
    const extMap: Record<string, string> = {
      'image/png': '.png',
      'image/jpeg': '.jpg',
      'image/jpg': '.jpg',
      'image/gif': '.gif',
      'image/webp': '.webp',
      'video/mp4': '.mp4',
      'video/webm': '.webm',
      'audio/mpeg': '.mp3',
      'audio/wav': '.wav',
      'audio/ogg': '.ogg',
      'text/plain': '.txt',
      'application/json': '.json'
    };
    const ext = extMap[mimeType] || '.bin';
    return { mimeType, ext };
  }
  return { mimeType: 'application/octet-stream', ext: '.bin' };
}

/**
 * 将 base64 data URL 转换为 Blob
 */
function dataURLToBlob(dataUrl: string): Blob {
  const arr = dataUrl.split(',');
  const mimeMatch = arr[0].match(/:(.*?);/);
  const mimeType = mimeMatch ? mimeMatch[1] : 'application/octet-stream';
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mimeType });
}

/**
 * 仅前端模式：将 data: URL 存入 IndexedDB，返回 local://key，供列表缩略图等使用。
 * 有后端时直接返回原 dataUrl（不写入 IndexedDB）。
 */
export async function persistDataUrlToLocal(dataUrl: string, keyPrefix: string): Promise<string> {
  if (!dataUrl || typeof dataUrl !== 'string' || !dataUrl.startsWith('data:')) return dataUrl;
  if (!isStandalone()) return dataUrl;
  try {
    const blob = dataURLToBlob(dataUrl);
    const key = `${keyPrefix}_${crypto.randomUUID()}`;
    await saveLocalFile(key, blob);
    return `local://${key}`;
  } catch (e) {
    console.warn('[WorkflowFileManager] persistDataUrlToLocal failed:', e);
    return dataUrl;
  }
}

export type SaveNodeOutputResult = {
  kind?: string;
  file_id?: string;
  file_url?: string;
  url?: string;
  mime_type?: string;
  ext?: string;
  task_id?: string;
  output_name?: string;
  is_cloud?: boolean;
  entries?: SaveNodeOutputResult[];
} | null;

/**
 * 将原始 value 包装为统一的 { type, data } 格式。
 * - data URL (data:xxx;base64,...) => { type: "base64", data: "data:..." }
 * - 纯文本字符串 => { type: "text", data: "..." }
 * - task ref ({ kind: "task", ... }) => { type: "task", data: {...} }
 * - file ref ({ kind: "file", ... }) => { type: "file", data: {...} }
 * - URL 字符串 (http/https/./assets) => { type: "url", data: "..." }
 * - 数组（多图）=> 每项各自 wrap 后返回数组
 */
/** 本地 task 结果 URL（./assets/task/result?task_id=xxx&name=yyy）解析为 task ref；仅云端（模型名 -cloud）用 type: 'url' */
function parseLocalTaskResultUrl(url: string): { kind: 'task'; task_id: string; output_name: string; is_cloud: false } | null {
  if (typeof url !== 'string' || !url.includes('task/result') || !url.includes('task_id=') || !url.includes('name=')) return null;
  const q = url.indexOf('?');
  if (q === -1) return null;
  const params = new URLSearchParams(url.slice(q));
  const task_id = params.get('task_id');
  const name = params.get('name');
  if (!task_id || !name) return null;
  return { kind: 'task', task_id, output_name: name, is_cloud: false };
}

function wrapOutputData(value: any): { type: string; data: any } | Array<{ type: string; data: any }> {
  if (Array.isArray(value)) {
    return value.map((item: any) => wrapOutputData(item) as { type: string; data: any });
  }
  if (value instanceof Blob) {
    // Blob 不应出现在这里——调用前应已转为 data URL
    throw new Error('Blob values must be converted to data URL before wrapOutputData');
  }
  if (typeof value === 'string') {
    if (value.startsWith('data:')) {
      return { type: 'base64', data: value };
    }
    if (
      value.startsWith('http://') || value.startsWith('https://') ||
      value.startsWith('./assets/') || value.startsWith('/assets/') ||
      value.startsWith('/api/')
    ) {
      return { type: 'url', data: value };
    }
    // 纯文本
    return { type: 'text', data: value };
  }
  if (typeof value === 'object' && value !== null) {
    const kind = value.kind || value.type;
    if (kind === 'task') return { type: 'task', data: value };
    if (kind === 'file') return { type: 'file', data: value };
    // 其他 object 当 JSON 文本存
    return { type: 'text', data: JSON.stringify(value) };
  }
  return { type: 'text', data: String(value ?? '') };
}

/**
 * 按端口保存节点输出。每个 port 调用一次 POST .../output/{port_id}/save。
 * 请求体格式：{ run_id: string, output_data: { type, data } | Array<{ type, data }> }
 * 多端口对象（如 text-generation customOutputs）按每个端口各调用一次 save，run_id 保持一致。
 */
export async function saveNodeOutputs(
  workflowId: string,
  nodeId: string,
  outputs: Record<string, string | object | any[]>,
  runId?: string
): Promise<Record<string, any> | null> {
  if (isStandalone()) return null;
  if (!outputs || Object.keys(outputs).length === 0) return null;
  try {
    console.log('[WorkflowFileManager] saveNodeOutputs:', outputs);
    // 准备每个端口要发送的数据
    const toSend: Array<{ portId: string; wrappedOutput: any }> = [];
    for (const [portId, value] of Object.entries(outputs)) {
      if (value === undefined || value === null) continue;
      if (typeof value === 'string' && value.length === 0) continue;
      let resolved: any = value;
      if (value instanceof Blob) {
        resolved = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(value);
        });
      }
      // 非仅前端且为 is_cloud 的 x2v 任务：用云端 result_url 得到 URL，以 { type: "url", data: url } 发给后端（云端以模型名 -cloud 判断）
      if (!isStandalone() && isLightX2VResultRef(resolved) && (resolved as { is_cloud?: boolean }).is_cloud) {
        resolved = await resolveLightX2VResultRef(resolved);
      } else if (!isStandalone() && Array.isArray(resolved)) {
        resolved = await Promise.all(resolved.map(async (item: any) => {
          if (isLightX2VResultRef(item) && (item as { is_cloud?: boolean }).is_cloud) {
            return await resolveLightX2VResultRef(item);
          }
          return item;
        }));
      }
      // 本地 task 结果 URL（./assets/task/result?task_id=...&name=...）统一转为 task ref(is_cloud: false)，避免误用 type: 'url'
      if (typeof resolved === 'string') {
        const taskRef = parseLocalTaskResultUrl(resolved);
        if (taskRef) resolved = taskRef;
      } else if (resolved && typeof resolved === 'object' && (resolved as { kind?: string }).kind === 'url' && typeof (resolved as { url?: string }).url === 'string') {
        const taskRef = parseLocalTaskResultUrl((resolved as { url: string }).url);
        if (taskRef) resolved = taskRef;
      } else if (Array.isArray(resolved)) {
        resolved = resolved.map((item: any) => {
          if (typeof item === 'string') {
            const tr = parseLocalTaskResultUrl(item);
            return tr ?? item;
          }
          if (item && typeof item === 'object' && item.kind === 'url' && typeof item.url === 'string') {
            const tr = parseLocalTaskResultUrl(item.url);
            return tr ?? item;
          }
          return item;
        });
      }
      const wrapped = wrapOutputData(resolved);
      toSend.push({ portId, wrappedOutput: wrapped });
    }
    if (toSend.length === 0) return null;

    const out: Record<string, any> = {};
    for (const { portId, wrappedOutput } of toSend) {
      const body: any = { output_data: wrappedOutput };
      if (runId) body.run_id = runId;
      const response = await apiRequest(
        `/api/v1/workflow/${workflowId}/node/${nodeId}/output/${encodeURIComponent(portId)}/save`,
        { method: 'POST', body: JSON.stringify(body) }
      );
      if (!response.ok) {
        const contentType = response.headers.get('content-type') || '';
        let errorMessage = `Failed to save node output ${nodeId}/${portId}: ${response.status} ${response.statusText}`;
        if (contentType.includes('application/json')) {
          try {
            const errorData = await response.json();
            errorMessage = (errorData as { message?: string }).message || errorMessage;
          } catch (_e) { /* ignore */ }
        }
        throw new Error(errorMessage);
      }
      const data = (await response.json()) as {
        user_id?: string;
        workflow_id?: string;
        kind?: string;
        file_id?: string;
        task_id?: string;
        run_id?: string;
        ouptut_name?: string;
        mime_type?: string;
        ext?: string;
        entries?: Array<{
          user_id?: string;
          workflow_id?: string;
          kind?: string;
          file_id?: string;
          task_id?: string;
          run_id?: string;
          ouptut_name?: string;
          mime_type?: string;
          ext?: string;
        }>;
      };

      // 多图 list 响应：后端返回 { entries: [...] }
      if (Array.isArray(data?.entries) && data.entries.length > 0) {
        out[portId] = data.entries;
      } else {
        out[portId] = data;
      }
    }
    return out;
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error(`[WorkflowFileManager] Error saving node outputs for ${nodeId}:`, msg);
    throw error;
  }
}

/**
 * 点击历史记录条目时专用：按 entry 的 output_value 格式（kind 为 task 或 file）调用 port/save。
 * kind=task 时上传 { output_data: { type: "task", data: { task_id, output_name } } }；
 * kind=file 时上传 { output_data: { type: "file", data: { file_id } } }。
 * run_id 通过 query_params 传递（通常为 history entry 的 id）。
 */
export async function saveNodeOutputsFromHistoryEntry(
  workflowId: string,
  nodeId: string,
  portKeyed: Record<string, { kind?: string; task_id?: string; output_name?: string; file_id?: string }>,
  runId?: string
): Promise<void> {
  if (isStandalone()) return;
  const outputPortKeys = Object.keys(portKeyed).filter((k) => k.startsWith('out-') || k === '__json__');
  const runIdParam = runId ? encodeURIComponent(runId) : '';
  for (const portId of outputPortKeys) {
    const v = portKeyed[portId];
    if (v == null || typeof v !== 'object') continue;
    const kind = v.kind ?? (v as any).type;
    let outputData: { type: string; data: Record<string, string> } | null = null;
    if ((kind === 'task' || kind === 'lightx2v_result') && (v.task_id != null || (v as any).taskId != null)) {
      outputData = {
        type: 'task',
        data: {
          task_id: v.task_id ?? (v as any).taskId,
          output_name: v.output_name ?? (v as any).outputName ?? 'output',
        },
      };
    } else if (kind === 'file' && (v.file_id != null || (v as any).fileId != null)) {
      outputData = {
        type: 'file',
        data: {
          file_id: v.file_id ?? (v as any).fileId,
        },
      };
    }
    if (!outputData) continue;
    const body = { output_data: outputData };
    const url = `/api/v1/workflow/${workflowId}/node/${nodeId}/output/${encodeURIComponent(portId)}/save${runIdParam ? `?run_id=${runIdParam}` : ''}`;
    const response = await apiRequest(url, { method: 'POST', body: JSON.stringify(body) });
    if (!response.ok) {
      const contentType = response.headers.get('content-type') || '';
      let errMsg = `Save history entry ${nodeId}/${portId}: ${response.status} ${response.statusText}`;
      if (contentType.includes('application/json')) {
        try {
          const errData = await response.json();
          errMsg = (errData as { message?: string }).message ?? errMsg;
        } catch (_) {}
      }
      throw new Error(errMsg);
    }
  }
}

/** 按 (workflowId, nodeId, portId, fileId?) 或 (workflowId, nodeId, portId) 永久缓存 */
const nodeOutputUrlCache = new Map<string, string>();
/** 同一 node+port+run 只允许一个请求在进行 */
const nodeOutputUrlInFlight = new Map<string, Promise<string | string[] | null>>();

function parseFileIdFromUrl(url: string): string | null {
  const m = url.match(/[?&]file_id=([^&]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

/**
 * 获取节点某端口输出的展示 URL（统一入口）。调用后端 /output/{port_id}/url 接口获取最终 URL。
 * 支持 file_id/run_id（文件引用）或 task_id（任务结果），统一走 /api/v1/workflow/.../node/.../output/.../url?task_id=xxx&file_id=xxx。
 * 传入 fileId 时按 (workflowId, nodeId, portId, fileId) 缓存并返回匹配项；多项时后端返回 urls，按 file_id 匹配。
 */
export async function getNodeOutputUrl(
  workflowId: string,
  nodeId: string,
  portId: string,
  fileId?: string,
  runId?: string,
  taskId?: string,
  fileIdForTask?: string
): Promise<string | null> {
  if (isStandalone()) return null;
  const cacheKey = taskId
    ? `${workflowId}:${nodeId}:${portId}:task:${taskId}:${fileIdForTask ?? ''}`
    : (fileId ? `${workflowId}:${nodeId}:${portId}:${fileId}` : `${workflowId}:${nodeId}:${portId}:${runId ?? ''}`);
  const cached = nodeOutputUrlCache.get(cacheKey);
  if (cached != null) return cached;
  const inFlightKey = taskId
    ? `${workflowId}:${nodeId}:${portId}:task:${taskId}`
    : `${workflowId}:${nodeId}:${portId}:${runId ?? ''}`;
  let promise = nodeOutputUrlInFlight.get(inFlightKey);
  if (promise) {
    const result = await promise;
    if (result == null) return null;
    if (Array.isArray(result)) {
      if (fileId) {
        const single = nodeOutputUrlCache.get(`${workflowId}:${nodeId}:${portId}:${fileId}`);
        return single ?? null;
      }
      return result[0];
    }
    return result;
  }
  promise = (async (): Promise<string | string[] | null> => {
    try {
      const params = new URLSearchParams();
      if (runId) params.set('run_id', runId);
      if (fileId) params.set('file_id', fileId);
      if (taskId) params.set('task_id', taskId);
      if (fileIdForTask) params.set('file_id', fileIdForTask);
      const qs = params.toString() ? `?${params.toString()}` : '';
      const response = await apiRequest(
        `/api/v1/workflow/${workflowId}/node/${encodeURIComponent(nodeId)}/output/${encodeURIComponent(portId)}/url${qs}`
      );
      if (!response.ok) return null;
      const data = await response.json();
      const toFinal = (url: string) => getAssetPath(url.startsWith('./') ? url.slice(1) : url);
      // 多项输出（如多图）
      if (Array.isArray(data?.urls) && data.urls.length > 0) {
        const finalUrls = data.urls.map((u: string) => toFinal(u));
        data.urls.forEach((u: string) => {
          const fid = parseFileIdFromUrl(u);
          if (fid) nodeOutputUrlCache.set(`${workflowId}:${nodeId}:${portId}:${fid}`, toFinal(u));
        });
        if (fileId) {
          const raw = data.urls.find((u: string) => parseFileIdFromUrl(u) === fileId);
          const single = raw ? toFinal(raw) : null;
          if (single) nodeOutputUrlCache.set(cacheKey, single);
          return single;
        }
        nodeOutputUrlCache.set(inFlightKey, finalUrls[0]);
        return finalUrls;
      }
      const url = data?.url;
      if (typeof url !== 'string' || !url) return null;
      const finalUrl = toFinal(url);
      const fid = parseFileIdFromUrl(url);
      if (fid) nodeOutputUrlCache.set(`${workflowId}:${nodeId}:${portId}:${fid}`, finalUrl);
      if (taskId) nodeOutputUrlCache.set(cacheKey, finalUrl);
      nodeOutputUrlCache.set(inFlightKey, finalUrl);
      return finalUrl;
    } catch (error) {
      console.error('[WorkflowFileManager] getNodeOutputUrl failed:', error);
      return null;
    } finally {
      nodeOutputUrlInFlight.delete(inFlightKey);
    }
  })();
  nodeOutputUrlInFlight.set(inFlightKey, promise);
  const result = await promise;
  if (result == null) return null;
  if (Array.isArray(result)) {
    if (fileId) return nodeOutputUrlCache.get(cacheKey) ?? null;
    return result[0];
  }
  return result;
}

/**
 * 统一根据节点输出值（file ref 或 task ref）获取展示 URL，走 /api/v1/workflow/.../node/.../output/.../url?task_id=xxx&file_id=xxx。
 * 用于任意节点的 image/video/audio 等展示，不再区分 resolveLightX2VResultRef 与 getNodeOutputUrl。
 */
export async function getNodeOutputDisplayUrl(
  workflowId: string,
  nodeId: string,
  portId: string,
  value: { kind?: string; file_id?: string; task_id?: string; output_name?: string; run_id?: string } | string | null
): Promise<string | null> {
  if (!value || !workflowId || !nodeId || !portId) return null;
  if (typeof value === 'string') return value.startsWith('data:') ? value : null;
  const kind = value.kind ?? (value as any).type;
  if (kind === 'task' || (value as any).task_id) {
    const taskId = (value as any).task_id;
    if (!taskId) return null;
    return getNodeOutputUrl(workflowId, nodeId, portId, undefined, undefined, taskId, (value as any).file_id);
  }
  if (kind === 'file' || (value as any).file_id) {
    const fileId = (value as any).file_id;
    if (!fileId) return null;
    return getNodeOutputUrl(workflowId, nodeId, portId, fileId, (value as any).run_id);
  }
  return null;
}

/**
 * 将后端返回的单个输出 URL 解析为 task ref 或 data URL。
 */
async function resolveOutputUrl(url: string): Promise<any | null> {
  // Task result: ./assets/task/result?task_id=...&name=...
  const taskMatch = url.match(/[?&]task_id=([^&]+)/);
  const nameMatch = url.match(/[?&]name=([^&]+)/);
  if (taskMatch && nameMatch) {
    return {
      kind: 'task',
      task_id: decodeURIComponent(taskMatch[1]),
      output_name: decodeURIComponent(nameMatch[1]),
      is_cloud: url.includes('is_cloud') ? true : undefined,
    };
  }
  // Workflow file 或其他 URL：添加 token 后 fetch 转为 data URL
  const resolvedUrl = getAssetPath(url.startsWith('./') ? url.slice(1) : url);
  return await fetchUrlAsDataUrl(resolvedUrl);
}

/**
 * 获取节点输出数据（当前输出）。调用 /output/{port_id}/url 接口，解析为下游可用的 data URL 或 task ref。
 * - 文件输出：fetch URL 转为 data URL
 * - 任务输出：返回 { kind: 'task', task_id, output_name } ref 供 resolveLightX2VResultRef
 * - 多项输出（如多图）：返回数组
 */
export async function getNodeOutputData(
  workflowId: string,
  nodeId: string,
  portId: string
): Promise<any | null> {
  if (isStandalone()) return null;
  try {
    const response = await apiRequest(
      `/api/v1/workflow/${workflowId}/node/${encodeURIComponent(nodeId)}/output/${encodeURIComponent(portId)}/url`
    );

    if (!response.ok) {
      if (response.status === 404) return null;
      const contentType = response.headers.get('content-type') || '';
      let errorMessage = `Failed to get node output: ${response.status} ${response.statusText}`;
      if (contentType.includes('application/json')) {
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch {
          // ignore
        }
      } else {
        try {
          const text = await response.text();
          if (text.includes('<!doctype') || text.includes('<html')) {
            errorMessage = `Server returned HTML error page (${response.status}). The node output may not exist yet.`;
          } else {
            errorMessage = text.substring(0, 200);
          }
        } catch {
          // ignore
        }
      }
      console.error(`[WorkflowFileManager] Error getting node output data for ${nodeId}/${portId}:`, errorMessage);
      throw new Error(errorMessage);
    }

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await response.text();
      throw new Error(`Expected JSON but got ${contentType}. Response: ${text.substring(0, 200)}`);
    }

    const result = await response.json();

    // 多项输出（如多图）：后端返回 { urls: [...] }
    if (Array.isArray(result?.urls)) {
      const resolved = await Promise.all(result.urls.map((u: string) => resolveOutputUrl(u)));
      // 过滤掉解析失败的项
      return resolved.filter((v: any) => v != null);
    }

    // 单项输出：后端返回 { url: "..." }
    const url = result?.url;
    if (typeof url !== 'string' || !url) return null;
    return await resolveOutputUrl(url);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`[WorkflowFileManager] Error getting node output data for ${nodeId}/${portId}:`, errorMessage);
    throw error;
  }
}

/** 通用：fetch URL 转为 data URL (base64)；可选传入 mime_type 强制使用指定类型 */
async function fetchUrlAsDataUrl(url: string, mimeType?: string): Promise<string | null> {
  const fetchUrl = url.startsWith('/') && !url.startsWith('//') ? `${typeof window !== 'undefined' ? window.location.origin : ''}${url}` : url;
  const fetchRes = await fetch(fetchUrl, { credentials: url.includes('/api/') || url.includes('/assets/') ? 'include' : 'omit' });
  if (!fetchRes.ok) return null;
  const blob = await fetchRes.blob();
  return await new Promise<string | null>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string | null;
      if (!result) {
        resolve(null);
        return;
      }
      // 默认 data URL：data:[<mediatype>][;base64],<data>
      if (!mimeType) {
        resolve(result);
        return;
      }
      const commaIndex = result.indexOf(',');
      if (commaIndex === -1) {
        resolve(result);
        return;
      }
      const dataPart = result.slice(commaIndex + 1);
      resolve(`data:${mimeType};base64,${dataPart}`);
    };
    reader.onerror = () => reject(new Error('Failed to read blob'));
    reader.readAsDataURL(blob);
  });
}

/** 通用：fetch URL 转为纯文本 */
async function fetchUrlAsText(url: string): Promise<string | null> {
  const fetchUrl = url.startsWith('/') && !url.startsWith('//') ? `${typeof window !== 'undefined' ? window.location.origin : ''}${url}` : url;
  const fetchRes = await fetch(fetchUrl, { credentials: url.includes('/api/') || url.includes('/assets/') ? 'include' : 'omit' });
  if (!fetchRes.ok) return null;
  return await fetchRes.text();
}

/**
 * 获取工作流对话历史（chat 已拆到独立存储）
 */
export async function getWorkflowChat(workflowId: string): Promise<{ messages: any[]; updated_at?: number } | null> {
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/chat`);
    if (!response.ok) return null;
    const data = await response.json();
    return { messages: data.messages || [], updated_at: data.updated_at };
  } catch (error) {
    console.error('[WorkflowFileManager] Error getting workflow chat:', error);
    return null;
  }
}

/**
 * 替换工作流对话历史（PUT 全量）
 */
export async function putWorkflowChat(workflowId: string, messages: any[]): Promise<boolean> {
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/chat`, {
      method: 'PUT',
      body: JSON.stringify({ messages })
    });
    return response.ok;
  } catch (error) {
    console.error('[WorkflowFileManager] Error putting workflow chat:', error);
    return false;
  }
}

/**
 * 获取节点历史记录（按节点拉取，再按 portId 过滤）
 */
export async function getNodeOutputHistory(
  workflowId: string,
  nodeId: string,
  portId: string
): Promise<any[]> {
  try {
    const response = await apiRequest(`/api/v1/workflow/${workflowId}/node/${nodeId}/output/history`);
    if (!response.ok) return [];
    const result = await response.json();
    const list = result.history || [];
    const normalizedPort = portId || 'output';
    return list.filter((e: any) => (e?.metadata?.port_id ?? 'output') === normalizedPort);
  } catch (error) {
    console.error('[WorkflowFileManager] Error getting node output history:', error);
    return [];
  }
}

/**
 * 从本地 workflow 的 nodeOutputHistory 中按「第 historyIndex 次 run」取一条历史记录（port_keyed）。
 * 每条 entry = 一次 run，output_value 为以 port_id 为键的字典；返回该 run 下指定 port 的展示值或整条 port_keyed。
 */
export function getNodeOutputHistoryEntry(
  nodeOutputHistory: Record<string, any[]> | undefined,
  nodeId: string,
  portId: string,
  historyIndex: number
): { data?: any; entry?: any } | null {
  if (!nodeOutputHistory || !nodeOutputHistory[nodeId] || !Array.isArray(nodeOutputHistory[nodeId])) return null;
  const entries = nodeOutputHistory[nodeId];
  if (historyIndex < 0 || historyIndex >= entries.length) return null;
  const entry = entries[historyIndex];
  if (!entry || typeof entry !== 'object') return null;
  const portKeyed = getEntryPortKeyedValue(entry as import('../../types').NodeHistoryEntry);
  const port = portId || 'output';
  const data = port ? portKeyed[port] : portKeyed;
  return { data, entry };
}

/** @deprecated Use getNodeOutputHistoryEntry for local workflow; persist changes via POST .../update. */
export async function reuseNodeOutputHistory(
  _workflowId: string,
  nodeId: string,
  portId: string,
  historyIndex: number,
  nodeOutputHistory?: Record<string, any[]>
): Promise<any | null> {
  if (nodeOutputHistory) {
    const r = getNodeOutputHistoryEntry(nodeOutputHistory, nodeId, portId, historyIndex);
    return r?.data ?? null;
  }
  return null;
}

/** saveInputFileViaOutputSave 的返回类型 */
type SaveInputFileRef = {
  kind: 'file';
  user_id?: string;
  workflow_id?: string;
  file_id: string;
  file_url: string;
  mime_type?: string;
  ext?: string;
  run_id?: string;
};

/** 同一 (workflowId, nodeId, portId) 的 save 去重：后发请求等待先发结果，避免轮询/双跑时重复 POST 产生两个文件 */
const pendingSaveByKey = new Map<string, Promise<SaveInputFileRef | null>>();

/** 按内容短期缓存：auto-save 与 run 时可能对同一内容各调一次 save，短时间内容相同则复用同一 ref，避免重复文件 */
const SAVE_INPUT_CACHE_TTL_MS = 4000;
const saveInputResultCache = new Map<string, { ref: SaveInputFileRef; expiresAt: number }>();

function pruneSaveInputCache(): void {
  if (saveInputResultCache.size <= 80) return;
  const now = Date.now();
  for (const [k, v] of saveInputResultCache.entries()) {
    if (v.expiresAt <= now) saveInputResultCache.delete(k);
  }
}

function contentKeyForInput(dataUrl: string, fileOrDataUrl?: File | string): string {
  if (typeof fileOrDataUrl === 'string') {
    let h = 0;
    const s = dataUrl;
    for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    return `${h.toString(36)}_${s.length}`;
  }
  if (fileOrDataUrl instanceof File) {
    return `f:${fileOrDataUrl.name}:${fileOrDataUrl.size}:${fileOrDataUrl.lastModified}`;
  }
  return String(Date.now());
}

/**
 * 将输入节点的文件通过 node output/save 存到数据库，与节点生成文件时的保存逻辑一致。
 * 返回 file 引用 { kind: 'file', file_id, file_url, mime_type, ext, run_id }，用于写入 node.output_value（port-keyed）。
 * 同一节点端口的并发调用会复用一次 save 结果，避免重复创建文件。
 */
export async function saveInputFileViaOutputSave(
  workflowId: string,
  nodeId: string,
  portId: string,
  fileOrDataUrl: File | string
): Promise<SaveInputFileRef | null> {
  try {
    let dataUrl: string;
    let ext = '.bin';
    if (typeof fileOrDataUrl === 'string') {
      dataUrl = fileOrDataUrl;
      if (dataUrl.startsWith('data:')) {
        const header = dataUrl.split(',')[0] || '';
        const mime = header.split(':')[1]?.split(';')[0] || '';
        const extMap: Record<string, string> = {
          'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp',
          'video/mp4': '.mp4', 'video/webm': '.webm', 'audio/mpeg': '.mp3', 'audio/wav': '.wav', 'audio/ogg': '.ogg',
          'text/plain': '.txt'
        };
        ext = extMap[mime] || '.bin';
      }
    } else {
      dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(fileOrDataUrl);
      });
      const name = fileOrDataUrl.name || '';
      if (name.includes('.')) ext = '.' + name.split('.').pop()!.toLowerCase();
    }

    if (isStandalone()) {
      const key = `localFile_${workflowId}_${nodeId}_${portId}_${Date.now()}_${crypto.randomUUID()}`;
      const file = typeof fileOrDataUrl === 'string'
        ? await (async () => {
            const res = await fetch(fileOrDataUrl);
            const blob = await res.blob();
            return new File([blob], 'file' + ext, { type: blob.type || 'application/octet-stream' });
          })()
        : fileOrDataUrl;
      await saveLocalFile(key, file);
      const localRef = `local://${key}`;
      const mime = extToMime(ext);
      return { kind: 'file', file_id: key, file_url: localRef, mime_type: mime, ext };
    }

    pruneSaveInputCache();
    const contentKey = contentKeyForInput(dataUrl, fileOrDataUrl);
    const cacheKey = `${workflowId}:${nodeId}:${portId}:${contentKey}`;
    const now = Date.now();
    const cached = saveInputResultCache.get(cacheKey);
    if (cached && cached.expiresAt > now) {
      return { ...cached.ref };
    }

    const dedupeKey = `${workflowId}:${nodeId}:${portId}`;
    const existing = pendingSaveByKey.get(dedupeKey);
    if (existing) {
      const ref = await existing;
      if (ref) {
        saveInputResultCache.set(cacheKey, { ref, expiresAt: now + SAVE_INPUT_CACHE_TTL_MS });
        return { ...ref };
      }
      return null;
    }

    const promise = (async (): Promise<SaveInputFileRef | null> => {
      try {
        const result = await saveNodeOutputs(workflowId, nodeId, { [portId]: dataUrl }, crypto.randomUUID());
        const r = result?.[portId];
        const fileId = r?.file_id ?? (r && typeof r === 'object' && (r as any).file_id);
        if (!fileId) return null;
        const outMime = (r as any)?.mime_type ?? ((r as any)?.ext ? extToMime((r as any).ext) : (dataUrl.startsWith('data:text/') ? 'text/plain' : undefined));
        const outExt: string | undefined = (r as any)?.ext != null
          ? (String((r as any).ext).startsWith('.') ? String((r as any).ext) : `.${(r as any).ext}`)
          : ext !== '.bin' ? ext : undefined;
        const outRunId: string | undefined = (r as any)?.run_id || undefined;
        const ref: SaveInputFileRef = {
          kind: 'file', file_id: String(fileId),
          file_url: "",
          user_id: r?.user_id,
          workflow_id: r?.workflow_id,
          ...(outMime != null && { mime_type: outMime }),
          ...(outExt != null && { ext: outExt }),
          ...(outRunId != null && { run_id: outRunId }),
        };
        saveInputResultCache.set(cacheKey, { ref, expiresAt: Date.now() + SAVE_INPUT_CACHE_TTL_MS });
        return ref;
      } finally {
        pendingSaveByKey.delete(dedupeKey);
      }
    })();
    pendingSaveByKey.set(dedupeKey, promise);
    const ref = await promise;
    return ref ? { ...ref } : null;
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error(`[WorkflowFileManager] saveInputFileViaOutputSave failed:`, msg);
    throw error;
  }
}

/**
 * 直接上传文件到工作流节点输出（用于输入节点）
 * 有后端时通过 output/save 保存；纯前端时保存到 IndexedDB。
 * @deprecated 新逻辑请用 saveInputFileViaOutputSave，并写入 node.output_value。
 */
export async function uploadNodeInputFile(
  workflowId: string,
  nodeId: string,
  portId: string,
  file: File,
  index: number = 0
): Promise<{ file_id: string; file_path: string; file_url: string; mime_type?: string } | null> {
  const ref = await saveInputFileViaOutputSave(workflowId, nodeId, portId, file);
  if (!ref) return null;
  return { file_id: ref.file_id, file_path: ref.file_url, file_url: ref.file_url, mime_type: ref.mime_type };
}

/**
 * 判断是否为“需上传的本地资源 URL”（如 /assets/girl.png），
 * 即不以 workflow/input、task/、http 等已保存路径为前缀。
 */
export function isLocalAssetUrlToUpload(url: string): boolean {
  if (typeof url !== 'string' || !url) return false;
  const u = url.replace(/^\.\//, '/');
  if (!u.startsWith('/assets/')) return false;
  if (u.startsWith('/assets/workflow/file') || u.startsWith('/assets/task/')) return false;
  if (u.startsWith('http://') || u.startsWith('https://')) return false;
  return true;
}

/**
 * 将本地资源 URL（如 /assets/girl.png）拉取为 File 并上传到节点输出，
 * 返回后端 file 引用，便于数据库以 file 类型存储而非 base64。
 * 有后端时使用；standalone 下不调用。
 */
export async function uploadLocalUrlAsNodeOutput(
  workflowId: string,
  nodeId: string,
  portId: string,
  localUrl: string,
  index: number = 0
): Promise<SaveInputFileRef | null> {
  try {
    const path = localUrl.startsWith('./') ? localUrl.slice(1) : localUrl;
    const urlToFetch = getAssetPath(path);
    const res = await fetch(urlToFetch);
    if (!res.ok) return null;
    const blob = await res.blob();
    const name = path.split('/').pop() || 'file';
    const file = new File([blob], name, { type: blob.type || 'application/octet-stream' });
    const ref = await saveInputFileViaOutputSave(workflowId, nodeId, portId, file);
    if (!ref) return null;
    return ref;
  } catch (e) {
    console.error('[WorkflowFileManager] uploadLocalUrlAsNodeOutput failed:', localUrl, e);
    return null;
  }
}

/** 构建 /assets/workflow/file?... 的完整 URL（含 token）。 nodeId/portId/runId 必须提供以匹配新存储路径。 */
function buildWorkflowFileUrl(
  workflowId: string,
  fileId: string,
  opts?: { nodeId?: string; portId?: string; runId?: string; mimeType?: string; ext?: string }
): string {
  const params = new URLSearchParams();
  params.set('workflow_id', workflowId);
  params.set('file_id', fileId);
  if (opts?.nodeId) params.set('node_id', opts.nodeId);
  if (opts?.portId) params.set('port_id', opts.portId);
  if (opts?.runId) params.set('run_id', opts.runId);
  if (opts?.mimeType) params.set('mime_type', opts.mimeType);
  if (opts?.ext) params.set('ext', opts.ext.startsWith('.') ? opts.ext : `.${opts.ext}`);
  return getAssetPath(`/assets/workflow/file?${params.toString()}`);
}

/**
 * 根据 file ref 获取工作流文件的 data URL（二进制转 base64）。
 * 纯前端模式：fileId 为 local://key 时从 IndexedDB 解析。
 * 有后端且提供 nodeId+portId 时：先通过 /output/{port_id}/url 获取展示 URL，再 fetch 转 data URL；否则用 /assets/workflow/file 直链。
 */
export async function getWorkflowFileByFileId(
  workflowId: string,
  fileId: string,
  mimeType?: string,
  ext?: string,
  nodeId?: string,
  portId?: string,
  runId?: string
): Promise<string | null> {
  try {
    if (fileId.startsWith('local://')) {
      return getLocalFileDataUrl(fileId);
    }
    if (isStandalone()) return null;
    if (nodeId && portId) {
      const displayUrl = await getNodeOutputUrl(workflowId, nodeId, portId, fileId, runId);
      if (displayUrl) {
        return await fetchUrlAsDataUrl(displayUrl, mimeType);
      }
    }
    const url = buildWorkflowFileUrl(workflowId, fileId, { nodeId, portId, runId, mimeType, ext });
    return await fetchUrlAsDataUrl(url, mimeType);
  } catch (error) {
    console.error('[WorkflowFileManager] Error getting workflow file by file_id:', error);
    return null;
  }
}

/**
 * 获取工作流文本文件内容（如 .txt 端口存为 file 后的预览）。
 * 有 nodeId+portId 时先通过 /output/{port_id}/url 获取 URL 再 fetch 文本；否则用 /assets/workflow/file 直链（需 node_id/port_id/run_id 否则可能 404）。
 */
export async function getWorkflowFileText(
  workflowId: string,
  fileId: string,
  nodeId?: string,
  portId?: string,
  runId?: string
): Promise<string | null> {
  try {
    if (isStandalone()) return null;
    if (nodeId && portId) {
      const displayUrl = await getNodeOutputUrl(workflowId, nodeId, portId, fileId, runId);
      if (displayUrl) return await fetchUrlAsText(displayUrl);
    }
    const url = buildWorkflowFileUrl(workflowId, fileId, { nodeId, portId, runId, mimeType: 'text/plain', ext: '.txt' });
    return await fetchUrlAsText(url);
  } catch {
    return null;
  }
}

/** 浏览器缓存 workflow 文件 URL，key: workflowId:fileId[:mimeType]，避免数据库存过期 file_url */
const WORKFLOW_FILE_URL_CACHE_KEY = 'canvas_workflow_file_url_cache';
const WORKFLOW_FILE_URL_TTL_MS = 50 * 60 * 1000; // 50 min，小于常见 presign 1h

function getWorkflowFileUrlCacheKey(workflowId: string, fileId: string, mimeType?: string): string {
  return mimeType ? `${workflowId}:${fileId}:${mimeType}` : `${workflowId}:${fileId}`;
}

function getWorkflowFileUrlFromStorage(workflowId: string, fileId: string, mimeType?: string): string | null {
  try {
    const raw = localStorage.getItem(WORKFLOW_FILE_URL_CACHE_KEY);
    if (!raw) return null;
    const cache = JSON.parse(raw) as Record<string, { url: string; expiresAt: number }>;
    const key = getWorkflowFileUrlCacheKey(workflowId, fileId, mimeType);
    const keyFallback = getWorkflowFileUrlCacheKey(workflowId, fileId);
    const entry = cache[key] ?? cache[keyFallback];
    if (!entry || entry.expiresAt <= Date.now()) return null;
    return entry.url;
  } catch {
    return null;
  }
}

function setWorkflowFileUrlInStorage(workflowId: string, fileId: string, url: string, mimeType?: string): void {
  try {
    const raw = localStorage.getItem(WORKFLOW_FILE_URL_CACHE_KEY);
    const cache: Record<string, { url: string; expiresAt: number }> = raw ? JSON.parse(raw) : {};
    const key = getWorkflowFileUrlCacheKey(workflowId, fileId, mimeType);
    cache[key] = { url, expiresAt: Date.now() + WORKFLOW_FILE_URL_TTL_MS };
    localStorage.setItem(WORKFLOW_FILE_URL_CACHE_KEY, JSON.stringify(cache));
  } catch (e) {
    console.warn('[WorkflowFileManager] setWorkflowFileUrlInStorage failed:', e);
  }
}

/**
 * 获取工作流文件 URL（用于直接访问，如 <img> 标签）。
 * 使用 /assets/workflow/file?... 格式，需要 nodeId/portId/runId 定位文件。
 */
export function getWorkflowFileUrl(
  workflowId: string,
  fileId: string,
  mimeType?: string,
  ext?: string,
  nodeId?: string,
  portId?: string,
  runId?: string
): string {
  return buildWorkflowFileUrl(workflowId, fileId, { nodeId, portId, runId, mimeType, ext });
}

/**
 * 带缓存的 workflow 文件 URL：先查缓存，过期或缺失时用 getWorkflowFileUrl 生成并写入缓存。
 * 用于展示（img src 等），避免数据库存 file_url 过期。
 */
export function getCachedWorkflowFileUrl(
  workflowId: string,
  fileId: string,
  mimeType?: string,
  ext?: string,
  nodeId?: string,
  portId?: string,
  runId?: string
): string {
  const cached = getWorkflowFileUrlFromStorage(workflowId, fileId, mimeType);
  if (cached) return cached;
  const url = getWorkflowFileUrl(workflowId, fileId, mimeType, ext, nodeId, portId, runId);
  setWorkflowFileUrlInStorage(workflowId, fileId, url, mimeType);
  return url;
}
