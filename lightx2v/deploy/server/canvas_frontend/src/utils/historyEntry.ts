import { NodeHistoryEntry, NodeHistoryEntryKind, NodeHistoryValue } from '../../types';
import { isLightX2VResultRef, type LightX2VResultRef } from './resultRef';

const FILE_STRING_PREFIXES = ['http://', 'https://', './', '/assets/', 'local://', 'blob:'];

const cloneJson = (value: any) => {
  if (Array.isArray(value)) return value.map(cloneJson);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, cloneJson(v)]));
  return value;
};

const looksLikeFileString = (str: unknown): str is string => {
  if (typeof str !== 'string') return false;
  if (str.startsWith('data:')) return true;
  return FILE_STRING_PREFIXES.some(prefix => str.startsWith(prefix));
};

const MIME_TO_EXT: Record<string, string> = {
  'image/png': '.png', 'image/jpeg': '.jpg', 'image/jpg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp',
  'video/mp4': '.mp4', 'video/webm': '.webm', 'audio/mpeg': '.mp3', 'audio/wav': '.wav', 'audio/ogg': '.ogg',
  'text/plain': '.txt', 'application/json': '.json'
};
const EXT_TO_MIME: Record<string, string> = Object.fromEntries(
  Object.entries(MIME_TO_EXT).map(([m, e]) => [e, m])
);
function extToMime(ext: string | undefined): string | undefined {
  if (!ext) return undefined;
  const norm = ext.startsWith('.') ? ext : `.${ext}`;
  return EXT_TO_MIME[norm] ?? 'application/octet-stream';
}

function inferExtFromDataUrl(dataUrl: string): string {
  const m = dataUrl.match(/data:([^;]+);/);
  const mime = m ? m[1] : '';
  return MIME_TO_EXT[mime] || '.bin';
}

function inferExtFromUrl(url: string): string | undefined {
  const m = url.match(/\.([a-z0-9]+)(?:\?|$)/i);
  return m ? `.${m[1].toLowerCase()}` : undefined;
}

const toTaskValue = (ref: LightX2VResultRef): { kind: NodeHistoryEntryKind; value: NodeHistoryValue } => ({
  kind: 'task',
  value: {
    task_id: ref.task_id,
    output_name: ref.output_name,
    is_cloud: !!ref.is_cloud
  }
});

type HistoryValueBuilderResult = { kind: NodeHistoryEntryKind; value: NodeHistoryValue };

export const buildHistoryValue = (value: any): HistoryValueBuilderResult | null => {
  if (value == null) return null;

  if (isLightX2VResultRef(value)) {
    return toTaskValue(value);
  }

  if (typeof value === 'string') {
    if (value.startsWith('data:')) {
      const ext = inferExtFromDataUrl(value);
      const mime = EXT_TO_MIME[ext] ?? 'application/octet-stream';
      return { kind: 'file', value: { dataUrl: value, mime_type: mime } };
    }
    if (looksLikeFileString(value)) {
      // 持久化用 file_id，不持久化 url（旧 /api/v1/workflow/.../file/{id} 或新 /assets/workflow/file?file_id=... 可解析出 file_id）
      const isOldFileUrl = value.includes('/api/v1/workflow/') && value.includes('/file/');
      const isNewFileUrl = value.includes('/assets/workflow/file');
      if (isOldFileUrl || isNewFileUrl) {
        const fileIdMatch = isNewFileUrl ? value.match(/[?&]file_id=([^&]+)/) : value.match(/\/file\/([^/?]+)/);
        if (fileIdMatch) {
          const file_id = fileIdMatch[1];
          let mime_type = 'application/octet-stream';
          const mimeMatch = value.match(/mime_type=([^&]+)/);
          if (mimeMatch) mime_type = decodeURIComponent(mimeMatch[1].replace(/\+/g, ' '));
          const extMatch = value.match(/[?&]ext=([^&]+)/);
          const extVal = extMatch ? decodeURIComponent(extMatch[1]) : undefined;
          const runIdMatch = value.match(/[?&]run_id=([^&]+)/);
          const runIdVal = runIdMatch ? decodeURIComponent(runIdMatch[1]) : undefined;
          return { kind: 'file', value: { file_id, mime_type, ...(extVal && { ext: extVal }), ...(runIdVal && { run_id: runIdVal }) } };
        }
      }
      const ext = inferExtFromUrl(value);
      const mime = ext ? (EXT_TO_MIME[ext] ?? 'application/octet-stream') : undefined;
      return { kind: 'file', value: { url: value, mime_type: mime } };
    }
    return { kind: 'text', value: { text: value } };
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return { kind: 'text', value: { text: String(value) } };
  }

  if (Array.isArray(value)) {
    return { kind: 'text', value: { text: JSON.stringify(cloneJson(value)) } };
  }

  if (typeof value === 'object') {
    if ('json_value' in value) {
      return buildHistoryValue((value as any).json_value);
    }
    if ('text_value' in value) {
      return { kind: 'text', value: { text: String((value as any).text_value ?? '') } };
    }
    if ((value as any).type === 'task' || (value as any).__type === 'lightx2v_result') {
      return toTaskValue(value as LightX2VResultRef);
    }
    // 文本结果统一为 kind: 'file'（上传文件形式），兼容旧 type: 'text'
    if (((value as any).kind === 'file' && 'data' in value) || ((value as any).type === 'text' && 'data' in value)) {
      const d = (value as any).data;
      const dataUrl = typeof d === 'string' && d.startsWith('data:') ? d : undefined;
      const ext = (value as any).ext ?? (dataUrl ? inferExtFromDataUrl(dataUrl) : undefined);
      const mime = (value as any).mime_type ?? (ext ? (EXT_TO_MIME[ext] ?? 'application/octet-stream') : undefined);
      return { kind: 'file', value: { dataUrl, url: typeof d === 'string' && !d.startsWith('data:') ? d : undefined, mime_type: mime } };
    }
    if (((value as any).kind === 'json' || (value as any).type === 'json') && 'data' in value) {
      return { kind: 'text', value: { text: JSON.stringify(cloneJson((value as any).data)) } };
    }
    if ((value as any).type === 'data_url') {
      const dataUrl = typeof value._full_data === 'string' ? value._full_data : value.data;
      const ext = typeof dataUrl === 'string' && dataUrl.startsWith('data:') ? inferExtFromDataUrl(dataUrl) : undefined;
      const mime = ext ? (EXT_TO_MIME[ext] ?? 'application/octet-stream') : undefined;
      return {
        kind: 'file',
        value: { dataUrl: typeof dataUrl === 'string' ? dataUrl : undefined, mime_type: mime }
      };
    }
    if ((value as any).kind === 'url' || (value as any).type === 'url') {
      const url = (value as any).data ?? (value as any).url;
      const mime = (value as any).mime_type ?? (typeof url === 'string' ? extToMime(inferExtFromUrl(url)) : undefined);
      const urlIsOldFile = typeof url === 'string' && url.includes('/api/v1/workflow/') && url.includes('/file/');
      const urlIsNewFile = typeof url === 'string' && url.includes('/assets/workflow/file');
      if (urlIsOldFile || urlIsNewFile) {
        const fileIdMatch = urlIsNewFile ? url.match(/[?&]file_id=([^&]+)/) : url.match(/\/file\/([^/?]+)/);
        if (fileIdMatch) {
          const mimeMatch = url.match(/mime_type=([^&]+)/);
          const mime_type = (value as any).mime_type ?? (mimeMatch ? decodeURIComponent(mimeMatch[1].replace(/\+/g, ' ')) : undefined) ?? mime ?? 'application/octet-stream';
          const extMatch = url.match(/[?&]ext=([^&]+)/);
          const extVal = extMatch ? decodeURIComponent(extMatch[1]) : undefined;
          const runIdMatch = url.match(/[?&]run_id=([^&]+)/);
          const runIdVal = runIdMatch ? decodeURIComponent(runIdMatch[1]) : undefined;
          return { kind: 'file', value: { file_id: fileIdMatch[1], mime_type, ...(extVal && { ext: extVal }), ...(runIdVal && { run_id: runIdVal }) } };
        }
      }
      return {
        kind: 'file',
        value: {
          url: typeof url === 'string' ? url : undefined,
          mime_type: mime ?? 'application/octet-stream'
        }
      };
    }
    // Backend-stored file: 持久化 file_id、mime_type、ext、run_id
    if ((value as any).kind === 'file' || (value as any).type === 'file' || value.file_id) {
      const mime = (value as any).mime_type ?? extToMime((value as any).ext);
      const extVal = (value as any).ext;
      const runIdVal = (value as any).run_id;
      return {
        kind: 'file',
        value: { file_id: value.file_id, mime_type: mime ?? 'application/octet-stream', ...(extVal && { ext: extVal }), ...(runIdVal && { run_id: runIdVal }) }
      };
    }
    if ('file_id' in value || 'file_url' in value) {
      const mime = (value as any).mime_type ?? extToMime((value as any).ext);
      const extVal = (value as any).ext;
      const runIdVal = (value as any).run_id;
      return {
        kind: 'file',
        value: { file_id: value.file_id, mime_type: mime ?? 'application/octet-stream', ...(extVal && { ext: extVal }), ...(runIdVal && { run_id: runIdVal }) }
      };
    }

    return { kind: 'text', value: { text: JSON.stringify(cloneJson(value)) } };
  }

  return null;
};

export const createHistoryEntryFromValue = (params: {
  id: string;
  timestamp: number;
  value: any;
  execution_time?: number;
  params?: Record<string, any>;
  valueOverride?: HistoryValueBuilderResult | null;
  /** 传入时 output_value 存为以 port_id 为键的字典 { [portId]: value }，与多端口格式一致 */
  portId?: string;
}): NodeHistoryEntry | null => {
  let { id, timestamp, value, execution_time, params: nodeParams, valueOverride, portId } = params;
  if (value != null && typeof value === 'object' && !Array.isArray(value)) {
    if (value.output_value != null && typeof value.output_value === 'object' && Object.keys(value.output_value).some((k: string) => k.startsWith('out-'))) {
      value = value.output_value;
    }
    const outKeys = Object.keys(value).filter((k: string) => k.startsWith('out-'));
    if (outKeys.length >= 1) {
      const portKeyed: Record<string, any> = {};
      for (const k of outKeys) {
        const payload = buildHistoryValue(value[k]);
        if (payload) portKeyed[k] = { kind: payload.kind, ...payload.value };
      }
      if (Object.keys(portKeyed).length > 0) {
        return {
          id,
          timestamp,
          execution_time,
          output_value: portKeyed as NodeHistoryEntry['output_value'],
          params: nodeParams ?? {}
        };
      }
    }
    if (outKeys.length === 1 && (portId == null || portId === '')) {
      portId = outKeys[0];
      value = value[outKeys[0]];
    } else if (portId != null && portId !== '' && portId in value) {
      value = value[portId];
    }
  }
  const payload = valueOverride ?? buildHistoryValue(value);
  if (!payload) return null;

  const singleValue = { kind: payload.kind, ...payload.value } as NodeHistoryEntry['output_value'];
  const output_value = portId != null && portId !== ''
    ? ({ [portId]: singleValue } as Record<string, any>)
    : singleValue;
  const entry: NodeHistoryEntry = {
    id,
    timestamp,
    execution_time,
    output_value: output_value as NodeHistoryEntry['output_value'],
    params: nodeParams ?? {}
  };
  return entry;
};

/** 创建一条 port_keyed 历史条目，仅写 output_value（以 port_id 为键），不再写 output_value_port_keyed */
export function createHistoryEntryFromPortKeyedValue(params: {
  id: string;
  timestamp: number;
  output_value_port_keyed: Record<string, any>;
  execution_time?: number;
  params?: Record<string, any>;
}): NodeHistoryEntry {
  const { id, timestamp, output_value_port_keyed, execution_time, params: nodeParams } = params;
  const portKeyedOnly = Object.fromEntries(
    Object.entries(output_value_port_keyed).filter(([k]) => k.startsWith('out-') || k === '__json__')
  );
  const output_value = Object.keys(portKeyedOnly).length > 0 ? portKeyedOnly : { ...output_value_port_keyed };
  return {
    id,
    timestamp,
    execution_time,
    output_value: output_value as NodeHistoryEntry['output_value'],
    params: nodeParams ?? {}
  };
}

/** 多端口节点历史：output_value 直接为以 port_id 为键的字典（无 kind/json 包装），只保留 out-* / __json__ 键 */
export function createHistoryEntryFromPortKeyedOutputValue(params: {
  id: string;
  timestamp: number;
  output_value: Record<string, any>;
  execution_time?: number;
  params?: Record<string, any>;
}): NodeHistoryEntry {
  const { id, timestamp, output_value: portKeyed, execution_time, params: nodeParams } = params;
  const onlyPortKeys = Object.fromEntries(
    Object.entries(portKeyed).filter(([k]) => k.startsWith('out-') || k === '__json__')
  );
  return {
    id,
    timestamp,
    execution_time,
    output_value: Object.keys(onlyPortKeys).length > 0 ? onlyPortKeys : { ...portKeyed },
    params: nodeParams ?? {}
  };
}

const normalizeValueForKind = (
  kind: NodeHistoryEntryKind | 'json',
  value: any,
  legacy?: any
): NodeHistoryValue => {
  switch (kind) {
    case 'text': {
      if (value && typeof value === 'object' && 'text' in value) {
        return { text: String(value.text ?? '') };
      }
      if (typeof value === 'string') return { text: value };
      if (legacy && typeof legacy.text === 'string') return { text: legacy.text };
      return { text: value != null ? String(value) : '' };
    }
    case 'json': {
      if (value && typeof value === 'object' && 'json' in value) {
        return { text: JSON.stringify(value.json ?? value) };
      }
      return { text: value !== undefined ? JSON.stringify(value) : '' };
    }
    case 'file': {
      const source = { ...(value || {}), ...(legacy || {}) };
      // 兼容旧数据：dataId 可当作 fileId 使用（历史格式迁移）；ext 兼容迁移为 mime_type
      const file_id = source.file_id ?? (source as any).fileId ?? source.dataId ?? source.data_id;
      const mime = source.mime_type ?? extToMime(source.ext);
      return {
        file_id: file_id || undefined,
        url: source.url ?? source.file_url,
        dataUrl: source.dataUrl,
        mime_type: mime
      };
    }
    case 'task': {
      const source = value || legacy || {};
      return {
        task_id: source.task_id ?? (source as any).taskId ?? '',
        output_name: source.output_name ?? (source as any).outputName ?? 'output',
        is_cloud: !!(source.is_cloud ?? (source as any).isCloud)
      };
    }
    default:
      return { text: value != null ? JSON.stringify(value) : '' };
  }
};

/** 从单条 output_value（含 kind）得到展示用值，用于归一化时填充 port_keyed */
function singleOutputValueToDisplay(ov: Record<string, any>): any {
  if (!ov) return null;
  const kind = ov.kind || 'json';
  switch (kind) {
    case 'text':
      return (ov as { text?: string }).text ?? '';
    case 'json':
      return cloneJson((ov as { json?: any }).json ?? ov);
    case 'file': {
      const fileValue = ov as { file_id?: string; url?: string; dataUrl?: string; mime_type?: string; ext?: string };
      const mime = fileValue.mime_type ?? extToMime(fileValue.ext);
      if (fileValue.dataUrl) return fileValue.dataUrl;
      if (fileValue.url) return { kind: 'url', data: fileValue.url, mime_type: mime };
      if (fileValue.file_id) {
        const ext = fileValue.ext != null ? (fileValue.ext.startsWith('.') ? fileValue.ext : `.${fileValue.ext}`) : undefined;
        return { kind: 'file', file_id: fileValue.file_id, file_url: (fileValue as { file_url?: string }).file_url ?? '', mime_type: mime, ...(ext != null && { ext }) };
      }
      return null;
    }
    case 'task':
      return {
        kind: 'task',
        task_id: (ov as { task_id?: string }).task_id || '',
        output_name: (ov as { output_name?: string }).output_name || 'output',
        is_cloud: !!(ov as { is_cloud?: boolean }).is_cloud
      };
    default:
      return null;
  }
}

/** 从单条 value 推断 port_id（用于旧历史归一化为 port_keyed 时） */
function inferPortIdFromValue(singleValue: any): string {
  if (!singleValue || typeof singleValue !== 'object') return 'output';
  const kind = singleValue.kind || singleValue.type;
  const mime = (singleValue.mime_type || singleValue.mimeType || '').toLowerCase();
  if (mime.includes('audio')) return 'out-audio';
  if (mime.includes('video')) return 'out-video';
  if (mime.includes('image')) return 'out-image';
  if (kind === 'task' || kind === 'lightx2v_result') {
    const name = (singleValue.output_name || singleValue.outputName || '').toLowerCase();
    if (name.includes('audio')) return 'out-audio';
    if (name.includes('video')) return 'out-video';
    if (name.includes('image')) return 'out-image';
  }
  if (kind === 'file' && (singleValue.ext === '.mp3' || singleValue.ext === '.wav' || singleValue.ext === '.ogg')) return 'out-audio';
  if (kind === 'file' && (singleValue.ext === '.mp4' || singleValue.ext === '.webm')) return 'out-video';
  if (kind === 'file' && /\.(png|jpg|jpeg|gif|webp|bmp)$/i.test(singleValue.ext || '')) return 'out-image';
  return 'out-text';
}

export const normalizeHistoryEntry = (raw: any): NodeHistoryEntry | null => {
  if (!raw || typeof raw !== 'object') return null;
  const portId = raw.port_id ?? raw.metadata?.port_id ?? 'output';
  if (raw.output_value_port_keyed && typeof raw.output_value_port_keyed === 'object' && Object.keys(raw.output_value_port_keyed).length > 0) {
    const portKeyed = raw.output_value_port_keyed as Record<string, any>;
    const firstKey = Object.keys(portKeyed)[0];
    const firstVal = portKeyed[firstKey];
    const singleValue = (raw.output_value && typeof raw.output_value === 'object')
      ? (raw.output_value as Record<string, any>)
      : (typeof firstVal === 'object' && firstVal !== null && (firstVal.kind === 'json' || firstVal.json !== undefined)
          ? { kind: 'text' as const, text: typeof (firstVal as any).json !== 'undefined' ? JSON.stringify((firstVal as any).json) : JSON.stringify(firstVal) }
          : firstVal);
    const output_value = Object.keys(portKeyed).some((k: string) => k.startsWith('out-')) ? { ...portKeyed } : { [firstKey || portId]: singleValue };
    const meta = raw.metadata && typeof raw.metadata === 'object' ? { ...raw.metadata, port_id: portId } : { port_id: portId };
    if ('run_timestamp' in meta) delete meta.run_timestamp;
    return {
      id: raw.id || `entry-${Date.now()}`,
      timestamp: typeof raw.timestamp === 'number' ? raw.timestamp : Date.now(),
      execution_time: raw.execution_time,
      output_value: output_value as NodeHistoryEntry['output_value'],
      params: raw.params ?? {},
      metadata: meta
    };
  }
  if (typeof raw === 'object' && raw.output_value) {
    const ov = raw.output_value;
    if (ov && typeof ov === 'object' && !Array.isArray(ov) && Object.keys(ov).some((k: string) => k.startsWith('out-'))) {
      const meta = raw.metadata && typeof raw.metadata === 'object' ? { ...raw.metadata } : {};
      if (!meta.port_id) meta.port_id = Object.keys(ov).find((k: string) => k.startsWith('out-')) || portId;
      if ('run_timestamp' in meta) delete meta.run_timestamp;
      return {
        id: raw.id || `entry-${Date.now()}`,
        timestamp: typeof raw.timestamp === 'number' ? raw.timestamp : Date.now(),
        execution_time: raw.execution_time,
        output_value: ov as NodeHistoryEntry['output_value'],
        params: raw.params ?? {},
        ...(Object.keys(meta).length > 0 ? { metadata: meta } : {})
      };
    }
    const kind = ov.kind || ov.type || 'text';
    if (kind === 'json' && typeof ov.json === 'object' && ov.json !== null && Object.keys(ov.json).some((k: string) => k.startsWith('out-'))) {
      const portKeyed = ov.json as Record<string, any>;
      const meta = raw.metadata && typeof raw.metadata === 'object' ? { ...raw.metadata, port_id: portId } : { port_id: portId };
      if ('run_timestamp' in meta) delete meta.run_timestamp;
      return {
        id: raw.id || `entry-${Date.now()}`,
        timestamp: typeof raw.timestamp === 'number' ? raw.timestamp : Date.now(),
        execution_time: raw.execution_time,
        output_value: portKeyed,
        params: raw.params ?? {},
        metadata: meta
      };
    }
    const normKind = kind === 'lightx2v_result' ? 'task' : (kind === 'json' ? 'text' : kind);
    const normalized = normalizeValueForKind(kind, ov, ov);
    const singleValue = (normKind === 'text' && kind === 'json' ? { kind: 'text' as const, text: typeof (normalized as any).text === 'string' ? (normalized as any).text : JSON.stringify(ov.json ?? ov) } : { kind: normKind, ...normalized }) as Record<string, any>;
    const effectivePortId = portId && portId !== 'output' ? portId : inferPortIdFromValue(singleValue);
    const output_value = { [effectivePortId]: singleValue };
    const meta = raw.metadata && typeof raw.metadata === 'object' ? { ...raw.metadata, port_id: effectivePortId } : { port_id: effectivePortId };
    if ('run_timestamp' in meta) delete meta.run_timestamp;
    return {
      id: raw.id || `entry-${Date.now()}`,
      timestamp: typeof raw.timestamp === 'number' ? raw.timestamp : Date.now(),
      execution_time: raw.execution_time,
      output_value,
      params: raw.params ?? {},
      metadata: meta
    };
  }
  if (typeof raw === 'object' && (raw.kind || raw.value)) {
    let kind = raw.kind || 'json';
    let value = raw.value;
    if (kind === 'json' && value?.json && typeof value.json === 'object' &&
        ((value.json as any).type === 'task' || (value.json as any).__type === 'lightx2v_result')) {
      kind = 'task';
      value = {
        task_id: value.json.task_id ?? value.json.taskId ?? '',
        output_name: value.json.output_name ?? value.json.outputName ?? 'output',
        is_cloud: !!(value.json.is_cloud ?? value.json.isCloud)
      };
    }
    if (kind === 'lightx2v_result') kind = 'task';
    const effectiveKind = kind === 'json' ? 'text' : (kind as NodeHistoryEntryKind);
    const normVal = normalizeValueForKind(kind, value, raw);
    const singleValue = { kind: effectiveKind, ...normVal } as Record<string, any>;
    const effectivePortId = portId && portId !== 'output' ? portId : inferPortIdFromValue(singleValue);
    const output_value = { [effectivePortId]: singleValue } as Record<string, any>;
    const meta = raw.metadata && typeof raw.metadata === 'object' ? { ...raw.metadata, port_id: effectivePortId } : { port_id: effectivePortId };
    if ('run_timestamp' in meta) delete meta.run_timestamp;
    return {
      id: raw.id || `entry-${Date.now()}`,
      timestamp: typeof raw.timestamp === 'number' ? raw.timestamp : Date.now(),
      execution_time: raw.execution_time,
      output_value,
      params: raw.params ?? {},
      metadata: meta
    };
  }
  return null;
};

export const normalizeHistoryEntries = (list: any[]): NodeHistoryEntry[] => {
  if (!Array.isArray(list)) return [];
  return list
    .map(normalizeHistoryEntry)
    .filter((entry): entry is NodeHistoryEntry => !!entry);
};

const RUN_TOLERANCE_MS = 2000;

/** 按 run 聚合：同一 timestamp 容差内的条目合并为一条，output_value 为各 port 合并（仅用 output_value，不再写 output_value_port_keyed） */
export function aggregateHistoryByRun(entries: NodeHistoryEntry[]): NodeHistoryEntry[] {
  if (!entries.length) return [];
  const groups = new Map<number, NodeHistoryEntry[]>();
  for (const e of entries) {
    const runKey = Math.floor((e.timestamp ?? 0) / RUN_TOLERANCE_MS) * RUN_TOLERANCE_MS;
    const list = groups.get(runKey) ?? [];
    list.push(e);
    groups.set(runKey, list);
  }
  const result: NodeHistoryEntry[] = [];
  const sortedKeys = Array.from(groups.keys()).sort((a, b) => b - a);
  for (const key of sortedKeys) {
    const list = groups.get(key)!;
    const first = list[0];
    const mergedPortKeyed: Record<string, any> = {};
    for (const e of list) {
      const pk = getEntryPortKeyedValue(e);
      for (const [portId, val] of Object.entries(pk)) {
        if (val === undefined || val === null) continue;
        if (portId in mergedPortKeyed) {
          const existing = mergedPortKeyed[portId];
          if (Array.isArray(existing)) (existing as any[]).push(val);
          else mergedPortKeyed[portId] = [existing, val];
        } else {
          mergedPortKeyed[portId] = val;
        }
      }
    }
    const meta = first.metadata && typeof first.metadata === 'object' ? { ...first.metadata } : undefined;
    if (meta && 'run_timestamp' in meta) delete meta.run_timestamp;
    result.push({
      id: first.id,
      timestamp: first.timestamp,
      execution_time: first.execution_time,
      output_value: mergedPortKeyed as NodeHistoryEntry['output_value'],
      params: first.params,
      ...(meta && Object.keys(meta).length > 0 ? { metadata: meta } : {})
    });
  }
  return result;
}

export const normalizeHistoryMap = (history?: Record<string, any[]>): Record<string, NodeHistoryEntry[]> => {
  if (!history) return {};
  return Object.fromEntries(
    Object.entries(history).map(([nodeId, entries]) => [nodeId, normalizeHistoryEntries(entries as any[])])
  );
};

/** 归一化并按 run 聚合，得到「每条 run 一条、port_keyed」的历史表 */
export function normalizeAndAggregateHistoryMap(history?: Record<string, any[]>): Record<string, NodeHistoryEntry[]> {
  if (!history) return {};
  return Object.fromEntries(
    Object.entries(history).map(([nodeId, entries]) => {
      const normalized = normalizeHistoryEntries(entries as any[]);
      const aggregated = aggregateHistoryByRun(normalized);
      return [nodeId, aggregated];
    })
  );
};

/** 判断是否为以 port_id 为键的 output_value（多端口节点历史） */
function isPortKeyedOutputValue(ov: any): ov is Record<string, any> {
  if (!ov || typeof ov !== 'object' || Array.isArray(ov)) return false;
  return Object.keys(ov).some(k => k.startsWith('out-'));
}

/** 取条目的 port_keyed 展示值；优先 output_value（已统一为 port_keyed），兼容旧数据 output_value_port_keyed，最后单端口推导 */
export function getEntryPortKeyedValue(entry: NodeHistoryEntry): Record<string, any> {
  const ov = entry.output_value as Record<string, any> | undefined;
  if (ov && isPortKeyedOutputValue(ov)) {
    return ov;
  }
  if (ov?.kind === 'json' && typeof ov.json === 'object' && isPortKeyedOutputValue(ov.json)) {
    return ov.json as Record<string, any>;
  }
  if (entry.output_value_port_keyed && Object.keys(entry.output_value_port_keyed).length > 0) {
    return entry.output_value_port_keyed;
  }
  const portId = entry.metadata?.port_id ?? (entry as any).port_id ?? 'output';
  const single = singleOutputValueToDisplay((entry.output_value || {}) as Record<string, any>);
  return { [portId]: single };
}

export const historyEntryToDisplayValue = (entry: NodeHistoryEntry, portId?: string): any => {
  const portKeyed = getEntryPortKeyedValue(entry);
  if (portId != null) {
    return portKeyed[portId];
  }
  if (Object.keys(portKeyed).length === 1) {
    return Object.values(portKeyed)[0];
  }
  return portKeyed;
};
