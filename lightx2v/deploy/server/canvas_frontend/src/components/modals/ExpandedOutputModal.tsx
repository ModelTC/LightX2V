import React, { useState, useEffect } from 'react';
import { X, Download, SaveAll } from 'lucide-react';
import { DataType } from '../../../types';
import { downloadFile } from '../../utils/download';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { pcmToWavUrl } from '../../utils/audio';
import { getAssetPath } from '../../utils/assetPath';
import { isLightX2VResultRef, type LightX2VResultRef } from '../../hooks/useWorkflowExecution';
import { collectLightX2VResultRefs } from '../../utils/resultRef';
import { AudioNodePreview } from '../previews/AudioNodePreview';
import { ResolvedImage } from '../editor/ResultsPanel';

interface ExpandedOutputModalProps {
  lang: Language;
  expandedOutput: { nodeId: string; fieldId?: string } | null;
  resolveLightX2VResultRef?: (ref: LightX2VResultRef) => Promise<string>;
  /** 仅用于本地 file 类输出（kind: 'file', file_id）；x2v 任务结果用 resolveLightX2VResultRef，不请求本地 node output url */
  getNodeOutputUrl?: (nodeId: string, portId: string, fileId?: string, runId?: string) => Promise<string | null>;
  workflowId?: string;
  /** 传入时优先从 node.output_value 取 ref，与节点小图一致，刚生成即可显示 */
  workflow?: { nodes: Array<{ id: string; output_value?: any }> } | null;
  /** 与节点右侧预览、运行结果栏同一数据源，传入后优先从此取 IMAGE ref，避免首次打开无图 */
  sourceOutputs?: Record<string, any>;
  /** 与运行结果栏同源：当前展开节点在 resultEntries 中的 output（可能已由 history 解包为 ref），弹窗优先用此显示图片 */
  resultEntryOutputForNode?: any;
  expandedResultData: {
    content: any;
    label: string;
    type: DataType;
    nodeId: string;
    originalOutput: any;
  } | null;
  isEditingResult: boolean;
  tempEditValue: string;
  onClose: () => void;
  onEditToggle: () => void;
  onSaveEdit: () => void;
  onTempEditValueChange: (value: string) => void;
}

function isFileRef(c: any): boolean {
  return c != null && typeof c === 'object' && (c.kind === 'file' || c._type === 'file') && !!c.file_id;
}

/** 当 content 为单端口包裹的 file ref（如 { "out-audio": { kind: "file", file_id, ... } }）时，返回内层 ref 与 portId，否则返回 content 与 undefined */
function unwrapPortFileRef(content: any, type: DataType): { effectiveContent: any; effectivePortId: string | undefined } {
  if (!content || typeof content !== 'object' || Array.isArray(content)) return { effectiveContent: content, effectivePortId: undefined };
  const keys = Object.keys(content);
  if (keys.length === 1) {
    const val = content[keys[0]];
    if (isFileRef(val)) return { effectiveContent: val, effectivePortId: keys[0] };
  }
  if (isFileRef(content)) {
    const defaultPort = type === DataType.AUDIO ? 'out-audio' : type === DataType.VIDEO ? 'out-video' : type === DataType.IMAGE ? 'out-image' : 'output';
    return { effectiveContent: content, effectivePortId: defaultPort };
  }
  return { effectiveContent: content, effectivePortId: undefined };
}

/** content 为 ref 或 port-keyed 包裹的 ref 时返回该 ref，否则返回 null */
function getLightX2VRefFromContent(content: any): LightX2VResultRef | null {
  if (!content || typeof content !== 'object') return null;
  if (isLightX2VResultRef(content)) return content;
  const refs = collectLightX2VResultRefs(content);
  return refs.length > 0 ? refs[0] : null;
}

/** 将可能为 camelCase 或 snake_case 的 ref 规范为 { kind, task_id, output_name, is_cloud }，无法识别则返回 null */
function normalizeTaskRef(val: any): { kind: 'task'; task_id: string; output_name: string; is_cloud: boolean } | null {
  if (val == null || typeof val !== 'object' || Array.isArray(val)) return null;
  const taskId = (val as any).task_id ?? (val as any).taskId;
  const outputName = (val as any).output_name ?? (val as any).outputName;
  if (typeof taskId !== 'string' || typeof outputName !== 'string') return null;
  const is_cloud = (val as any).is_cloud === true;
  return { kind: 'task', task_id: taskId, output_name: outputName, is_cloud };
}

export const ExpandedOutputModal: React.FC<ExpandedOutputModalProps> = ({
  lang,
  expandedOutput,
  expandedResultData,
  resolveLightX2VResultRef,
  getNodeOutputUrl,
  workflowId,
  workflow,
  sourceOutputs,
  resultEntryOutputForNode,
  isEditingResult,
  tempEditValue,
  onClose,
  onEditToggle,
  onSaveEdit,
  onTempEditValueChange
}) => {
  const { t } = useTranslation(lang);
  const [resolvedMediaUrl, setResolvedMediaUrl] = useState<string | null>(null);
  const [resolvedFileUrl, setResolvedFileUrl] = useState<string | null>(null);
  /** 首次打开时 content 可能尚未就绪（workflow 未更新），从 workflow 再读一次 ref，触发重渲染以显示图片 */
  const [lateWorkflowRef, setLateWorkflowRef] = useState<LightX2VResultRef | null>(null);

  // 与节点右侧预览、运行结果栏同源，优先从此取 ref（IMAGE/VIDEO/AUDIO 均避免 expandedResultData 未就绪时无媒体）
  const portIdByType = React.useMemo(() => ({
    [DataType.IMAGE]: 'out-image',
    [DataType.VIDEO]: 'out-video',
    [DataType.AUDIO]: 'out-audio'
  }), []);
  const refFromSourceOutputsTop = React.useMemo(() => {
    if (!expandedOutput?.nodeId || !sourceOutputs) return null;
    const type = expandedResultData?.type;
    const portId = type != null && type in portIdByType ? portIdByType[type as DataType] : 'out-image';
    const soRaw = sourceOutputs[expandedOutput.nodeId];
    if (soRaw == null) return null;
    const soPortVal =
      typeof soRaw === 'object' && !Array.isArray(soRaw) && portId in soRaw
        ? (soRaw as Record<string, unknown>)[portId]
        : typeof soRaw === 'object' && !Array.isArray(soRaw) && Object.keys(soRaw).length === 1
          ? (soRaw as Record<string, unknown>)[Object.keys(soRaw)[0]]
          : soRaw;
    return soPortVal != null ? (normalizeTaskRef(soPortVal) ?? getLightX2VRefFromContent(soPortVal)) : null;
  }, [expandedOutput?.nodeId, sourceOutputs, expandedResultData?.type, portIdByType]);

  /** 与运行结果栏同源：从 resultEntryOutputForNode 解包出 ref（与 ResolvedImage 收到的 content 一致），供 IMAGE 优先显示 */
  const refFromResultEntryTop = React.useMemo(() => {
    const out = resultEntryOutputForNode;
    if (out == null) return null;
    const unwrapped = typeof out === 'object' && !Array.isArray(out) && Object.keys(out).length === 1
      ? (out as Record<string, unknown>)[Object.keys(out)[0]]
      : out;
    return unwrapped != null ? (normalizeTaskRef(unwrapped) ?? (isLightX2VResultRef(unwrapped) ? unwrapped : null) ?? getLightX2VRefFromContent(unwrapped)) : null;
  }, [resultEntryOutputForNode]);

  /** 从 sourceOutputs 取当前类型的端口内容（供 AUDIO/VIDEO 与 content 同源优先使用） */
  const contentFromSourceOutputsTop = React.useMemo(() => {
    if (!expandedOutput?.nodeId || !sourceOutputs) return null;
    const type = expandedResultData?.type;
    const portId = type != null && type in portIdByType ? portIdByType[type as DataType] : 'out-image';
    const soRaw = sourceOutputs[expandedOutput.nodeId];
    if (soRaw == null) return null;
    if (typeof soRaw === 'object' && !Array.isArray(soRaw) && portId in soRaw) return (soRaw as Record<string, unknown>)[portId];
    if (typeof soRaw === 'object' && !Array.isArray(soRaw) && Object.keys(soRaw).length === 1) return (soRaw as Record<string, unknown>)[Object.keys(soRaw)[0]];
    return soRaw;
  }, [expandedOutput?.nodeId, sourceOutputs, expandedResultData?.type, portIdByType]);

  // 优先用 workflow 里节点最新 output_value 取 ref，与节点小图一致，刚生成即可显示
  const contentForRef =
    workflow && expandedOutput?.nodeId
      ? (() => {
          const node = workflow.nodes.find((n: { id: string }) => n.id === expandedOutput!.nodeId);
          const ov = node?.output_value;
          if (expandedOutput?.fieldId && ov && typeof ov === 'object') return ov[expandedOutput.fieldId] ?? ov;
          return ov;
        })()
      : expandedResultData?.content;
  const lightX2VRef = contentForRef != null ? getLightX2VRefFromContent(contentForRef) : null;

  // 首次打开时 expandedResultData.content 可能尚未包含 output_value，从 workflow 直接再读一次（IMAGE/VIDEO），以便 workflow 更新后能立即显示
  useEffect(() => {
    const type = expandedResultData?.type;
    if (!expandedOutput?.nodeId || !workflow || (type !== DataType.IMAGE && type !== DataType.VIDEO)) {
      setLateWorkflowRef(null);
      return;
    }
    const node = workflow.nodes.find((n: { id: string }) => n.id === expandedOutput.nodeId);
    const ov = node?.output_value;
    if (ov == null || typeof ov !== 'object') {
      setLateWorkflowRef(null);
      return;
    }
    const portId = type === DataType.VIDEO ? 'out-video' : 'out-image';
    const portVal = typeof ov === 'object' && !Array.isArray(ov) && portId in ov
      ? (ov as Record<string, unknown>)[portId]
      : Object.keys(ov).length >= 1 ? (ov as Record<string, unknown>)[Object.keys(ov)[0]] : null;
    const ref = portVal != null ? (normalizeTaskRef(portVal) ?? getLightX2VRefFromContent(portVal)) : getLightX2VRefFromContent(ov);
    setLateWorkflowRef(ref);
  }, [workflow, expandedOutput?.nodeId, expandedResultData?.type]);

  const refToResolve = refFromResultEntryTop ?? refFromSourceOutputsTop ?? lightX2VRef ?? lateWorkflowRef;
  const defaultPortIdByType = expandedResultData?.type === DataType.VIDEO ? 'out-video' : expandedResultData?.type === DataType.AUDIO ? 'out-audio' : 'out-image';
  const effectivePortIdForResolve = expandedOutput?.fieldId ?? defaultPortIdByType;

  useEffect(() => {
    if (!refToResolve || !resolveLightX2VResultRef) {
      setResolvedMediaUrl(null);
      return;
    }
    let cancelled = false;
    const ctx = workflowId && expandedResultData?.nodeId ? { workflow_id: workflowId, node_id: expandedResultData.nodeId, port_id: effectivePortIdForResolve } : undefined;
    resolveLightX2VResultRef(refToResolve, ctx).then(url => {
      if (!cancelled) setResolvedMediaUrl(url);
    }).catch(() => { if (!cancelled) setResolvedMediaUrl(null); });
    return () => { cancelled = true; };
  }, [refToResolve?.task_id, refToResolve?.output_name, refToResolve?.is_cloud, resolveLightX2VResultRef, workflowId, expandedResultData?.nodeId, effectivePortIdForResolve]);

  // 仅对本地 file 类输出（kind: 'file', file_id）请求本地后端；x2v ref 已在上方处理，此处不再请求
  useEffect(() => {
    if (!expandedResultData || !expandedOutput || !workflowId) {
      setResolvedFileUrl(null);
      return;
    }
    const content = expandedResultData.content;
    if (getLightX2VRefFromContent(content) != null) {
      setResolvedFileUrl(null);
      return;
    }
    const type = expandedResultData.type;
    if (type !== DataType.AUDIO && type !== DataType.IMAGE && type !== DataType.VIDEO) {
      setResolvedFileUrl(null);
      return;
    }
    const { effectiveContent, effectivePortId } = unwrapPortFileRef(content, type);
    if (!isFileRef(effectiveContent)) {
      setResolvedFileUrl(null);
      return;
    }
    const fileId = (effectiveContent as { file_id?: string }).file_id;
    const runId = (effectiveContent as { run_id?: string }).run_id;
    const defaultPort = type === DataType.VIDEO ? 'out-video' : type === DataType.AUDIO ? 'out-audio' : 'out-image';
    const portId = expandedOutput.fieldId || effectivePortId || defaultPort;
    // Avoid requesting URL when file_id looks like a port id (e.g. "out-image") to prevent 404
    const looksLikePortId = !fileId || fileId === portId || /^out-/.test(fileId);
    if (!workflowId || !fileId || !getNodeOutputUrl || looksLikePortId) {
      setResolvedFileUrl(null);
      return;
    }
    let cancelled = false;
    getNodeOutputUrl(expandedResultData.nodeId, portId, fileId, runId).then(url => {
      if (!cancelled && url) setResolvedFileUrl(url);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [expandedResultData?.content, expandedResultData?.nodeId, expandedResultData?.type, expandedOutput?.fieldId, workflowId, getNodeOutputUrl]);

  if (!expandedOutput || !expandedResultData) return null;

  const isTextType = expandedResultData.type === DataType.TEXT;
  const originalText = isTextType
    ? (typeof expandedResultData.content === 'object'
        ? JSON.stringify(expandedResultData.content, null, 2)
        : (expandedResultData.content ?? ''))
    : '';
  const hasTextChanges = isTextType && tempEditValue !== originalText;

  const getDownloadUrl = (): string | null => {
    const c = expandedResultData.content;
    if (resolvedFileUrl) return getAssetPath(resolvedFileUrl) || resolvedFileUrl;
    if (expandedResultData.type !== DataType.TEXT && getLightX2VRefFromContent(c) && resolvedMediaUrl) return resolvedMediaUrl;
    if (typeof c === 'string') return c || null;
    if (Array.isArray(c)) {
      const first = c.find((item: any) => typeof item === 'string');
      return first ?? null;
    }
    if (c && typeof c === 'object') {
      return (
        (typeof (c as any).url === 'string' && (c as any).url) ||
        (typeof (c as any).file_url === 'string' && (c as any).file_url) ||
        (typeof (c as any).path === 'string' && (c as any).path) ||
        (typeof (c as any).src === 'string' && (c as any).src) ||
        (typeof (c as any).data === 'string' && (c as any).data) ||
        (typeof (c as any)._full_data === 'string' && (c as any)._full_data) ||
        null
      );
    }
    return null;
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-8 bg-slate-950/90 backdrop-blur-2xl animate-in fade-in duration-300">
      <div className="bg-slate-900 border border-slate-800 rounded-[40px] shadow-2xl w-full max-w-5xl h-full flex flex-col relative overflow-hidden">
        <div className="p-6 border-b border-slate-800 flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-4">
              <h2 className="text-xl font-black uppercase tracking-widest">{expandedResultData.label}</h2>
              {expandedResultData.type === DataType.TEXT && (
                <button
                  onClick={onEditToggle}
                  className={`px-3 py-1.5 rounded-lg text-[10px] font-black uppercase transition-all ${
                    isEditingResult ? 'bg-#90dce1 text-white shadow-lg' : 'bg-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  {isEditingResult ? t('save_changes') : t('edit_mode')}
                </button>
              )}
            </div>
            {isEditingResult && expandedResultData.type === DataType.TEXT && (
              <p className="text-[10px] text-#90dce1 font-bold uppercase animate-pulse">{t('manual_edit_hint')}</p>
            )}
          </div>
          <div className="flex items-center gap-3">
            {isTextType && isEditingResult ? (
              <button
                onClick={onSaveEdit}
                disabled={!hasTextChanges}
                className={`p-3 rounded-2xl transition-all flex items-center gap-2 px-6 ${
                  hasTextChanges
                    ? 'bg-emerald-600 hover:bg-emerald-500 text-white active:scale-90 shadow-lg shadow-emerald-500/20 cursor-pointer'
                    : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                }`}
              >
                <SaveAll size={20} />
                <span className="text-sm font-black uppercase">{t('save_changes')}</span>
              </button>
            ) : (
              <button
                onClick={async () => {
                  const url = getDownloadUrl();
                  if (url == null) return;
                  try {
                    await downloadFile(url, expandedResultData.label, expandedResultData.type);
                  } catch (e) {
                    console.error('[ExpandedOutputModal] Download failed:', e);
                  }
                }}
                className="p-3 bg-slate-800 hover:bg-slate-700 rounded-2xl transition-all active:scale-90"
              >
                <Download size={20} />
              </button>
            )}
            <button onClick={onClose} className="p-3 text-slate-400 hover:text-white transition-all">
              <X size={24} />
            </button>
          </div>
        </div>
        <div className="flex-1 min-h-0 p-12 overflow-y-auto overflow-x-auto flex items-center justify-center custom-scrollbar">
          {expandedResultData.type === DataType.TEXT ? (
            isEditingResult ? (
              <textarea
                value={tempEditValue}
                onChange={e => onTempEditValueChange(e.target.value)}
                className="w-full h-full bg-slate-950 border-2 border-#90dce1/50 rounded-3xl p-8 text-base text-indigo-100 resize-none focus:ring-0 focus:border-#90dce1 font-mono transition-all custom-scrollbar selection:bg-#90dce1/30"
                placeholder="Manually edit the AI output..."
                autoFocus
              />
            ) : typeof expandedResultData.content === 'object' ? (
              <pre className="text-xs bg-slate-950/50 p-8 rounded-3xl border border-slate-800/50 text-indigo-300 max-w-3xl w-full overflow-auto selection:bg-#90dce1/20">
                {JSON.stringify(expandedResultData.content, null, 2)}
              </pre>
            ) : (
              <p className="text-lg leading-relaxed max-w-3xl whitespace-pre-wrap selection:bg-#90dce1/20">
                {expandedResultData.content}
              </p>
            )
          ) : expandedResultData.type === DataType.IMAGE ? (
            (() => {
              // 与下方运行结果栏完全一致：用同一份 output，同一套 (Array.isArray ? res : [res])，同一组件 ResolvedImage
              const res = resultEntryOutputForNode ?? expandedResultData.content;
              const items = Array.isArray(res) ? res : res != null ? [res] : [];
              if (items.length === 0) return <div className="text-sm text-slate-500">No image data</div>;
              return (
                <div className="w-full h-full min-h-0 flex items-center justify-center">
                  <div className="flex gap-4 flex-wrap justify-center items-center max-h-full max-w-full">
                    {items.map((img, i) => (
                      <div key={i} className="flex items-center justify-center max-h-full">
                        <ResolvedImage
                          content={img}
                          resolveLightX2VResultRef={resolveLightX2VResultRef}
                          workflowId={workflowId}
                          nodeId={expandedResultData.nodeId}
                          portId={effectivePortIdForResolve}
                          getNodeOutputUrl={getNodeOutputUrl}
                          className="max-h-full max-w-full w-auto h-auto rounded-2xl shadow-2xl border border-slate-800 object-contain"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()
          ) : expandedResultData.type === DataType.AUDIO ? (
            (() => {
              const { effectiveContent: audioContent } = unwrapPortFileRef(expandedResultData.content, DataType.AUDIO);
              const audioContentForDisplay = contentFromSourceOutputsTop ?? audioContent;
              const getMediaValue = (value: any) => {
                if (!value) return '';
                if (typeof value === 'string') return value;
                if (Array.isArray(value)) {
                  return value.find(item => typeof item === 'string') || '';
                }
                if (typeof value === 'object') {
                  return (
                    (typeof value.url === 'string' && value.url) ||
                    (typeof value.file_url === 'string' && value.file_url) ||
                    (typeof value.path === 'string' && value.path) ||
                    (typeof value.src === 'string' && value.src) ||
                    (typeof value.data === 'string' && value.data) ||
                    ''
                  );
                }
                return '';
              };

              const fromApi = resolvedFileUrl ? (getAssetPath(resolvedFileUrl) || resolvedFileUrl) : '';
              const audioValue = getMediaValue(audioContentForDisplay);
              const fallback = !audioValue ? '' : (audioValue.startsWith('data:') ? audioValue : (audioValue.startsWith('http') || audioValue.startsWith('/') || audioValue.startsWith('./assets') || audioValue.startsWith('blob:') ? getAssetPath(audioValue) : pcmToWavUrl(audioValue)));
              const audioSrc = (resolvedMediaUrl || fromApi || fallback) || '';
              if (!audioSrc) {
                return <div className="text-sm text-slate-500">{isFileRef(audioContentForDisplay) && !resolvedFileUrl ? 'Loading...' : 'No audio data'}</div>;
              }
              return (
                <div className="w-full max-w-2xl">
                  <AudioNodePreview
                    audioData={{
                      original: audioSrc,
                      trimmed: audioSrc,
                      range: { start: 0, end: 100 }
                    }}
                    readOnly
                  />
                </div>
              );
            })()
          ) : (
            (() => {
              const { effectiveContent: raw } = unwrapPortFileRef(expandedResultData.content, DataType.VIDEO);
              const rawForDisplay = contentFromSourceOutputsTop ?? raw;
              const refFromContent = getLightX2VRefFromContent(rawForDisplay);
              const refForDisplay = refFromSourceOutputsTop ?? refFromContent ?? lightX2VRef ?? lateWorkflowRef ?? null;
              const isRef = refForDisplay != null || getLightX2VRefFromContent(raw) != null;
              const fromApi = resolvedFileUrl ? (getAssetPath(resolvedFileUrl) || resolvedFileUrl) : '';
              const getMediaValue = (val: any): string => {
                if (val == null) return '';
                if (typeof val === 'string') return val;
                if (Array.isArray(val)) {
                  const first = val.find((item: any) => typeof item === 'string');
                  return first != null ? first : '';
                }
                if (typeof val === 'object') {
                  return (
                    (typeof val.data === 'string' && val.data) ||
                    (typeof val.url === 'string' && val.url) ||
                    (typeof val.file_url === 'string' && val.file_url) ||
                    (typeof val._full_data === 'string' && val._full_data) ||
                    ''
                  );
                }
                return '';
              };
              const v = getMediaValue(rawForDisplay);
              const fallback = v !== '' ? (v.startsWith('http') || v.startsWith('data:') || v.startsWith('blob:') ? v : getAssetPath(v)) : '';
              const videoSrc = (resolvedMediaUrl || fromApi || fallback) || '';
              if (!videoSrc || videoSrc === '') {
                return <div className="text-sm text-slate-500">{(isRef && !resolvedMediaUrl) || (isFileRef(rawForDisplay) && !resolvedFileUrl) ? 'Loading...' : 'No video data'}</div>;
              }
              return (
                <video
                  controls
                  autoPlay
                  src={videoSrc}
                  className="max-h-full rounded-2xl shadow-2xl"
                />
              );
            })()
          )}
        </div>
      </div>
    </div>
  );
};
