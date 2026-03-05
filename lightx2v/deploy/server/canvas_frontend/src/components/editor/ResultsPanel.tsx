import React, { useState, useEffect } from 'react';
import {
  History,
  ChevronUp,
  ChevronDown,
  ToggleLeft,
  ToggleRight,
  RefreshCw,
  Maximize2,
  ArrowUpRight,
  AlertCircle,
  Volume2,
  Video as VideoIcon,
  Clock
} from 'lucide-react';
import { WorkflowState, NodeStatus, DataType } from '../../../types';
import { TOOLS } from '../../../constants';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { getIcon } from '../../utils/icons';
import { formatTime } from '../../utils/format';
import { getAssetPath } from '../../utils/assetPath';
import { getWorkflowFileText } from '../../utils/workflowFileManager';
import type { ResultEntry } from '../../hooks/useResultManagement';
import { isLightX2VResultRef, type LightX2VResultRef } from '../../hooks/useWorkflowExecution';

interface ResultsPanelProps {
  lang: Language;
  workflow: WorkflowState;
  showIntermediateResults: boolean;
  resultsCollapsed: boolean;
  onToggleCollapsed: () => void;
  resultEntries: ResultEntry[];
  onToggleShowIntermediate: () => void;
  onExpandOutput: (nodeId: string, fieldId?: string, runId?: string) => void;
  onPinOutputToCanvas?: (content: any, type: DataType) => void;
  resolveLightX2VResultRef?: (ref: LightX2VResultRef, context?: { workflow_id?: string; node_id?: string; port_id?: string }) => Promise<string>;
  getNodeOutputUrl?: (nodeId: string, portId: string, fileId?: string, runId?: string) => Promise<string | null>;
}

/** 运行结果里「文本」为 file 引用（file_id + text/plain）时按输出预览方式解析为可显示文本 */
function isTextFileRef(x: any): boolean {
  if (x == null || typeof x !== 'object') return false;
  const t = (x as any)._type === 'file' || (x as any).kind === 'file';
  const fileId = !!(x as any).file_id;
  const mime = (x as any).mime_type === 'text/plain';
  return (t && fileId) || (fileId && mime);
}

/** 文本文件引用：拉取正文并显示。传入 nodeId/portId/run_id 时通过 /output/url 解析，避免 404。 */
function ResolvedTextContent({
  workflowId,
  fileRef,
  nodeId,
  portId,
  className
}: {
  workflowId: string;
  fileRef: { file_id: string; run_id?: string };
  nodeId?: string;
  portId?: string;
  className?: string;
}) {
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    getWorkflowFileText(workflowId, fileRef.file_id, nodeId, portId, fileRef.run_id).then((content) => {
      if (!cancelled) setText(content ?? '');
    }).catch(() => { if (!cancelled) setText(''); });
    return () => { cancelled = true; };
  }, [workflowId, fileRef.file_id, fileRef.run_id, nodeId, portId]);
  if (text === null) {
    return <span className={className}>Loading…</span>;
  }
  return <span className={className}>{text || '—'}</span>;
}

function textResultDisplay(val: any, t: (key: string) => string): string {
  if (val == null) return '—';
  if (typeof val === 'string') return val;
  if (typeof val === 'object') {
    if ((val as any).kind === 'text' && typeof (val as any).text === 'string') return (val as any).text;
    if ((val as any).type === 'text' && typeof (val as any).data === 'string') return (val as any).data;
    if (isTextFileRef(val)) return t('result_text_file_placeholder'); // fallback when no workflowId
    return t('result_text_file_placeholder');
  }
  return String(val);
}

/** 渲染时通过 getNodeOutputUrl 将 file/task ref 转为 url 给 img，不修改 output_value。与运行结果栏、弹窗共用。 */
export function ResolvedImage({
  content,
  resolveLightX2VResultRef,
  workflowId,
  nodeId,
  portId = 'out-image',
  getNodeOutputUrl,
  className
}: {
  content: any;
  resolveLightX2VResultRef?: (ref: LightX2VResultRef, context?: { workflow_id?: string; node_id?: string; port_id?: string }) => Promise<string>;
  workflowId?: string;
  nodeId?: string;
  portId?: string;
  getNodeOutputUrl?: (nodeId: string, portId: string, fileId?: string, runId?: string) => Promise<string | null>;
  className?: string;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const displayContent = content != null && typeof content === 'object' && !Array.isArray(content) && Object.keys(content).length === 1
    ? (content as Record<string, unknown>)[Object.keys(content)[0]]
    : content;
  useEffect(() => {
    if (displayContent == null) { setUrl(null); return; }
    if (isLightX2VResultRef(displayContent) && resolveLightX2VResultRef) {
      let cancelled = false;
      const ctx = workflowId && nodeId && portId ? { workflow_id: workflowId, node_id: nodeId, port_id: portId } : undefined;
      resolveLightX2VResultRef(displayContent, ctx).then(u => { if (!cancelled) setUrl(u); }).catch(() => { if (!cancelled) setUrl(null); });
      return () => { cancelled = true; };
    }
    const fileRef = displayContent && typeof displayContent === 'object' && (displayContent as any).kind === 'file' && (displayContent as any).file_id;
    if (fileRef && workflowId && nodeId && getNodeOutputUrl) {
      const fileId = (displayContent as any).file_id;
      const runId = (displayContent as any).run_id;
      let cancelled = false;
      getNodeOutputUrl(nodeId, portId, fileId, runId).then(u => { if (!cancelled && u) setUrl(getAssetPath(u) || u); }).catch(() => { if (!cancelled) setUrl(null); });
      return () => { cancelled = true; };
    }
    if (typeof displayContent === 'string' && (displayContent.startsWith('http') || displayContent.startsWith('data:'))) {
      setUrl(displayContent);
      return;
    }
    if (typeof displayContent === 'string') {
      setUrl(getAssetPath(displayContent) || null);
      return;
    }
    setUrl(null);
  }, [displayContent, resolveLightX2VResultRef, workflowId, nodeId, portId, getNodeOutputUrl]);
  if (url) return <img src={url} className={className} alt="" />;
  if (displayContent != null && isLightX2VResultRef(displayContent)) return <div className={className + ' flex items-center justify-center text-[9px] text-slate-500 uppercase'}>Loading...</div>;
  if (displayContent != null && typeof displayContent === 'object' && (displayContent as any).kind === 'file' && workflowId && nodeId && getNodeOutputUrl) return <div className={className + ' flex items-center justify-center text-[9px] text-slate-500 uppercase'}>Loading...</div>;
  return <img src={getAssetPath(displayContent) ?? ''} className={className} alt="" />;
}

export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  lang,
  workflow,
  showIntermediateResults,
  resultsCollapsed,
  onToggleCollapsed,
  resultEntries,
  onToggleShowIntermediate,
  onExpandOutput,
  onPinOutputToCanvas,
  resolveLightX2VResultRef,
  getNodeOutputUrl
}) => {
  const { t } = useTranslation(lang);

  return (
    <footer
      className={`${
        resultsCollapsed ? 'h-12' : 'h-80'
      } border-t border-slate-800/60 bg-slate-900/60 backdrop-blur-3xl z-40 flex flex-col overflow-hidden transition-all duration-300`}
    >
      <div className="px-8 py-4 border-b border-slate-800/60 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <button
            onClick={onToggleCollapsed}
            className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.3em] text-#90dce1 hover:text-indigo-300 transition-all"
          >
            <History size={16} />
            {t('execution_results')}
            {resultsCollapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {!resultsCollapsed && (
            <>
              <div className="h-4 w-px bg-slate-800"></div>
              <button
                onClick={onToggleShowIntermediate}
                className="flex items-center gap-2 text-[9px] font-black uppercase text-slate-400 hover:text-indigo-300 transition-all"
              >
                {showIntermediateResults ? (
                  <ToggleRight size={16} className="text-#90dce1" />
                ) : (
                  <ToggleLeft size={16} />
                )}
                {t('show_intermediate')}
              </button>
            </>
          )}
        </div>
      </div>
      {!resultsCollapsed && (
        <div className="flex-1 overflow-x-auto p-8 flex gap-8 items-start custom-scrollbar flex-row">
          {resultEntries.length === 0 && !workflow.isRunning ? (
            <div className="flex-1 flex flex-col items-center justify-center opacity-10 animate-pulse">
              <RefreshCw size={48} className="mb-4 animate-spin-slow" />
              <span className="text-[10px] font-black uppercase tracking-widest">{t('awaiting_execution')}</span>
            </div>
          ) : (
            resultEntries.map(entry => {
              const { node, output: res, runId } = entry;
              const tool = TOOLS.find(t => t.id === node.tool_id);
              if (!tool) return null;
              const type = tool?.outputs[0]?.type || DataType.TEXT;
              const isTerminal = !workflow.connections.some(c => c.source_node_id === node.id);

              const elapsed =
                node.status === NodeStatus.RUNNING
                  ? ((performance.now() - (node.start_time || performance.now())) / 1000).toFixed(1) + 's'
                  : node.status === NodeStatus.PENDING
                    ? (lang === 'zh' ? '排队中' : 'Pending')
                    : formatTime(node.execution_time);

              const expand = (fieldId?: string) => onExpandOutput(node.id, fieldId, runId);

              return (
                <div
                  key={`${runId}-${node.id}`}
                  className={`min-w-[320px] max-w-[420px] bg-slate-900/50 rounded-[32px] border p-6 flex flex-col shadow-2xl relative overflow-hidden group transition-all h-[190px] ${
                    node.status === NodeStatus.ERROR
                      ? 'border-red-500/30 bg-red-500/5'
                      : node.status === NodeStatus.PENDING
                      ? 'border-amber-500/30 bg-amber-500/5'
                      : isTerminal
                      ? 'border-emerald-500/30'
                      : 'border-slate-800/60'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex flex-col gap-0.5">
                      <span className="text-[9px] font-black text-slate-500 uppercase flex items-center gap-2">
                        {React.createElement(getIcon(tool.icon), { size: 12 })}{' '}
                        {node.name ?? (lang === 'zh' ? tool.name_zh : tool.name)}
                      </span>
                      <span className="text-[8px] font-bold text-slate-600">
                        {runId === 'current' && (node.status === NodeStatus.RUNNING || node.status === NodeStatus.PENDING || node.execution_time !== undefined)
                          ? `${t('run_time')}: ${elapsed}`
                          : new Date(entry.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="flex gap-2">
                      {node.status !== NodeStatus.ERROR && (
                        <>
                          <button
                            onClick={() => expand()}
                            className="p-1.5 text-slate-500 hover:text-white transition-all"
                          >
                            <Maximize2 size={14} />
                          </button>
                          {onPinOutputToCanvas && res != null && (
                            <button
                              onClick={() => onPinOutputToCanvas(res, type)}
                              className="p-1.5 text-slate-500 hover:text-#90dce1 transition-all"
                            >
                              <ArrowUpRight size={14} />
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                  <div
                    className={`flex-1 overflow-y-auto rounded-xl p-3 custom-scrollbar ${
                      node.status === NodeStatus.ERROR
                        ? 'bg-red-500/10 border border-red-500/20'
                        : node.status === NodeStatus.PENDING
                        ? 'bg-amber-500/10 border border-amber-500/20'
                        : 'bg-slate-950/40'
                    }`}
                  >
                    {node.status === NodeStatus.ERROR ? (
                      <div className="h-full flex flex-col items-center justify-center text-center p-2">
                        <AlertCircle size={24} className="text-red-500 mb-2" />
                        <p className="text-[10px] text-red-400 font-bold uppercase mb-1">{t('execution_failed')}</p>
                        <p className="text-[9px] text-slate-400 line-clamp-3 leading-relaxed">{node.error}</p>
                      </div>
                    ) : node.status === NodeStatus.PENDING ? (
                      <div className="h-full flex flex-col items-center justify-center text-center p-2">
                        <Clock size={24} className="text-amber-400 mb-2" />
                        <p className="text-[10px] text-amber-400 font-bold uppercase">{lang === 'zh' ? '排队中' : 'Pending'}</p>
                      </div>
                    ) : type === DataType.TEXT ? (
                      typeof res === 'object' && res !== null && !Array.isArray(res) && !isTextFileRef(res) ? (
                        <div className="space-y-2">
                          {Object.entries(res).map(([k, v]) => (
                            <div
                              key={k}
                              onClick={() => expand(k)}
                              className="group/field p-2 bg-slate-900/60 rounded-lg border border-slate-800 hover:border-#90dce1/50 cursor-pointer transition-all"
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-[8px] font-black text-#90dce1 uppercase tracking-tighter">
                                  {k}
                                </span>
                                <Maximize2 size={8} className="text-slate-600 group-hover/field:text-#90dce1" />
                              </div>
                              <p className="text-[9px] text-slate-400 line-clamp-1">
                                {workflow?.id && isTextFileRef(v) && (v as { file_id: string }).file_id ? (
                                  <ResolvedTextContent workflowId={workflow.id} fileRef={v as { file_id: string; run_id?: string }} nodeId={node.id} portId={k} className="block" />
                                ) : (
                                  textResultDisplay(v, t)
                                )}
                              </p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p
                          onClick={() => expand()}
                          className="text-[10px] text-slate-400 line-clamp-4 leading-relaxed cursor-pointer hover:text-slate-200 transition-colors"
                        >
                          {workflow?.id && isTextFileRef(res) && (res as { file_id?: string }).file_id ? (
                            <ResolvedTextContent workflowId={workflow.id} fileRef={res as { file_id: string; run_id?: string }} nodeId={node.id} portId={tool?.outputs?.[0]?.id ?? 'out-text'} className="block" />
                          ) : (
                            textResultDisplay(res, t)
                          )}
                        </p>
                      )
                    ) : type === DataType.IMAGE ? (
                      <div
                        onClick={() => expand()}
                        className="flex gap-2 overflow-x-auto h-full pb-1 custom-scrollbar cursor-pointer items-center"
                      >
                        {(Array.isArray(res) ? res : res != null ? [res] : []).map((img, i) => (
                          <React.Fragment key={i}>
                            <ResolvedImage
                              content={img}
                              resolveLightX2VResultRef={resolveLightX2VResultRef}
                              workflowId={workflow?.id}
                              nodeId={node.id}
                              portId={tool?.outputs?.[0]?.id ?? 'out-image'}
                              getNodeOutputUrl={getNodeOutputUrl}
                              className="max-h-full w-auto object-contain rounded-lg border border-slate-800"
                            />
                          </React.Fragment>
                        ))}
                      </div>
                    ) : (
                      <div
                        onClick={() => expand()}
                        className="flex items-center justify-center h-full text-#90dce1 cursor-pointer hover:text-[#90dce1] transition-colors"
                      >
                        {type === DataType.AUDIO ? <Volume2 size={24} /> : <VideoIcon size={24} />}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}
    </footer>
  );
};
