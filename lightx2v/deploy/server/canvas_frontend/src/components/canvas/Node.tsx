import React, { useRef } from 'react';
import {
  RefreshCw,
  Trash2,
  CheckCircle2,
  AlertCircle,
  TriangleAlert,
  PlayCircle as PlayIcon,
  FastForward,
  X,
  Plus,
  Upload,
  Volume2,
  Video as VideoIcon,
  Maximize2,
  Play,
  Square,
  ChevronDown,
  CheckCircle2 as CheckCircle,
  Globe,
  Bot,
  Clock,
  History,
  FileText,
  Image as ImageIcon
} from 'lucide-react';
import { WorkflowNode, WorkflowState, NodeStatus, DataType, Port, ToolDefinition, NodeHistoryEntry, NodeRunState } from '../../../types';
import { TOOLS } from '../../../constants';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { getIcon } from '../../utils/icons';
import { formatTime, formatRunTime } from '../../utils/format';
import { screenToWorld, ViewState } from '../../utils/canvas';
import { getAssetPath } from '../../utils/assetPath';
import { historyEntryToDisplayValue, normalizeHistoryEntries, getEntryPortKeyedValue } from '../../utils/historyEntry';
import { getLocalFileDataUrl, getWorkflowFileByFileId, getWorkflowFileText, getNodeOutputUrl, getNodeOutputDisplayUrl, getWorkflowFileUrl, persistDataUrlToLocal, saveNodeOutputsFromHistoryEntry } from '../../utils/workflowFileManager';
import { isStandalone } from '../../config/runtimeMode';
import { isLightX2VResultRef, type LightX2VResultRef } from '../../hooks/useWorkflowExecution';
import { getOutputValueByPort, setOutputValueByPort, INPUT_PORT_IDS } from '../../utils/outputValuePort';
import { TextNodePreview } from '../previews/TextNodePreview';
import { ImageNodePreview } from '../previews/ImageNodePreview';
import { AudioNodePreview } from '../previews/AudioNodePreview';
import { VideoNodePreview } from '../previews/VideoNodePreview';

const HistoryEntryItem: React.FC<{
  entry: NodeHistoryEntry;
  lang: Language;
  onSelect: () => void;
  resolveLightX2VResultRef?: (ref: LightX2VResultRef) => Promise<string>;
  outputDataType?: string;
  outputPortId?: string;
  resolveAudioUrl?: (entry: NodeHistoryEntry) => Promise<string | null>;
  resolveVideoUrl?: (entry: NodeHistoryEntry) => Promise<string | null>;
  resolveImageUrl?: (entry: NodeHistoryEntry) => Promise<string | null>;
  resolveTextFileContent?: (entry: NodeHistoryEntry) => Promise<string | null>;
}> = ({ entry, lang, onSelect, resolveLightX2VResultRef, outputDataType, outputPortId, resolveAudioUrl, resolveVideoUrl, resolveImageUrl, resolveTextFileContent }) => {
  const portKeyed = getEntryPortKeyedValue(entry);
  const effectiveOv = (outputPortId ? portKeyed[outputPortId] : null) ?? (Object.keys(portKeyed).length === 1 ? Object.values(portKeyed)[0] : null) ?? entry.output_value ?? {};
  const ov = (effectiveOv != null && typeof effectiveOv === 'object' && !Array.isArray(effectiveOv) ? effectiveOv : (entry.output_value ?? {})) as Record<string, any>;
  const kind = ov.kind ?? (entry as any).kind;
  let preview = '';
  let isLightX2VResult = false;
  let lightX2VRef: LightX2VResultRef | null = null;
  if (kind === 'text') {
    const textVal = (ov as { text?: string })?.text ?? '';
    preview = textVal.length > 40 ? textVal.slice(0, 40) + '…' : textVal;
  } else if (kind === 'json') {
    const jsonVal = (ov as { json?: any })?.json ?? ov;
    const s = typeof jsonVal === 'string' ? jsonVal : JSON.stringify(jsonVal);
    preview = s.length > 40 ? s.slice(0, 40) + '…' : s;
  } else if (kind === 'file') {
    const fileVal = ov as { dataUrl?: string; url?: string; file_id?: string; mime_type?: string };
    if (fileVal.dataUrl) preview = lang === 'zh' ? '[内联文件]' : '[Inline file]';
    else if (fileVal.file_id && (fileVal.mime_type === 'text/plain' || outputDataType === DataType.TEXT)) preview = lang === 'zh' ? '[已存文本]' : '[Text file]';
    else if (fileVal.file_id) preview = lang === 'zh' ? '[已存文件]' : '[Stored file]';
    else if (fileVal.url) preview = lang === 'zh' ? '[URL]' : '[URL]';
    else preview = lang === 'zh' ? '[文件]' : '[File]';
  } else if (kind === 'task' || kind === 'lightx2v_result') {
    const val = ov as { task_id?: string; output_name?: string; is_cloud?: boolean; workflow_id?: string; node_id?: string; port_id?: string };
    preview = `${val.task_id || ''} ${val.output_name || ''}`.trim() || (lang === 'zh' ? '[视频结果]' : '[Video]');
    isLightX2VResult = true;
    if (val.task_id) {
      lightX2VRef = {
        kind: 'task',
        workflow_id: val.workflow_id,
        node_id: val.node_id,
        port_id: val.port_id,
        task_id: val.task_id,
        output_name: val.output_name || 'output',
        is_cloud: !!val.is_cloud,
      };
    }
  }
  const ovFileId = (ov as { file_id?: string })?.file_id;
  const ovRunId = (ov as { run_id?: string })?.run_id;
  const [resolvedUrl, setResolvedUrl] = React.useState<string | null>(null);
  const [textFilePreview, setTextFilePreview] = React.useState<string | null>(null);
  const isTextFileRef = kind === 'file' && (outputDataType === DataType.TEXT || (ov as { mime_type?: string }).mime_type === 'text/plain') && !!ovFileId;
  const resolveTextFileContentRef = React.useRef(resolveTextFileContent);
  resolveTextFileContentRef.current = resolveTextFileContent;
  React.useEffect(() => {
    if (!isTextFileRef || !resolveTextFileContentRef.current) return;
    let cancelled = false;
    resolveTextFileContentRef.current(entry).then((text) => {
      if (!cancelled && text != null) setTextFilePreview(text.length > 40 ? text.slice(0, 40) + '…' : text);
    }).catch(() => {
      if (!cancelled) setTextFilePreview(null);
    });
    return () => { cancelled = true; setTextFilePreview(null); };
  }, [isTextFileRef, entry.id, ovFileId]);
  const [audioUrl, setAudioUrl] = React.useState<string | null>(null);
  const audioElRef = React.useRef<HTMLAudioElement | null>(null);
  React.useEffect(() => {
    if (!lightX2VRef || !resolveLightX2VResultRef) return;
    let cancelled = false;
    resolveLightX2VResultRef(lightX2VRef).then((url) => {
      if (!cancelled) setResolvedUrl(url);
    }).catch(() => {
      if (!cancelled) setResolvedUrl(null);
    });
    return () => { cancelled = true; setResolvedUrl(null); };
  }, [lightX2VRef?.task_id, lightX2VRef?.output_name, lightX2VRef?.is_cloud, resolveLightX2VResultRef]);
  const isAudio = outputDataType === DataType.AUDIO && (kind === 'file' || kind === 'task' || kind === 'lightx2v_result') && !!resolveAudioUrl;
  const resolveAudioUrlRef = React.useRef(resolveAudioUrl);
  resolveAudioUrlRef.current = resolveAudioUrl;
  React.useEffect(() => {
    if (!isAudio || !resolveAudioUrlRef.current) return;
    let cancelled = false;
    resolveAudioUrlRef.current(entry).then((url) => {
      if (!cancelled && url) setAudioUrl(url);
    }).catch(() => {
      if (!cancelled) setAudioUrl(null);
    });
    return () => { cancelled = true; setAudioUrl(null); };
  }, [isAudio, entry.id, ovFileId, ovRunId]);
  const handlePlayAudio = React.useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (!audioUrl) return;
    if (audioElRef.current) {
      audioElRef.current.pause();
      audioElRef.current.currentTime = 0;
    }
    const audio = new Audio(audioUrl);
    audioElRef.current = audio;
    audio.play().catch(() => {});
    audio.onended = () => { audioElRef.current = null; };
  }, [audioUrl]);
  const isImage = isLightX2VResult && (ov as { output_name?: string })?.output_name === 'output_image';
  const output_name = (ov as { output_name?: string })?.output_name || '';
  const isLightX2VAudio = isLightX2VResult && output_name === 'output_audio';
  const hasThumbnail = isLightX2VResult && !isLightX2VAudio && resolveLightX2VResultRef && lightX2VRef;
  const isVideo = outputDataType === DataType.VIDEO && (kind === 'file' || kind === 'task' || kind === 'lightx2v_result') && !!resolveVideoUrl;
  const [videoUrl, setVideoUrl] = React.useState<string | null>(null);
  const resolveVideoUrlRef = React.useRef(resolveVideoUrl);
  resolveVideoUrlRef.current = resolveVideoUrl;
  React.useEffect(() => {
    if (!isVideo || !resolveVideoUrlRef.current || (kind === 'task' || kind === 'lightx2v_result')) return;
    let cancelled = false;
    resolveVideoUrlRef.current(entry).then((url) => {
      if (!cancelled && url) setVideoUrl(url);
    }).catch(() => {
      if (!cancelled) setVideoUrl(null);
    });
    return () => { cancelled = true; setVideoUrl(null); };
  }, [isVideo, entry.id, kind, ovFileId, ovRunId]);
  const hasVideoFileThumbnail = isVideo && kind === 'file' && videoUrl;
  const isImageFile = outputDataType === DataType.IMAGE && kind === 'file' && !!resolveImageUrl;
  const [imageUrl, setImageUrl] = React.useState<string | null>(null);
  const resolveImageUrlRef = React.useRef(resolveImageUrl);
  resolveImageUrlRef.current = resolveImageUrl;
  React.useEffect(() => {
    if (!isImageFile || !resolveImageUrlRef.current) return;
    let cancelled = false;
    resolveImageUrlRef.current(entry).then((url) => {
      if (!cancelled && url) setImageUrl(url);
    }).catch(() => {
      if (!cancelled) setImageUrl(null);
    });
    return () => { cancelled = true; setImageUrl(null); };
  }, [isImageFile, entry.id, ovFileId, ovRunId]);
  const hasImageFileThumbnail = isImageFile && imageUrl;
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
      className="w-full px-3 py-2 text-left text-[10px] text-slate-300 hover:bg-[#90dce1]/20 hover:text-white transition-colors flex items-center gap-2 border-b border-slate-700/50 last:border-b-0"
    >
      {(hasThumbnail || hasVideoFileThumbnail || hasImageFileThumbnail) ? (
        <span className="flex-shrink-0 w-1/2 min-w-[72px] aspect-square rounded overflow-hidden bg-slate-700 flex items-center justify-center">
          {hasImageFileThumbnail ? (
            <img src={imageUrl!} alt="" className="w-full h-full object-contain" />
          ) : hasVideoFileThumbnail ? (
            <video
              src={videoUrl!}
              className="w-full h-full object-cover"
              muted
              preload="metadata"
              onMouseOver={(e) => {
                const v = e.currentTarget;
                if (v.readyState < 2) v.load();
                v.play().catch(() => {});
              }}
              onMouseOut={(e) => {
                e.currentTarget.pause();
                e.currentTarget.currentTime = 0;
              }}
            />
          ) : resolvedUrl ? (
            isImage ? (
              <img src={resolvedUrl} alt="" className="w-full h-full object-contain" />
            ) : (
              <video
                src={resolvedUrl}
                className="w-full h-full object-contain"
                muted
                preload="metadata"
                onMouseOver={(e) => {
                  const v = e.currentTarget;
                  if (v.readyState < 2) v.load();
                  v.play().catch(() => {});
                }}
                onMouseOut={(e) => {
                  e.currentTarget.pause();
                  e.currentTarget.currentTime = 0;
                }}
              />
            )
          ) : (
            <span className="text-[8px] text-slate-500">…</span>
          )}
        </span>
      ) : isAudio ? (
        <span className="flex-shrink-0 w-8 h-8 rounded-lg bg-slate-700/80 flex items-center justify-center text-slate-400" title={kind === 'file' ? (lang === 'zh' ? '音频' : 'Audio') : ''}>
          <Volume2 size={14} />
        </span>
      ) : isImageFile ? (
        <span className="flex-shrink-0 w-8 h-8 rounded-lg bg-slate-700/80 flex items-center justify-center text-slate-400 relative" title={lang === 'zh' ? '图片' : 'Image'}>
          <ImageIcon size={14} />
          {!imageUrl && <RefreshCw size={10} className="animate-spin absolute" />}
        </span>
      ) : isTextFileRef ? (
        <span className="flex-shrink-0 w-8 h-8 rounded-lg bg-slate-700/80 flex items-center justify-center text-slate-400 relative" title={lang === 'zh' ? '文本' : 'Text'}>
          {textFilePreview != null ? <FileText size={14} /> : <RefreshCw size={12} className="animate-spin" />}
        </span>
      ) : null}
      <span className="flex-1 min-w-0 flex flex-col gap-0.5">
        <span className="truncate">
          {isAudio && ovFileId
            ? (lang === 'zh' ? '已存音频' : 'Stored audio')
            : isImageFile && ovFileId
              ? (imageUrl ? (lang === 'zh' ? '已存图片' : 'Stored image') : (lang === 'zh' ? '加载中…' : 'Loading…'))
              : isTextFileRef
                ? (textFilePreview != null ? textFilePreview : (lang === 'zh' ? '加载中…' : 'Loading…'))
                : (preview || (lang === 'zh' ? '—' : '—'))}
        </span>
        {entry.timestamp ? (
          <span className="text-[9px] text-slate-500">{new Date(entry.timestamp).toLocaleTimeString()}</span>
        ) : null}
      </span>
      {isAudio && (
        <span className="flex-shrink-0">
          {audioUrl ? (
            <button
              type="button"
              onClick={handlePlayAudio}
              className="p-1.5 rounded-lg bg-[#90dce1]/20 hover:bg-[#90dce1]/40 text-[#90dce1] transition-colors"
              title={lang === 'zh' ? '播放' : 'Play'}
            >
              <PlayIcon size={14} />
            </button>
          ) : (
            <span className="inline-block p-1.5 text-slate-500" title={lang === 'zh' ? '加载中…' : 'Loading…'}>
              <RefreshCw size={12} className="animate-spin" />
            </span>
          )}
        </span>
      )}
    </button>
  );
};

interface NodeProps {
  node: WorkflowNode;
  workflow: WorkflowState;
  isSelected: boolean;
  sourceOutputs: Record<string, any>;
  nodeHeight: number;
  onSelect: (nodeId: string) => void;
  onDragStart: (nodeId: string, offsetX: number, offsetY: number) => void;
  onDrag: (nodeId: string, x: number, y: number) => void;
  onDragEnd: () => void;
  getNodeOutputs: (node: WorkflowNode) => Port[];
  canvasRef: React.RefObject<HTMLDivElement>;
  view: ViewState;
  lang: Language;
  // Additional props for node interactions
  showReplaceMenu?: string | null;
  showOutputQuickAdd?: { nodeId: string; portId: string } | null;
  showModelSelect?: string | null;
  showVoiceSelect?: string | null;
  lightX2VVoiceList?: { voices?: any[]; emotions?: string[]; languages?: any[] } | null;
  cloneVoiceList?: any[];
  onUpdateNodeData: (nodeId: string, key: string, value: any) => void;
  onDeleteNode: (nodeId: string) => void;
  onReplaceNode: (nodeId: string, newToolId: string) => void;
  onRunWorkflow: (nodeId?: string, runThisOnly?: boolean) => void;
  onCancelNodeRun?: (nodeId: string) => void;
  pendingRunNodeIds?: string[];
  onSetReplaceMenu: (nodeId: string | null) => void;
  onSetOutputQuickAdd: (value: { nodeId: string; portId: string } | null) => void;
  onSetModelSelect: (nodeId: string | null) => void;
  onSetVoiceSelect: (nodeId: string | null) => void;
  onSetExpandedOutput: (value: { nodeId: string; fieldId?: string } | null) => void;
  onSetShowAudioEditor: (nodeId: string | null) => void;
  onSetShowVideoEditor: (nodeId: string | null) => void;
  onSetConnecting: (value: {
    nodeId: string;
    portId: string;
    type: DataType;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null) => void;
  onAddConnection: (connection: {
    id: string;
    source_node_id: string;
    source_port_id: string;
    target_node_id: string;
    target_port_id: string;
  }) => void;
  onClearSelectedRunId: () => void;
  getReplaceableTools: (nodeId: string) => ToolDefinition[];
  getCompatibleToolsForOutput: (outputType: DataType) => ToolDefinition[];
  quickAddInput: (node: WorkflowNode, port: Port) => void;
  quickAddOutput: (node: WorkflowNode, port: Port, toolId: string) => void;
  onAddNodeToChat?: (nodeId: string, name: string) => void;
  resolveLightX2VResultRef?: (ref: LightX2VResultRef, context?: { workflow_id?: string; node_id?: string; port_id?: string }) => Promise<string>;
  connecting: {
    nodeId: string;
    portId: string;
    type: DataType;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null;
  onNodeHeightChange?: (nodeId: string, height: number) => void;
  getNodeOutputUrl?: (nodeId: string, portId: string, fileId?: string, runId?: string) => Promise<string | null>;
  /** 可选：保存后刷新 workflow（用于历史条目选中后同步后端） */
  refreshWorkflowFromBackend?: (workflowId: string) => Promise<void>;
}

export const Node: React.FC<NodeProps> = ({
  node,
  workflow,
  isSelected,
  sourceOutputs,
  nodeHeight,
  onSelect,
  onDragStart,
  onDrag,
  onDragEnd,
  getNodeOutputs,
  canvasRef,
  view,
  lang,
  showReplaceMenu,
  showOutputQuickAdd,
  showModelSelect,
  showVoiceSelect,
  lightX2VVoiceList,
  cloneVoiceList = [],
  onUpdateNodeData,
  onDeleteNode,
  onReplaceNode,
  onRunWorkflow,
  onCancelNodeRun = (_nodeId?: string) => {},
  pendingRunNodeIds = [],
  resolveLightX2VResultRef,
  onSetReplaceMenu,
  onSetOutputQuickAdd,
  onSetModelSelect,
  onSetVoiceSelect,
  onSetExpandedOutput,
  onSetShowAudioEditor,
  onSetShowVideoEditor,
  onSetConnecting,
  onAddConnection,
  onClearSelectedRunId = () => {},
  getReplaceableTools,
  getCompatibleToolsForOutput,
  quickAddInput,
  quickAddOutput,
  onAddNodeToChat,
  connecting,
  onNodeHeightChange,
  getNodeOutputUrl,
  screenToWorldCoords,
  refreshWorkflowFromBackend,
}) => {
  const { t } = useTranslation(lang);
  const nodeRef = useRef<HTMLDivElement>(null);
  const lastHeightRef = useRef<number>(0);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  const tool = TOOLS.find((t) => t.id === node.tool_id);
  if (!tool) return null;

  const outputs = getNodeOutputs(node);

  // Measure and report node height when it changes
  React.useEffect(() => {
    if (nodeRef.current && onNodeHeightChange) {
      const currentHeight = nodeRef.current.offsetHeight;
      if (currentHeight !== lastHeightRef.current) {
        lastHeightRef.current = currentHeight;
        onNodeHeightChange(node.id, currentHeight);
      }
    }
  }, [node.id, onNodeHeightChange, node.data, outputs.length, node.output_value, node.status]);
  // sourceOutputs 或 node.output_value 可能为 port-keyed；Input 按 port 取值，输出节点按首端口取值以便预览 task ref 等
  const primaryPortId = tool.category === 'Input' ? INPUT_PORT_IDS[node.tool_id] ?? outputs[0]?.id : undefined;
  const firstOutputPortId = outputs[0]?.id;
  const nodeResultRaw = sourceOutputs[node.id] != null
    ? (primaryPortId && typeof sourceOutputs[node.id] === 'object' && !Array.isArray(sourceOutputs[node.id]) && primaryPortId in sourceOutputs[node.id]
        ? sourceOutputs[node.id][primaryPortId]
        : firstOutputPortId && typeof sourceOutputs[node.id] === 'object' && !Array.isArray(sourceOutputs[node.id]) && firstOutputPortId in sourceOutputs[node.id]
        ? sourceOutputs[node.id][firstOutputPortId]
        : sourceOutputs[node.id])
    : (firstOutputPortId ? getOutputValueByPort(node, firstOutputPortId) : null) ?? (primaryPortId ? getOutputValueByPort(node, primaryPortId) : null) ?? node.output_value ?? (tool.category === 'Input' ? node.data.value : null);
  // Extract actual value from reference/optimized objects (for history or 纯前端 run.outputs)
  const r = (x: string) => (nodeResultRaw as any).kind === x || (nodeResultRaw as any).type === x;
  const nodeResult =
    nodeResultRaw && typeof nodeResultRaw === 'object' && !Array.isArray(nodeResultRaw)
      ? r('url')
        ? (typeof (nodeResultRaw as any).url === 'string' ? (nodeResultRaw as any).url : typeof (nodeResultRaw as any).data === 'string' ? (nodeResultRaw as any).data : undefined)
        : r('text')
        ? (nodeResultRaw as any).data
        : r('file')
        ? (nodeResultRaw as any).data ?? ((nodeResultRaw as any).file_id ? nodeResultRaw : undefined)
        : r('data_url') && typeof (nodeResultRaw as any)._full_data === 'string'
        ? (nodeResultRaw as any)._full_data
        : r('json') && (nodeResultRaw as any).data != null
        ? (nodeResultRaw as any).data
        : nodeResultRaw
      : nodeResultRaw;
  const firstOutputType = outputs[0]?.type || DataType.TEXT;
  const previewValue = Array.isArray(nodeResult) ? (nodeResult.length > 0 ? nodeResult[0] : null) : (nodeResult ?? null);
  const isPreviewRef = previewValue != null && isLightX2VResultRef(previewValue);
  const previewRefObj = isPreviewRef && typeof previewValue === 'object' ? (previewValue as LightX2VResultRef) : null;
  const [resolvedPreviewUrl, setResolvedPreviewUrl] = React.useState<string | null>(null);
  const resolveTaskRefContext = React.useMemo(() => (workflow?.id && node.id && firstOutputPortId
    ? { workflow_id: workflow.id, node_id: node.id, port_id: firstOutputPortId }
    : undefined), [workflow?.id, node.id, firstOutputPortId]);
  const resolveTaskRefForHistory = React.useCallback((ref: LightX2VResultRef) => {
    if (!resolveLightX2VResultRef) return Promise.reject(new Error('no resolver'));
    return resolveLightX2VResultRef(ref, resolveTaskRefContext);
  }, [resolveLightX2VResultRef, resolveTaskRefContext]);
  React.useEffect(() => {
    if (!previewRefObj || !resolveLightX2VResultRef) return;
    let cancelled = false;
    resolveLightX2VResultRef(previewRefObj, resolveTaskRefContext).then((url) => {
      if (!cancelled) setResolvedPreviewUrl(url);
    }).catch(() => {
      if (!cancelled) setResolvedPreviewUrl(null);
    });
    return () => { cancelled = true; setResolvedPreviewUrl(null); };
  }, [previewRefObj?.task_id, previewRefObj?.output_name, previewRefObj?.is_cloud, resolveLightX2VResultRef, resolveTaskRefContext]);
  // 统一通过 resolveLightX2VResultRef 获取 URL（本地/云端均走接口）
  const refPreviewUrl = isPreviewRef && previewRefObj ? resolvedPreviewUrl : null;
  // 文本端口为 file ref（.txt）时拉取内容用于预览
  const isTextFileRef = firstOutputType === DataType.TEXT && previewValue != null && typeof previewValue === 'object' && (previewValue as any).kind === 'file' && (previewValue as any).file_id;
  const [resolvedTextFromFile, setResolvedTextFromFile] = React.useState<string | null>(null);
  React.useEffect(() => {
    if (!isTextFileRef || !workflow?.id) return;
    const ref = previewValue as { file_id: string; run_id?: string };
    const portId = firstOutputPortId ?? 'out-text';
    let cancelled = false;
    getWorkflowFileText(workflow.id, ref.file_id, node.id, portId, ref.run_id).then((text) => {
      if (!cancelled) setResolvedTextFromFile(text ?? null);
    });
    return () => { cancelled = true; setResolvedTextFromFile(null); };
  }, [isTextFileRef, workflow?.id, node.id, firstOutputPortId, (previewValue as any)?.file_id, (previewValue as any)?.run_id]);
  const [resolvedLocalUrls, setResolvedLocalUrls] = React.useState<Record<string, string>>({});
  React.useEffect(() => {
    const values = Array.isArray(node.data.value) ? node.data.value : (node.data.value ? [node.data.value] : []);
    const localRefs = values.filter((v: unknown) => typeof v === 'string' && (v as string).startsWith('local://'));
    if (localRefs.length === 0) return;
    let cancelled = false;
    Promise.all(
      localRefs.map(async (ref: string) => {
        const dataUrl = await getLocalFileDataUrl(ref);
        return [ref, dataUrl] as const;
      })
    ).then((pairs) => {
      if (cancelled) return;
      setResolvedLocalUrls((prev) => {
        const next = { ...prev };
        pairs.forEach(([ref, url]) => {
          if (url) next[ref] = url;
        });
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [node.id, node.data.value]);

  // 载入工作流后：output_value["out-image"] 或 data.value 可能为 file_id 引用，通过 file_id 拉取并显示预览
  // 优先用 output_value（保存/加载后多为 port-keyed），避免上传为 file 后 data.value 未同步导致输入节点变空
  const imagePortValue = node.tool_id === 'image-input' ? getOutputValueByPort(node, 'out-image') : undefined;
  const fromData = Array.isArray(node.data.value) ? node.data.value : (node.data.value != null ? [node.data.value] : []);
  const fromOutputPort =
    node.output_value && typeof node.output_value === 'object' && !Array.isArray(node.output_value) && 'out-image' in node.output_value
      ? (Array.isArray((node.output_value as Record<string, unknown>)['out-image'])
          ? (node.output_value as Record<string, unknown>)['out-image'] as unknown[]
          : [(node.output_value as Record<string, unknown>)['out-image']].filter(Boolean))
      : [];
  const effectiveImageValues =
    (Array.isArray(imagePortValue) && imagePortValue.length > 0 ? imagePortValue : null) ??
    (fromData.length > 0 ? fromData : null) ??
    (fromOutputPort.length > 0 ? fromOutputPort : []);
  const [resolvedFileRefUrls, setResolvedFileRefUrls] = React.useState<Record<string, string>>({});
  React.useEffect(() => {
    if (node.tool_id !== 'image-input' || !workflow?.id || !getNodeOutputUrl) return;
    const values = effectiveImageValues;
    const fileRefs = values.filter((v: unknown) => v && typeof v === 'object' && (v as { file_id?: string }).file_id) as { file_id: string; mime_type?: string; ext?: string; run_id?: string }[];
    if (fileRefs.length === 0) return;
    let cancelled = false;
    Promise.all(fileRefs.map((ref) => getNodeOutputUrl(node.id, 'out-image', ref.file_id, ref.run_id))).then((urls) => {
      if (cancelled) return;
      setResolvedFileRefUrls((prev) => {
        const next = { ...prev };
        fileRefs.forEach((ref, i) => {
          if (urls[i]) next[ref.file_id] = urls[i];
        });
        return next;
      });
    });
    return () => { cancelled = true; };
  }, [node.id, node.tool_id, workflow?.id, node.output_value, node.data?.value, getNodeOutputUrl, effectiveImageValues]);

  // 载入工作流后：audio-input / video-input 的 output_value 可能为 file 引用，需拉取为 data URL 再展示
  const mediaValueForRef = node.tool_id === 'audio-input' || node.tool_id === 'video-input'
    ? (Array.isArray(node.data.value) ? node.data.value[0] : node.data.value)
    : undefined;
  const mediaFileRef = mediaValueForRef && typeof mediaValueForRef === 'object' && (mediaValueForRef as { file_id?: string }).file_id
    ? (mediaValueForRef as { file_id: string; mime_type?: string; ext?: string })
    : null;
  React.useEffect(() => {
    if (!mediaFileRef?.file_id || !workflow?.id) return;
    const mediaPortId = node.tool_id === 'audio-input' ? 'out-audio' : 'out-video';
    let cancelled = false;
    getNodeOutputUrl(node.id, mediaPortId, mediaFileRef.file_id, (mediaFileRef as any).run_id).then((url) => {
      if (!cancelled && url) setResolvedFileRefUrls((prev) => ({ ...prev, [mediaFileRef.file_id]: url }));
    });
    return () => { cancelled = true; };
  }, [node.id, workflow?.id, mediaFileRef?.file_id]);

  const resolveMediaSrc = (value?: string | { file_id?: string; file_url?: string }) => {
    if (value == null) return '';
    if (typeof value !== 'string') {
      if (value?.file_id && resolvedFileRefUrls[value.file_id]) return resolvedFileRefUrls[value.file_id];
      // 后端返回的 file_url 可直接用于预览（与生图节点一致），避免等 getNodeOutputUrl 才显示
      const url = (value as { file_url?: string }).file_url;
      if (typeof url === 'string' && (url.startsWith('/') || url.startsWith('http'))) return getAssetPath(url);
      return value?.file_id ? resolvedFileRefUrls[value.file_id] ?? '' : '';
    }
    if (value.startsWith('local://')) return resolvedLocalUrls[value] || '';
    if (value.startsWith('data:') || value.startsWith('http') || value.startsWith('/api/')) return value;
    return getAssetPath(value);
  };

  // output_value["out-image"] 或 data.value 为数据源；image_edits 仅存 crop_box，不存 base64；载入时用 crop_box + 值显示
  const rawImageValues = effectiveImageValues;
  const imageEdits = Array.isArray(node.data.image_edits) ? node.data.image_edits : [];
  const imageEntries = rawImageValues.map((value: string | { type?: string; file_id?: string; file_url?: string; mime_type?: string; ext?: string; run_id?: string }, index: number) => {
    let display = resolveMediaSrc(value);
    // 无 file_url 且未解析到 URL 时，用 file_id + run_id 直接拼 /assets/workflow/file 展示（与生图节点一致）
    if (!display && typeof value === 'object' && value?.file_id && workflow?.id) {
      display = getWorkflowFileUrl(
        workflow.id,
        value.file_id,
        value.mime_type,
        value.ext,
        node.id,
        'out-image',
        value.run_id
      );
    }
    const existing = imageEdits[index] as { crop_box?: { x: number; y: number; w: number; h: number }; cropped?: string } | undefined;
    const crop_box = existing?.crop_box ?? { x: 10, y: 10, w: 80, h: 80 };
    // 不使用存储的 base64；仅用 URL 或载入时用 display（output_value/data.value）作为显示
    const cropped = (existing?.cropped && existing.cropped !== '' && !existing.cropped.startsWith('data:')) ? existing.cropped : display;
    return {
      source: value,
      original: display,
      cropped,
      crop_box
    };
  });

  // 与主应用一致：排队显示等待个数，运行中才计时，成功显示运行时间；进度条用整体进度 (已完成数/总数)，与主应用 getOverallProgress 一致
  const runState = node.run_state as NodeRunState | undefined;
  const taskStatus = runState?.status;
  const subtasks = runState?.subtasks ?? [];
  const firstSubtask = subtasks[0];
  const queueOrder = firstSubtask?.estimated_pending_order;
  const isTaskPending = taskStatus === 'PENDING' || (firstSubtask?.status === 'PENDING');
  const isTaskRunning = taskStatus === 'RUNNING' || (firstSubtask?.status === 'RUNNING');
  const getOverallProgress = (): number => {
    if (!subtasks.length) return 0;
    const completedCount = subtasks.filter((s: { status: string }) => s.status === 'SUCCEED').length;
    return Math.round((completedCount / subtasks.length) * 100);
  };
  const progressPercent = getOverallProgress();

  const durationText =
    node.error === 'Cancelled'
      ? (lang === 'zh' ? '已取消' : 'Cancelled')
      : node.status === NodeStatus.PENDING || (node.status === NodeStatus.RUNNING && isTaskPending)
        ? (lang === 'zh' ? '排队中' : 'Queued') + (queueOrder != null && queueOrder >= 0 ? ` (${lang === 'zh' ? '等待' : 'Wait'}: ${queueOrder})` : '')
        : node.status === NodeStatus.RUNNING
          ? ((performance.now() - (node.start_time || performance.now())) / 1000).toFixed(1) + 's'
          : node.status === NodeStatus.SUCCESS && node.execution_time != null
            ? formatRunTime(node.execution_time)
            : node.execution_time != null
              ? formatTime(node.execution_time)
              : '';

  const isInputNode = tool.category === 'Input';
  const hasOutputValue = !isInputNode && node.output_value != null && (
    typeof node.output_value !== 'object' || Array.isArray(node.output_value) || Object.keys(node.output_value).length > 0
  );
  const hasData =
    (isInputNode && node.data.value && (Array.isArray(node.data.value) ? node.data.value.length > 0 : true)) ||
    (!isInputNode && (sourceOutputs[node.id] || hasOutputValue));
  const shouldShowPreview = hasData && !isInputNode && node.tool_id !== 'text-input';
  const isMultiPortOutput = outputs.length > 1;

  // 多端口小预览：每个端口的文本 file ref 拉取后直接显示
  const [resolvedPortText, setResolvedPortText] = React.useState<Record<string, string | null>>({});
  React.useEffect(() => {
    if (!isMultiPortOutput || !workflow?.id || !outputs.length) {
      setResolvedPortText({});
      return;
    }
    const refs: { portId: string; fileId: string; runId?: string }[] = [];
    outputs.forEach((p) => {
      const portVal = typeof sourceOutputs[node.id] === 'object' && sourceOutputs[node.id] != null && p.id in sourceOutputs[node.id]
        ? sourceOutputs[node.id][p.id]
        : getOutputValueByPort(node, p.id);
      const isFileRef = portVal && typeof portVal === 'object' && (portVal as any).file_id && (portVal as any).mime_type === 'text/plain';
      if (isFileRef) refs.push({ portId: p.id, fileId: (portVal as any).file_id, runId: (portVal as any).run_id });
    });
    if (refs.length === 0) {
      setResolvedPortText({});
      return;
    }
    let cancelled = false;
    Promise.all(
      refs.map(async ({ portId, fileId, runId }) => {
        const text = await getWorkflowFileText(workflow.id!, fileId, node.id, portId, runId);
        return [portId, text ?? null] as const;
      })
    ).then((pairs) => {
      if (cancelled) return;
      setResolvedPortText((prev) => {
        const next = { ...prev };
        pairs.forEach(([portId, text]) => { next[portId] = text; });
        return next;
      });
    });
    return () => { cancelled = true; setResolvedPortText({}); };
  }, [isMultiPortOutput, workflow?.id, node.id, outputs.length, node.output_value, sourceOutputs[node.id]]);

  const handleNodeClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (connecting) return; // 拖拽连线时不选中节点，避免误选
    onClearSelectedRunId?.();
    onSelect(node.id);
  };

  const handleNodeMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('input, textarea, button, label')) return;
    e.preventDefault(); // 避免拖拽时触发其他节点文字选中（与连接线端口拖拽一致）
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const world = screenToWorldCoords
      ? screenToWorldCoords(x, y)
      : screenToWorld(x, y, view, rect);
    onDragStart(node.id, world.x - node.x, world.y - node.y);
  };

  // Note: onDrag is handled by parent component (Canvas) via mouse move events
  // The parent will call onDrag when draggingNode is set

  const handlePortMouseDown = (port: Port, direction: 'in' | 'out') => (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault(); // 避免从端口拖拽时触发文字选中

    if (direction === 'out') {
      const nodeWidth = isInputNode ? 320 : 224;
      // For output ports, calculate position from node bottom (same as Connection component)
      const outputPortIndex = outputs.findIndex((p) => p.id === port.id);
      const nodeBottomY = node.y + nodeHeight;
      const startX = node.x + 224; // Output port center X
      const startY = nodeBottomY - ((outputs.length - 1 - outputPortIndex) * 30) - 24; // Output port center Y
      onSetConnecting({
        nodeId: node.id,
        portId: port.id,
        type: port.type,
        direction,
        startX,
        startY
      });
    } else {
      // For input ports, calculate position from node top (same as Connection component)
      const tool = TOOLS.find((t) => t.id === node.tool_id);
      const inputPortIndex = tool?.inputs.findIndex((p) => p.id === port.id) ?? -1;
      const startX = node.x; // Input port center X
      const startY = node.y + 71 + (inputPortIndex * 30); // Input port center Y
      onSetConnecting({
        nodeId: node.id,
        portId: port.id,
        type: port.type,
        direction,
        startX,
        startY
      });
    }
  };

  const handlePortMouseUp = (port: Port, direction: 'in' | 'out') => () => {
    if (connecting && connecting.direction !== direction && connecting.type === port.type) {
      onClearSelectedRunId?.();
      if (direction === 'in') {
        onAddConnection({
          id: `conn-${Date.now()}`,
          source_node_id: connecting.nodeId,
          source_port_id: connecting.portId,
          target_node_id: node.id,
          target_port_id: port.id
        });
      } else {
        onAddConnection({
          id: `conn-${Date.now()}`,
          source_node_id: node.id,
          source_port_id: port.id,
          target_node_id: connecting.nodeId,
          target_port_id: connecting.portId
        });
      }
      onSetConnecting(null);
    }
  };

  const [isUploading, setIsUploading] = React.useState(false);
  const [showHistoryDropdown, setShowHistoryDropdown] = React.useState(false);
  const historyDropdownRef = React.useRef<HTMLDivElement>(null);

  const nodeHistoryEntries = React.useMemo(() => {
    const raw = workflow.nodeOutputHistory?.[node.id];
    const list = Array.isArray(raw) ? raw : (raw && typeof raw === 'object' ? Object.values(raw) : []);
    return normalizeHistoryEntries(list as any[]);
  }, [workflow.nodeOutputHistory, node.id]);

  // 历史与 nodes 一致：port_keyed。输入节点用 INPUT_PORT_IDS 取对应 port（out-image/out-audio/out-video/out-text）；非输入节点取整条 port_keyed 或单端口值
  const handleHistoryEntrySelect = React.useCallback(
    (entry: NodeHistoryEntry) => {
      const portIdForInput = isInputNode ? INPUT_PORT_IDS[node.tool_id] : undefined;
      const displayValue = historyEntryToDisplayValue(entry, portIdForInput);
      if (displayValue != null) {
        if (isInputNode) {
          const portId = INPUT_PORT_IDS[node.tool_id];
          if (portId) {
            const nextOutput = setOutputValueByPort(node.output_value, node.tool_id, portId, displayValue);
            onUpdateNodeData(node.id, 'output_value', nextOutput);
            if (node.tool_id === 'text-input') {
              if (typeof displayValue === 'string' && !displayValue.startsWith('data:')) {
                onUpdateNodeData(node.id, 'value', displayValue);
              } else if (typeof displayValue === 'object' && (displayValue as { file_id?: string }).file_id) {
                onUpdateNodeData(node.id, 'value', displayValue);
              }
            } else {
              const valueForData = node.tool_id === 'image-input' && !Array.isArray(displayValue) ? [displayValue] : displayValue;
              onUpdateNodeData(node.id, 'value', valueForData);
            }
          } else {
            onUpdateNodeData(node.id, 'output_value', displayValue);
          }
        } else {
          onUpdateNodeData(node.id, 'output_value', displayValue);
        }
      }
      if (entry.params && typeof entry.params === 'object' && Object.keys(entry.params).length > 0) {
        onUpdateNodeData(node.id, '__mergeData', entry.params);
      }
      setShowHistoryDropdown(false);

      // 非 standalone 时按 task/file 格式调用 port/save，将选中的历史结果写回后端（使用 history entry 的 id 作为 run_id）
      if (!isStandalone() && workflow?.id) {
        const portKeyed = getEntryPortKeyedValue(entry);
        saveNodeOutputsFromHistoryEntry(workflow.id, node.id, portKeyed, entry.id)
          .then(() => refreshWorkflowFromBackend?.(workflow.id))
          .catch((e) => console.warn('[Node] Save history selection to backend failed:', e));
      }
    },
    [node.id, node.tool_id, node.output_value, isInputNode, onUpdateNodeData, workflow?.id, refreshWorkflowFromBackend]
  );

  const resolveAudioUrlForEntry = React.useCallback(async (entry: NodeHistoryEntry, portId?: string): Promise<string | null> => {
    const portKeyed = getEntryPortKeyedValue(entry);
    const keys = Object.keys(portKeyed);
    const portIdToUse = (keys.length === 1 ? keys[0] : undefined) ?? portId ?? (entry as { port_id?: string }).port_id;
    const ov = (portIdToUse ? portKeyed[portIdToUse] : null) ?? entry.output_value ?? (entry as any).value;
    const kind = ov?.kind ?? (entry as any).kind;
    const pid = portIdToUse ?? (entry as { port_id?: string }).port_id ?? 'out-audio';
    if (workflow?.id && (kind === 'task' || kind === 'file' || kind === 'lightx2v_result' || (ov as any)?.task_id || (ov as any)?.file_id)) {
      const url = await getNodeOutputDisplayUrl(workflow.id, node.id, pid, ov as any);
      if (url) return getAssetPath(url) ?? url;
    }
    if (kind === 'task' || kind === 'lightx2v_result') {
      const val = (ov || entry) as { task_id?: string; output_name?: string; is_cloud?: boolean; workflow_id?: string; node_id?: string; port_id?: string };
      if (val.output_name !== 'output_audio' && val.output_name !== 'output') return null;
      if (!resolveLightX2VResultRef || !val.task_id) return null;
      const ref: LightX2VResultRef = {
        kind: 'task',
        workflow_id: val.workflow_id ?? workflow?.id,
        node_id: val.node_id ?? node.id,
        port_id: val.port_id ?? portIdToUse,
        task_id: val.task_id,
        output_name: val.output_name || 'output',
        is_cloud: !!val.is_cloud
      };
      const ctx = workflow?.id && node.id && portIdToUse ? { workflow_id: workflow.id, node_id: node.id, port_id: portIdToUse } : undefined;
      return resolveLightX2VResultRef(ref, ctx).catch(() => null);
    }
    if (kind === 'file') {
      const fileVal = ov as { dataUrl?: string; url?: string; file_id?: string; mime_type?: string; run_id?: string };
      if (fileVal?.dataUrl?.startsWith('data:audio/')) return fileVal.dataUrl;
      if (fileVal?.url) return resolveMediaSrc(fileVal.url) || getAssetPath(fileVal.url);
      const fid = fileVal?.file_id;
      if (fid && workflow?.id && getNodeOutputUrl) {
        const url = await getNodeOutputUrl(node.id, pid, fid, fileVal?.run_id);
        if (url) return getAssetPath(url) ?? url;
      }
      return null;
    }
    return null;
  }, [workflow?.id, node.id, getNodeOutputUrl, resolveLightX2VResultRef, resolveMediaSrc]);

  const resolveVideoUrlForEntry = React.useCallback(async (entry: NodeHistoryEntry, portId?: string): Promise<string | null> => {
    const portKeyed = getEntryPortKeyedValue(entry);
    const keys = Object.keys(portKeyed);
    const portIdToUse = (keys.length === 1 ? keys[0] : undefined) ?? portId ?? (entry as { port_id?: string }).port_id;
    const ov = (portIdToUse ? portKeyed[portIdToUse] : null) ?? entry.output_value ?? (entry as any).value;
    const kind = ov?.kind ?? (entry as any).kind;
    const pid = portIdToUse ?? (entry as { port_id?: string }).port_id ?? 'out-video';
    if (workflow?.id && (kind === 'task' || kind === 'file' || kind === 'lightx2v_result' || (ov as any)?.task_id || (ov as any)?.file_id)) {
      const url = await getNodeOutputDisplayUrl(workflow.id, node.id, pid, ov as any);
      if (url) return getAssetPath(url) ?? url;
    }
    if (kind === 'task' || kind === 'lightx2v_result') {
      const val = (ov || entry) as {task_id?: string; output_name?: string; is_cloud?: boolean; workflow_id?: string; node_id?: string; port_id?: string };
      const output_name = val.output_name || 'output';
      if (output_name !== 'output_video' && output_name !== 'output') return null;
      if (!resolveLightX2VResultRef || !val.task_id) return null;
      const ref: LightX2VResultRef = {
        kind: 'task',
        workflow_id: val.workflow_id ?? workflow?.id,
        node_id: val.node_id ?? node.id,
        port_id: val.port_id ?? portIdToUse,
        task_id: val.task_id,
        output_name: output_name,
        is_cloud: !!val.is_cloud
      };
      const ctx = workflow?.id && node.id && portIdToUse ? { workflow_id: workflow.id, node_id: node.id, port_id: portIdToUse } : undefined;
      return resolveLightX2VResultRef(ref, ctx).catch(() => null);
    }
    if (kind === 'file') {
      const fileVal = ov as { dataUrl?: string; url?: string; file_id?: string; mime_type?: string; ext?: string; run_id?: string };
      if (fileVal?.dataUrl?.startsWith('data:video/')) return fileVal.dataUrl;
      if (fileVal?.url) return resolveMediaSrc(fileVal.url) || getAssetPath(fileVal.url);
      const fid = fileVal?.file_id;
      if (fid && workflow?.id) {
        const url = await getNodeOutputUrl(node.id, pid, fid, fileVal?.run_id);
        return url ? (getAssetPath(url) ?? url) : null;
      }
      return null;
    }
    return null;
  }, [workflow?.id, node.id, getNodeOutputUrl, resolveLightX2VResultRef, resolveMediaSrc]);

  const resolveImageUrlForEntry = React.useCallback(async (entry: NodeHistoryEntry, portId?: string): Promise<string | null> => {
    const portKeyed = getEntryPortKeyedValue(entry);
    const keys = Object.keys(portKeyed);
    const portIdToUse = (keys.length === 1 ? keys[0] : undefined) ?? portId ?? (entry as { port_id?: string }).port_id;
    const ov = (portIdToUse ? portKeyed[portIdToUse] : null) ?? entry.output_value ?? (entry as any).value;
    const kind = ov?.kind ?? (entry as any).kind;
    const pid = portIdToUse ?? (entry as { port_id?: string }).port_id ?? 'out-image';
    if (workflow?.id && (kind === 'task' || kind === 'file' || kind === 'lightx2v_result' || (ov as any)?.task_id || (ov as any)?.file_id)) {
      const url = await getNodeOutputDisplayUrl(workflow.id, node.id, pid, ov as any);
      if (url) return getAssetPath(url) ?? url;
    }
    if (kind === 'task' || kind === 'lightx2v_result') {
      const val = (ov || entry) as {task_id?: string; output_name?: string; is_cloud?: boolean; workflow_id?: string; node_id?: string; port_id?: string };
      const output_name = val.output_name || 'output';
      if (output_name !== 'output_image' && output_name !== 'output') return null;
      if (!resolveLightX2VResultRef || !val.task_id) return null;
      const ref: LightX2VResultRef = {
        kind: 'task',
        workflow_id: val.workflow_id ?? workflow?.id,
        node_id: val.node_id ?? node.id,
        port_id: val.port_id ?? portIdToUse,
        task_id: val.task_id,
        output_name: output_name,
        is_cloud: !!val.is_cloud
      };
      const ctx = workflow?.id && node.id && portIdToUse ? { workflow_id: workflow.id, node_id: node.id, port_id: portIdToUse } : undefined;
      const url = await resolveLightX2VResultRef(ref, ctx).catch(() => null);
      return url;
    }
    if (kind === 'file') {
      const fileVal = ov as { dataUrl?: string; url?: string; file_id?: string; mime_type?: string; run_id?: string };
      if (fileVal?.dataUrl?.startsWith('data:image/')) return fileVal.dataUrl;
      if (fileVal?.url) return resolveMediaSrc(fileVal.url) || getAssetPath(fileVal.url);
      const fid = fileVal?.file_id;
      if (fid && workflow?.id) {
        const url = await getNodeOutputUrl(node.id, pid, fid, fileVal?.run_id);
        return url ? (getAssetPath(url) ?? url) : null;
      }
      return null;
    }
    return null;
  }, [workflow?.id, node.id, getNodeOutputUrl, resolveLightX2VResultRef, resolveMediaSrc]);

  // 当前输出为音频时（如 TTS），在节点卡片右侧显示小播放键，不进拓展框也可直接播放
  const currentAudioVal = firstOutputType === DataType.AUDIO && firstOutputPortId ? getOutputValueByPort(node, firstOutputPortId) : null;
  const portValFileId = (currentAudioVal && typeof currentAudioVal === 'object' && (currentAudioVal as any).file_id) ? (currentAudioVal as any).file_id : '';
  const portValTaskId = (currentAudioVal && typeof currentAudioVal === 'object' && (currentAudioVal as any).task_id) ? (currentAudioVal as any).task_id : '';
  const [currentAudioUrl, setCurrentAudioUrl] = React.useState<string | null>(null);
  const currentAudioElRef = React.useRef<HTMLAudioElement | null>(null);
  const nodeForAudioRef = React.useRef(node);
  nodeForAudioRef.current = node;
  const resolveAudioUrlForEntryRef = React.useRef(resolveAudioUrlForEntry);
  resolveAudioUrlForEntryRef.current = resolveAudioUrlForEntry;
  // 仅当「要解析的音频身份」变化时重新请求 URL，避免 node.output_value 引用或回调引用导致重复请求
  const currentAudioResolutionKey = React.useMemo(() => {
    if (firstOutputType !== DataType.AUDIO || !firstOutputPortId) return null;
    if (portValFileId) return `file:${node.id}:${firstOutputPortId}:${portValFileId}`;
    if (portValTaskId) return `task:${portValTaskId}`;
    return `${node.id}:${firstOutputPortId}`;
  }, [node.id, firstOutputPortId, firstOutputType, portValFileId, portValTaskId]);
  React.useEffect(() => {
    const n = nodeForAudioRef.current;
    const val = firstOutputType === DataType.AUDIO && firstOutputPortId ? getOutputValueByPort(n, firstOutputPortId) : null;
    if (!val || !firstOutputPortId || !currentAudioResolutionKey) {
      setCurrentAudioUrl(null);
      return;
    }
    let cancelled = false;
    const entry: NodeHistoryEntry = { id: 'current', timestamp: 0, output_value: typeof val === 'object' && val !== null ? val : {} };
    resolveAudioUrlForEntryRef.current(entry, firstOutputPortId).then((url) => {
      if (!cancelled && url) setCurrentAudioUrl(url);
      else if (!cancelled) setCurrentAudioUrl(null);
    });
    return () => { cancelled = true; setCurrentAudioUrl(null); };
  }, [currentAudioResolutionKey]);
  const handlePlayCurrentAudio = React.useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (!currentAudioUrl) return;
    if (currentAudioElRef.current) {
      currentAudioElRef.current.pause();
      currentAudioElRef.current.currentTime = 0;
    }
    const audio = new Audio(currentAudioUrl);
    currentAudioElRef.current = audio;
    audio.play().catch(() => {});
    audio.onended = () => { currentAudioElRef.current = null; };
  }, [currentAudioUrl]);

  React.useEffect(() => {
    if (!showHistoryDropdown) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (historyDropdownRef.current && !historyDropdownRef.current.contains(e.target as Node)) {
        setShowHistoryDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showHistoryDropdown]);

  const outputPortId = outputs[0]?.id;
  const isWorkflowReady = (id?: string) => {
    if (!id) return false;
    if (isStandalone()) return id.length > 0;
    return id.startsWith('workflow-') ||
      id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
  };

  const dataUrlToFile = (dataUrl: string, namePrefix: string) => {
    const arr = dataUrl.split(',');
    const mimeMatch = arr[0].match(/:(.*?);/);
    const mimeType = mimeMatch ? mimeMatch[1] : 'application/octet-stream';
    const extMap: Record<string, string> = {
      'image/png': '.png',
      'image/jpeg': '.jpg',
      'image/jpg': '.jpg',
      'image/webp': '.webp',
      'image/gif': '.gif',
      'audio/wav': '.wav',
      'audio/mpeg': '.mp3',
      'audio/ogg': '.ogg',
      'video/webm': '.webm',
      'video/mp4': '.mp4'
    };
    const ext = extMap[mimeType] || '.bin';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], `${namePrefix}-${Date.now()}${ext}`, { type: mimeType });
  };

  // 与文本输入一致：有后端时仅保留 data URL，执行时才存库；standalone 时写入 IndexedDB 得到 local://
  const persistDataUrl = async (dataUrl: string, namePrefix: string): Promise<string> => {
    if (!outputPortId || !isWorkflowReady(workflow.id)) return dataUrl;
    try {
      const localRef = await persistDataUrlToLocal(dataUrl, namePrefix);
      return localRef ?? dataUrl;
    } catch (err) {
      console.error('[Node] Failed to persist data URL:', err);
      return dataUrl;
    }
  };

  // 同步 image_edits 与 data.value 长度；只持久化 { crop_box }，不存 cropped/base64，载入时用 output_value 显示
  React.useEffect(() => {
    if (node.tool_id !== 'image-input') return;
    const values = Array.isArray(node.data.value) ? node.data.value : [];
    const edits = Array.isArray(node.data.image_edits) ? node.data.image_edits : [];
    if (values.length === edits.length && values.length === 0) return;
    const needsSync = values.length !== edits.length;
    if (!needsSync) return;
    const defaultCrop = { x: 10, y: 10, w: 80, h: 80 };
    const nextEdits = values.map((_: unknown, index: number) => {
      const existing = edits[index] as { crop_box?: { x: number; y: number; w: number; h: number } } | undefined;
      return { crop_box: existing?.crop_box ?? defaultCrop };
    });
    onUpdateNodeData(node.id, 'image_edits', nextEdits);
  }, [node.id, node.tool_id, node.data.value, node.data.image_edits]);

  React.useEffect(() => {
    if (!isWorkflowReady(workflow.id)) return;
    if (node.tool_id === 'audio-input' && typeof node.data.value === 'string' && node.data.value.startsWith('data:')) {
      persistDataUrl(node.data.value, 'audio-input').then((url) => {
        if (url !== node.data.value) {
          onUpdateNodeData(node.id, 'value', url);
          onUpdateNodeData(node.id, 'output_value', url);
        }
      });
    }
    if (node.tool_id === 'image-input' && Array.isArray(node.data.value) && node.data.value.some((v: unknown) => typeof v === 'string' && v.startsWith('data:'))) {
      const values = node.data.value as (string | { file_id?: string })[];
      const edits = Array.isArray(node.data.image_edits) ? node.data.image_edits : [];
      const persist = async () => {
        const updatedValues = await Promise.all(values.map((val, idx) =>
          typeof val === 'string' && val.startsWith('data:') ? persistDataUrl(val, `image-input-${idx}`) : val
        ));
        if (updatedValues.some((val, idx) => val !== values[idx])) {
          const defaultCrop = { x: 10, y: 10, w: 80, h: 80 };
          const updatedEdits = values.map((_: unknown, idx: number) => {
            const existing = edits[idx] as { crop_box?: { x: number; y: number; w: number; h: number } } | undefined;
            return { crop_box: existing?.crop_box ?? defaultCrop };
          });
          onUpdateNodeData(node.id, 'image_edits', updatedEdits);
          onUpdateNodeData(node.id, 'value', updatedValues);
          onUpdateNodeData(node.id, 'output_value', updatedValues);
        }
      };
      persist();
    }
    if (node.tool_id === 'video-input' && typeof node.data.value === 'string' && node.data.value.startsWith('data:')) {
      persistDataUrl(node.data.value, 'video-input').then((url) => {
        if (url !== node.data.value) {
          onUpdateNodeData(node.id, 'value', url);
          onUpdateNodeData(node.id, 'output_value', url);
        }
      });
    }
  }, [workflow.id, node.id, node.tool_id, node.data.value]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, isMultiple: boolean = false) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    // 需要 workflow.id 才能上传
    if (!isWorkflowReady(workflow.id)) {
      console.error('[Node] Cannot upload file: workflow ID is not available');
      return;
    }

    setIsUploading(true);

    try {
      const tool = TOOLS.find(t => t.id === node.tool_id);
      if (!tool || tool.category !== 'Input') {
        console.error('[Node] Cannot upload file: node is not an input node');
        return;
      }

      const outputPort = tool.outputs[0];
      if (!outputPort) {
        console.error('[Node] Cannot upload file: output port not found');
        return;
      }

      // 与文本输入一致：仅暂存 data URL，执行该节点（下游用到）时才存库
      const dataUrlPromises = files.map((file: File) =>
        new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(file);
        })
      );
      const dataUrls = await Promise.all(dataUrlPromises);

      if (dataUrls.length > 0) {
        const portId = outputPort?.id ?? INPUT_PORT_IDS[node.tool_id] ?? outputs[0]?.id;
        if (!portId) return;
        // 使用 output_value ?? data.value 作为已有值；写入 port-keyed output_value
        const currentValue = getOutputValueByPort(node, portId) ?? node.data.value;
        const existing = isMultiple
          ? (Array.isArray(currentValue) ? currentValue : currentValue != null ? [currentValue] : [])
          : [];
        const newVal = isMultiple ? [...existing, ...dataUrls] : dataUrls[0];
        const nextPortKeyed = setOutputValueByPort(node.output_value, node.tool_id, portId, newVal);
        onUpdateNodeData(node.id, 'output_value', nextPortKeyed);
        if (node.tool_id === 'audio-input' && newVal) {
          onUpdateNodeData(node.id, 'value', newVal);
          onUpdateNodeData(node.id, 'audio_range', { start: 0, end: 100 });
        }
        if (node.tool_id === 'video-input') {
          onUpdateNodeData(node.id, 'trimStart', 0);
          onUpdateNodeData(node.id, 'trimEnd', undefined);
        }
      }
    } catch (err) {
      console.error('[Node] Error uploading files:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const modelNotInList =
    Array.isArray(tool.models) &&
    tool.models.length > 0 &&
    node.data.model != null &&
    String(node.data.model).trim() !== '' &&
    !tool.models.some((m) => m.id === node.data.model);

  const modelsListEmpty = Array.isArray(tool.models) && tool.models.length === 0;
  // 仅对需要模型的节点（AI 模型类）显示该警告；语音合成、音色克隆等不显示；文本生成（大模型）不显示空列表警告
  const noModelNeededToolIds = ['tts', 'lightx2v-voice-clone'];
  const showModelsListEmptyWarning =
    modelsListEmpty &&
    tool.category === 'AI Model' &&
    !noModelNeededToolIds.includes(tool.id) &&
    tool.id !== 'text-generation';

  // 当前模型不在支持列表中时，自动选为列表第一项
  React.useEffect(() => {
    if (!modelNotInList || !tool.models?.length) return;
    const firstId = tool.models[0]?.id;
    if (firstId != null) onUpdateNodeData(node.id, 'model', firstId);
  }, [modelNotInList, tool.models, node.id, onUpdateNodeData]);

  return (
    <div
      ref={nodeRef}
      className={`node-element absolute bg-slate-900 border transition-all group pointer-events-auto ${
        isSelected ? 'z-[100]' : 'z-10'
      } ${isInputNode
        ? 'w-80 rounded-[2.5rem] shadow-2xl'
        : 'w-56 rounded-3xl shadow-2xl'
      } ${isSelected
        ? 'border-[#90dce1] ring-8 ring-#90dce1/10 shadow-[0_0_40px_-10px_rgba(144,220,225,0.35)]'
        : 'border-slate-800/80 hover:border-[#90dce1]/60 hover:shadow-[0_0_30px_-10px_rgba(144,220,225,0.2)]'
      }`}
      style={{ left: node.x, top: node.y }}
      onClick={handleNodeClick}
      onMouseDown={handleNodeMouseDown}
    >
      {/* 模型列表为空时显示警告（仅 AI 模型类）；模型不在列表中时已由 effect 自动切到第一项 */}
      {showModelsListEmptyWarning && (
        <div
          className="absolute bottom-full left-0 right-0 mb-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg bg-amber-500/20 border border-amber-500/50 text-amber-200 text-[9px] font-bold z-20"
          title={t('model_list_empty')}
        >
          <TriangleAlert size={10} className="shrink-0" />
          <span className="truncate">{t('model_list_empty')}</span>
        </div>
      )}
      {/* Replace and Delete Menu */}
      {isSelected && (
        <div className="absolute -top-14 left-1/2 -translate-x-1/2 flex items-center gap-2 z-20 replace-menu-container">
          <div className="relative">
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onSetReplaceMenu(showReplaceMenu === node.id ? null : node.id);
              }}
              className="p-2 bg-[#90dce1] text-white rounded-full shadow-lg hover:bg-[#90dce1] transition-all active:scale-90"
              title={lang === 'zh' ? '替换节点' : 'Replace Node'}
            >
              <RefreshCw size={16} />
            </button>
            {showReplaceMenu === node.id && (
              <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-48 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar">
                {getReplaceableTools(node.id).length > 0 ? (
                  getReplaceableTools(node.id).map((replaceTool) => (
                    <button
                      key={replaceTool.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        onReplaceNode(node.id, replaceTool.id);
                      }}
                      className="w-full px-4 py-2 text-left text-xs text-slate-300 hover:bg-[#90dce1]/20 hover:text-white transition-colors flex items-center gap-2"
                    >
                      <div className="p-1 rounded bg-slate-700">
                        {React.createElement(getIcon(replaceTool.icon), { size: 12 })}
                      </div>
                      <span>{lang === 'zh' ? replaceTool.name_zh : replaceTool.name}</span>
                    </button>
                  ))
                ) : (
                  <div className="px-4 py-3 text-xs text-slate-500 text-center">
                    {lang === 'zh' ? '没有可替换的工具' : 'No replaceable tools'}
                  </div>
                )}
              </div>
            )}
          </div>
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onDeleteNode(node.id);
              onSetReplaceMenu(null);
            }}
            className="p-2 bg-red-500 text-white rounded-full shadow-lg hover:bg-red-600 transition-all active:scale-90"
          >
            <Trash2 size={16} />
          </button>
          {onAddNodeToChat && (
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                const nodeName = node.name ?? (lang === 'zh' ? tool.name_zh : tool.name);
                onAddNodeToChat(node.id, nodeName);
              }}
              className="p-2 bg-[#90dce1] text-slate-900 rounded-full shadow-lg hover:bg-[#7accd0] transition-all active:scale-90"
              title={lang === 'zh' ? '加入对话' : 'Add to chat'}
            >
              <Bot size={16} />
            </button>
          )}
        </div>
      )}

      {/* Node Header */}
      <div
        className={`px-4 py-3 border-b bg-slate-800/40 rounded-t-3xl ${
          node.status === NodeStatus.RUNNING
            ? 'animate-pulse bg-[#90dce1]/10 border-[#90dce1]/20'
            : node.status === NodeStatus.PENDING
              ? 'bg-amber-500/10 border-amber-500/20'
              : ''
        } ${node.status === NodeStatus.RUNNING && isTaskRunning && progressPercent > 0 ? 'pb-2' : ''}`}
      >
        <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 truncate flex-1 min-w-0">
          <div
            className={`p-1.5 rounded-lg shrink-0 ${
              node.status === NodeStatus.RUNNING
                ? 'bg-[#90dce1] text-white'
                : node.status === NodeStatus.PENDING
                  ? 'bg-amber-500/80 text-white'
                  : 'bg-slate-800 text-slate-400'
            }`}
          >
            {React.createElement(getIcon(tool.icon), { size: 10 })}
          </div>
          <span className="text-[10px] font-black uppercase truncate tracking-widest">
            {node.name ?? (lang === 'zh' ? tool.name_zh : tool.name)}
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {(node.tool_id === 'audio-input' || node.tool_id === 'video-input') && (
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                if (node.tool_id === 'audio-input') {
                  onSetShowAudioEditor(node.id);
                } else {
                  onSetShowVideoEditor(node.id);
                }
              }}
              className="p-1.5 rounded-lg bg-slate-900/70 border border-slate-800 text-slate-400 hover:text-[#90dce1] hover:border-[#90dce1]/60 transition-all"
              title={lang === 'zh' ? '放大编辑' : 'Expand'}
            >
              <Maximize2 size={12} />
            </button>
          )}
          {(node.status === NodeStatus.RUNNING || node.status === NodeStatus.PENDING || node.execution_time !== undefined || node.error === 'Cancelled') && durationText && (
            <span
              className={`text-[8px] font-bold ${
                node.error === 'Cancelled'
                  ? 'text-slate-500'
                  : node.status === NodeStatus.RUNNING
                    ? 'text-[#90dce1]'
                    : node.status === NodeStatus.PENDING
                      ? 'text-amber-400'
                      : 'text-slate-500'
              }`}
            >
              {durationText}
            </span>
          )}

          {node.status === NodeStatus.RUNNING && isTaskRunning && progressPercent > 0 && (
            <span className="text-[8px] font-bold text-[#90dce1] tabular-nums">{progressPercent}%</span>
          )}

          {firstOutputType === DataType.AUDIO && currentAudioVal != null && (
            <span className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
              {currentAudioUrl ? (
                <button
                  type="button"
                  onMouseDown={(e) => e.stopPropagation()}
                  onClick={handlePlayCurrentAudio}
                  className="p-1.5 rounded-lg bg-[#90dce1]/20 hover:bg-[#90dce1]/40 text-[#90dce1] transition-colors"
                  title={lang === 'zh' ? '播放' : 'Play'}
                >
                  <PlayIcon size={12} />
                </button>
              ) : (
                <span className="inline-block p-1.5 text-slate-500" title={lang === 'zh' ? '加载中…' : 'Loading…'}>
                  <RefreshCw size={12} className="animate-spin opacity-70" />
                </span>
              )}
            </span>
          )}
          {node.tool_id !== 'image-input' && (
            <div className="relative" ref={historyDropdownRef}>
              <button
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  setShowHistoryDropdown((v) => !v);
                }}
                className={`p-1 rounded transition-colors ${
                  nodeHistoryEntries.length > 0
                    ? 'text-slate-400 hover:text-[#90dce1] opacity-0 group-hover:opacity-100'
                    : 'text-slate-600 cursor-default opacity-0 group-hover:opacity-50'
                }`}
                title={lang === 'zh' ? '历史结果' : 'History'}
              >
                <History size={12} />
              </button>
              {showHistoryDropdown && (
                <div
                  className="absolute right-0 top-full mt-1 w-64 h-72 overflow-y-auto bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 custom-scrollbar"
                  onMouseDown={(e) => e.stopPropagation()}
                  onWheel={(e) => {
                    e.stopPropagation();
                    if (e.ctrlKey) e.preventDefault();
                  }}
                >
                  {nodeHistoryEntries.length > 0 ? (
                    nodeHistoryEntries.map((entry) => (
                      <HistoryEntryItem
                        key={entry.id}
                        entry={entry}
                        lang={lang}
                        onSelect={() => handleHistoryEntrySelect(entry)}
                        resolveLightX2VResultRef={resolveLightX2VResultRef ? resolveTaskRefForHistory : undefined}
                        outputDataType={firstOutputType}
                        outputPortId={firstOutputPortId ?? undefined}
                        resolveAudioUrl={(e) => resolveAudioUrlForEntry(e, firstOutputPortId ?? undefined)}
                        resolveVideoUrl={(e) => resolveVideoUrlForEntry(e, firstOutputPortId ?? undefined)}
                        resolveImageUrl={(e) => resolveImageUrlForEntry(e, firstOutputPortId ?? undefined)}
                        resolveTextFileContent={workflow?.id && firstOutputType === DataType.TEXT ? (e) => {
                          const portKeyed = getEntryPortKeyedValue(e);
                          const keys = Object.keys(portKeyed);
                          const portId = (keys.length === 1 ? keys[0] : undefined) ?? firstOutputPortId ?? 'out-text';
                          const ov = (portId ? portKeyed[portId] : null) ?? (e?.output_value as { file_id?: string; run_id?: string });
                          const fid = ov?.file_id;
                          return fid ? getWorkflowFileText(workflow.id!, fid, node.id, portId, ov?.run_id) : Promise.resolve(null);
                        } : undefined}
                      />
                    ))
                  ) : (
                    <div className="px-3 py-4 text-[10px] text-slate-500 text-center">
                      {lang === 'zh' ? '暂无历史记录' : 'No history'}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          {!isInputNode && (
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              {node.status === NodeStatus.RUNNING || node.status === NodeStatus.PENDING || (pendingRunNodeIds && pendingRunNodeIds.includes(node.id)) ? (
                <button
                  title={lang === 'zh' ? '取消运行' : 'Cancel run'}
                  onClick={(e) => {
                    e.stopPropagation();
                    onCancelNodeRun(node.id);
                  }}
                  className="p-1 text-slate-400 hover:text-red-400 transition-colors"
                >
                  <Square size={12} />
                </button>
              ) : (
                <>
                  <button
                    title={t('run_this_only')}
                    onClick={(e) => {
                      e.stopPropagation();
                      onRunWorkflow(node.id, true);
                    }}
                    className="p-1 text-slate-400 hover:text-[#90dce1] transition-colors"
                  >
                    <PlayIcon size={12} />
                  </button>
                  <button
                    title={t('run_from_here')}
                    onClick={(e) => {
                      e.stopPropagation();
                      onRunWorkflow(node.id, false);
                    }}
                    className="p-1 text-slate-400 hover:text-emerald-400 transition-colors"
                  >
                    <FastForward size={12} />
                  </button>
                </>
              )}
            </div>
          )}

          {node.status === NodeStatus.PENDING && <Clock size={12} className="text-amber-400" />}
          {node.status === NodeStatus.SUCCESS && <CheckCircle2 size={12} className="text-emerald-500" />}
          {node.status === NodeStatus.ERROR && <AlertCircle size={12} className="text-red-500" />}
        </div>
        {node.status === NodeStatus.RUNNING && isTaskRunning && progressPercent > 0 && (
          <div className="mt-2 w-full h-1 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden">
            <div className="h-full bg-[#90dce1] rounded-full transition-all duration-500" style={{ width: `${progressPercent}%` }} />
          </div>
        )}
        </div>
      </div>

      {/* Node Content */}
      <div className="p-4 space-y-4">
        {/* Input Node Content */}
        {isInputNode && (
          <div onMouseDown={(e) => e.stopPropagation()} className="space-y-3">
            {node.tool_id === 'text-input' && (
              <TextNodePreview
                value={(() => {
                  const plainFromData = typeof node.data?.value === 'string' && !node.data.value.startsWith('data:') ? node.data.value : null;
                  if (plainFromData != null) return plainFromData;
                  if (isTextFileRef) return resolvedTextFromFile ?? '';
                  const pv = getOutputValueByPort(node, 'out-text');
                  return typeof pv === 'string' ? pv : '';
                })()}
                onChange={(value) => {
                  onUpdateNodeData(node.id, 'value', value);
                  // 仅前端模式：无后端，文本始终为纯文本，需写入 output_value 以便持久化；有后端时仅更新 data.value，output_value 由执行/保存时上传为 file ref
                  if (isStandalone()) {
                    onUpdateNodeData(node.id, 'output_value', setOutputValueByPort(node.output_value ?? {}, 'text-input', 'out-text', value));
                  }
                }}
                placeholder={lang === 'zh' ? '在此输入文本...' : 'Enter input text here...'}
                charsLabel={lang === 'zh' ? '字符' : 'Characters'}
              />
            )}
            {node.tool_id === 'image-input' && (
              <div className="space-y-2">
                <input
                  ref={imageInputRef}
                  type="file"
                  multiple
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => handleFileUpload(e, true)}
                />
                {imageEntries.length === 0 ? (
                  <label
                    onClick={() => imageInputRef.current?.click()}
                    className="flex flex-col items-center justify-center w-full py-10 border-2 border-dashed border-slate-800 rounded-[2rem] cursor-pointer hover:bg-slate-800/40 hover:border-[#90dce1]/40 transition-all group"
                  >
                    <div className="p-3 rounded-2xl bg-slate-900 group-hover:bg-[#90dce1]/10 transition-colors mb-3">
                      <Upload size={24} className="text-slate-600 group-hover:text-[#90dce1] transition-colors" />
                    </div>
                    <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest group-hover:text-slate-200">
                      {lang === 'zh' ? '点击上传图片' : 'Upload Images'}
                    </span>
                  </label>
                ) : (
                  <div className="relative group/content-container">
                    <button
                      onClick={() => {
                        onUpdateNodeData(node.id, 'value', []);
                        onUpdateNodeData(node.id, 'output_value', []);
                        onUpdateNodeData(node.id, 'image_edits', []);
                      }}
                      className="absolute -top-3 -right-3 p-2 bg-red-500 hover:bg-red-600 text-white rounded-full shadow-2xl z-20 opacity-0 group-hover/content-container:opacity-100 transition-all scale-90 group-hover/content-container:scale-100 active:scale-90"
                    >
                      <X size={12} />
                    </button>
                    <ImageNodePreview
                      images={imageEntries}
                      onAddMore={() => imageInputRef.current?.click()}
                      onUpdate={(nextImages) => {
                        const nextValues = nextImages.map((entry: any) =>
                          entry.cropped && entry.cropped !== entry.original ? entry.cropped : entry.source
                        );
                        onUpdateNodeData(node.id, 'value', nextValues);
                        onUpdateNodeData(node.id, 'output_value', nextValues);
                        onUpdateNodeData(node.id, 'image_edits', nextImages.map((entry: any) => ({ crop_box: entry.crop_box })));
                        if (isWorkflowReady(workflow.id) && outputPortId) {
                          const persist = async () => {
                            const newValues = await Promise.all(nextImages.map(async (entry: any, idx: number) => {
                              if (typeof entry.cropped === 'string' && entry.cropped.startsWith('data:')) {
                                return await persistDataUrl(entry.cropped, `image-input-${idx}`) || entry.cropped;
                              }
                              return entry.cropped || entry.source;
                            }));
                            onUpdateNodeData(node.id, 'value', newValues);
                            onUpdateNodeData(node.id, 'output_value', newValues);
                          };
                          persist();
                        }
                      }}
                    />
                  </div>
                )}
              </div>
            )}
            {(node.tool_id === 'audio-input' || node.tool_id === 'video-input') && (
              <div className="space-y-2">
                {(() => {
                  const mediaValue = Array.isArray(node.data.value) ? node.data.value[0] : node.data.value;
                  const hasMedia = !!mediaValue;
                  if (!hasMedia) {
                    return (
                      <>
                        <input
                          ref={node.tool_id === 'audio-input' ? audioInputRef : videoInputRef}
                          type="file"
                          accept={node.tool_id === 'audio-input' ? 'audio/*' : 'video/*'}
                          className="hidden"
                          disabled={isUploading}
                          onChange={(e) => handleFileUpload(e, false)}
                        />
                        <label
                          onClick={() => (node.tool_id === 'audio-input' ? audioInputRef.current : videoInputRef.current)?.click()}
                          className="flex flex-col items-center justify-center w-full py-10 border-2 border-dashed border-slate-800 rounded-[2rem] cursor-pointer hover:bg-slate-800/40 hover:border-[#90dce1]/40 transition-all group"
                        >
                          {isUploading ? (
                            <>
                              <RefreshCw size={24} className="text-[#90dce1] animate-spin mb-3" />
                              <span className="text-[10px] font-black text-[#90dce1] uppercase tracking-widest">
                                {lang === 'zh' ? '上传中...' : 'Uploading...'}
                              </span>
                            </>
                          ) : (
                            <>
                              <div className="p-3 rounded-2xl bg-slate-900 group-hover:bg-[#90dce1]/10 transition-colors mb-3">
                                <Upload size={24} className="text-slate-600 group-hover:text-[#90dce1] transition-colors" />
                              </div>
                              <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest group-hover:text-slate-200">
                                {lang === 'zh'
                                  ? `点击上传${node.tool_id === 'audio-input' ? '音频' : '视频'}`
                                  : `Upload ${node.tool_id === 'audio-input' ? 'Audio' : 'Video'}`}
                              </span>
                            </>
                          )}
                        </label>
                      </>
                    );
                  }

                  return (
                    <div className="relative group/content-container">
                      <button
                        onClick={() => {
                          onUpdateNodeData(node.id, 'value', null);
                          if (node.tool_id === 'audio-input') {
                            onUpdateNodeData(node.id, 'audio_range', null);
                          }
                          if (node.tool_id === 'video-input') {
                            onUpdateNodeData(node.id, 'trimStart', null);
                            onUpdateNodeData(node.id, 'trimEnd', null);
                          }
                        }}
                        className="absolute -top-3 -right-3 p-2 bg-red-500 hover:bg-red-600 text-white rounded-full shadow-2xl z-20 opacity-0 group-hover/content-container:opacity-100 transition-all scale-90 group-hover/content-container:scale-100 active:scale-90"
                      >
                        <X size={12} />
                      </button>
                      {node.tool_id === 'audio-input' ? (
                        <>
                        <AudioNodePreview
                          audioData={{
                            original: (typeof node.data.value === 'string' ? node.data.value : '') || resolveMediaSrc(mediaValue),
                            trimmed: typeof mediaValue === 'string' ? mediaValue : resolveMediaSrc(mediaValue),
                            range: node.data.audio_range || { start: 0, end: 100 }
                          }}
                          onUpdate={(trimmed, range) => {
                            if (!node.data.value) {
                              onUpdateNodeData(node.id, 'value', mediaValue);
                            }
                            onUpdateNodeData(node.id, 'audio_range', range);
                            const portId = INPUT_PORT_IDS[node.tool_id] ?? 'out-audio';
                            const nextOutput = setOutputValueByPort(node.output_value, node.tool_id, portId, trimmed);
                            onUpdateNodeData(node.id, 'output_value', nextOutput);
                            if (typeof trimmed === 'string' && trimmed.startsWith('data:') && isWorkflowReady(workflow.id)) {
                              persistDataUrl(trimmed, 'audio-input-trim').then((url) => {
                                if (url !== trimmed) {
                                  const next = setOutputValueByPort(node.output_value, node.tool_id, portId, url);
                                  onUpdateNodeData(node.id, 'output_value', next);
                                }
                              });
                            }
                            if (typeof node.data.value === 'string' && node.data.value.startsWith('data:') && isWorkflowReady(workflow.id)) {
                              persistDataUrl(node.data.value, 'audio-input-original').then((url) => {
                                if (url !== node.data.value) {
                                  onUpdateNodeData(node.id, 'value', url);
                                }
                              });
                            }
                          }}
                        />
                        </>
                      ) : (
                        <VideoNodePreview
                          videoUrl={resolveMediaSrc(node.data.videoOriginal || mediaValue)}
                          initialStart={node.data.trimStart}
                          initialEnd={node.data.trimEnd}
                          onRangeChange={(start, end) => {
                            onUpdateNodeData(node.id, 'trimStart', start);
                            onUpdateNodeData(node.id, 'trimEnd', end);
                          }}
                          onUpdate={(start, end, trimmedUrl) => {
                            onUpdateNodeData(node.id, 'trimStart', start);
                            onUpdateNodeData(node.id, 'trimEnd', end);
                            if (!node.data.videoOriginal) {
                              onUpdateNodeData(node.id, 'videoOriginal', mediaValue);
                            }
                            if (trimmedUrl) {
                              onUpdateNodeData(node.id, 'value', trimmedUrl);
                              if (trimmedUrl.startsWith('data:') && isWorkflowReady(workflow.id)) {
                                persistDataUrl(trimmedUrl, 'video-input-trim').then((url) => {
                                  if (url !== trimmedUrl) {
                                    onUpdateNodeData(node.id, 'value', url);
                                  }
                                });
                              }
                            }
                            if (typeof node.data.videoOriginal === 'string' && node.data.videoOriginal.startsWith('data:') && isWorkflowReady(workflow.id)) {
                              persistDataUrl(node.data.videoOriginal, 'video-input-original').then((url) => {
                                if (url !== node.data.videoOriginal) {
                                  onUpdateNodeData(node.id, 'videoOriginal', url);
                                }
                              });
                            }
                          }}
                        />
                      )}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {node.status === NodeStatus.ERROR && (
          <div className="bg-red-500/10 border border-red-500/20 p-2 rounded-xl text-[8px] text-red-400 leading-tight">
            <span className="font-bold uppercase mb-1 block">{t('execution_error')}</span>
            {node.error}
          </div>
        )}

        {/* Input Ports */}
        {tool.inputs.map((p) => {
          const isConnected = workflow.connections.some(
            (c) => c.target_node_id === node.id && c.target_port_id === p.id
          );
          return (
            <div key={p.id} className="flex items-center gap-2 text-[9px] font-bold text-slate-500 relative group/port">
              {!isConnected && (
                <button
                  onMouseDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    quickAddInput(node, p);
                  }}
                  className="opacity-0 group-hover/port:opacity-100 transition-opacity p-1 bg-[#90dce1] text-white rounded-lg absolute -left-12 z-20 shadow-xl hover:bg-[#90dce1] active:scale-90 flex items-center justify-center"
                  title={t('quick_add_source')}
                >
                  <Plus size={14} />
                </button>
              )}
              <div
                className="port w-3 h-3 rounded-full bg-slate-800 border-2 border-slate-950 absolute -left-[24px] cursor-crosshair hover:bg-[#90dce1] transition-colors"
                onMouseDown={handlePortMouseDown(p, 'in')}
                onMouseUp={handlePortMouseUp(p, 'in')}
              />
              <span className="truncate">{p.label}</span>
            </div>
          );
        })}

        {/* Output Ports - 固定行高 30px、底留 24px 与 Connection 公式一致，mb-[-16px] 抵消 p-4 底边距使端口块底对齐节点底 */}
        <div className="flex flex-col pb-[9px] -mb-4" style={{ minHeight: outputs.length ? outputs.length * 30 + 9 : 0 }}>
        {outputs.map((p) => {
          const isConnected = workflow.connections.some(
            (c) => c.source_node_id === node.id && c.source_port_id === p.id
          );
          const compatibleTools = getCompatibleToolsForOutput(p.type);
          const showMenu = showOutputQuickAdd?.nodeId === node.id && showOutputQuickAdd?.portId === p.id;
          return (
            <div key={p.id} className="h-[30px] flex items-center justify-end gap-2 text-[9px] font-bold text-slate-500 relative group/port shrink-0">
              {!isConnected && compatibleTools.length > 0 && (
                <button
                  onMouseDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSetOutputQuickAdd(showMenu ? null : { nodeId: node.id, portId: p.id });
                  }}
                  className="opacity-0 group-hover/port:opacity-100 transition-opacity p-1 bg-[#90dce1] text-white rounded-lg absolute -right-12 z-20 shadow-xl hover:bg-[#90dce1] active:scale-90 flex items-center justify-center"
                  title={lang === 'zh' ? '快速添加节点' : 'Quick Add Node'}
                >
                  <Plus size={14} />
                </button>
              )}
              {showMenu && (
                <div
                  className="absolute top-0 left-full ml-2 w-48 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar output-quick-add-menu"
                  onClick={(e) => e.stopPropagation()}
                >
                  {compatibleTools.map((tool) => (
                    <button
                      key={tool.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        quickAddOutput(node, p, tool.id);
                      }}
                      className="w-full px-4 py-2 text-left text-xs text-slate-300 hover:bg-[#90dce1]/20 hover:text-white transition-colors flex items-center gap-2"
                    >
                      <div className="p-1 rounded bg-slate-700">
                        {React.createElement(getIcon(tool.icon), { size: 12 })}
                      </div>
                      <span>{lang === 'zh' ? tool.name_zh : tool.name}</span>
                    </button>
                  ))}
                </div>
              )}
              <span className="truncate">{lang === 'zh' && p.label_zh ? p.label_zh : p.label}</span>
              <div
                className="port w-3 h-3 rounded-full bg-slate-800 border-2 border-slate-950 absolute -right-[24px] cursor-crosshair hover:bg-[#90dce1] transition-colors"
                onMouseDown={handlePortMouseDown(p, 'out')}
                onMouseUp={handlePortMouseUp(p, 'out')}
              />
            </div>
          );
        })}
        </div>
      </div>

      {/* Preview: multi-port = one box per port; single = one box */}
      {shouldShowPreview && isMultiPortOutput && (
        <div className="absolute -right-42 top-0 flex flex-col gap-2 z-30 max-w-50">
          {outputs.map((p) => {
            const portVal = typeof sourceOutputs[node.id] === 'object' && sourceOutputs[node.id] != null && p.id in sourceOutputs[node.id]
              ? sourceOutputs[node.id][p.id]
              : getOutputValueByPort(node, p.id);
            const isFileRef = portVal && typeof portVal === 'object' && (portVal as any).file_id && (portVal as any).mime_type === 'text/plain';
            const fetched = resolvedPortText[p.id];
            const displayText = typeof portVal === 'string'
              ? (portVal.length > 60 ? portVal.slice(0, 60) + '…' : portVal)
              : isFileRef && typeof fetched === 'string'
                ? (fetched.length > 60 ? fetched.slice(0, 60) + '…' : fetched)
                : isFileRef
                  ? (fetched === null ? (lang === 'zh' ? '[已存文本]' : '[Text file]') : (lang === 'zh' ? '加载中…' : 'Loading…'))
                  : (lang === 'zh' ? '[结果]' : '[Result]');
            return (
              <div
                key={p.id}
                onClick={(e) => {
                  e.stopPropagation();
                  onSetExpandedOutput({ nodeId: node.id, fieldId: p.id });
                }}
                onMouseDown={(e) => e.stopPropagation()}
                className="relative max-w-36 bg-slate-800/95 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden cursor-pointer hover:scale-105 hover:border-[#90dce1] transition-all group/thumb flex flex-col"
              >
                <div className="px-2.5 pt-2 pb-2 text-[8px] font-bold text-slate-500 uppercase tracking-wider truncate">
                  {p.label}
                </div>
                <div className="p-2.5 pt-0 text-[8px] text-slate-300 overflow-hidden leading-snug font-medium selection:bg-transparent line-clamp-3 h-[1.5rem]">
                  {displayText}
                </div>
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-0 group-hover/thumb:opacity-100 transition-opacity duration-200 bg-slate-900/40 rounded-2xl">
                  <Maximize2 size={20} className="text-white drop-shadow-lg scale-75 group-hover/thumb:scale-100 transition-transform duration-200" />
                </div>
              </div>
            );
          })}
        </div>
      )}
      {shouldShowPreview && !isMultiPortOutput && (
        <div
          onClick={(e) => {
            e.stopPropagation();
            onSetExpandedOutput({ nodeId: node.id });
          }}
          onMouseDown={(e) => e.stopPropagation()}
          className="absolute -right-36 top-0 max-w-32 max-h-32 bg-slate-800/95 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden cursor-pointer hover:scale-110 hover:border-[#90dce1] transition-all z-30 group/thumb flex items-center justify-center"
        >
          {firstOutputType === DataType.IMAGE ? (
            isPreviewRef ? (
              refPreviewUrl ? (
                <img
                  src={refPreviewUrl}
                  className="max-w-full max-h-full w-auto h-auto object-contain"
                  alt="Preview"
                />
              ) : (
                <div className="text-[9px] text-slate-500 uppercase">Loading...</div>
              )
            ) : (
              <img
                src={getAssetPath(
                  Array.isArray(nodeResult)
                    ? (nodeResult.length > 0 ? nodeResult[0] : '')
                    : (nodeResult || '')
                )}
                className="max-w-full max-h-full w-auto h-auto object-contain"
                alt="Preview"
                onError={(e) => {
                  const target = e.currentTarget;
                  const originalSrc = Array.isArray(nodeResult)
                    ? (nodeResult.length > 0 ? nodeResult[0] : '')
                    : (nodeResult || '');
                  if (target.src !== originalSrc) {
                    target.src = originalSrc;
                  }
                }}
              />
            )
          ) : firstOutputType === DataType.TEXT ? (
            <div className="p-3 text-[8px] text-slate-300 overflow-hidden leading-snug font-medium selection:bg-transparent w-full h-full">
              {(resolvedTextFromFile != null
                ? resolvedTextFromFile
                : typeof nodeResult === 'object'
                ? JSON.stringify(nodeResult).slice(0, 100)
                : nodeResult?.toString?.() ?? '').slice(0, 100)}
              ...
            </div>
          ) : firstOutputType === DataType.AUDIO ? (
            <div className="w-full h-full flex flex-col items-center justify-center text-[#90dce1] bg-[#90dce1]/5 min-w-32 min-h-24">
              <Volume2 size={32} className="mb-1" />
              <div className="w-20 h-1.5 bg-[#90dce1]/20 rounded-full overflow-hidden">
                <div className="w-1/2 h-full bg-[#90dce1] animate-pulse"></div>
              </div>
            </div>
          ) : (
            <div className="w-full h-full relative bg-black group/video min-w-32 min-h-24 flex items-center justify-center">
              {isPreviewRef ? (
                refPreviewUrl ? (
                  <>
                    <video
                      src={refPreviewUrl}
                      className="max-w-full max-h-full w-auto h-auto object-contain opacity-60 group-hover/thumb:opacity-100 transition-opacity"
                      muted
                      preload="none"
                      loading="lazy"
                      onMouseOver={(e) => {
                        const video = e.currentTarget;
                        if (video.readyState < 2) video.load();
                        video.play().catch(() => {});
                      }}
                      onMouseOut={(e) => {
                        e.currentTarget.pause();
                        e.currentTarget.currentTime = 0;
                      }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center text-white pointer-events-none group-hover/thumb:scale-125 transition-transform">
                      <Play size={24} className="drop-shadow-lg" fill="currentColor" />
                    </div>
                  </>
                ) : (
                  <div className="text-[9px] text-slate-500 uppercase">Loading...</div>
                )
              ) : (
                <>
                  <video
                    src={getAssetPath(
                      Array.isArray(nodeResult)
                        ? (nodeResult.length > 0 ? nodeResult[0] : '')
                        : (nodeResult || '')
                    )}
                    className="max-w-full max-h-full w-auto h-auto object-contain opacity-60 group-hover/thumb:opacity-100 transition-opacity"
                    muted
                    preload="none"
                    loading="lazy"
                    onMouseOver={(e) => {
                      const video = e.currentTarget;
                      if (video.readyState < 2) {
                        video.load();
                      }
                      video.play().catch(() => {});
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.pause();
                      e.currentTarget.currentTime = 0;
                    }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center text-white pointer-events-none group-hover/thumb:scale-125 transition-transform">
                    <Play size={24} className="drop-shadow-lg" fill="currentColor" />
                  </div>
                </>
              )}
            </div>
          )}
          <div className="absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-slate-950/90 to-transparent flex items-center px-2.5">
            <span className="text-[7px] font-black uppercase text-white/80 tracking-widest flex items-center gap-1">
              <Maximize2 size={10} /> {t('inspect_result')}
            </span>
          </div>
        </div>
      )}

      {/* Model and Voice Selectors */}
      <div className="absolute top-full left-0 mt-2 flex items-center gap-2 z-20">
        {/* Model selector */}
        {tool.models && tool.models.length > 0 && (
          <div className="relative model-select-container">
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onSetModelSelect(showModelSelect === node.id ? null : node.id);
                onSetVoiceSelect(null);
              }}
              className="px-2 py-1 text-[9px] font-bold bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors flex items-center gap-1 shadow-lg"
              title={lang === 'zh' ? '选择模型' : 'Select Model'}
            >
              <span className="truncate max-w-[80px]">
                {tool.models.find((m) => m.id === node.data.model)?.name || tool.models[0]?.name || 'Model'}
              </span>
              <ChevronDown size={10} className={showModelSelect === node.id ? 'rotate-180' : ''} />
            </button>
            {showModelSelect === node.id && (
              <div
                className="absolute top-full left-0 mt-1 w-48 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar"
                onClick={(e) => e.stopPropagation()}
                onWheel={(e) => {
                  e.stopPropagation();
                  if (e.ctrlKey) e.preventDefault();
                }}
              >
                {tool.models.map((model) => (
                  <button
                    key={model.id}
                    onClick={(e) => {
                      e.stopPropagation();
                      onUpdateNodeData(node.id, 'model', model.id);
                      onSetModelSelect(null);
                    }}
                    className={`w-full px-4 py-2 text-left text-xs transition-colors flex items-center gap-2 ${
                      node.data.model === model.id
                        ? 'bg-[#90dce1]/20 text-white'
                        : 'text-slate-300 hover:bg-[#90dce1]/20 hover:text-white'
                    }`}
                  >
                    <span>{model.name}</span>
                    {node.data.model === model.id && <CheckCircle size={12} className="ml-auto" />}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Web search toggle for DeepSeek and Doubao models */}
        {node.tool_id === 'text-generation' &&
          (node.data.model?.startsWith('deepseek-') || node.data.model?.startsWith('doubao-')) && (
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onUpdateNodeData(node.id, 'useSearch', !node.data.useSearch);
            }}
            className={`px-2 py-1 text-[9px] font-bold rounded-md transition-colors flex items-center gap-1 shadow-lg ${
              node.data.useSearch
                ? 'bg-[#90dce1]/80 hover:bg-[#90dce1] text-white'
                : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
            }`}
            title={lang === 'zh' ? '联网搜索' : 'Web Search'}
          >
            <Globe size={10} />
          </button>
        )}

        {/* Voice selector for TTS nodes */}
        {node.tool_id === 'tts' &&
          (node.data.model === 'lightx2v' || node.data.model?.startsWith('lightx2v')) &&
          lightX2VVoiceList?.voices && (
            <div className="relative voice-select-container">
              <button
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  onSetVoiceSelect(showVoiceSelect === node.id ? null : node.id);
                  onSetModelSelect(null);
                }}
                className="px-2 py-1 text-[9px] font-bold bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors flex items-center gap-1 shadow-lg"
                title={lang === 'zh' ? '选择音色' : 'Select Voice'}
              >
                <span className="truncate max-w-[80px]">
                  {lightX2VVoiceList.voices.find((v: any) => v.voice_type === node.data.voiceType)?.name ||
                    node.data.voiceType ||
                    'Voice'}
                </span>
                <ChevronDown size={10} className={showVoiceSelect === node.id ? 'rotate-180' : ''} />
              </button>
              {showVoiceSelect === node.id && (
                <div
                  className="absolute top-full left-0 mt-1 w-56 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar"
                  onClick={(e) => e.stopPropagation()}
                  onWheel={(e) => {
                    e.stopPropagation();
                    if (e.ctrlKey) e.preventDefault();
                  }}
                >
                  {lightX2VVoiceList.voices.map((voice: any) => (
                    <button
                      key={voice.voice_type}
                      onClick={(e) => {
                        e.stopPropagation();
                        onUpdateNodeData(node.id, 'voiceType', voice.voice_type);
                        if (voice.resource_id) {
                          onUpdateNodeData(node.id, 'resourceId', voice.resource_id);
                        }
                        onSetVoiceSelect(null);
                      }}
                      className={`w-full px-4 py-2 text-left text-xs transition-colors flex items-center gap-2 ${
                        node.data.voiceType === voice.voice_type
                          ? 'bg-[#90dce1]/20 text-white'
                          : 'text-slate-300 hover:bg-[#90dce1]/20 hover:text-white'
                      }`}
                    >
                      <span className="truncate">{voice.name || voice.voice_type}</span>
                      {node.data.voiceType === voice.voice_type && <CheckCircle size={12} className="ml-auto" />}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

        {/* Voice selector for voice clone nodes */}
        {node.tool_id === 'lightx2v-voice-clone' && cloneVoiceList.length > 0 && (
          <div className="relative voice-select-container">
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onSetVoiceSelect(showVoiceSelect === node.id ? null : node.id);
                onSetModelSelect(null);
              }}
              className="px-2 py-1 text-[9px] font-bold bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-md transition-colors flex items-center gap-1 shadow-lg"
              title={lang === 'zh' ? '选择克隆音色' : 'Select Clone Voice'}
            >
              <span className="truncate max-w-[80px]">
                {cloneVoiceList.find((v) => v.speaker_id === node.data.speakerId)?.name ||
                  node.data.speakerId ||
                  'Voice'}
              </span>
              <ChevronDown size={10} className={showVoiceSelect === node.id ? 'rotate-180' : ''} />
            </button>
            {showVoiceSelect === node.id && (
              <div
                className="absolute top-full left-0 mt-1 w-56 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-30 max-h-64 overflow-y-auto custom-scrollbar"
                onClick={(e) => e.stopPropagation()}
              >
                {cloneVoiceList.map((voice: any) => (
                  <button
                    key={voice.speaker_id}
                    onClick={(e) => {
                      e.stopPropagation();
                      onUpdateNodeData(node.id, 'speakerId', voice.speaker_id);
                      onSetVoiceSelect(null);
                    }}
                    className={`w-full px-4 py-2 text-left text-xs transition-colors flex items-center gap-2 ${
                      node.data.speakerId === voice.speaker_id
                        ? 'bg-[#90dce1]/20 text-white'
                        : 'text-slate-300 hover:bg-[#90dce1]/20 hover:text-white'
                    }`}
                  >
                    <span className="truncate">{voice.name || voice.speaker_id}</span>
                    {node.data.speakerId === voice.speaker_id && <CheckCircle size={12} className="ml-auto" />}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
