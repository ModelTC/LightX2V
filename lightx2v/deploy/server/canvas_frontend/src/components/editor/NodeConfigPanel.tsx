import React, { useState, useMemo, useEffect, useCallback } from 'react';
import {
  Settings,
  Trash2,
  Boxes,
  MessageSquare,
  RefreshCw,
  Target,
  Upload,
  Search,
  ListPlus,
  CheckCircle2,
  X,
  Globe,
  Tag,
  Plus
} from 'lucide-react';
import { WorkflowState, WorkflowNode, Port, DataType } from '../../../types';
import { TOOLS } from '../../../constants';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { getOutputValueByPort, setOutputValueByPort } from '../../utils/outputValuePort';
import { getWorkflowFileText, saveNodeOutputs } from '../../utils/workflowFileManager';
import { isStandalone } from '../../config/runtimeMode';
import { getAssetPath } from '../../utils/assetPath';
import { isLightX2VResultRef } from '../../hooks/useWorkflowExecution';

interface NodeConfigPanelProps {
  lang: Language;
  workflow: WorkflowState;
  selectedNodeId: string | null;
  sourceOutputs: Record<string, any>;
  disconnectedInputs: Array<{
    nodeId: string;
    port: Port;
    toolName: string;
    isSourceNode?: boolean;
    dataType: DataType;
  }>;
  lightX2VVoiceList?: { voices?: any[]; emotions?: string[]; languages?: any[] } | null;
  loadingVoiceList: boolean;
  voiceSearchQuery: string;
  setVoiceSearchQuery: (query: string) => void;
  showVoiceFilter: boolean;
  setShowVoiceFilter: (show: boolean) => void;
  voiceFilterGender: string;
  setVoiceFilterGender: (gender: string) => void;
  filteredVoices: any[];
  isFemaleVoice: (voiceType: string) => boolean;
  cloneVoiceList: any[];
  loadingCloneVoiceList: boolean;
  onUpdateNodeData: (nodeId: string, key: string, value: any) => void;
  onUpdateNodeName?: (nodeId: string, name: string) => void;
  onDeleteNode: (nodeId: string) => void;
  onGlobalInputChange: (nodeId: string, portId: string, value: any) => void;
  onDescriptionChange?: (description: string) => void;
  onTagsChange?: (tags: string[]) => void;
  onShowCloneVoiceModal: () => void;
  collapsed?: boolean;
  style?: React.CSSProperties;
  resolveLightX2VResultRef?: (ref: any) => Promise<string>;
  getNodeOutputUrl?: (nodeId: string, portId: string, fileId?: string, runId?: string) => Promise<string | null>;
}

export const NodeConfigPanel: React.FC<NodeConfigPanelProps> = ({
  lang,
  workflow,
  selectedNodeId,
  sourceOutputs,
  disconnectedInputs,
  lightX2VVoiceList,
  loadingVoiceList,
  voiceSearchQuery,
  setVoiceSearchQuery,
  showVoiceFilter,
  setShowVoiceFilter,
  voiceFilterGender,
  setVoiceFilterGender,
  filteredVoices,
  isFemaleVoice,
  cloneVoiceList,
  loadingCloneVoiceList,
  onUpdateNodeData,
  onUpdateNodeName,
  onDeleteNode,
  onGlobalInputChange,
  onDescriptionChange,
  onTagsChange,
  onShowCloneVoiceModal,
  collapsed = false,
  style,
  resolveLightX2VResultRef,
  getNodeOutputUrl
}) => {
  const { t } = useTranslation(lang);
  const [uploadingNodes, setUploadingNodes] = useState<Set<string>>(new Set());
  const [newTagInput, setNewTagInput] = useState('');
  const selectedNode = selectedNodeId ? workflow.nodes.find(n => n.id === selectedNodeId) : null;
  const tags = workflow.tags ?? [];

  const connectedInputsList = useMemo(() => {
    if (!selectedNode) return [];
    const tool = TOOLS.find(t => t.id === selectedNode.tool_id);
    if (!tool) return [];
    return tool.inputs
      .map(port => {
        const conn = workflow.connections.find(c => c.target_node_id === selectedNode.id && c.target_port_id === port.id);
        if (!conn) return null;
        const sourceNode = workflow.nodes.find(n => n.id === conn.source_node_id);
        if (!sourceNode) return null;
        const sourceTool = TOOLS.find(t => t.id === sourceNode.tool_id);
        let sourceDataType: DataType = DataType.TEXT;
        let fieldLabel: string = conn.source_port_id;
        if (sourceNode.tool_id === 'text-generation' && sourceNode.data.custom_outputs) {
          const custom = sourceNode.data.custom_outputs.find((o: any) => o.id === conn.source_port_id);
          if (custom) {
            sourceDataType = DataType.TEXT;
            fieldLabel = custom.label ?? conn.source_port_id;
          } else {
            sourceDataType = DataType.TEXT;
            fieldLabel = conn.source_port_id;
          }
        } else {
          const outPort = sourceTool?.outputs?.find((o: any) => o.id === conn.source_port_id);
          sourceDataType = (outPort as any)?.type ?? DataType.TEXT;
          fieldLabel = (outPort as any)?.label ?? conn.source_port_id;
        }
        return { port, conn, sourceNode, fieldLabel, sourceDataType };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);
  }, [selectedNode, workflow?.nodes, workflow?.connections]);

  const [resolvedFieldText, setResolvedFieldText] = useState<Record<string, string>>({});
  const [localEdits, setLocalEdits] = useState<Record<string, string>>({});
  const [applyingFieldKey, setApplyingFieldKey] = useState<string | null>(null);
  const [resolvedMediaUrls, setResolvedMediaUrls] = useState<Record<string, string | string[]>>({});

  const textOnlyList = useMemo(() => connectedInputsList.filter(i => i.sourceDataType === DataType.TEXT), [connectedInputsList]);
  const mediaList = useMemo(() => connectedInputsList.filter(i => i.sourceDataType === DataType.IMAGE || i.sourceDataType === DataType.AUDIO || i.sourceDataType === DataType.VIDEO), [connectedInputsList]);

  useEffect(() => {
    if (textOnlyList.length === 0) {
      setResolvedFieldText({});
      return;
    }
    const key = (nodeId: string, portId: string) => `${nodeId}:${portId}`;
    const next: Record<string, string> = {};
    const refs: { k: string; fileId: string; sourceNodeId: string; sourcePortId: string; runId?: string }[] = [];
    textOnlyList.forEach(({ conn, sourceNode }) => {
      const raw = getOutputValueByPort(sourceNode, conn.source_port_id) ?? sourceOutputs[conn.source_node_id]?.[conn.source_port_id];
      const k = key(conn.source_node_id, conn.source_port_id);
      if (typeof raw === 'string') next[k] = raw;
      else if (raw && typeof raw === 'object' && (raw as any).file_id && (raw as any).mime_type === 'text/plain')
        refs.push({ k, fileId: (raw as any).file_id, sourceNodeId: conn.source_node_id, sourcePortId: conn.source_port_id, runId: (raw as any).run_id });
    });
    if (refs.length === 0) {
      setResolvedFieldText(prev => (Object.keys(next).length ? { ...prev, ...next } : prev));
      return;
    }
    setResolvedFieldText(prev => ({ ...prev, ...next }));
    if (!workflow?.id) return;
    let cancelled = false;
    Promise.all(refs.map(async ({ k, fileId, sourceNodeId, sourcePortId, runId }) => {
      const text = await getWorkflowFileText(workflow.id!, fileId, sourceNodeId, sourcePortId, runId);
      return [k, text ?? ''] as const;
    })).then(pairs => {
      if (cancelled) return;
      setResolvedFieldText(prev => { const n = { ...prev }; pairs.forEach(([k, v]) => { n[k] = v; }); return n; });
    });
    return () => { cancelled = true; };
  }, [workflow?.id, workflow?.nodes, textOnlyList, sourceOutputs]);

  useEffect(() => {
    if (mediaList.length === 0 || !workflow?.id) {
      setResolvedMediaUrls({});
      return;
    }
    const key = (nodeId: string, portId: string) => `${nodeId}:${portId}`;
    const resolveOne = (raw: any): string | null => {
      if (!raw || typeof raw === 'string') return raw && typeof raw === 'string' ? raw : null;
      return null;
    };
    let cancelled = false;
    Promise.all(mediaList.map(async ({ conn, sourceNode, sourceDataType }) => {
      const raw = getOutputValueByPort(sourceNode, conn.source_port_id) ?? sourceOutputs[conn.source_node_id]?.[conn.source_port_id];
      const k = key(conn.source_node_id, conn.source_port_id);
      const arr = Array.isArray(raw) ? raw : (raw != null ? [raw] : []);
      const urls = await Promise.all(arr.map(async (v: any) => {
        if (isLightX2VResultRef(v) && resolveLightX2VResultRef) return resolveLightX2VResultRef(v, workflow?.id ? { workflow_id: workflow.id, node_id: conn.source_node_id, port_id: conn.source_port_id } : undefined);
        if (v && typeof v === 'object' && (v as any).kind === 'url' && typeof (v as any).url === 'string') return (v as any).url;
        if (v && typeof v === 'object' && (v as any).file_id && getNodeOutputUrl) {
          const url = await getNodeOutputUrl(conn.source_node_id, conn.source_port_id, (v as any).file_id, (v as any).run_id);
          if (url) return getAssetPath(url) || url;
        }
        return resolveOne(v);
      }));
      const resolved = urls.filter((u): u is string => u != null);
      return [k, sourceDataType, resolved] as const;
    })).then(pairs => {
      if (cancelled) return;
      setResolvedMediaUrls(prev => {
        const n = { ...prev };
        pairs.forEach(([k, sourceDataType, resolved]) => {
          if (resolved.length) n[k] = sourceDataType === DataType.IMAGE && resolved.length > 1 ? resolved : resolved[0];
        });
        return n;
      });
    });
    return () => { cancelled = true; };
  }, [workflow?.id, workflow?.nodes, mediaList, sourceOutputs, resolveLightX2VResultRef, getNodeOutputUrl]);

  const handleApplyConnectedField = useCallback(async (sourceNode: WorkflowNode, conn: { source_node_id: string; source_port_id: string }, newText: string) => {
    const fieldKey = `${conn.source_node_id}:${conn.source_port_id}`;
    setApplyingFieldKey(fieldKey);
    try {
      let valueToSet: any = newText;
      if (workflow?.id && !isStandalone()) {
        const result = await saveNodeOutputs(workflow.id, sourceNode.id, { [conn.source_port_id]: newText }, crypto.randomUUID());
        if (result?.[conn.source_port_id]) valueToSet = result[conn.source_port_id];
      }
      const nextOutput = setOutputValueByPort(sourceNode.output_value, sourceNode.tool_id, conn.source_port_id, valueToSet);
      onUpdateNodeData(sourceNode.id, 'output_value', nextOutput);
      setLocalEdits(prev => { const n = { ...prev }; delete n[fieldKey]; return n; });
    } finally {
      setApplyingFieldKey(null);
    }
  }, [workflow?.id, onUpdateNodeData]);

  if (selectedNodeId && selectedNode) {
      return (
        <aside
          className={`flex flex-col z-30 transition-all ${collapsed ? 'h-0 overflow-hidden' : 'flex-1 min-h-0 p-6 overflow-y-auto'}`}
          style={style}
      >
        <div className="space-y-8 animate-in slide-in-from-right-4 duration-300">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
              <Settings size={14} /> {t('settings')}
            </h2>
            <button
              onClick={() => onDeleteNode(selectedNode.id)}
              className="p-2 text-slate-500 hover:text-red-400 transition-colors"
            >
              <Trash2 size={16} />
            </button>
          </div>
          <div className="space-y-6">
            {/* Node name */}
            {onUpdateNodeName && (
              <div className="space-y-2">
                <span className="text-[10px] text-slate-500 font-black uppercase flex items-center gap-2">
                  <Boxes size={12} /> {t('node_name')}
                </span>
                <input
                  type="text"
                  value={selectedNode.name ?? ''}
                  onChange={e => onUpdateNodeName(selectedNode.id, e.target.value)}
                  placeholder={t('node_name_placeholder')}
                  className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all placeholder:text-slate-500"
                />
              </div>
            )}
            {/* Model Selection */}
            {TOOLS.find(t => t.id === selectedNode.tool_id)?.models && (
              <div className="space-y-2">
                <span className="text-[10px] text-slate-500 font-black uppercase flex items-center gap-2">
                  <Boxes size={12} /> {t('select_model')}
                </span>
                <select
                  value={selectedNode.data.model}
                  onChange={e => onUpdateNodeData(selectedNode.id, 'model', e.target.value)}
                  className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                >
                  {TOOLS.find(t => t.id === selectedNode.tool_id)?.models?.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Web Search Toggle for DeepSeek and Doubao */}
            {selectedNode.tool_id === 'text-generation' &&
              (selectedNode.data.model?.startsWith('deepseek-') || selectedNode.data.model?.startsWith('doubao-')) && (
              <div className="space-y-2">
                <span className="text-[10px] text-slate-500 font-black uppercase flex items-center gap-2">
                  <Globe size={12} /> {lang === 'zh' ? '联网搜索' : 'Web Search'}
                </span>
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedNode.data.useSearch || false}
                    onChange={e => onUpdateNodeData(selectedNode.id, 'useSearch', e.target.checked)}
                    className="w-5 h-5 rounded border-slate-600 bg-slate-800 text-[#90dce1] focus:ring-#90dce1 focus:ring-offset-slate-900 cursor-pointer"
                  />
                  <span className="text-xs text-slate-300">
                    {lang === 'zh' ? '启用联网搜索功能' : 'Enable web search'}
                  </span>
                </label>
              </div>
            )}

            {/* Gemini Text Mode */}
            {selectedNode.tool_id === 'text-generation' && (
              <div className="space-y-6">
                <div className="space-y-2">
                  <span className="text-[10px] text-slate-500 font-black uppercase">{t('mode')}</span>
                  <select
                    value={selectedNode.data.mode}
                    onChange={e => onUpdateNodeData(selectedNode.id, 'mode', e.target.value)}
                    className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                  >
                    {['basic', 'enhance', 'summarize', 'polish', 'custom'].map(m => (
                      <option key={m} value={m}>{m.toUpperCase()}</option>
                    ))}
                  </select>
                </div>

                {selectedNode.data.mode === 'custom' && (
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">System Instruction</span>
                    <textarea
                      value={selectedNode.data.customInstruction || ''}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'customInstruction', e.target.value)}
                      className="w-full h-32 bg-slate-800/40 border border-slate-700/50 rounded-2xl p-4 text-xs resize-none focus:border-[#90dce1] transition-all"
                      placeholder="Set global AI behavior..."
                    />
                  </div>
                )}

                {/* Structured Outputs */}
                <div className="space-y-4">
                  <span className="text-[10px] text-slate-500 font-black uppercase">{t('structured_outputs')}</span>
                  {selectedNode.data.custom_outputs?.map((o: any, i: number) => (
                    <div key={o.id ?? i} className="p-4 bg-slate-950/40 border border-slate-800 rounded-[24px] space-y-3">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-slate-500 shrink-0">{`ID: ${o.id ?? `out-text${i + 1}`}`}</span>
                        <input
                          value={o.label ?? ''}
                          placeholder={t('field_id_placeholder')}
                          onChange={e => {
                            const n = [...selectedNode.data.custom_outputs];
                            n[i] = { ...n[i], label: e.target.value };
                            onUpdateNodeData(selectedNode.id, 'custom_outputs', n);
                          }}
                          className="bg-transparent border-none text-[10px] font-black text-[#90dce1] flex-1 p-0 focus:ring-0"
                        />
                        <button
                          onClick={() => {
                            const n = selectedNode.data.custom_outputs.filter((_: any, idx: number) => idx !== i);
                            onUpdateNodeData(selectedNode.id, 'custom_outputs', n);
                          }}
                          className="text-slate-600 hover:text-red-400 transition-colors"
                        >
                          <X size={14} />
                        </button>
                      </div>
                      <textarea
                        value={o.description || ''}
                        placeholder={t('custom_instruction_placeholder')}
                        onChange={e => {
                          const n = [...selectedNode.data.custom_outputs];
                          n[i].description = e.target.value;
                          onUpdateNodeData(selectedNode.id, 'custom_outputs', n);
                        }}
                        className="w-full h-16 bg-slate-900/50 border border-slate-800 rounded-xl p-2 text-[10px] text-slate-400 resize-none focus:border-[#90dce1] focus:ring-0 transition-all"
                      />
                    </div>
                  ))}
                  <button
                    onClick={() => {
                      const len = selectedNode.data.custom_outputs?.length ?? 0;
                      const n = [...(selectedNode.data.custom_outputs || []), { id: `out-text${len + 1}`, label: 'Output', description: '' }];
                      onUpdateNodeData(selectedNode.id, 'custom_outputs', n);
                    }}
                    className="w-full py-2 border border-dashed border-slate-700 rounded-xl text-[10px] text-slate-500 uppercase hover:text-[#90dce1] transition-all"
                  >
                    + {t('add_output')}
                  </button>
                </div>
              </div>
            )}

            {/* Aspect Ratio */}
            {(selectedNode.tool_id === 'text-to-image' || selectedNode.tool_id === 'image-to-image' || selectedNode.tool_id.includes('video-gen')) && (
              <div className="space-y-2">
                <span className="text-[10px] text-slate-500 font-black uppercase">{t('aspect_ratio')}</span>
                <select
                  value={selectedNode.data.aspectRatio}
                  onChange={e => onUpdateNodeData(selectedNode.id, 'aspectRatio', e.target.value)}
                  className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                >
                  {selectedNode.tool_id.includes('video-gen')
                    ? ['16:9', '9:16'].map(r => <option key={r} value={r}>{r}</option>)
                    : ['1:1', '4:3', '3:4', '16:9', '9:16'].map(r => <option key={r} value={r}>{r}</option>)
                  }
                </select>
              </div>
            )}

            {/* TTS Settings */}
            {selectedNode.tool_id === 'tts' && (() => {
              const isLightX2V = selectedNode.data.model === 'lightx2v' || selectedNode.data.model?.startsWith('lightx2v');

              if (!isLightX2V) {
                return (
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">{t('voice')}</span>
                    <select
                      value={selectedNode.data.voice || 'Kore'}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'voice', e.target.value)}
                      className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                    >
                      {['Kore', 'Puck', 'Fenrir', 'Charon', 'Zephyr'].map(v => (
                        <option key={v} value={v}>{v}</option>
                      ))}
                    </select>
                  </div>
                );
              }

              return (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Voice Type</span>
                    {loadingVoiceList ? (
                      <div className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700 text-slate-400 text-center">
                        Loading voices...
                      </div>
                    ) : lightX2VVoiceList?.voices && lightX2VVoiceList.voices.length > 0 ? (
                      <div className="space-y-3">
                        {/* Search and Filter */}
                        <div className="flex items-center gap-2">
                          <div className="relative flex-1">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 w-4 h-4 pointer-events-none" />
                            <input
                              type="text"
                              value={voiceSearchQuery}
                              onChange={e => setVoiceSearchQuery(e.target.value)}
                              placeholder={t('search_voices_placeholder')}
                              className="w-full bg-slate-800 rounded-xl pl-10 pr-3 py-2 text-xs border border-slate-700 text-slate-300 placeholder-slate-500 focus:outline-none focus:border-[#90dce1]"
                            />
                          </div>
                          <button
                            onClick={() => setShowVoiceFilter(!showVoiceFilter)}
                            className={`px-3 py-2 bg-slate-800 rounded-xl text-xs border border-slate-700 text-slate-300 hover:bg-slate-700 transition-colors flex items-center gap-1 ${showVoiceFilter ? 'border-[#90dce1] bg-slate-700' : ''}`}
                          >
                            <ListPlus className="w-3 h-3" />
                            Filter
                          </button>
                        </div>

                        {/* Filter Panel */}
                        {showVoiceFilter && (
                          <div className="bg-slate-800 rounded-xl p-3 border border-slate-700">
                            <div className="space-y-1">
                              <span className="text-[10px] text-slate-500 font-black uppercase">Gender</span>
                              <select
                                value={voiceFilterGender}
                                onChange={e => setVoiceFilterGender(e.target.value)}
                                className="w-full bg-slate-900/80 hover:bg-slate-900 rounded-lg px-3 py-1.5 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-300 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:16px_16px] bg-[right_8px_center] bg-no-repeat pr-8 shadow-sm"
                              >
                                <option value="all">All</option>
                                <option value="female">Female</option>
                                <option value="male">Male</option>
                              </select>
                            </div>
                          </div>
                        )}

                        {/* Voice Cards Grid */}
                        <div className="bg-slate-800 rounded-xl border border-slate-700 p-3 max-h-[400px] overflow-y-auto">
                          {filteredVoices.length === 0 ? (
                            <div className="text-center py-8 text-slate-500 text-xs">
                              No voices found matching your criteria
                            </div>
                          ) : (
                            <div className="grid grid-cols-2 gap-2">
                              {filteredVoices.map((voice: any, index: number) => {
                                const uniqueKey = voice.voice_type
                                  ? `${voice.voice_type}_${voice.resource_id || ''}_${index}`
                                  : `voice_${index}`;
                                const isSelected = selectedNode.data.voiceType === voice.voice_type;

                                return (
                                  <label
                                    key={uniqueKey}
                                    className={`relative cursor-pointer m-0 p-3 rounded-lg border-2 transition-all ${
                                      isSelected
                                        ? 'border-[#90dce1] bg-[#90dce1]/10 shadow-lg shadow-[#90dce1]/20'
                                        : 'border-slate-700 bg-slate-900/50 hover:border-slate-600 hover:bg-slate-800'
                                    }`}
                                  >
                                    <input
                                      type="radio"
                                      name="voice-selection"
                                      value={voice.voice_type}
                                      checked={isSelected}
                                      onChange={() => {
                                        onUpdateNodeData(selectedNode.id, 'voiceType', voice.voice_type);
                                        if (voice.resource_id) {
                                          onUpdateNodeData(selectedNode.id, 'resourceId', voice.resource_id);
                                        }
                                      }}
                                      className="sr-only"
                                    />
                                    {voice.version === '2.0' && (
                                      <div className="absolute top-1 right-1 px-1 py-0.5 bg-[#90dce1]/90 text-white text-[7px] font-semibold rounded z-10">
                                        v2.0
                                      </div>
                                    )}
                                    {isSelected && (
                                      <div className="absolute bottom-2 right-2 w-4 h-4 bg-[#90dce1] rounded-full flex items-center justify-center z-20">
                                        <CheckCircle2 className="w-2.5 h-2.5 text-white" />
                                      </div>
                                    )}
                                    <div className="text-xs font-medium text-slate-200 text-center truncate w-full pt-1">
                                      {voice.name || voice.voice_name || voice.voice_type}
                                    </div>
                                  </label>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <input
                        type="text"
                        value={selectedNode.data.voiceType || 'zh_female_vv_uranus_bigtts'}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'voiceType', e.target.value)}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                        placeholder="zh_female_vv_uranus_bigtts"
                      />
                    )}
                  </div>
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Emotion</span>
                    {lightX2VVoiceList?.emotions && lightX2VVoiceList.emotions.length > 0 ? (
                      <select
                        value={selectedNode.data.emotion || ''}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'emotion', e.target.value)}
                        className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                      >
                        <option value="">None</option>
                        {lightX2VVoiceList.emotions.map((emotion: string, index: number) => (
                          <option key={`emotion_${emotion}_${index}`} value={emotion}>{emotion}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={selectedNode.data.emotion || ''}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'emotion', e.target.value)}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                        placeholder="Optional emotion"
                      />
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-2">
                      <span className="text-[10px] text-slate-500 font-black uppercase">Emotion Scale (1-5)</span>
                      <input
                        type="number"
                        min="1"
                        max="5"
                        value={selectedNode.data.emotionScale || 3}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'emotionScale', parseInt(e.target.value))}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-[10px] text-slate-500 font-black uppercase">Speech Rate (-50~100)</span>
                      <input
                        type="number"
                        min="-50"
                        max="100"
                        value={selectedNode.data.speechRate || 0}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'speechRate', parseInt(e.target.value))}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-[10px] text-slate-500 font-black uppercase">Pitch (-12~12)</span>
                      <input
                        type="number"
                        min="-12"
                        max="12"
                        value={selectedNode.data.pitch || 0}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'pitch', parseInt(e.target.value))}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-[10px] text-slate-500 font-black uppercase">Loudness (-50~100)</span>
                      <input
                        type="number"
                        min="-50"
                        max="100"
                        value={selectedNode.data.loudnessRate || 0}
                        onChange={e => onUpdateNodeData(selectedNode.id, 'loudnessRate', parseInt(e.target.value))}
                        className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                      />
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* Voice Clone Settings */}
            {selectedNode.tool_id === 'lightx2v-voice-clone' && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Cloned Voice</span>
                    <button
                      onClick={onShowCloneVoiceModal}
                      className="px-2 py-1 text-[9px] bg-[#90dce1]/20 hover:bg-[#90dce1]/30 text-[#90dce1] rounded border border-[#90dce1]/30 transition-all"
                    >
                      + New
                    </button>
                  </div>
                  {loadingCloneVoiceList ? (
                    <div className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700 text-slate-400 text-center">
                      Loading voices...
                    </div>
                  ) : cloneVoiceList.length > 0 ? (
                    <select
                      value={selectedNode.data.speakerId || ''}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'speakerId', e.target.value)}
                      className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                    >
                      <option value="">Select a cloned voice...</option>
                      {cloneVoiceList.map((voice: any) => (
                        <option key={voice.speaker_id} value={voice.speaker_id}>
                          {voice.name || voice.speaker_id}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-xs text-amber-400">
                      No cloned voices found. Click "+ New" to create one.
                    </div>
                  )}
                  {selectedNode.data.speakerId && (
                    <div className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                      <span className="text-[9px] text-emerald-400 font-mono">ID: {selectedNode.data.speakerId}</span>
                    </div>
                  )}
                </div>
                <div className="space-y-2">
                  <span className="text-[10px] text-slate-500 font-black uppercase">Style</span>
                  <input
                    type="text"
                    value={selectedNode.data.style || '正常'}
                    onChange={e => onUpdateNodeData(selectedNode.id, 'style', e.target.value)}
                    className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                    placeholder="正常"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Speed (0.5~2.0)</span>
                    <input
                      type="number"
                      step="0.1"
                      min="0.5"
                      max="2.0"
                      value={selectedNode.data.speed || 1.0}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'speed', parseFloat(e.target.value))}
                      className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Volume (-12~12)</span>
                    <input
                      type="number"
                      min="-12"
                      max="12"
                      value={selectedNode.data.volume || 0}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'volume', parseFloat(e.target.value))}
                      className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Pitch (-24~24)</span>
                    <input
                      type="number"
                      min="-24"
                      max="24"
                      value={selectedNode.data.pitch || 0}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'pitch', parseFloat(e.target.value))}
                      className="w-full bg-slate-800 rounded-xl p-3 text-xs border border-slate-700"
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-[10px] text-slate-500 font-black uppercase">Language</span>
                    <select
                      value={selectedNode.data.language || 'ZH_CN'}
                      onChange={e => onUpdateNodeData(selectedNode.id, 'language', e.target.value)}
                      className="w-full bg-slate-800/80 hover:bg-slate-800 rounded-xl p-3 text-xs border border-slate-700/60 hover:border-slate-600 text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:20px_20px] bg-[right_12px_center] bg-no-repeat pr-10 shadow-sm"
                    >
                      {['ZH_CN', 'EN_US', 'ZH_CN_SICHUAN', 'ZH_CN_HK'].map(l => (
                        <option key={l} value={l}>{l}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* 连接的节点输出：显示上游节点端口预览文本/媒体，可编辑写回上游节点 */}
            {connectedInputsList.length > 0 && (
              <div className="space-y-4">
                <span className="text-[10px] text-slate-500 font-black uppercase flex items-center gap-2">
                  <MessageSquare size={12} />
                  {lang === 'zh' ? '连接的节点输出' : 'Connected Node Outputs'}
                </span>
                {connectedInputsList.map(({ port, conn, sourceNode, fieldLabel, sourceDataType }) => {
                  const fieldKey = `${conn.source_node_id}:${conn.source_port_id}`;
                  const raw = getOutputValueByPort(sourceNode, conn.source_port_id) ?? sourceOutputs[conn.source_node_id]?.[conn.source_port_id];
                  const isMedia = sourceDataType === DataType.IMAGE || sourceDataType === DataType.AUDIO || sourceDataType === DataType.VIDEO;
                  const mediaUrl = resolvedMediaUrls[fieldKey];
                  const mediaUrls = typeof mediaUrl === 'string' ? [mediaUrl] : (Array.isArray(mediaUrl) ? mediaUrl : []);

                  if (isMedia) {
                    return (
                      <div key={port.id} className="p-4 bg-slate-950/40 border border-slate-800 rounded-[24px] space-y-3">
                        <div className="flex flex-col gap-1">
                          <span className="text-[9px] font-black text-[#90dce1] uppercase">{fieldLabel}</span>
                          <span className="text-[8px] text-slate-500">
                            {lang === 'zh' ? '来自' : 'From'}: {lang === 'zh' ? TOOLS.find(t => t.id === sourceNode.tool_id)?.name_zh : TOOLS.find(t => t.id === sourceNode.tool_id)?.name} → {port.label}
                          </span>
                        </div>
                        {mediaUrls.length === 0 ? (
                          <div className="text-[10px] text-slate-500 py-4">{lang === 'zh' ? '加载中…' : 'Loading…'}</div>
                        ) : sourceDataType === DataType.IMAGE ? (
                          <div className="flex flex-wrap gap-2">
                            {mediaUrls.map((url, i) => (
                              <img key={i} src={url} alt="" className="max-w-full max-h-48 rounded-xl object-contain bg-slate-900/50 border border-slate-800" />
                            ))}
                          </div>
                        ) : sourceDataType === DataType.AUDIO ? (
                          <audio controls src={mediaUrls[0]} className="w-full max-w-full rounded-xl" />
                        ) : (
                          <video controls src={mediaUrls[0]} className="w-full max-w-full rounded-xl" />
                        )}
                      </div>
                    );
                  }

                  const isFileRef = raw && typeof raw === 'object' && (raw as any).file_id && (raw as any).mime_type === 'text/plain';
                  const displayValue = isFileRef
                    ? (resolvedFieldText[fieldKey] ?? (lang === 'zh' ? '加载中…' : 'Loading…'))
                    : (typeof raw === 'string' ? raw : (raw != null ? JSON.stringify(raw, null, 2) : ''));
                  const editedValue = localEdits[fieldKey] !== undefined ? localEdits[fieldKey] : displayValue;
                  const hasChanges = editedValue !== displayValue && displayValue !== (lang === 'zh' ? '加载中…' : 'Loading…');
                  const applying = applyingFieldKey === fieldKey;
                  return (
                    <div key={port.id} className="p-4 bg-slate-950/40 border border-slate-800 rounded-[24px] space-y-3">
                      <div className="flex flex-col gap-1">
                        <span className="text-[9px] font-black text-[#90dce1] uppercase">{fieldLabel}</span>
                        <span className="text-[8px] text-slate-500">
                          {lang === 'zh' ? '来自' : 'From'}: {lang === 'zh' ? TOOLS.find(t => t.id === sourceNode.tool_id)?.name_zh : TOOLS.find(t => t.id === sourceNode.tool_id)?.name} → {port.label}
                        </span>
                      </div>
                      <textarea
                        value={editedValue}
                        onChange={e => setLocalEdits(prev => ({ ...prev, [fieldKey]: e.target.value }))}
                        className="w-full h-32 bg-slate-900/50 border border-slate-800 rounded-xl p-3 text-[10px] text-slate-300 resize-none focus:border-[#90dce1] focus:ring-0 transition-all font-mono"
                        placeholder={t('edit_field_placeholder')}
                      />
                      {hasChanges && (
                        <button
                          onClick={() => handleApplyConnectedField(sourceNode, conn, editedValue)}
                          disabled={applying}
                          className="w-full py-2 rounded-xl text-[10px] font-bold uppercase transition-all bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {applying ? (lang === 'zh' ? '保存中…' : 'Saving…') : (lang === 'zh' ? '应用修改' : 'Apply changes')}
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </aside>
    );
  }

  // Global Inputs Panel (includes workflow description + global inputs)
  return (
    <aside className="w-80 border-l border-slate-800/60 bg-slate-900/40 backdrop-blur-xl flex flex-col flex-1 min-h-0 z-30 p-6 overflow-y-auto">
      <div className="space-y-12">
        {/* Workflow description */}
        <div className="space-y-6">
          <h2 className="text-xs font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
            <Globe size={14} /> {t('workflow_description')}
          </h2>
          <textarea
            value={workflow.description ?? ''}
            onChange={e => onDescriptionChange?.(e.target.value)}
            placeholder={t('workflow_description_placeholder')}
            className="w-full min-h-[80px] bg-slate-900 border border-slate-800 rounded-xl p-3 text-xs resize-y focus:border-[#90dce1] transition-all custom-scrollbar placeholder:text-slate-500"
          />
        </div>
        {/* Workflow tags */}
        <div className="space-y-6">
          <h2 className="text-xs font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
            <Tag size={14} /> {t('workflow_tags')}
          </h2>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag, idx) => (
              <span
                key={`${tag}-${idx}`}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700 text-[11px] text-slate-300"
              >
                {tag}
                <button
                  type="button"
                  onClick={() => onTagsChange?.(tags.filter((_, i) => i !== idx))}
                  className="p-0.5 rounded hover:bg-slate-600 text-slate-400 hover:text-white transition-colors"
                  aria-label={t('remove_tag')}
                >
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={newTagInput}
              onChange={e => setNewTagInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  const v = newTagInput.trim();
                  if (v && !tags.includes(v)) {
                    onTagsChange?.([...tags, v]);
                    setNewTagInput('');
                  }
                }
              }}
              placeholder={t('workflow_tags_placeholder')}
              className="flex-1 min-w-0 bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs focus:border-[#90dce1] transition-all placeholder:text-slate-500"
            />
            <button
              type="button"
              onClick={() => {
                const v = newTagInput.trim();
                if (v && !tags.includes(v)) {
                  onTagsChange?.([...tags, v]);
                  setNewTagInput('');
                }
              }}
              className="shrink-0 px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 hover:border-[#90dce1] hover:text-[#90dce1] transition-all flex items-center gap-1.5 text-xs font-bold uppercase"
            >
              <Plus size={12} /> {t('add_tag')}
            </button>
          </div>
        </div>
        <div className="space-y-6">
          <h2 className="text-xs font-black uppercase tracking-widest text-slate-500 flex items-center gap-2">
            <Target size={14} /> {t('global_inputs')}
          </h2>
          {disconnectedInputs.length === 0 ? (
            <p className="text-[10px] text-slate-600 italic px-2">{t('all_inputs_automated')}</p>
          ) : (
            <div className="space-y-4">
              {disconnectedInputs.map(item => (
                <div key={`${item.nodeId}-${item.port.id}`} className="space-y-2 p-5 bg-slate-800/20 border border-slate-800 rounded-[32px]">
                  <span className="text-[9px] font-black text-slate-500 uppercase px-1">
                    {item.port.label} ({lang === 'zh' ? '针对' : 'for'} {item.toolName})
                  </span>
                  {item.dataType === DataType.TEXT ? (
                    <textarea
                      value={item.isSourceNode ? (workflow.nodes.find(n => n.id === item.nodeId)?.data.value || '') : (workflow.globalInputs[`${item.nodeId}-${item.port.id}`] || '')}
                      onChange={e => item.isSourceNode ? onUpdateNodeData(item.nodeId, 'value', e.target.value) : onGlobalInputChange(item.nodeId, item.port.id, e.target.value)}
                      className="w-full h-24 bg-slate-900 border border-slate-800 rounded-xl p-3 text-xs resize-none focus:border-[#90dce1] transition-all custom-scrollbar"
                    />
                  ) : (
                    <div className="space-y-3">
                      <label className="flex items-center justify-center w-full h-12 border border-dashed border-slate-700 rounded-xl cursor-pointer hover:border-[#90dce1] hover:bg-[#90dce1]/5 transition-all gap-2">
                        <Upload size={14} className="text-slate-500" />
                        <span className="text-[10px] text-slate-500 font-bold uppercase">{lang === 'zh' ? '上传文件' : 'Upload File'}</span>
                        <input
                          type="file"
                          accept={item.dataType === DataType.IMAGE ? "image/*" : item.dataType === DataType.AUDIO ? "audio/*" : "video/*"}
                          className="hidden"
                          disabled={uploadingNodes.has(item.nodeId)}
                          onChange={async (e) => {
                            const file = e.target.files?.[0];
                            if (!file) return;

                            // 需要 workflow.id 才能上传
                            if (!workflow.id || (!workflow.id.startsWith('workflow-') && !workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i))) {
                              console.error('[NodeConfigPanel] Cannot upload file: workflow ID is not available');
                              return;
                            }

                            setUploadingNodes(prev => new Set(prev).add(item.nodeId));

                            try {
                              if (item.isSourceNode) {
                                const node = workflow.nodes.find(n => n.id === item.nodeId);
                                if (!node) return;

                                const tool = TOOLS.find(t => t.id === node.tool_id);
                                if (!tool || tool.category !== 'Input') {
                                  console.error('[NodeConfigPanel] Cannot upload file: node is not an input node');
                                  return;
                                }

                                const outputPort = tool.outputs[0];
                                if (!outputPort) {
                                  console.error('[NodeConfigPanel] Cannot upload file: output port not found');
                                  return;
                                }

                                const dataUrl = await new Promise<string>((resolve, reject) => {
                                  const reader = new FileReader();
                                  reader.onloadend = () => resolve(reader.result as string);
                                  reader.onerror = reject;
                                  reader.readAsDataURL(file);
                                });
                                // 与 Node 一致：以 output_value ?? data.value 为已有值，只更新 output_value，由 useNodeManagement 同步到 data.value
                                const currentValue = node.output_value ?? node.data?.value;
                                const existing = item.dataType === DataType.IMAGE
                                  ? (Array.isArray(currentValue) ? currentValue : currentValue != null ? [currentValue] : [])
                                  : null;
                                const newValue = item.dataType === DataType.IMAGE
                                  ? [...(existing ?? []), dataUrl]
                                  : dataUrl;
                                onUpdateNodeData(item.nodeId, 'output_value', newValue);
                              } else {
                                // 对于非源节点的全局输入，暂时保持原逻辑（base64）
                                const base64 = await new Promise<string>((resolve) => {
                                  const reader = new FileReader();
                                  reader.onloadend = () => resolve(reader.result as string);
                                  reader.readAsDataURL(file);
                                });
                                onGlobalInputChange(item.nodeId, item.port.id, base64);
                              }
                            } catch (err) {
                              console.error('[NodeConfigPanel] Error uploading file:', err);
                            } finally {
                              setUploadingNodes(prev => {
                                const next = new Set(prev);
                                next.delete(item.nodeId);
                                return next;
                              });
                            }
                          }}
                        />
                      </label>
                      {uploadingNodes.has(item.nodeId) ? (
                        <div className="flex items-center gap-1 text-[9px] text-[#90dce1]">
                          <RefreshCw size={8} className="animate-spin" />
                          <span>{lang === 'zh' ? '上传中...' : 'Uploading...'}</span>
                        </div>
                      ) : (
                        <>
                          {item.isSourceNode ? (
                            workflow.nodes.find(n => n.id === item.nodeId)?.data.value && (
                              <div className="text-[9px] text-slate-400">
                                {lang === 'zh' ? '已上传文件' : 'File uploaded'}
                              </div>
                            )
                          ) : (
                            workflow.globalInputs[`${item.nodeId}-${item.port.id}`] && (
                              <div className="text-[9px] text-slate-400">
                                {lang === 'zh' ? '已上传文件' : 'File uploaded'}
                              </div>
                            )
                          )}
                        </>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
