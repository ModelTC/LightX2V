import { useMemo, useCallback, useEffect, useState } from 'react';
import { WorkflowState, WorkflowNode, NodeStatus, DataType } from '../../types';
import { TOOLS } from '../../constants';
import { Language } from '../i18n/useTranslation';
import { historyEntryToDisplayValue } from '../utils/historyEntry';
import { getWorkflowFileText, saveNodeOutputs } from '../utils/workflowFileManager';
import { isStandalone } from '../config/runtimeMode';

/** 单条结果：某节点某次执行的输出；entryId 为 NodeHistoryEntry.id，'current' 表示当前输出 */
export interface ResultEntry {
  runId: string;       // historyEntryId (e.g. node-xxx-ts) or 'current'
  timestamp: number;
  nodeId: string;
  node: WorkflowNode;
  output: any;         // 已归一化，可直接展示
}

function normalizeOutputForDisplay(v: any): any {
  const k = (x: string) => (v as any).kind === x || (v as any).type === x;
  if (v == null) return v;
  if (typeof v === 'object' && !Array.isArray(v)) {
    if (k('data_url') && typeof (v as any)._full_data === 'string') return (v as any)._full_data;
    if (k('url') && typeof (v as any).data === 'string') return (v as any).data;
    if (k('file') && (v as any).file_id) return { ...v, _type: 'file' };
    if (k('text')) return typeof (v as any).text === 'string' ? (v as any).text : ((v as any).data !== undefined ? (v as any).data : v);
    if (k('file')) return (v as any).data !== undefined ? (v as any).data : v;
    if (k('json') && (v as any).data !== undefined) return (v as any).data;
  }
  if (Array.isArray(v)) return v.map(normalizeOutputForDisplay);
  return v;
}

interface UseResultManagementProps {
  workflow: WorkflowState | null;
  showIntermediateResults: boolean;
  expandedOutput: { nodeId: string; fieldId?: string; historyEntryId?: string } | null;
  tempEditValue: string;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  setExpandedOutput: (output: { nodeId: string; fieldId?: string; historyEntryId?: string } | null) => void;
  setIsEditingResult: (editing: boolean) => void;
  lang: Language;
}

export const useResultManagement = ({
  workflow,
  showIntermediateResults,
  expandedOutput,
  tempEditValue,
  setWorkflow,
  setExpandedOutput,
  setIsEditingResult,
  lang
}: UseResultManagementProps) => {
  const [resolvedPortText, setResolvedPortText] = useState<string | null>(null);

  useEffect(() => {
    if (!expandedOutput?.nodeId || !expandedOutput?.fieldId || !workflow?.id) {
      setResolvedPortText(null);
      return;
    }
    const node = workflow.nodes.find(n => n.id === expandedOutput.nodeId);
    const content = node?.output_value;
    const portContent = content && typeof content === 'object' && expandedOutput.fieldId in content
      ? content[expandedOutput.fieldId]
      : null;
    const isTextFileRef = portContent && typeof portContent === 'object' && (portContent as any).file_id && (portContent as any).mime_type === 'text/plain';
    if (!isTextFileRef) {
      setResolvedPortText(null);
      return;
    }
    let cancelled = false;
    getWorkflowFileText(workflow.id, (portContent as any).file_id, expandedOutput.nodeId, expandedOutput.fieldId, (portContent as any).run_id).then((text) => {
      if (!cancelled) setResolvedPortText(text ?? '');
    }).catch(() => { if (!cancelled) setResolvedPortText(null); });
    return () => { cancelled = true; setResolvedPortText(null); };
  }, [expandedOutput?.nodeId, expandedOutput?.fieldId, workflow?.id, workflow?.nodes]);

  const expandedResultData = useMemo(() => {
    if (!expandedOutput || !workflow) return null;
    const node = workflow.nodes.find(n => n.id === expandedOutput.nodeId);
    if (!node) return null;
    const tool = TOOLS.find(t => t.id === node.tool_id);
    let content: any;
    if (expandedOutput.historyEntryId) {
      const raw = workflow.nodeOutputHistory?.[expandedOutput.nodeId];
      const list = Array.isArray(raw) ? raw : (raw && typeof raw === 'object' ? Object.values(raw) : []);
      const entry = list.find((e: any) => e.id === expandedOutput.historyEntryId);
      content = entry ? historyEntryToDisplayValue(entry) : undefined;
    } else {
      content = node.output_value;
    }
    const ck = (x: string) => (content as any).kind === x || (content as any).type === x;
    if (content && typeof content === 'object' && !Array.isArray(content)) {
      if (ck('data_url') && typeof (content as any)._full_data === 'string') content = (content as any)._full_data;
      else if (ck('url') && typeof (content as any).data === 'string') content = (content as any).data;
      else if (ck('file') && (content as any).file_id) {
        // Keep file_url/url so preview can load the file (path must be passed to modal)
        content = { ...(content as any), _type: 'file' };
      } else if (ck('text') || ck('file')) content = (content as any).data !== undefined ? (content as any).data : content;
      else if (ck('json') && (content as any).data !== undefined) content = (content as any).data;
    }
    if (!content && tool?.category === 'Input') content = node.data.value;
    let label = (lang === 'zh' ? tool?.name_zh : tool?.name) || 'Output';
    let type = tool?.outputs?.[0]?.type || DataType.TEXT;
    if (content && typeof content === 'object' && !Array.isArray(content)) {
      const outName = (content as any).output_name;
      const firstVal = outName == null && Object.keys(content).length >= 1 ? (content as any)[Object.keys(content)[0]] : null;
      const resolvedOutName = outName ?? (firstVal && typeof firstVal === 'object' ? (firstVal as any).output_name : null);
      if (resolvedOutName === 'output_image') type = DataType.IMAGE;
      else if (resolvedOutName === 'output_video') type = DataType.VIDEO;
      else if (firstVal && firstVal.kind == 'file' && firstVal.mime_type == "text/plain"){
        type = DataType.TEXT;
        if (!expandedOutput.fieldId) {
          expandedOutput.fieldId = Object.keys(content)[0];
        }
      }
    }
    if (expandedOutput.fieldId && content && typeof content === 'object') {
      content = content[expandedOutput.fieldId];
      // Use port label from custom_outputs for text-generation multi-port
      const customLabel = (node.data?.custom_outputs as { id: string; label?: string }[] | undefined)?.find((o: any) => o.id === expandedOutput.fieldId)?.label;
      label = customLabel ?? expandedOutput.fieldId;
    }
    const isTextFileRef = content && typeof content === 'object' && (content as any).file_id && (content as any).mime_type === 'text/plain';
    const displayContent = isTextFileRef && resolvedPortText != null ? resolvedPortText : content;
    return { content: displayContent, label, type, nodeId: node.id, originalOutput: content };
  }, [expandedOutput, workflow, lang, resolvedPortText]);

  const resultEntries = useMemo((): ResultEntry[] => {
    if (!workflow) return [];
    const entries: ResultEntry[] = [];
    const isTerminal = (n: WorkflowNode) => !workflow.connections.some(c => c.source_node_id === n.id);

    // From nodeOutputHistory: one entry per (nodeId, historyEntry)
    const nodeHistory = workflow.nodeOutputHistory || {};
    for (const nodeId of Object.keys(nodeHistory)) {
      const node = workflow.nodes.find(n => n.id === nodeId);
      if (!node) continue;
      const raw = nodeHistory[nodeId];
      const list = Array.isArray(raw) ? raw : (raw && typeof raw === 'object' ? Object.values(raw) : []);
      for (const entry of list) {
        if (!showIntermediateResults && !isTerminal(node)) continue;
        const historyValue = historyEntryToDisplayValue(entry);
        const output = normalizeOutputForDisplay(historyValue);
        entries.push({
          runId: entry.id,
          runTimestamp: entry.timestamp,
          nodeId,
          node,
          output
        });
      }
    }

    // Current run (running or just finished)
    if (workflow.isRunning) {
      const now = Date.now();
      for (const node of workflow.nodes) {
        if (node.status !== NodeStatus.RUNNING && node.status !== NodeStatus.PENDING && node.status !== NodeStatus.SUCCESS && node.status !== NodeStatus.ERROR) continue;
        if (!showIntermediateResults && !isTerminal(node) && node.status !== NodeStatus.ERROR) continue;
        const raw = node.output_value;
        const tool = TOOLS.find(t => t.id === node.tool_id);
        const output = raw != null ? normalizeOutputForDisplay(raw) : (tool?.category === 'Input' ? node.data?.value : undefined);
        entries.push({ runId: 'current', timestamp: now, nodeId: node.id, node, output: output ?? null });
      }
    }

    return entries.sort((a, b) => b.timestamp - a.timestamp);
  }, [workflow, showIntermediateResults]);

  const activeResultsList = useMemo(() => {
    if (!workflow) return [];
    const data = workflow ? Object.fromEntries(workflow.nodes.filter(n => n.output_value != null).map(n => [n.id, n.output_value])) : {};
    const nodes = workflow.nodes;
    const filtered = nodes.filter(n => {
      if (n.status === NodeStatus.ERROR) return true;
      const hasOutput = data[n.id] != null || (workflow.nodeOutputHistory?.[n.id]?.length ?? 0) > 0;
      const tool = TOOLS.find(t => t.id === n.tool_id);
      const inputHasValue = tool?.category === 'Input' && n.data?.value;
      if (!hasOutput && !inputHasValue) return false;
      return true;
    }).filter(n => {
      if (!showIntermediateResults) {
        const isTerminal = !workflow.connections.some(c => c.source_node_id === n.id);
        return isTerminal || n.status === NodeStatus.ERROR;
      }
      return true;
    });
    const getCompletedTime = (node: WorkflowNode) => {
      if (typeof node.completed_at === 'number') return node.completed_at;
      const list = workflow.nodeOutputHistory?.[node.id];
      if (list?.length) return list[0].timestamp;
      return 0;
    };
    return [...filtered].sort((a, b) => getCompletedTime(b) - getCompletedTime(a));
  }, [workflow, showIntermediateResults]);

  const handleManualResultEdit = useCallback(async () => {
    if (!expandedResultData || !expandedOutput || !workflow) return;

    let finalValue: any = tempEditValue;

    // Try to parse JSON if editing the entire object and it looks like JSON
    if (!expandedOutput.fieldId && (tempEditValue.trim().startsWith('{') || tempEditValue.trim().startsWith('['))) {
      try {
        finalValue = JSON.parse(tempEditValue);
      } catch (e) {}
    }

    const nodeId = expandedOutput.nodeId;
    const fieldId = expandedOutput.fieldId;
    const node = workflow.nodes.find(n => n.id === nodeId);
    const tool = node ? TOOLS.find(t => t.id === node.tool_id) : undefined;
    const portId = fieldId ?? (node?.tool_id === 'text-generation' && (node?.data?.custom_outputs as { id: string }[] | undefined)?.length
      ? (node?.data?.custom_outputs as { id: string }[])[0]?.id
      : tool?.outputs?.[0]?.id);

    // 修改后点保存：文本直接上传为文件并替换为 file ref
    if (workflow.id && !isStandalone() && portId && typeof finalValue === 'string') {
      try {
        const result = await saveNodeOutputs(workflow.id, nodeId, { [portId]: finalValue }, crypto.randomUUID());
        if (result?.[portId]) finalValue = result[portId];
      } catch (e) {
        console.error('[useResultManagement] Save node output failed:', e);
      }
    }

    setWorkflow(prev => {
      if (!prev) return prev;
      const n = prev.nodes.find(x => x.id === nodeId);
      if (!n) return prev;
      const existingNodeOutput = n.output_value;
      let newNodeOutput: any;
      if (portId && typeof existingNodeOutput === 'object' && existingNodeOutput !== null && !Array.isArray(existingNodeOutput)) {
        newNodeOutput = { ...existingNodeOutput, [portId]: finalValue };
      } else if (portId) {
        newNodeOutput = { [portId]: finalValue };
      } else {
        newNodeOutput = finalValue;
      }
      return {
        ...prev,
        isDirty: true,
        nodes: prev.nodes.map(x => x.id === nodeId ? { ...x, output_value: newNodeOutput, status: NodeStatus.SUCCESS } : x)
      };
    });

    setIsEditingResult(false);
  }, [expandedResultData, expandedOutput, tempEditValue, workflow, setWorkflow, setIsEditingResult]);

  return {
    expandedResultData,
    activeResultsList,
    resultEntries,
    handleManualResultEdit
  };
};
