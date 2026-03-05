import React, { useState, useCallback, useEffect, useRef } from 'react';
import { WorkflowState, WorkflowNode, NodeHistoryEntry, Connection, NodeStatus } from '../../types';
import { apiRequest } from '../utils/apiClient';
import { getWorkflowFileByFileId, getWorkflowFileText, getLocalFileDataUrl, persistDataUrlToLocal, uploadLocalUrlAsNodeOutput, isLocalAssetUrlToUpload, saveInputFileViaOutputSave } from '../utils/workflowFileManager';
import { TOOLS } from '../../constants';
import { checkWorkflowOwnership, getCurrentUserId } from '../utils/workflowUtils';
import { workflowSaveQueue } from '../utils/workflowSaveQueue';
import { workflowOfflineQueue } from '../utils/workflowOfflineQueue';
import { isStandalone } from '../config/runtimeMode';
import { getOutputValueByPort, setOutputValueByPort, INPUT_PORT_IDS } from '../utils/outputValuePort';
import { createHistoryEntryFromValue, createHistoryEntryFromPortKeyedOutputValue, normalizeHistoryMap, normalizeAndAggregateHistoryMap } from '../utils/historyEntry';

/** 不再强制规范 text-generation 端口 id：支持 default 的 out-text1/out-text2 与 AI 等指定的自定义 id，原样保留 nodes/connections */
function migrateTextGenerationPortIds(nodes: WorkflowNode[], connections: Connection[]): { nodes: WorkflowNode[]; connections: Connection[] } {
  return { nodes, connections };
}

/** 判断节点输出值是否有效（非空），用于加载后恢复已完成节点的状态，避免刷新后仍显示「排队中」 */
function hasValidOutputValue(outVal: any): boolean {
  if (outVal == null) return false;
  if (typeof outVal !== 'object') return true;
  if (Array.isArray(outVal)) return outVal.length > 0;
  return Object.keys(outVal).length > 0;
}

export const useWorkflow = () => {
  const [myWorkflows, setMyWorkflows] = useState<WorkflowState[]>([]);
  const [workflow, setWorkflow] = useState<WorkflowState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // 使用 ref 存储最新的工作流状态，用于保存时获取最新数据
  const workflowRef = useRef<WorkflowState | null>(null);
  useEffect(() => {
    workflowRef.current = workflow;
  }, [workflow]);

  // Load workflows from API (with localStorage fallback)
  const loadWorkflows = useCallback(async () => {
    setIsLoading(true);
    if (isStandalone()) {
      const saved = localStorage.getItem('omniflow_user_data');
      if (saved) {
        try {
          const workflows = JSON.parse(saved);
          setMyWorkflows(Array.isArray(workflows) ? workflows : []);
          console.log('[Workflow] [Standalone] Loaded workflows from localStorage:', (workflows || []).length);
        } catch (e) {
          console.error('[Workflow] Failed to parse omniflow_user_data:', e);
        }
      }
      setIsLoading(false);
      return;
    }
    try {
      // Try to load from API first
      const response = await apiRequest('/api/v1/workflow/list?page=1&page_size=100');
      if (response.ok) {
        const data = await response.json();
        const workflows: WorkflowState[] = data.workflows.map((wf: any) => ({
          id: wf.workflow_id,
          name: wf.name,
          nodes: wf.nodes || [],
          connections: wf.connections || [],
          isDirty: false,
          isRunning: false,
          globalInputs: wf.global_inputs || {},
          updatedAt: wf.update_t * 1000, // Convert to milliseconds
          visibility: wf.visibility || 'private',
          thumsupCount: wf.thumsup_count ?? 0,
          thumsupLiked: wf.thumsup_liked ?? false
        }));
        setMyWorkflows(workflows);
        console.log('[Workflow] Loaded workflows from API:', workflows.length);
        setIsLoading(false);
        return;
      }
    } catch (error) {
      console.warn('[Workflow] Failed to load workflows from API, falling back to localStorage:', error);
    }

    // Fallback to localStorage
    const saved = localStorage.getItem('omniflow_user_data');
    if (saved) {
      try {
        const workflows = JSON.parse(saved);
        setMyWorkflows(workflows);
        console.log('[Workflow] Loaded workflows from localStorage:', workflows.length);
      } catch (e) {
        console.error('Failed to load workflows from localStorage:', e);
      }
    }
    setIsLoading(false);
  }, []);

  // Load workflows on mount
  useEffect(() => {
    loadWorkflows();
  }, [loadWorkflows]);

  // Save workflow to localStorage (for backup and local development)
  // Define this BEFORE saveWorkflowToDatabase to avoid "Cannot access before initialization" error
  const saveWorkflowToLocal = useCallback(async (current: WorkflowState) => {
    let nodesToSave = current.nodes;
    // 仅前端：先把 data: 存入 IndexedDB 得到 local://，再保存列表，这样 MY 列表缩略图能解析显示
    if (isStandalone()) {
      nodesToSave = await Promise.all(current.nodes.map(async (node) => {
        const tool = TOOLS.find(t => t.id === node.tool_id);
        if (!tool || tool.category !== 'Input' || !node.data?.value) return node;
        const nodeValue = node.data.value;
        const prefix = `wf_${current.id}_${node.id}`;
        if (Array.isArray(nodeValue)) {
          const replaced = await Promise.all(nodeValue.map((v: string, i: number) =>
            typeof v === 'string' && v.startsWith('data:') ? persistDataUrlToLocal(v, `${prefix}_${i}`) : v
          ));
          return { ...node, data: { ...node.data, value: replaced } };
        }
        if (typeof nodeValue === 'string' && nodeValue.startsWith('data:')) {
          const localRef = await persistDataUrlToLocal(nodeValue, `${prefix}_0`);
          return { ...node, data: { ...node.data, value: localRef } };
        }
        return node;
      }));
    }
    // Clean input node values: remove base64 data URLs, keep only file paths (standalone 下已是 local://)
    const cleanedNodes = nodesToSave.map(node => {
      const tool = TOOLS.find(t => t.id === node.tool_id);
      if (!tool || tool.category !== 'Input') {
        // Not an input node, return as-is
        return node;
      }

      const nodeValue = node.data.value;
      if (!nodeValue) {
        return node;
      }

      // Check if it's a base64 data URL
      const isDataURL = (val: any): boolean => {
        if (typeof val === 'string') {
          return val.startsWith('data:');
        }
        if (Array.isArray(val)) {
          return val.some(item => typeof item === 'string' && item.startsWith('data:'));
        }
        return false;
      };

      // Check if it's already a saved path (including standalone local://)
      const isSavedPath = (val: any): boolean => {
        if (typeof val === 'string') {
          return val.startsWith('local://') ||
                 val.startsWith('./assets/workflow/file') ||
                 val.startsWith('./assets/task/') ||
                 val.startsWith('/assets/workflow/file') ||
                 val.startsWith('/assets/task/') ||
                 val.startsWith('/api/v1/workflow/') ||
                 (val.startsWith('http://') || val.startsWith('https://'));
        }
        if (Array.isArray(val)) {
          return val.every(item => typeof item === 'string' && (
            item.startsWith('local://') ||
            item.startsWith('./assets/workflow/file') ||
            item.startsWith('./assets/task/') ||
            item.startsWith('/assets/workflow/file') ||
            item.startsWith('/assets/task/') ||
            item.startsWith('/api/v1/workflow/') ||
            item.startsWith('http://') ||
            item.startsWith('https://')
          ));
        }
        return false;
      };

      // If it's base64 and not saved, remove it (don't save base64 to localStorage)
      if (isDataURL(nodeValue) && !isSavedPath(nodeValue)) {
        // Remove base64 data - user will need to re-upload or it will be saved during execution
        return {
          ...node,
          data: {
            ...node.data,
            value: Array.isArray(nodeValue) ? [] : undefined
          }
        };
      }

      return node;
    });

    const updated = {
      ...current,
      nodes: cleanedNodes, // Use cleaned nodes (no base64 data)
      updatedAt: Date.now(),
      isDirty: false
    };

    setMyWorkflows(prev => {
      const next = prev.find(w => w.id === updated.id)
        ? prev.map(w => w.id === updated.id ? updated : w)
        : [updated, ...prev];

      try {
        localStorage.setItem('omniflow_user_data', JSON.stringify(next));
      } catch (e: any) {
        console.error('Failed to save workflows:', e);
      }
      return next;
    });
  }, []);

  /** 仅保存到本地（localStorage + 列表），不请求后端。用于仅部署前端时 autosave 后端失败后的回退。 */
  const saveWorkflowToLocalOnly = useCallback(async (current: WorkflowState, options?: { name?: string }): Promise<string> => {
    const workflowId = current.id;
    const toSave = options?.name ? { ...current, name: options.name } : current;
    await saveWorkflowToLocal(toSave);
    return workflowId;
  }, [saveWorkflowToLocal]);

  // Save workflow to database (and optionally localStorage)
  const saveWorkflowToDatabase = useCallback(async (current: WorkflowState, options?: { name?: string; description?: string }): Promise<string> => {
    const workflowId = current.id;

    // 使用保存队列，避免并发保存冲突
    return workflowSaveQueue.enqueue(workflowId, async () => {
      setIsSaving(true);
      const latestWorkflow = (workflowRef.current && workflowRef.current.id === workflowId)
        ? workflowRef.current
        : current;

      // preset 工作流（仅前端或有后端）保存时都先分配新 UUID，再按环境保存
      let effectiveId = workflowId;
      let toSave: WorkflowState = options?.name ? { ...latestWorkflow, name: options.name } : latestWorkflow;

      if (isStandalone()) {
        await saveWorkflowToLocal(toSave);
        setIsSaving(false);
        return effectiveId;
      }

      let workflowData: any = null;

      try {
        // 检查工作流是否仍然存在（通过检查是否在 myWorkflows 中）
        // 如果工作流已被删除，跳过保存
        // 注意：对于新创建的工作流（UUID格式），可能不在 myWorkflows 中，这是正常的，不需要查询
        const workflows = myWorkflows;
        const workflowExists = workflows.some(w => w.id === effectiveId);

        // 只有在工作流已存在且不在列表中时才检查（可能是被删除了）
        // 对于新创建的工作流（UUID格式），即使不在列表中也是正常的，不需要查询
        if (!workflowExists && workflowId.match(/^workflow-/)) {
          // 只有 workflow- 前缀的工作流才需要检查（这些是已存在的工作流）
          // 尝试查询工作流是否存在
          try {
            const checkResponse = await apiRequest(`/api/v1/workflow/${workflowId}`);
            if (!checkResponse.ok) {
              throw new Error(`Workflow ${workflowId} was deleted, save task cancelled`);
            }
          } catch (error) {
            if (error instanceof Error && error.message.includes('deleted')) {
              throw error;
            }
            // 其他错误继续处理
          }
        }

        // Extract execution state from nodes (status, error, executionTime)
        // Nodes already contain execution state, so we just save them as-is

        // 输入节点与文本输入一致：执行时才存库，保存工作流时保留 data URL / ref
        const cleanedNodes = [...latestWorkflow.nodes];

        const stripBase64FromOutput = (value: any): any => {
          if (typeof value === 'string') {
            return value.startsWith('data:') ? '' : value;
          }
          if (Array.isArray(value)) {
            return value
              .map(stripBase64FromOutput)
              .filter(v => v !== '' && v !== null && v !== undefined);
          }
          if (value && typeof value === 'object') {
            if (value.type === 'file' || value.file_id) {
              return { kind: (value as any).kind || 'file', file_id: (value as any).file_id, ...((value as any).mime_type != null && { mime_type: (value as any).mime_type }) };
            }
            // Preserve LightX2VResultRef (task_id + output_name) so saved workflow can resolve result_url
            if (((value as any).type === 'task' || (value as any).__type === 'lightx2v_result') && typeof value.task_id === 'string' && typeof value.output_name === 'string') {
              return value;
            }
            const cleaned: Record<string, any> = {};
            Object.entries(value).forEach(([key, val]) => {
              const nextVal = stripBase64FromOutput(val);
              if (nextVal !== '' && nextVal !== null && nextVal !== undefined) {
                cleaned[key] = nextVal;
              }
            });
            return cleaned;
          }
          return value;
        };

        // Helper function to clean base64 data from node data.value
        const cleanNodeDataValue = (nodeData: any): any => {
          if (!nodeData || !nodeData.value) return nodeData;

          const nodeValue = nodeData.value;
          const isDataURL = (val: any): boolean => {
            if (typeof val === 'string') return val.startsWith('data:');
            if (Array.isArray(val)) return val.some(item => typeof item === 'string' && item.startsWith('data:'));
            return false;
          };

          const isSavedPath = (val: any): boolean => {
            if (val && typeof val === 'object' && (val.type === 'file' || val.file_id)) {
              return true;
            }
            if (typeof val === 'string') {
              return val.startsWith('./assets/workflow/file') ||
                     val.startsWith('./assets/task/') ||
                     val.startsWith('/assets/workflow/file') ||
                     val.startsWith('/assets/task/') ||
                     val.startsWith('/api/v1/workflow/') ||
                     (val.startsWith('http://') || val.startsWith('https://'));
            }
            if (Array.isArray(val)) {
              return val.every(item => {
                if (item && typeof item === 'object' && (item.type === 'file' || item.file_id)) return true;
                if (typeof item !== 'string') return false;
                return item.startsWith('./assets/workflow/file') ||
                       item.startsWith('./assets/task/') ||
                       item.startsWith('/assets/workflow/file') ||
                       item.startsWith('/assets/task/') ||
                       item.startsWith('/api/v1/workflow/') ||
                       item.startsWith('http://') ||
                       item.startsWith('https://');
              });
            }
            return false;
          };

          // If it's base64 and not saved, remove it
          if (isDataURL(nodeValue) && !isSavedPath(nodeValue)) {
            return {
              ...nodeData,
              value: Array.isArray(nodeValue) ? [] : undefined
            };
          }

          return nodeData;
        };

        // Clean node output_value to avoid persisting base64 in workflow nodes
        // 输入节点与文本输入一致：执行时才存库，保存工作流时保留 data URL 不 strip
        for (let i = 0; i < cleanedNodes.length; i++) {
          const node = cleanedNodes[i];
          const nodeTool = TOOLS.find(t => t.id === node.tool_id);
          if (nodeTool?.category === 'Input') continue;
          if (node.output_value !== undefined) {
            const cleanedOutputValue = stripBase64FromOutput(node.output_value);
            if (Array.isArray(cleanedOutputValue)) {
              cleanedNodes[i] = {
                ...cleanedNodes[i],
                output_value: cleanedOutputValue.length > 0 ? cleanedOutputValue : undefined
              };
            } else if (cleanedOutputValue && typeof cleanedOutputValue === 'object') {
              cleanedNodes[i] = {
                ...cleanedNodes[i],
                output_value: Object.keys(cleanedOutputValue).length > 0 ? cleanedOutputValue : undefined
              };
            } else {
              cleanedNodes[i] = {
                ...cleanedNodes[i],
                output_value: cleanedOutputValue !== '' && cleanedOutputValue !== null && cleanedOutputValue !== undefined
                  ? cleanedOutputValue
                  : undefined
              };
            }
          }
        }

        // 使用后端时，将输入节点中的本地 URL（如 /assets/girl.png）通过 save/upload 存到后端，数据库以 file 类型存储
        const resolveInputNodeLocalUrls = async (wfId: string, nodes: WorkflowNode[]): Promise<WorkflowNode[]> => {
          const next = [...nodes];
          for (let i = 0; i < next.length; i++) {
            const node = next[i];
            const tool = TOOLS.find(t => t.id === node.tool_id);
            if (!tool || tool.category !== 'Input') continue;
            const portId = INPUT_PORT_IDS[node.tool_id];
            if (!portId) continue;
            const nodeValue = node.data?.value;
            if (!nodeValue) continue;
            const isLocalUrl = (s: string) => typeof s === 'string' && isLocalAssetUrlToUpload(s);
            if (Array.isArray(nodeValue)) {
              const resolved: any[] = [];
              for (let idx = 0; idx < nodeValue.length; idx++) {
                const item = nodeValue[idx];
                if (item && typeof item === 'object' && ((item as any).kind === 'file' || (item as any).type === 'file' || item.file_id)) {
                  resolved.push(item);
                  continue;
                }
                if (!isLocalUrl(item)) {
                  resolved.push(item);
                  continue;
                }
                const ref = await uploadLocalUrlAsNodeOutput(wfId, node.id, portId, item, idx);
                resolved.push(ref || item);
              }
              next[i] = { ...next[i], data: { ...next[i].data, value: resolved } };
            } else if (typeof nodeValue === 'string' && isLocalUrl(nodeValue)) {
              const ref = await uploadLocalUrlAsNodeOutput(wfId, node.id, portId, nodeValue, 0);
              next[i] = { ...next[i], data: { ...next[i].data, value: ref || nodeValue } };
            }
          }
          return next;
        };

        /** 将节点中的 base64（data: URL）上传为 file_id，返回替换后的节点列表，保证写入 DB 的 payload 不含 base64 */
        const uploadBase64InNodesToFileRefs = async (wfId: string, nodes: WorkflowNode[]): Promise<WorkflowNode[]> => {
          const next = [...nodes];
          for (let i = 0; i < next.length; i++) {
            const node = next[i];
            const tool = TOOLS.find(t => t.id === node.tool_id);
            if (!tool || tool.category !== 'Input') continue;
            const portId = INPUT_PORT_IDS[node.tool_id];
            if (!portId) continue;
            // text-input 优先用 data.value（当前编辑内容），保存时上传为 file ref 再写回 output_value；若已是 file ref 且展示文本与文件内容一致则不再上传
            const existingPort = getOutputValueByPort(node, portId);
            if (node.tool_id === 'text-input' && existingPort && typeof existingPort === 'object' && (existingPort as { file_id?: string }).file_id) {
              const displayText = typeof node.data?.value === 'string' ? node.data.value : null;
              if (displayText != null) {
                try {
                  const fileText = await getWorkflowFileText(wfId, (existingPort as { file_id: string }).file_id, node.id, portId, (existingPort as any).run_id);
                  if (fileText === displayText) continue;
                } catch (_e) { /* 比较失败则继续走上传 */ }
              }
            }
            const nodeValue = node.tool_id === 'text-input'
              ? (node.data?.value ?? getOutputValueByPort(node, portId))
              : (getOutputValueByPort(node, portId) ?? node.data?.value);
            if (nodeValue == null) continue;
            const isDataUrl = (v: any) => typeof v === 'string' && v.startsWith('data:');
            if (Array.isArray(nodeValue) && nodeValue.length > 0) {
              const newValue: any[] = [];
              for (const item of nodeValue) {
                if (isDataUrl(item)) {
                  try {
                    const ref = await saveInputFileViaOutputSave(wfId, node.id, portId, item);
                    newValue.push(ref ?? item);
                  } catch (e) {
                    console.warn('[Workflow] uploadBase64InNodesToFileRefs failed for item:', e);
                    newValue.push(item);
                  }
                } else {
                  newValue.push(item);
                }
              }
              const finalVal = portId === 'out-image' ? newValue : newValue[0] ?? null;
              const nextPortKeyed = setOutputValueByPort(node.output_value, node.tool_id, portId, finalVal);
              next[i] = { ...next[i], data: { ...next[i].data, value: finalVal }, output_value: nextPortKeyed };
            } else if (isDataUrl(nodeValue)) {
              try {
                const ref = await saveInputFileViaOutputSave(wfId, node.id, portId, nodeValue);
                if (ref) {
                  const nextPortKeyed = setOutputValueByPort(node.output_value, node.tool_id, portId, ref);
                  next[i] = { ...next[i], data: { ...next[i].data, value: ref }, output_value: nextPortKeyed };
                }
              } catch (e) {
                console.warn('[Workflow] uploadBase64InNodesToFileRefs failed:', e);
              }
            } else if (node.tool_id === 'text-input' && typeof nodeValue === 'string' && nodeValue.trim()) {
              try {
                const dataUrl = `data:text/plain;charset=utf-8;base64,${btoa(unescape(encodeURIComponent(nodeValue)))}`;
                const ref = await saveInputFileViaOutputSave(wfId, node.id, portId, dataUrl);
                if (ref) {
                  const nextPortKeyed = setOutputValueByPort(node.output_value, node.tool_id, portId, ref);
                  next[i] = { ...next[i], data: { ...next[i].data, value: ref }, output_value: nextPortKeyed };
                }
              } catch (e) {
                console.warn('[Workflow] uploadBase64InNodesToFileRefs text-input failed:', e);
              }
            }
          }
          return next;
        };

        let nodesToSave = cleanedNodes;
        const isUuid = (id: string) => /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
        if (!isStandalone() && effectiveId && isUuid(effectiveId)) {
          nodesToSave = await resolveInputNodeLocalUrls(effectiveId, cleanedNodes);
        }

        // 先上传再保存：有 UUID 时把 base64 上传为 file_id，再写入 DB；无 UUID（新建）时先剥离 base64，创建后再上传并 PUT
        if (!isStandalone() && effectiveId && isUuid(effectiveId)) {
          nodesToSave = await uploadBase64InNodesToFileRefs(effectiveId, nodesToSave);
          setWorkflow(prev => (prev && prev.id === workflowId ? { ...prev, nodes: nodesToSave } : prev));
          if (workflowRef.current && workflowRef.current.id === workflowId) {
            workflowRef.current = { ...workflowRef.current, nodes: nodesToSave };
          }
        } else if (!isStandalone() && effectiveId) {
          // 创建且尚无 UUID：剥离 base64，避免 POST body 带 base64
          nodesToSave = nodesToSave.map(n => {
            const nodeTool = TOOLS.find(t => t.id === n.tool_id);
            if (nodeTool?.category === 'Input') {
              const cleanedData = cleanNodeDataValue(n.data);
              const cleanedOutput = stripBase64FromOutput(n.output_value);
              const outVal = (cleanedOutput === '' || (Array.isArray(cleanedOutput) && cleanedOutput.length === 0)) ? undefined : cleanedOutput;
              return { ...n, data: cleanedData, output_value: outVal };
            }
            return n;
          });
        }

        // 使用最新的工作流状态构建保存数据（不再写入 history_metadata，执行历史按节点存 nodeOutputHistory / data_store）
        const MAX_NODE_HISTORY = 20;
        const trimNodeOutputHistory = (hist: Record<string, any[]>) => {
          if (!hist || typeof hist !== 'object') return {};
          const out: Record<string, any[]> = {};
          for (const [nodeId, entries] of Object.entries(hist)) {
            if (Array.isArray(entries)) {
              out[nodeId] = entries.slice(0, MAX_NODE_HISTORY);
            }
          }
          return out;
        };

        // 有后端时：输入节点只持久化 output_value（file ref），不存 data.value；仅前端时：不写后端，保留 data.value（base64/纯文本）以便本地持久化
        const nodesForPayload = nodesToSave.map((n: WorkflowNode) => {
          const tool = TOOLS.find(t => t.id === n.tool_id);
          if (tool?.category === 'Input') {
            if (isStandalone()) return n;
            const { value: _dropped, ...restData } = n.data || {};
            return { ...n, data: restData };
          }
          return n;
        });

        workflowData = {
          name: options?.name || latestWorkflow.name,
          description: options?.description ?? latestWorkflow.description ?? '',
          nodes: nodesForPayload,
          connections: latestWorkflow.connections,
          visibility: latestWorkflow.visibility || 'private',
          tags: latestWorkflow.tags ?? [],
          node_output_history: trimNodeOutputHistory(latestWorkflow.nodeOutputHistory ?? {})
        };

        // Check if workflow exists in database and user owns it (else create new UUID first)
        const isSavedWorkflow = workflowId && workflowId.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
        const isTemporaryId = workflowId && (workflowId.startsWith('temp-') || workflowId.startsWith('flow-'));

        const currentUserId = getCurrentUserId();
        const { owned } = await checkWorkflowOwnership(effectiveId, currentUserId);
        if (!owned) {
          console.log('[Workflow] Workflow not owned (preset/404), creating new with ID:', effectiveId);
          return effectiveId;
        }

        // For saved workflows (UUID) that user owns, try to update first
        if (isSavedWorkflow && owned) {
          const updateResponse = await apiRequest(`/api/v1/workflow/${effectiveId}/update`, {
            method: 'POST',
            body: JSON.stringify(workflowData)
          });

          if (updateResponse.ok) {
            console.log('[Workflow] Workflow updated in database:', effectiveId);
            await saveWorkflowToLocal(latestWorkflow);
            return effectiveId;
          } else {
            const errorText = await updateResponse.text();
            if (updateResponse.status === 404) {
              console.log('[Workflow] Workflow not found on update, creating new one');
            } else {
              throw new Error(`Failed to update workflow: ${errorText}`);
            }
          }
        }
        return effectiveId;
      } catch (error) {
        console.error('[Workflow] Failed to save workflow to database:', error);

        // Check if it's a network error
        const isNetworkError = error instanceof TypeError ||
                               (error instanceof Error && error.message.includes('fetch')) ||
                               !navigator.onLine;

        if (isNetworkError && workflowData) {
          // Add to offline queue
          console.log('[Workflow] Network error detected, adding to offline queue');
          workflowOfflineQueue.addTask(workflowId, workflowData);
        }

        // Fallback to localStorage
        await saveWorkflowToLocal(latestWorkflow);
        throw error;
      } finally {
        setIsSaving(false);
      }
    });
  }, [saveWorkflowToLocal, setWorkflow, myWorkflows]);

  // Load a single workflow from API (or from localStorage in standalone)
  const loadWorkflow = useCallback(async (workflowId: string): Promise<{ workflow: WorkflowState } | null> => {
    try {
      if (isStandalone()) {
        const saved = localStorage.getItem('omniflow_user_data');
        if (!saved) {
          setWorkflow(null);
          return null;
        }
        let workflows: WorkflowState[];
        try {
          workflows = JSON.parse(saved);
          if (!Array.isArray(workflows)) workflows = [];
        } catch {
          setWorkflow(null);
          return null;
        }
        const found = workflows.find((w: WorkflowState) => w.id === workflowId);
        if (!found) {
          setWorkflow(null);
          return null;
        }
        const { nodes: nodesMigrated, connections: connectionsMigrated } = migrateTextGenerationPortIds(found.nodes, found.connections || []);
        const foundMigrated = { ...found, nodes: nodesMigrated, connections: connectionsMigrated };
        const loadedOutputs: Record<string, any> = {};
        for (const node of foundMigrated.nodes) {
          const tool = TOOLS.find(t => t.id === node.tool_id);
          if (!tool || tool.category !== 'Input') continue;
          const nodeValue = node.data?.value;
          if (!nodeValue) continue;
          const resolveOne = async (path: string): Promise<string | null> => {
            if (typeof path !== 'string') return null;
            if (path.startsWith('local://')) return getLocalFileDataUrl(path);
            return path;
          };
          if (Array.isArray(nodeValue)) {
            const urls = await Promise.all(nodeValue.map((p: string) => resolveOne(p)));
            loadedOutputs[node.id] = urls.filter((u): u is string => u != null);
          } else {
            const url = await resolveOne(nodeValue);
            if (url) loadedOutputs[node.id] = url;
          }
        }
        const nodesWithOutputs = foundMigrated.nodes.map((n: WorkflowNode) => {
          const outVal = loadedOutputs[n.id] ?? n.output_value;
          const isInput = TOOLS.find(t => t.id === n.tool_id)?.category === 'Input';
          if (isInput) {
            const canonical = outVal ?? n.data?.value;
            return { ...n, output_value: canonical, data: { ...n.data, value: canonical } };
          }
          // 刷新后重载：若节点已有有效输出（如 text-generation/语音合成/语音克隆已执行完），恢复为 SUCCESS 并清除 run_state，避免显示「排队中」
          const completed = hasValidOutputValue(outVal);
          return {
            ...n,
            output_value: outVal,
            ...(completed ? { status: NodeStatus.SUCCESS, run_state: { status: 'SUCCEED', subtasks: [] } } : {}),
          };
        });
        const workflowWithOutputs = { ...foundMigrated, nodes: nodesWithOutputs };
        setWorkflow(workflowWithOutputs);
        return { workflow: workflowWithOutputs };
      }

      const response = await apiRequest(`/api/v1/workflow/${workflowId}`);
      if (response.ok) {
        const wf = await response.json();

        const nodesRaw = wf.nodes || [];
        const { nodes, connections: connectionsMigrated } = migrateTextGenerationPortIds(nodesRaw, wf.connections || []);
        const loadPromises: Promise<unknown>[] = [];
        const loadedOutputs: Record<string, any> = {};

        for (const node of nodes) {
          const tool = TOOLS.find(t => t.id === node.tool_id);
          if (!tool) continue;

          if (tool.category === 'Input') {
            const portId = INPUT_PORT_IDS[node.tool_id];
            const nodeValue = node.data?.value ?? (portId ? getOutputValueByPort(node, portId) : undefined) ?? node.output_value;
            if (!nodeValue) continue;

            // text-input 的 file ref：拉取纯文本用于预览，不转为 data URL，且保留 output_value 为 file ref
            const textFileRef = node.tool_id === 'text-input' && nodeValue && typeof nodeValue === 'object' && (nodeValue as { file_id?: string }).file_id;
            if (textFileRef) {
              const fileId = (nodeValue as { file_id: string }).file_id;
              const runId = (nodeValue as { run_id?: string }).run_id;
              loadPromises.push(
                getWorkflowFileText(wf.workflow_id, fileId, node.id, portId, runId).then((text) => {
                  if (text != null) loadedOutputs[node.id] = text;
                  else loadedOutputs[node.id] = nodeValue;
                }).catch((err) => {
                  console.error(`[Workflow] Error loading text file for node ${node.id}:`, err);
                  loadedOutputs[node.id] = nodeValue;
                })
              );
              continue;
            }

            const isFilePath = (val: any): boolean => {
              if (typeof val === 'string') {
                return val.startsWith('/api/v1/workflow/') ||
                       val.startsWith('./assets/') ||
                       val.startsWith('http://') ||
                       val.startsWith('https://');
              }
              if (Array.isArray(val)) {
                return val.every(item => typeof item === 'string' && (
                  item.startsWith('/api/v1/workflow/') ||
                  item.startsWith('./assets/') ||
                  item.startsWith('http://') ||
                  item.startsWith('https://')
                ));
              }
              return false;
            };

            const pathOrUrlFromVal = (val: any): string | null => {
              if (typeof val === 'string') return val;
              if (val && typeof val === 'object' && typeof val.file_url === 'string') return val.file_url;
              return null;
            };
            const pathToFetch = pathOrUrlFromVal(nodeValue);
            if (pathToFetch && (pathToFetch.startsWith('/api/v1/workflow/') || pathToFetch.startsWith('./assets/') || pathToFetch.startsWith('http://') || pathToFetch.startsWith('https://'))) {
              if (pathToFetch.startsWith('/api/v1/workflow/')) {
                const match = pathToFetch.match(/\/api\/v1\/workflow\/([^/]+)\/file\/(.+)$/);
                if (match) {
                  const [, workflowId, fileId] = match;
                  loadPromises.push(
                    getWorkflowFileByFileId(workflowId, fileId).then(dataUrl => {
                      if (dataUrl) loadedOutputs[node.id] = dataUrl;
                      else loadedOutputs[node.id] = nodeValue;
                    }).catch(err => {
                      console.error(`[Workflow] Error loading input file for node ${node.id}:`, err);
                      loadedOutputs[node.id] = nodeValue;
                    })
                  );
                } else {
                  loadedOutputs[node.id] = nodeValue;
                }
              } else {
                loadedOutputs[node.id] = nodeValue;
              }
            } else if (isFilePath(nodeValue)) {
              if (Array.isArray(nodeValue)) {
                const fileLoadPromises = nodeValue.map((path: string) => {
                  if (path.startsWith('/api/v1/workflow/')) {
                    const match = path.match(/\/api\/v1\/workflow\/([^/]+)\/file\/(.+)$/);
                    if (match) {
                      const [, workflowId, fileId] = match;
                      return getWorkflowFileByFileId(workflowId, fileId);
                    }
                  }
                  return Promise.resolve(path);
                });
                loadPromises.push(
                  Promise.all(fileLoadPromises).then(dataUrls => {
                    loadedOutputs[node.id] = dataUrls.filter((url: any) => url !== null);
                    return dataUrls;
                  }).catch(err => {
                    console.error(`[Workflow] Error loading input files for node ${node.id}:`, err);
                    loadedOutputs[node.id] = nodeValue;
                    return nodeValue;
                  })
                );
              } else {
                if (typeof nodeValue === 'string' && nodeValue.startsWith('/api/v1/workflow/')) {
                  const match = nodeValue.match(/\/api\/v1\/workflow\/([^/]+)\/file\/(.+)$/);
                  if (match) {
                    const [, workflowId, fileId] = match;
                    loadPromises.push(
                      getWorkflowFileByFileId(workflowId, fileId).then(dataUrl => {
                        if (dataUrl) {
                          loadedOutputs[node.id] = dataUrl;
                        } else {
                          loadedOutputs[node.id] = nodeValue;
                        }
                      }).catch(err => {
                        console.error(`[Workflow] Error loading input file for node ${node.id}:`, err);
                        loadedOutputs[node.id] = nodeValue;
                      })
                    );
                  } else {
                    loadedOutputs[node.id] = nodeValue;
                  }
                } else {
                  loadedOutputs[node.id] = nodeValue;
                }
              }
            } else {
              loadedOutputs[node.id] = nodeValue;
            }
            continue;
          }

          if (node.output_value != null) {
            loadedOutputs[node.id] = node.output_value;
          }
        }

        if (loadPromises.length > 0) {
          await Promise.race([
            Promise.all(loadPromises),
            new Promise(resolve => setTimeout(resolve, 5000))
          ]);
        }

        const extraInfo = wf.extra_info || {};
        if (extraInfo.active_outputs && typeof extraInfo.active_outputs === 'object') {
          Object.entries(extraInfo.active_outputs).forEach(([nodeId, value]) => {
            if (loadedOutputs[nodeId] == null) {
              loadedOutputs[nodeId] = value;
            }
          });
        }

        let normalizedHistory = normalizeAndAggregateHistoryMap(wf.node_output_history || {});
        if ((!wf.node_output_history || Object.keys(normalizedHistory).length === 0) && wf.data_store?.outputs) {
          const dataStoreOutputs = wf.data_store.outputs;
          const legacyHistory: Record<string, NodeHistoryEntry[]> = {};
          for (const nodeId of Object.keys(dataStoreOutputs)) {
            const portData = dataStoreOutputs[nodeId];
            if (!portData || typeof portData !== 'object') continue;
            const portIds = Object.keys(portData);
            if (portIds.length === 0) continue;
            const histories = portIds.map((pid: string) => {
              const h = portData[pid]?.history;
              return Array.isArray(h) ? h : [];
            });
            const minLen = Math.min(...histories.map(h => h.length));
            if (minLen === 0) continue;
            const entries: NodeHistoryEntry[] = [];
            for (let i = 0; i < minLen; i++) {
              const firstRef = histories[0][i];
              const ts = firstRef?.metadata?.created_at ?? Math.floor(Date.now() / 1000) * 1000;
              const timestamp = typeof ts === 'number' ? (ts < 1e12 ? ts * 1000 : ts) : Date.now();
              const id = `node-${nodeId}-${timestamp}`;
              let output: any;
              if (portIds.length === 1) {
                output = histories[0][i];
              } else {
                output = {} as Record<string, any>;
                portIds.forEach((pid, idx) => {
                  output[pid] = histories[idx][i];
                });
              }
              const entry = portIds.length === 1
                ? createHistoryEntryFromValue({ id, timestamp, value: output, params: {}, portId: portIds[0] })
                : createHistoryEntryFromPortKeyedOutputValue({ id, timestamp, output_value: output, params: {} });
              if (entry) entries.push(entry);
            }
            if (entries.length > 0) {
              legacyHistory[nodeId] = entries;
            }
          }
          normalizedHistory = normalizeAndAggregateHistoryMap(legacyHistory);
        }

        const createT = wf.create_t ?? wf.create_at;
        const createAtMs = createT != null ? (createT < 1e12 ? createT * 1000 : createT) : undefined;

        const nodesWithOutputs = nodes.map((n: WorkflowNode) => {
          const outVal = loadedOutputs[n.id] ?? n.output_value;
          const tool = TOOLS.find(t => t.id === n.tool_id);
          const isInput = tool?.category === 'Input';
          if (isInput) {
            const portVal = getOutputValueByPort(n, INPUT_PORT_IDS[n.tool_id]);
            const isTextInputFileRef = n.tool_id === 'text-input' && portVal && typeof portVal === 'object' && (portVal as { file_id?: string }).file_id;
            if (isTextInputFileRef) {
              return { ...n, output_value: n.output_value, data: { ...n.data, value: loadedOutputs[n.id] ?? n.data?.value } };
            }
            if (n.tool_id === 'image-input') {
              return n;
            }
            const canonical = outVal ?? n.data?.value;
            return { ...n, output_value: canonical, data: { ...n.data, value: canonical } };
          }
          // 刷新后重载：若节点已有有效输出（如 text-generation/语音合成/语音克隆已执行完），恢复为 SUCCESS 并清除 run_state，避免显示「排队中」
          const completed = hasValidOutputValue(outVal);
          return {
            ...n,
            output_value: outVal,
            ...(completed ? { status: NodeStatus.SUCCESS, run_state: { status: 'SUCCEED', subtasks: [] } } : {}),
          };
        });
        const workflow: WorkflowState = {
          id: wf.workflow_id,
          name: wf.name,
          description: wf.description ?? '',
          nodes: nodesWithOutputs,
          connections: connectionsMigrated,
          isDirty: false,
          isRunning: false,
          globalInputs: wf.global_inputs || {},
          nodeOutputHistory: normalizedHistory,
          chatHistory: wf.chat_history || [],
          updatedAt: wf.update_t * 1000,
          createAt: createAtMs,
          visibility: wf.visibility || 'private',
          thumsupCount: wf.thumsup_count ?? 0,
          thumsupLiked: wf.thumsup_liked ?? false,
          userId: wf.user_id,
          tags: wf.tags ?? []
        };
        console.log('[Workflow] loaded workflow:', workflow);
        setWorkflow(workflow);
        return { workflow };
      } else {
        console.warn('[Workflow] Failed to load workflow from API, trying localStorage');
        // Fallback to localStorage：用节点的 output_value 恢复执行结果预览（不依赖 nodeOutputHistory）
        const saved = localStorage.getItem('omniflow_user_data');
        if (saved) {
          const workflows = JSON.parse(saved);
          const found = workflows.find((w: WorkflowState) => w.id === workflowId);
          if (found) {
            const { nodes: nodesMigrated, connections: connectionsMigrated } = migrateTextGenerationPortIds(found.nodes, found.connections || []);
            const normalizedFound: WorkflowState = {
              ...found,
              nodes: nodesMigrated,
              connections: connectionsMigrated,
              nodeOutputHistory: normalizeAndAggregateHistoryMap(found.nodeOutputHistory ?? {})
            };
            setWorkflow(normalizedFound);
            return { workflow: normalizedFound };
          }
        }
        return null;
      }
    } catch (error) {
      console.error('[Workflow] Error loading workflow:', error);
      return null;
    }
  }, [setWorkflow]);

  // Delete workflow
  const deleteWorkflow = useCallback(async (id: string) => {
    workflowSaveQueue.removeTask(id);
    workflowOfflineQueue.removeTask(id);

    if (!isStandalone()) {
      try {
        if (id.startsWith('workflow-') || id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)) {
          const response = await apiRequest(`/api/v1/workflow/${id}`, {
            method: 'DELETE'
          });
          if (response.ok) {
            console.log('[Workflow] Workflow deleted from database:', id);
          }
        }
      } catch (error) {
        console.warn('[Workflow] Failed to delete workflow from database, deleting from localStorage:', error);
      }
    }

    setMyWorkflows(prev => {
      const next = prev.filter(w => w.id !== id);
      try {
        localStorage.setItem('omniflow_user_data', JSON.stringify(next));
      } catch (e) {
        console.error('Failed to delete workflow from localStorage:', e);
      }
      return next;
    });
  }, [setMyWorkflows]);

  const updateWorkflowVisibility = useCallback(async (id: string, visibility: 'private' | 'public') => {
    if (!id) return;
    if (visibility !== 'private' && visibility !== 'public') return;

    if (isStandalone()) {
      setMyWorkflows(prev => {
        const next = prev.map(w => w.id === id ? { ...w, visibility } : w);
        try {
          localStorage.setItem('omniflow_user_data', JSON.stringify(next));
        } catch (e) {
          console.error('Failed to save workflows to localStorage:', e);
        }
        return next;
      });
      setWorkflow(prev => prev ? (prev.id === id ? { ...prev, visibility, isDirty: true } : prev) : prev);
      return;
    }

    try {
      const response = await apiRequest(`/api/v1/workflow/${id}/update`, {
        method: 'POST',
        body: JSON.stringify({ visibility })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to update visibility: ${errorText}`);
      }

      setMyWorkflows(prev => {
        const next = prev.map(w => w.id === id ? { ...w, visibility } : w);
        try {
          localStorage.setItem('omniflow_user_data', JSON.stringify(next));
        } catch (e) {
          console.error('Failed to save workflows to localStorage:', e);
        }
        return next;
      });

      setWorkflow(prev => prev ? (prev.id === id ? { ...prev, visibility, isDirty: true } : prev) : prev);
    } catch (error) {
      console.error('[Workflow] Failed to update workflow visibility:', error);
      throw error;
    }
  }, [setMyWorkflows, setWorkflow]);

  /**
   * 确保当前工作流属于当前用户；若不拥有（预设/404），则先创建新 UUID 并更新状态。
   * 若为预设工作流，则调用/create 创建用户自己的 workflow
   * 若为公共工作流，则调用/copy 复制公共工作流
   * @returns 实际应使用的 workflow id（原 id 或新建的 UUID）
   */
  const ensureWorkflowOwned = useCallback(async (current: WorkflowState): Promise<string> => {
    if (isStandalone()) return current.id;
    const { owned, isPreset } = await checkWorkflowOwnership(current.id, getCurrentUserId());
    if (owned) return current.id;
    const newId = typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    const MAX_NODE = 20;
    const trimNodeHist = (hist: Record<string, any[]>) => {
      if (!hist || typeof hist !== 'object') return {};
      const out: Record<string, any[]> = {};
      for (const [nodeId, entries] of Object.entries(hist)) {
        if (Array.isArray(entries)) out[nodeId] = entries.slice(0, MAX_NODE);
      }
      return out;
    };

    // preset workflow, create it
    if (isPreset) {
      const createData = {
        name: current.name,
        description: current.description ?? '',
        nodes: current.nodes,
        connections: current.connections,
        workflow_id: newId,
        tags: current.tags ?? [],
        node_output_history: trimNodeHist(current.nodeOutputHistory ?? {})
      };
      const res = await apiRequest('/api/v1/workflow/create', { method: 'POST', body: JSON.stringify(createData) });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Failed to create workflow: ${text}`);
      } else {
        console.log('[Workflow] ensureWorkflowOwned: created new preset workflow', newId);
        // save input nodes and refresh
        // saveWorkflowToDatabase({ ...createData, id: newId }, { name: current.name });
      }
    }
    // public workflow, copy it
    else {
      const copyResponse = await apiRequest(`/api/v1/workflow/${current.id}/copy`, {
        method: 'POST',
        body: JSON.stringify({ workflow_id: newId })
      });
      if (!copyResponse.ok) {
        const text = await copyResponse.text();
        throw new Error(`Failed to copy workflow: ${text}`);
      } else {
        console.log('[Workflow] ensureWorkflowOwned: copied public workflow', newId);
      }
    }
    setWorkflow(prev => (prev && prev.id === current.id ? { ...prev, id: newId } : prev));
    if (typeof window !== 'undefined' && window.history?.replaceState) {
      window.history.replaceState(null, '', `#workflow/${newId}`);
    }
    return newId;
  }, [setWorkflow]);

  /** 获取当前最新工作流（用于 AI 等需要“实时”状态的场景，避免闭包拿到旧状态） */
  const getWorkflow = useCallback(() => workflowRef.current, []);

  /**
   * 从后端拉取 workflow 并仅合并 nodes 的 output_value 与 node_output_history，不替换整棵 state（用于输入节点 save / 任务节点 save 后刷新展示）。
   */
  const refreshWorkflowFromBackend = useCallback(async (workflowId: string): Promise<void> => {
    if (isStandalone()) return;
    try {
      const response = await apiRequest(`/api/v1/workflow/${workflowId}`);
      if (!response.ok) return;
      const wf = await response.json();
      const serverNodes: WorkflowNode[] = Array.isArray(wf.nodes) ? wf.nodes : [];
      const serverHistory: Record<string, NodeHistoryEntry[]> =
        typeof wf.node_output_history === 'object' && wf.node_output_history != null ? wf.node_output_history : {};
      setWorkflow((prev) => {
        if (!prev || prev.id !== workflowId) return prev;
        return {
          ...prev,
          nodes: prev.nodes.map((n) => {
            const sn = serverNodes.find((x: WorkflowNode) => x.id === n.id);
            if (!sn || sn.output_value === undefined) return n;
            return { ...n, output_value: sn.output_value };
          }),
          nodeOutputHistory: { ...(prev.nodeOutputHistory ?? {}), ...serverHistory },
        };
      });
    } catch (e) {
      console.warn('[Workflow] refreshWorkflowFromBackend failed:', workflowId, e);
    }
  }, []);

  return {
    myWorkflows,
    workflow,
    setWorkflow,
    getWorkflow,
    setMyWorkflows,
    saveWorkflowToLocal,
    saveWorkflowToLocalOnly,
    saveWorkflowToDatabase,
    loadWorkflow,
    loadWorkflows,
    refreshWorkflowFromBackend,
    deleteWorkflow,
    updateWorkflowVisibility,
    ensureWorkflowOwned,
    isSaving,
    isLoading
  };
};
