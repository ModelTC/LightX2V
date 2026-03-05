/**
 * Workflow 运行逻辑（仅前端）
 *
 * 1. 输入节点：所有输入类型节点的 text/image/video/audio 通过
 *    POST /api/v1/workflow/{workflow_id}/node/{node_id}/output/{port_id}/save 保存到后端，
 *    然后调用远端 GET workflow 更新节点的 output_value、node_history_output。
 *
 * 2. 拓扑顺序运行：按拓扑序分批执行，前序依赖执行完毕后才执行下一批；提交任务时，
 *    若输入连接前序节点输出，仅用 [workflow_id, node_id, port_id] 提交给后端，不解析为 url/bytes。
 *
 * 3. Task 节点完成时：调用 /save 将该节点输出的 { task_id, output_name } 等保存到后端，
 *    再调用远端 get_workflow 更新 output_value、node_history_output。
 *
 * 4. 前端资源展示：任意节点统一通过
 *    GET /api/v1/workflow/{workflow_id}/node/{node_id}/output/{port_id}/url?task_id=xxx&file_id=xxx
 *    获取 URL 后嵌入 <img>/<video>/<audio> 等（由 Node.tsx / workflowFileManager 处理，本 hook 不负责）。
 */

import React, { useCallback, useRef } from 'react';
import { flushSync } from 'react-dom';
import { WorkflowState, Connection, NodeStatus } from '../../types';
import { TOOLS } from '../../constants';
import {
  geminiText,
  geminiImage,
  geminiSpeech,
  geminiVideo,
  lightX2VTask,
  lightX2VTTS,
  lightX2VVoiceCloneTTS,
  deepseekText,
  doubaoText,
  ppchatGeminiText,
  getLightX2VConfigForModel,
  lightX2VCancelTask,
  lightX2VTaskQuery,
  type WorkflowRefsPayload,
} from '../../services/geminiService';
import { isStandalone } from '../config/runtimeMode';
import { removeGeminiWatermark } from '../../services/watermarkRemover';
import { useTranslation, Language } from '../i18n/useTranslation';
import {
  saveNodeOutputs,
  saveInputFileViaOutputSave,
  uploadLocalUrlAsNodeOutput,
  isLocalAssetUrlToUpload,
  getWorkflowFileByFileId,
  getWorkflowFileText,
} from '../utils/workflowFileManager';
import { apiRequest } from '../utils/apiClient';
import {
  resolveLightX2VResultRef as resolveLightX2VResultRefUtil,
  isLightX2VResultRef as isLightX2VResultRefUtil,
  type LightX2VResultRef,
} from '../utils/resultRef';
import {
  getOutputValueByPort,
  setOutputValueByPort,
  INPUT_PORT_IDS,
  isPortKeyedOutputValue,
  toFileRefForOutputValue,
} from '../utils/outputValuePort';
import { workflowSaveQueue } from '../utils/workflowSaveQueue';
import { workflowOfflineQueue } from '../utils/workflowOfflineQueue';
import { checkWorkflowOwnership, getCurrentUserId } from '../utils/workflowUtils';
import { getAssetPath, getAssetBasePath } from '../utils/assetPath';

export type { LightX2VResultRef };
export const isLightX2VResultRef = isLightX2VResultRefUtil;

const MAX_NODE_HISTORY = 20;

// --- 工具函数 ----------------------------------------------------------------

/** 从 connections 组装 [workflow_id, node_id, port_id] 供提交任务使用；仅用前序节点引用，不解析为 url/bytes */
function buildWorkflowRefs(
  workflowId: string,
  incomingConns: Connection[],
  toolInputs: { id: string }[]
): WorkflowRefsPayload | undefined {
  const refs: WorkflowRefsPayload = { workflowId };
  const tuple = (c: Connection): [string, string, string] => [
    workflowId,
    c.source_node_id,
    c.source_port_id,
  ];
  for (const port of toolInputs) {
    const conns = incomingConns.filter((c) => c.target_port_id === port.id);
    if (conns.length === 0) continue;
    if (port.id === 'in-text') {
      refs.prompt = tuple(conns[0]);
    } else if (port.id === 'in-image') {
      refs.input_image = conns.length === 1 ? tuple(conns[0]) : conns.map(tuple);
    } else if (port.id === 'in-image-start') {
      refs.input_image = tuple(conns[0]);
    } else if (port.id === 'in-image-end') {
      refs.input_last_frame = tuple(conns[0]);
    } else if (port.id === 'in-audio') {
      refs.input_audio = tuple(conns[0]);
    } else if (port.id === 'in-video') {
      refs.input_video = tuple(conns[0]);
    }
  }
  const hasRefs =
    refs.prompt !== undefined ||
    refs.input_image !== undefined ||
    refs.input_audio !== undefined ||
    refs.input_video !== undefined ||
    refs.input_last_frame !== undefined;
  return hasRefs ? refs : undefined;
}

/** 多路 in-image 扁平为一维 */
function flattenImageInput(raw: any): string[] {
  if (raw == null) return [];
  if (Array.isArray(raw)) {
    const flat = raw.flatMap((item: any) =>
      Array.isArray(item) ? item : item != null ? [item] : []
    );
    return flat.filter((x: any) => x != null && typeof x === 'string') as string[];
  }
  return typeof raw === 'string' ? [raw] : [];
}

/** 图片 URL/路径转 data URL（供需要 base64 的 Gemini 等使用） */
async function ensureImageInputsAsDataUrls(imgs: string[]): Promise<string[]> {
  return Promise.all(
    imgs.map(async (img) => {
      if (typeof img !== 'string') return img;
      if (img.startsWith('data:')) return img;
      const isHttp = img.startsWith('http');
      const isLocalAsset =
        img.includes('/assets/task/result') || img.includes('/assets/workflow/file');
      const isSameOrigin =
        typeof window !== 'undefined' &&
        isHttp &&
        img.startsWith(window.location.origin);
      if (isHttp && !isSameOrigin && !isLocalAsset) return img;
      if (!img.startsWith('/')) return img;
      const url = isHttp ? img : getAssetPath(img);
      try {
        const res = await fetch(url);
        const blob = await res.blob();
        return await new Promise<string>((resolve, reject) => {
          const r = new FileReader();
          r.onloadend = () => resolve(r.result as string);
          r.onerror = reject;
          r.readAsDataURL(blob);
        });
      } catch (e) {
        console.warn('[WorkflowExecution] Failed to fetch image:', img, e);
        return img;
      }
    })
  );
}

// --- Props & state ----------------------------------------------------------

interface UseWorkflowExecutionProps {
  workflow: WorkflowState | null;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  isPausedRef: React.MutableRefObject<boolean>;
  setIsPaused: (paused: boolean) => void;
  runningTaskIdsRef: React.MutableRefObject<Map<string, string>>;
  abortControllerRef: React.MutableRefObject<AbortController | null>;
  getLightX2VConfig: (workflow: WorkflowState | null) => { url: string; token: string };
  setValidationErrors: (errors: { message: string; type: 'ENV' | 'INPUT' }[]) => void;
  setGlobalError: (error: { message: string; details?: string } | null) => void;
  updateNodeData: (nodeId: string, key: string, value: any) => void;
  voiceList: { lightX2VVoiceList: any };
  lang: Language;
  onSaveExecutionToLocal?: (workflowState: WorkflowState) => Promise<void>;
  saveWorkflowBeforeRun?: (workflow: WorkflowState) => Promise<string | void>;
  refreshWorkflowFromBackend?: (workflowId: string, options?: { getCurrentWorkflow?: () => WorkflowState | null }) => Promise<WorkflowState | null>;
  /** 执行前、执行后各调用一次，用于同步 workflow（含 node 参数）到后端 */
  updateWorkflowSync?: (workflowId: string, workflow: WorkflowState) => Promise<void>;
  /** 获取当前 workflow（用于执行结束后 refresh 时合并，再调 update 避免旧 output_value 覆盖远端） */
  getWorkflow?: () => WorkflowState | null;
}

// --- Hook 实现 --------------------------------------------------------------

function useWorkflowExecutionImpl({
  workflow,
  setWorkflow,
  isPausedRef,
  setIsPaused,
  runningTaskIdsRef,
  abortControllerRef,
  getLightX2VConfig,
  setValidationErrors,
  setGlobalError,
  updateNodeData,
  voiceList,
  lang,
  onSaveExecutionToLocal,
  saveWorkflowBeforeRun,
  refreshWorkflowFromBackend,
  updateWorkflowSync,
  getWorkflow,
}: UseWorkflowExecutionProps) {
  const { t } = useTranslation(lang);

  const getDescendants = useCallback((nodeId: string, connections: Connection[]): Set<string> => {
    const descendants = new Set<string>();
    const stack = [nodeId];
    while (stack.length > 0) {
      const current = stack.pop()!;
      connections
        .filter((c) => c.source_node_id === current)
        .forEach((c) => {
          if (!descendants.has(c.target_node_id)) {
            descendants.add(c.target_node_id);
            stack.push(c.target_node_id);
          }
        });
    }
    return descendants;
  }, []);

  const getTopologicalBatches = useCallback(
    (nodesToRunIds: Set<string>, connections: Connection[]): string[][] => {
      const nodeIds = Array.from(nodesToRunIds);
      const inDegree: Record<string, number> = {};
      nodeIds.forEach((id) => {
        inDegree[id] = 0;
      });
      connections.forEach((c) => {
        if (
          nodesToRunIds.has(c.source_node_id) &&
          nodesToRunIds.has(c.target_node_id) &&
          c.source_node_id !== c.target_node_id
        ) {
          inDegree[c.target_node_id] = (inDegree[c.target_node_id] ?? 0) + 1;
        }
      });
      const batches: string[][] = [];
      const added = new Set<string>();
      while (added.size < nodeIds.length) {
        const layer = nodeIds.filter((id) => !added.has(id) && inDegree[id] === 0);
        if (layer.length === 0) break;
        batches.push(layer);
        layer.forEach((id) => added.add(id));
        layer.forEach((sourceId) => {
          connections
            .filter((c) => c.source_node_id === sourceId)
            .forEach((c) => {
              const tId = c.target_node_id;
              if (nodesToRunIds.has(tId) && inDegree[tId] != null) inDegree[tId]--;
            });
        });
      }
      if (added.size < nodeIds.length) {
        const remaining = nodeIds.filter((id) => !added.has(id));
        batches.push(remaining);
      }
      return batches;
    },
    []
  );

  const validateWorkflow = useCallback(
    (nodesToRunIds: Set<string>): { message: string; type: 'ENV' | 'INPUT' }[] => {
      if (!workflow) return [];
      const errors: { message: string; type: 'ENV' | 'INPUT' }[] = [];

      const usesLightX2V = Array.from(nodesToRunIds).some((id) => {
        const node = workflow.nodes.find((n) => n.id === id);
        return (
          node &&
          (node.tool_id.includes('lightx2v') ||
            node.tool_id.includes('video') ||
            node.tool_id === 'avatar-gen' ||
            ((node.tool_id === 'text-to-image' || node.tool_id === 'image-to-image') &&
              node.data.model?.startsWith('Qwen')))
        );
      });

      if (usesLightX2V && !isStandalone()) {
        const config = getLightX2VConfig(workflow);
        const apiClient = (window as any).__API_CLIENT__;
        if (apiClient) {
          if (!config.token?.trim()) {
            errors.push({ message: t('missing_env_msg'), type: 'ENV' });
          }
        } else {
          if (!config.url?.trim() || !config.token?.trim()) {
            errors.push({ message: t('missing_env_msg'), type: 'ENV' });
          }
        }
      }

      workflow.nodes.forEach((node) => {
        if (!nodesToRunIds.has(node.id)) return;
        const tool = TOOLS.find((t) => t.id === node.tool_id);
        if (!tool) return;

        if (tool.category === 'Input') {
          const val = node.data.value;
          const isEmpty = (Array.isArray(val) && val.length === 0) || !val;
          if (isEmpty) {
            errors.push({
              message: `${node.name ?? (lang === 'zh' ? tool.name_zh : tool.name)} (${t('executing')})`,
              type: 'INPUT',
            });
          }
          return;
        }

        tool.inputs.forEach((port) => {
          const isOptional =
            port.label.toLowerCase().includes('optional') ||
            port.label.toLowerCase().includes('(opt)');
          if (isOptional) return;
          const isConnected = workflow.connections.some(
            (c) => c.target_node_id === node.id && c.target_port_id === port.id
          );
          const hasGlobalVal = !!workflow.globalInputs[`${node.id}-${port.id}`]?.toString().trim();
          if (!isConnected && !hasGlobalVal) {
            errors.push({
              message: `${node.name ?? (lang === 'zh' ? tool.name_zh : tool.name)} -> ${port.label}`,
              type: 'INPUT',
            });
          }
        });

        if (node.tool_id === 'lightx2v-voice-clone' && !node.data.speakerId) {
          errors.push({
            message: `${node.name ?? (lang === 'zh' ? tool.name_zh : tool.name)}: ${t('select_cloned_voice')}`,
            type: 'INPUT',
          });
        }
      });

      return errors;
    },
    [workflow, getLightX2VConfig, t, lang]
  );

  type QueueJob = {
    id: string;
    startNodeId?: string;
    onlyOne?: boolean;
    cancelled?: boolean;
    affectedNodeIds: Set<string>;
  };
  const executionQueueRef = useRef<QueueJob[]>([]);
  const runningJobCountRef = useRef(0);
  const MAX_CONCURRENT_JOBS = 3;
  const runningJobsByJobIdRef = useRef<
    Map<
      string,
      {
        affectedNodeIds: Set<string>;
        taskIdsByNodeId: Map<string, string>;
        nodeIdToIsCloud: Map<string, boolean>;
        abortController: AbortController;
        pollIntervalId?: number;
      }
    >
  >(new Map());
  const [pendingRunNodeIds, setPendingRunNodeIds] = React.useState<string[]>([]);
  const workflowRef = useRef<WorkflowState | null>(workflow);
  const effectiveWorkflowIdRef = useRef<string | null>(null);

  /** port/save 与执行后同步：入队后按顺序依次执行，避免多 Node 同时请求 /port/save */
  type PortSaveTask = { wfId: string; nodeId: string; toSave: Record<string, any>; runId: string };
  const portSaveQueueRef = useRef<PortSaveTask[]>([]);
  const portSaveProcessorRunningRef = useRef(false);
  /** 用于「批量执行完一批后等待 port save 排空」的 Promise resolve */
  const portSaveDrainResolveRef = useRef<(() => void) | null>(null);

  const processPortSaveQueue = useCallback(async () => {
    if (portSaveProcessorRunningRef.current || portSaveQueueRef.current.length === 0) return;
    portSaveProcessorRunningRef.current = true;
    while (portSaveQueueRef.current.length > 0) {
      const task = portSaveQueueRef.current.shift()!;
      try {
        const newOutputs = await saveNodeOutputs(task.wfId, task.nodeId, task.toSave, task.runId);
        console.log('[WorkflowExecution] New outputs:', newOutputs);
        if (refreshWorkflowFromBackend) await refreshWorkflowFromBackend(task.wfId);
      } catch (e) {
        console.warn('[WorkflowExecution] Port save queue task failed:', e);
      }
    }
    portSaveProcessorRunningRef.current = false;
    // 通知「等待排空」的调用方：当前批次产生的 port save 已全部处理完，可开始下一轮
    if (portSaveDrainResolveRef.current) {
      portSaveDrainResolveRef.current();
      portSaveDrainResolveRef.current = null;
    }
  }, [refreshWorkflowFromBackend]);

  /** 等待 port save 队列排空（当前批次产生的所有 port/save 调用完成后再 resolve），用于批量执行时「一批结束后再开始下一批」 */
  const waitForPortSaveQueueDrain = useCallback((): Promise<void> => {
    if (portSaveQueueRef.current.length === 0 && !portSaveProcessorRunningRef.current) {
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      portSaveDrainResolveRef.current = resolve;
    });
  }, []);

  const enqueuePortSave = useCallback(
    (task: PortSaveTask) => {
      portSaveQueueRef.current.push(task);
      processPortSaveQueue();
    },
    [processPortSaveQueue]
  );

  React.useEffect(() => {
    workflowRef.current = workflow;
  }, [workflow]);

  const getAffectedNodeIds = useCallback(
    (startNodeId?: string, onlyOne?: boolean): Set<string> => {
      if (!workflow) return new Set();
      if (startNodeId) {
        if (onlyOne) return new Set([startNodeId]);
        const desc = getDescendants(startNodeId, workflow.connections);
        desc.add(startNodeId);
        return desc;
      }
      return new Set(
        workflow.nodes
          .filter((n) => TOOLS.find((t) => t.id === n.tool_id)?.category !== 'Input')
          .map((n) => n.id)
      );
    },
    [workflow, getDescendants]
  );

  type JobInfo = {
    affectedNodeIds: Set<string>;
    taskIdsByNodeId: Map<string, string>;
    nodeIdToIsCloud: Map<string, boolean>;
    abortController: AbortController;
    pollIntervalId?: number;
  };
  const refreshPendingNodeIds = useCallback(() => {
    const fromQueue = executionQueueRef.current
      .filter((j) => !j.cancelled)
      .flatMap((j) => [...j.affectedNodeIds]);
    const fromRunning = (
      Array.from(runningJobsByJobIdRef.current.values()) as JobInfo[]
    ).flatMap((j) => [...j.affectedNodeIds]);
    setPendingRunNodeIds([...new Set([...fromQueue, ...fromRunning])]);
  }, []);

  const resolveLightX2VResultRef = useCallback(
    (
      ref: LightX2VResultRef,
      context?: { workflow_id?: string; node_id?: string; port_id?: string }
    ) => resolveLightX2VResultRefUtil(ref, context),
    []
  );

  /**
   * 阶段一：保存所有输入节点输出到后端，并刷新 workflow。
   * 保证 input 节点的 text/image/video/audio 通过 /save 落库，再 GET workflow 更新 output_value、node_output_history。
   */
  const saveInputNodesAndRefresh = useCallback(
    async (
      wfId: string,
      nodesToRunIds: Set<string>,
      connections: Connection[],
      sessionOutputs: Record<string, any>
    ): Promise<void> => {
      if (isStandalone() || !refreshWorkflowFromBackend) return;

      const nodesNeededAsInputs = new Set<string>();
      connections.forEach((c) => {
        if (nodesToRunIds.has(c.target_node_id) && !nodesToRunIds.has(c.source_node_id)) {
          nodesNeededAsInputs.add(c.source_node_id);
        }
      });

      for (const node of workflow?.nodes ?? []) {
        const tool = TOOLS.find((t) => t.id === node.tool_id);
        if (!tool || tool.category !== 'Input') continue;
        const inScope = nodesToRunIds.has(node.id) || nodesNeededAsInputs.has(node.id);
        if (!inScope) continue;

        const portId = INPUT_PORT_IDS[node.tool_id];
        if (!portId) continue;

        if (node.tool_id === 'text-input') {
          const rawPort = getOutputValueByPort(node, 'out-text');
          const textVal = node.data?.value ?? rawPort;
          const str = typeof textVal === 'string' ? textVal : textVal?.text ?? '';
          if (str && typeof str === 'string') {
            const dataUrl = `data:text/plain;charset=utf-8;base64,${btoa(unescape(encodeURIComponent(str)))}`;
            try {
              const saveRef = await saveInputFileViaOutputSave(wfId, node.id, 'out-text', dataUrl);
              const ref = toFileRefForOutputValue(saveRef);
              if (ref) {
                sessionOutputs[node.id] = setOutputValueByPort(
                  node.output_value,
                  node.tool_id,
                  'out-text',
                  ref
                );
              }
            } catch (e) {
              console.warn('[WorkflowExecution] Save text-input failed:', e);
            }
          } else if (rawPort != null) {
            sessionOutputs[node.id] =
              typeof node.output_value === 'object' && node.output_value !== null
                ? node.output_value
                : setOutputValueByPort(node.output_value, node.tool_id, 'out-text', rawPort);
          }
          continue;
        }

        const nodeValue = getOutputValueByPort(node, portId);
        if (nodeValue == null) continue;
        console.log('[WorkflowExecution saveInputNodesAndRefresh] Node value:', node, portId, nodeValue);
        const isDataUrl = (v: any) => typeof v === 'string' && v.startsWith('data:');
        const isLocalUrl = (s: string) => typeof s === 'string' && isLocalAssetUrlToUpload(s);
        const isImage = portId === 'out-image';
        const arr = Array.isArray(nodeValue) ? nodeValue : [nodeValue].filter(Boolean);

        if (arr.length > 0 && (isImage || arr.length === 1)) {
          const newOutputValue: any[] = [];
          for (const item of arr) {
            try {
              if (isDataUrl(item)) {
                const saveRef = await saveInputFileViaOutputSave(wfId, node.id, portId, item);
                const ref = toFileRefForOutputValue(saveRef);
                newOutputValue.push(ref || item);
              }
              else if (isLocalUrl(item)) {
                const saveRef = await uploadLocalUrlAsNodeOutput(wfId, node.id, portId, item, 0);
                const ref = toFileRefForOutputValue(saveRef)
                newOutputValue.push(ref || item);
              }
              else {
                newOutputValue.push(item);
              }
            } catch (e) {
              console.warn('[WorkflowExecution] Save input file failed:', e);
              newOutputValue.push(item);
            }
          }
          const finalVal = isImage ? newOutputValue : newOutputValue[0] ?? null;
          sessionOutputs[node.id] = setOutputValueByPort(
            node.output_value,
            node.tool_id,
            portId,
            finalVal
          );
        } else if (!Array.isArray(nodeValue)) {
          try {
            let ref: any | null = nodeValue;
            if (isDataUrl(nodeValue)) {
                const saveRef = await saveInputFileViaOutputSave(wfId, node.id, portId, nodeValue);
                ref = toFileRefForOutputValue(saveRef);
            } else if(isLocalUrl(nodeValue)) {
                const saveRef = await uploadLocalUrlAsNodeOutput(wfId, node.id, portId, nodeValue, 0);
                ref = toFileRefForOutputValue(saveRef);
            }
            sessionOutputs[node.id] = setOutputValueByPort(
              node.output_value,
              node.tool_id,
              portId,
              ref
            );
          } catch (e) {
            console.warn('[WorkflowExecution] Save input file failed:', e);
            sessionOutputs[node.id] = setOutputValueByPort(
              node.output_value,
              node.tool_id,
              portId,
              nodeValue
            );
          }
        } else {
          sessionOutputs[node.id] = isPortKeyedOutputValue(node.output_value)
            ? node.output_value
            : setOutputValueByPort(node.output_value, node.tool_id, portId, nodeValue);
        }
      }

      await refreshWorkflowFromBackend(wfId);
    },
    [workflow, refreshWorkflowFromBackend]
  );

  /**
   * 从 sessionOutputs 或 workflow 取某端口值；若为 file ref 且需要内容（如 Gemini），再按需解析。
   * 提交任务时若使用 workflowRefs，则不需要解析为 bytes，仅用 [workflow_id, node_id, port_id]。
   */
  const getInputValueForPort = useCallback(
    async (
      wfId: string,
      sourceNodeId: string,
      sourcePortId: string,
      sessionOutputs: Record<string, any>,
      resolveFileRef: boolean
    ): Promise<any> => {
      const sourceNode = workflow?.nodes.find((n) => n.id === sourceNodeId);
      const out = sessionOutputs[sourceNodeId] ?? sourceNode?.output_value;
      if (out == null) return undefined;

      let raw =
        (typeof out === 'object' && sourcePortId in out ? out[sourcePortId] : undefined) ??
        getOutputValueByPort(
          sourceNode ? { ...sourceNode, output_value: out } : { output_value: out, tool_id: '', data: {} },
          sourcePortId
        );

      if (isLightX2VResultRef(raw)) raw = await resolveLightX2VResultRefUtil(raw);
      if (!resolveFileRef) return raw;

      const isRef = (v: any) =>
        v && typeof v === 'object' && ((v as any).kind === 'file' || (v as any).file_id);
      const isTextFileRef = (v: any) =>
        isRef(v) &&
        ((v as any).mime_type === 'text/plain' ||
          (v as any).ext === '.txt' ||
          (v as any).ext === 'txt');
      if (isRef(raw)) {
        if (isTextFileRef(raw)) {
          return (
            (await getWorkflowFileText(
              wfId,
              (raw as any).file_id,
              sourceNodeId,
              sourcePortId,
              (raw as any).run_id
            )) ?? raw
          );
        }
        return (
          (await getWorkflowFileByFileId(
            wfId,
            (raw as any).file_id,
            (raw as any).mime_type,
            (raw as any).ext,
            sourceNodeId,
            sourcePortId,
            (raw as any).run_id
          )) || (raw as any).file_url ||
          raw
        );
      }
      if (Array.isArray(raw) && raw.some((v: any) => isRef(v))) {
        return Promise.all(
          raw.map((v: any) =>
            isRef(v)
              ? getWorkflowFileByFileId(
                  wfId,
                  (v as any).file_id,
                  (v as any).mime_type,
                  (v as any).ext,
                  sourceNodeId,
                  sourcePortId,
                  (v as any).run_id
                ).then((url) => url || v)
              : Promise.resolve(v)
          )
        );
      }
      return raw;
    },
    [workflow]
  );

  const executeOneRun = useCallback(
    async (jobId: string, startNodeId?: string, onlyOne?: boolean) => {
      if (!workflow) return;

      const jobInfo = runningJobsByJobIdRef.current.get(jobId);
      if (!jobInfo) return;

      const wfId = effectiveWorkflowIdRef.current ?? workflow.id ?? '';
      effectiveWorkflowIdRef.current = null;
      const jobAbortSignal = jobInfo.abortController.signal;

      const registerTaskId = (nodeId: string, taskId: string, isCloud?: boolean) => {
        runningTaskIdsRef.current.set(nodeId, taskId);
        jobInfo.taskIdsByNodeId.set(nodeId, taskId);
        jobInfo.nodeIdToIsCloud.set(nodeId, isCloud ?? false);
        if (!jobInfo.pollIntervalId) {
          jobInfo.pollIntervalId = window.setInterval(async () => {
            const j = runningJobsByJobIdRef.current.get(jobId);
            if (!j) return;
            for (const [nid, tid] of j.taskIdsByNodeId) {
              try {
                let run_state: { status: string; subtasks?: any[] };
                if (isStandalone() || j.nodeIdToIsCloud.get(nid)) {
                  const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
                  const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
                  const info = await lightX2VTaskQuery(cloudUrl, cloudToken, String(tid));
                  run_state = { status: info.status || 'UNKNOWN', subtasks: [] };
                } else {
                  const res = await apiRequest(`/api/v1/task/query?task_id=${tid}`, { method: 'GET' });
                  const data = res.ok
                    ? ((await res.json().catch(() => ({}))) as { status?: string; subtasks?: any[] })
                    : {};
                  run_state = { status: data.status || 'UNKNOWN', subtasks: data.subtasks || [] };
                }
                setWorkflow((prev) =>
                  prev
                    ? { ...prev, nodes: prev.nodes.map((n) => (n.id === nid ? { ...n, run_state } : n)) }
                    : null
                );
                if (['SUCCEED', 'FAILED', 'CANCEL', 'CANCELLED'].includes(run_state.status)) {
                  j.taskIdsByNodeId.delete(nid);
                  j.nodeIdToIsCloud.delete(nid);
                }
              } catch (_) {}
            }
            if (j.taskIdsByNodeId.size === 0 && j.pollIntervalId) {
              clearInterval(j.pollIntervalId);
              j.pollIntervalId = undefined;
            }
          }, 2000);
        }
      };

      setIsPaused(false);
      isPausedRef.current = false;

      let nodesToRunIds: Set<string>;
      if (startNodeId) {
        if (onlyOne) nodesToRunIds = new Set([startNodeId]);
        else {
          nodesToRunIds = getDescendants(startNodeId, workflow.connections);
          nodesToRunIds.add(startNodeId);
        }
      } else {
        // 整图运行：跳过已成功的节点，只运行尚无结果或已失败的节点
        nodesToRunIds = new Set(
          workflow.nodes
            .filter((n) => TOOLS.find((t) => t.id === n.tool_id)?.category !== 'Input')
            .filter((n) => n.status !== NodeStatus.SUCCESS)
            .map((n) => n.id)
        );
      }

      setWorkflow((prev) =>
        prev
          ? {
              ...prev,
              nodes: prev.nodes.map((n) =>
                nodesToRunIds.has(n.id) ? { ...n, status: NodeStatus.IDLE, error: undefined } : n
              ),
            }
          : null
      );

      const errors = validateWorkflow(nodesToRunIds);
      if (errors.length > 0) {
        setValidationErrors(errors);
        return;
      }
      setValidationErrors([]);

      const requiresUserApiKey = workflow.nodes
        .filter((n) => nodesToRunIds.has(n.id))
        .some(
          (n) =>
            n.tool_id.includes('video') ||
            n.tool_id === 'avatar-gen' ||
            n.data.model === 'gemini-3-pro-image-preview' ||
            n.data.model === 'gemini-2.5-flash-image'
        );
      if (requiresUserApiKey) {
        try {
          if (!(await (window as any).aistudio?.hasSelectedApiKey?.())) {
            await (window as any).aistudio?.openSelectKey?.();
          }
        } catch (_) {}
      }

      setWorkflow((prev) =>
        prev
          ? {
              ...prev,
              isRunning: true,
              nodes: prev.nodes.map((n) =>
                nodesToRunIds.has(n.id)
                  ? { ...n, status: NodeStatus.PENDING, error: undefined }
                  : n
              ),
            }
          : null
      );

      const sessionOutputs: Record<string, any> = {};
      workflow.nodes.forEach((n) => {
        if (!nodesToRunIds.has(n.id) && n.output_value != null) {
          sessionOutputs[n.id] = n.output_value;
        }
      });

      const nodesNeededAsInputs = new Set<string>();
      workflow.connections.forEach((c) => {
        if (nodesToRunIds.has(c.target_node_id) && !nodesToRunIds.has(c.source_node_id)) {
          nodesNeededAsInputs.add(c.source_node_id);
        }
      });
      nodesNeededAsInputs.forEach((nodeId) => {
        const n = workflow.nodes.find((nn) => nn.id === nodeId);
        if (n?.output_value != null && sessionOutputs[nodeId] === undefined) {
          sessionOutputs[nodeId] = n.output_value;
        }
      });

      const hasValidDbId =
        wfId &&
        (wfId.startsWith('workflow-') ||
          /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(wfId));

      // 执行前：同步 workflow（含 node 参数）到后端
      if (!isStandalone() && hasValidDbId && updateWorkflowSync) {
        const wf = workflowRef.current ?? workflow;
        console.log('[WorkflowExecution] Update workflow before execution:', wf);
        if (wf) await updateWorkflowSync(wfId, wf);
      }

      // 阶段一：输入节点通过 /save 保存到后端，再 GET workflow 更新
      await saveInputNodesAndRefresh(wfId, nodesToRunIds, workflow.connections, sessionOutputs);

      const lightX2VConfig = getLightX2VConfig(workflow);
      const shouldSaveOutputs = hasValidDbId && !isStandalone();
      const cancelledByFailure = new Set<string>();
      const topologicalBatches = getTopologicalBatches(nodesToRunIds, workflow.connections);
      // 本次批量/单节点执行共用同一个 run_id，供各节点 port/save 使用
      const runId = crypto.randomUUID();

      for (const batchNodeIds of topologicalBatches) {
        const batch = batchNodeIds
          .filter((id) => !cancelledByFailure.has(id))
          .map((id) => workflow.nodes.find((n) => n.id === id))
          .filter((n): n is NonNullable<typeof n> => n != null);
        if (batch.length === 0) continue;

        while (isPausedRef.current) {
          await new Promise((r) => setTimeout(r, 100));
          if (!workflow?.isRunning) return;
        }

        const executionPromises = batch.map(async (node) => {
          const tool = TOOLS.find((t) => t.id === node.tool_id)!;
          const incomingConns = workflow.connections.filter((c) => c.target_node_id === node.id);
          const nodeStart = performance.now();

          setWorkflow((prev) =>
            prev
              ? {
                  ...prev,
                  nodes: prev.nodes.map((n) =>
                    n.id === node.id ? { ...n, status: NodeStatus.RUNNING, start_time: nodeStart } : n
                  ),
                }
              : null
          );

          try {
            const nodeInputs: Record<string, any> = {};
            for (const port of tool.inputs) {
              const conns = incomingConns.filter((c) => c.target_port_id === port.id);
              if (conns.length > 0) {
                const values = await Promise.all(
                  conns.map((c) =>
                    getInputValueForPort(
                      wfId,
                      c.source_node_id,
                      c.source_port_id,
                      sessionOutputs,
                      true
                    )
                  )
                );
                const filtered = values.filter((v) => v !== undefined);
                if (filtered.length === 1) nodeInputs[port.id] = filtered[0];
                else if (filtered.length > 0) nodeInputs[port.id] = filtered;
              } else {
                nodeInputs[port.id] = workflow.globalInputs[`${node.id}-${port.id}`];
              }
            }

            const ensureStringIfObjectFromTextGen = (raw: any) => {
              const sourceNode = workflow.nodes.find((n) => n.id === node.id);
              if (
                sourceNode?.tool_id === 'text-generation' &&
                typeof raw === 'object' &&
                raw !== null &&
                !Array.isArray(raw)
              )
                return JSON.stringify(raw);
              return raw;
            };
            if (nodeInputs['in-text'] != null && typeof nodeInputs['in-text'] !== 'string') {
              nodeInputs['in-text'] = ensureStringIfObjectFromTextGen(nodeInputs['in-text']);
            }

            let result: any;
            const model = node.data.model;

            switch (node.tool_id) {
              case 'text-input':
                result = node.data.value ?? '';
                break;
              case 'image-input': {
                const rawImage = node.output_value ?? node.data.value;
                const imageValue = Array.isArray(rawImage)
                  ? rawImage
                  : rawImage != null
                    ? [rawImage]
                    : [];
                result = imageValue;
                break;
              }
              case 'audio-input':
                result = node.output_value ?? node.data.value;
                break;
              case 'video-input':
                result = node.output_value ?? node.data.value;
                break;
              case 'web-search':
                result = await geminiText(
                  nodeInputs['in-text'] || 'Search query',
                  true,
                  'basic',
                  undefined,
                  model
                );
                break;
              case 'text-generation': {
                const customOutputs = (node.data.custom_outputs || []) as {
                  id: string;
                  label?: string;
                  description?: string;
                }[];
                const outputFields = customOutputs.map((o: any) => ({
                  id: o.id,
                  description: o.description || o.label,
                }));
                const useSearch = node.data.useSearch || false;
                let rawResult: any;
                const imageInputRaw = flattenImageInput(nodeInputs['in-image']);
                const imageInput =
                  imageInputRaw.length > 0 ? await ensureImageInputsAsDataUrls(imageInputRaw) : undefined;
                if (model?.startsWith('deepseek-')) {
                  rawResult = await deepseekText(
                    nodeInputs['in-text'] || '...',
                    node.data.mode,
                    node.data.customInstruction,
                    model,
                    outputFields,
                    useSearch
                  );
                } else if (model?.startsWith('doubao-')) {
                  rawResult = await doubaoText(
                    nodeInputs['in-text'] || '...',
                    node.data.mode,
                    node.data.customInstruction,
                    model,
                    outputFields,
                    imageInput?.length ? imageInput : undefined,
                    useSearch
                  );
                } else if (model?.startsWith('ppchat-')) {
                  rawResult = await ppchatGeminiText(
                    nodeInputs['in-text'] || '...',
                    node.data.mode,
                    node.data.customInstruction,
                    model.replace('ppchat-', ''),
                    outputFields,
                    imageInput?.length ? imageInput : undefined
                  );
                } else {
                  rawResult = await geminiText(
                    nodeInputs['in-text'] || '...',
                    false,
                    node.data.mode,
                    node.data.customInstruction,
                    model,
                    outputFields,
                    imageInput?.length ? imageInput : undefined
                  );
                }
                if (
                  customOutputs.length > 0 &&
                  rawResult != null &&
                  typeof rawResult === 'object' &&
                  !Array.isArray(rawResult)
                ) {
                  const portIds = customOutputs.map((o) => o.id);
                  const portKeyedOnly: Record<string, any> = {};
                  portIds.forEach((portId, i) => {
                    const fromRaw = rawResult[portId];
                    if (fromRaw !== undefined && fromRaw !== null) {
                      portKeyedOnly[portId] = fromRaw;
                    } else {
                      portKeyedOnly[portId] =
                        (node.output_value && typeof node.output_value === 'object' && portId in node.output_value)
                          ? node.output_value[portId]
                          : (i === 0 && typeof rawResult === 'object' && Object.keys(rawResult).length > 0
                              ? Object.values(rawResult)[0]
                              : '...');
                    }
                  });
                  result = portKeyedOnly;
                } else {
                  result = rawResult;
                }
                break;
              }
              case 'text-to-image':
                if (model === 'gemini-2.5-flash-image') {
                  result = await geminiImage(
                    nodeInputs['in-text'] || 'Artistic portrait',
                    undefined,
                    node.data.aspectRatio,
                    model
                  );
                } else {
                  const t2iModelConfig = getLightX2VConfigForModel(model || 'Qwen-Image-2512');
                  const t2iWorkflowRefs =
                    !t2iModelConfig.isCloud && wfId
                      ? buildWorkflowRefs(wfId, incomingConns, tool.inputs)
                      : undefined;
                  result = await lightX2VTask(
                    wfId,
                    node.id,
                    'out-image',
                    t2iModelConfig.url,
                    t2iModelConfig.token,
                    't2i',
                    model || 'Qwen-Image-2512',
                    nodeInputs['in-text'] || '',
                    undefined,
                    undefined,
                    undefined,
                    'output_image',
                    node.data.aspectRatio,
                    undefined,
                    (taskId) => registerTaskId(node.id, taskId, t2iModelConfig.isCloud),
                    jobAbortSignal,
                    t2iWorkflowRefs
                  );
                }
                break;
              case 'image-to-image':
                if (model === 'gemini-2.5-flash-image') {
                  const geminiImgs = flattenImageInput(nodeInputs['in-image']);
                  result = await geminiImage(
                    nodeInputs['in-text'] || 'Transform',
                    geminiImgs.length > 0 ? geminiImgs : undefined,
                    node.data.aspectRatio || '1:1',
                    model
                  );
                } else {
                  const i2iModelConfig = getLightX2VConfigForModel(model || 'Qwen-Image-Edit-2511');
                  const i2iWorkflowRefs =
                    !i2iModelConfig.isCloud && wfId
                      ? buildWorkflowRefs(wfId, incomingConns, tool.inputs)
                      : undefined;
                  let imageInput: string | string[] | undefined;
                  if (i2iWorkflowRefs?.input_image != null) {
                    imageInput = undefined;
                  } else {
                    const i2iImgs = flattenImageInput(nodeInputs['in-image']);
                    imageInput =
                      i2iImgs.length === 0
                        ? undefined
                        : i2iImgs.length === 1
                          ? i2iImgs[0]
                          : i2iImgs;
                    if (Array.isArray(imageInput)) {
                      imageInput = await ensureImageInputsAsDataUrls(imageInput);
                    } else if (typeof imageInput === 'string' && !imageInput.startsWith('data:')) {
                      imageInput = (await ensureImageInputsAsDataUrls([imageInput]))[0];
                    }
                  }
                  result = await lightX2VTask(
                    wfId,
                    node.id,
                    'out-image',
                    i2iModelConfig.url,
                    i2iModelConfig.token,
                    'i2i',
                    model || 'Qwen-Image-Edit-2511',
                    nodeInputs['in-text'] || '',
                    imageInput,
                    undefined,
                    undefined,
                    'output_image',
                    node.data.aspectRatio,
                    undefined,
                    (taskId) => registerTaskId(node.id, taskId, i2iModelConfig.isCloud),
                    jobAbortSignal,
                    i2iWorkflowRefs
                  );
                }
                break;
              case 'tts': {
                const isLightX2V = model === 'lightx2v' || model?.startsWith('lightx2v');
                if (isLightX2V) {
                  let resourceIdToUse = node.data.resourceId;
                  if (voiceList.lightX2VVoiceList?.voices?.length) {
                    const voiceTypeToUse = node.data.voiceType || 'zh_female_vv_uranus_bigtts';
                    const matching = voiceList.lightX2VVoiceList.voices.find(
                      (v: any) => v.voice_type === voiceTypeToUse
                    );
                    if (matching?.resource_id) {
                      resourceIdToUse = matching.resource_id;
                      if (node.data.resourceId !== resourceIdToUse) {
                        updateNodeData(node.id, 'resourceId', resourceIdToUse);
                      }
                    }
                  }
                  resourceIdToUse = resourceIdToUse || 'seed-tts-1.0';
                  const contextTone = nodeInputs['in-context-tone'] || '';
                  result = await lightX2VTTS(
                    lightX2VConfig.url,
                    lightX2VConfig.token,
                    nodeInputs['in-text'] || '',
                    node.data.voiceType || 'zh_female_vv_uranus_bigtts',
                    contextTone,
                    node.data.emotion || '',
                    node.data.emotionScale ?? 3,
                    node.data.speechRate ?? 0,
                    node.data.pitch ?? 0,
                    node.data.loudnessRate ?? 0,
                    resourceIdToUse
                  );
                } else {
                  const contextTone = nodeInputs['in-context-tone'] || '';
                  result = await geminiSpeech(
                    nodeInputs['in-text'] || 'Script',
                    node.data.voice || 'Kore',
                    model || 'gemini-2.5-flash-preview-tts',
                    contextTone
                  );
                }
                break;
              }
              case 'lightx2v-voice-clone': {
                const speakerId = node.data.speakerId;
                if (!speakerId)
                  throw new Error(
                    'Please select a cloned voice. Use the node settings to choose or create a new cloned voice.'
                  );
                const ttsText = nodeInputs['in-tts-text'] || nodeInputs['in-text'] || '';
                if (!ttsText) throw new Error('TTS text is required');
                result = await lightX2VVoiceCloneTTS(
                  lightX2VConfig.url,
                  lightX2VConfig.token,
                  ttsText,
                  speakerId,
                  node.data.style || '正常',
                  node.data.speed ?? 1.0,
                  node.data.volume ?? 0,
                  node.data.pitch ?? 0,
                  node.data.language || 'ZH_CN'
                );
                break;
              }
              case 'video-gen-text': {
                const t2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_T2V_A14B_distilled');
                const t2vWorkflowRefs =
                  !t2vModelConfig.isCloud && wfId
                    ? buildWorkflowRefs(wfId, incomingConns, tool.inputs)
                    : undefined;
                result = await lightX2VTask(
                  wfId,
                  node.id,
                  'out-video',
                  t2vModelConfig.url,
                  t2vModelConfig.token,
                  't2v',
                  model || 'Wan2.2_T2V_A14B_distilled',
                  nodeInputs['in-text'] || '',
                  undefined,
                  undefined,
                  undefined,
                  'output_video',
                  node.data.aspectRatio,
                  undefined,
                  (taskId) => registerTaskId(node.id, taskId),
                  jobAbortSignal,
                  t2vWorkflowRefs
                );
                break;
              }
              case 'video-gen-image': {
                const startImg = Array.isArray(nodeInputs['in-image'])
                  ? nodeInputs['in-image'][0]
                  : nodeInputs['in-image'];
                const i2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_I2V_A14B_distilled');
                const i2vWorkflowRefs =
                  !i2vModelConfig.isCloud && wfId
                    ? buildWorkflowRefs(wfId, incomingConns, tool.inputs)
                    : undefined;
                const startImgForSubmit =
                  i2vWorkflowRefs?.input_image != null
                    ? undefined
                    : startImg;
                result = await lightX2VTask(
                  wfId,
                  node.id,
                  'out-video',
                  i2vModelConfig.url,
                  i2vModelConfig.token,
                  'i2v',
                  model || 'Wan2.2_I2V_A14B_distilled',
                  nodeInputs['in-text'] || '',
                  startImgForSubmit,
                  undefined,
                  undefined,
                  'output_video',
                  node.data.aspectRatio,
                  undefined,
                  (taskId) => registerTaskId(node.id, taskId, i2vModelConfig.isCloud),
                  jobAbortSignal,
                  i2vWorkflowRefs
                );
                break;
              }
              case 'video-gen-dual-frame': {
                const dualStart = Array.isArray(nodeInputs['in-image-start'])
                  ? nodeInputs['in-image-start'][0]
                  : nodeInputs['in-image-start'];
                const dualEnd = Array.isArray(nodeInputs['in-image-end'])
                  ? nodeInputs['in-image-end'][0]
                  : nodeInputs['in-image-end'];
                const flf2vModelConfig = getLightX2VConfigForModel(model || 'Wan2.2_I2V_A14B_distilled');
                const flf2vWorkflowRefs =
                  !flf2vModelConfig.isCloud && wfId
                    ? buildWorkflowRefs(wfId, incomingConns, tool.inputs)
                    : undefined;
                result = await lightX2VTask(
                  wfId,
                  node.id,
                  'out-video',
                  flf2vModelConfig.url,
                  flf2vModelConfig.token,
                  'flf2v',
                  model || 'Wan2.2_I2V_A14B_distilled',
                  nodeInputs['in-text'] || '',
                  flf2vWorkflowRefs?.input_image != null ? undefined : dualStart,
                  undefined,
                  flf2vWorkflowRefs?.input_last_frame != null ? undefined : dualEnd,
                  'output_video',
                  node.data.aspectRatio,
                  undefined,
                  (taskId) => registerTaskId(node.id, taskId),
                  jobAbortSignal,
                  flf2vWorkflowRefs
                );
                break;
              }
              case 'character-swap': {
                const swapImg = Array.isArray(nodeInputs['in-image'])
                  ? nodeInputs['in-image'][0]
                  : nodeInputs['in-image'];
                const swapVid = Array.isArray(nodeInputs['in-video'])
                  ? nodeInputs['in-video'][0]
                  : nodeInputs['in-video'];
                if (model === 'wan2.2_animate' || model?.endsWith('-cloud')) {
                  const animateModelConfig = getLightX2VConfigForModel(model);
                  const animateWorkflowRefs =
                    !animateModelConfig.isCloud && wfId
                      ? buildWorkflowRefs(wfId, incomingConns, tool.inputs)
                      : undefined;
                  result = await lightX2VTask(
                    wfId,
                    node.id,
                    'out-video',
                    animateModelConfig.url,
                    animateModelConfig.token,
                    'animate',
                    model,
                    nodeInputs['in-text'] || '',
                    animateWorkflowRefs?.input_image != null ? undefined : swapImg,
                    undefined,
                    undefined,
                    'output_video',
                    node.data.aspectRatio,
                    animateWorkflowRefs?.input_video != null ? undefined : swapVid,
                    (taskId) => registerTaskId(node.id, taskId, animateModelConfig.isCloud),
                    jobAbortSignal,
                    animateWorkflowRefs
                  );
                } else {
                  result = await geminiVideo(
                    nodeInputs['in-text'] || 'Swap character',
                    swapImg,
                    '16:9',
                    '720p',
                    swapVid,
                    model
                  );
                }
                break;
              }
              case 'avatar-gen': {
                const avatarImg = Array.isArray(nodeInputs['in-image'])
                  ? nodeInputs['in-image'][0]
                  : nodeInputs['in-image'];
                const avatarAudio = Array.isArray(nodeInputs['in-audio'])
                  ? nodeInputs['in-audio'][0]
                  : nodeInputs['in-audio'];
                const s2vModelConfig = getLightX2VConfigForModel(model || 'SekoTalk');
                const s2vWorkflowRefs =
                  !s2vModelConfig.isCloud && wfId
                    ? buildWorkflowRefs(wfId, incomingConns, tool.inputs)
                    : undefined;
                result = await lightX2VTask(
                  wfId,
                  node.id,
                  'out-video',
                  s2vModelConfig.url,
                  s2vModelConfig.token,
                  's2v',
                  model || 'SekoTalk',
                  nodeInputs['in-text'] || 'A person talking naturally.',
                  s2vWorkflowRefs?.input_image != null ? undefined : avatarImg || '',
                  s2vWorkflowRefs?.input_audio != null ? undefined : avatarAudio || '',
                  undefined,
                  'output_video',
                  undefined,
                  undefined,
                  (taskId) => registerTaskId(node.id, taskId, s2vModelConfig.isCloud),
                  jobAbortSignal,
                  s2vWorkflowRefs
                );
                break;
              }
              default:
                result = 'Processed';
            }

            const nodeDuration = performance.now() - nodeStart;
            let valueToStore: any = result;
            if (tool?.outputs?.length && !isPortKeyedOutputValue(valueToStore)) {
              const portId = tool.outputs[0].id;
              valueToStore = { [portId]: valueToStore };
            }
            sessionOutputs[node.id] = valueToStore;

            setWorkflow((prev) => {
              if (!prev) return null;
              const idx = prev.nodes.findIndex((n) => n.id === node.id);
              if (idx < 0) return prev;
              const updated = [...prev.nodes];
              updated[idx] = {
                ...updated[idx],
                status: NodeStatus.SUCCESS,
                execution_time: nodeDuration,
                output_value: valueToStore,
                completed_at: Date.now(),
              };
              return { ...prev, nodes: updated };
            });

            // Task 节点完成：将 /save 放入队列依次执行，避免多 Node 同时请求 port/save
            if (shouldSaveOutputs && valueToStore != null) {
              const toSave: Record<string, any> =
                typeof valueToStore === 'object' && !Array.isArray(valueToStore)
                  ? valueToStore
                  : tool.outputs?.[0]
                    ? { [tool.outputs[0].id]: valueToStore }
                    : {};
              if (Object.keys(toSave).length > 0) {
                enqueuePortSave({ wfId, nodeId: node.id, toSave, runId });
              }
            }

            return { nodeId: node.id, result: valueToStore, duration: nodeDuration };
          } catch (err: any) {
            const nodeDuration = performance.now() - nodeStart;
            if (err?.message?.includes?.('Requested entity was not found')) {
              await (window as any).aistudio?.openSelectKey?.();
            }
            const errorMessage =
              err?.message ?? (typeof err === 'string' ? err : String(err ?? 'Unknown execution error'));
            flushSync(() => {
              setWorkflow((prev) =>
                prev
                  ? {
                      ...prev,
                      nodes: prev.nodes.map((n) =>
                        n.id === node.id
                          ? {
                              ...n,
                              status: NodeStatus.ERROR,
                              error: errorMessage,
                              execution_time: nodeDuration,
                              completed_at: Date.now(),
                            }
                          : n
                      ),
                    }
                  : null
              );
            });
            throw { nodeId: node.id, error: err, duration: nodeDuration };
          }
        });

        const results = await Promise.allSettled(executionPromises);
        const successfulResults: Array<{ nodeId: string; result: any; duration: number }> = [];
        const failedNodes: Array<{ nodeId: string; error: any; duration: number }> = [];

        results.forEach((settledResult, index) => {
          const node = batch[index];
          if (settledResult.status === 'fulfilled') {
            const v = settledResult.value;
            successfulResults.push({ nodeId: v.nodeId, result: v.result, duration: v.duration });
          } else {
            const errorInfo = settledResult.reason;
            if (errorInfo?.nodeId) {
              failedNodes.push({
                nodeId: errorInfo.nodeId,
                error:
                  errorInfo.error instanceof Error
                    ? errorInfo.error.message
                    : typeof errorInfo.error === 'string'
                      ? errorInfo.error
                      : String(errorInfo.error ?? 'Unknown error'),
                duration: errorInfo.duration ?? 0,
              });
            } else {
              failedNodes.push({
                nodeId: node.id,
                error:
                  settledResult.reason instanceof Error
                    ? settledResult.reason.message
                    : String(settledResult.reason ?? 'Unknown error'),
                duration: 0,
              });
            }
          }
        });

        if (failedNodes.length > 0) {
          const downstreamToCancel = new Set<string>();
          failedNodes.forEach(({ nodeId }) => {
            const desc = getDescendants(nodeId, workflow.connections);
            desc.forEach((d) => {
              if (nodesToRunIds.has(d)) downstreamToCancel.add(d);
            });
          });
          downstreamToCancel.forEach((id) => cancelledByFailure.add(id));
          setWorkflow((prev) => {
            if (!prev) return null;
            return {
              ...prev,
              nodes: prev.nodes.map((n) => {
                if (downstreamToCancel.has(n.id)) {
                  return { ...n, status: NodeStatus.IDLE, error: undefined };
                }
                const failed = failedNodes.find((f) => f.nodeId === n.id);
                if (failed)
                  return {
                    ...n,
                    status: NodeStatus.ERROR,
                    error: failed.error,
                    execution_time: failed.duration,
                  };
                return n;
              }),
            };
          });
        }

        // 本批次产生的 port/save 全部完成后再开始下一轮，避免与下一批执行重叠
        if (shouldSaveOutputs) {
          await waitForPortSaveQueueDrain();
        }
        // 执行后：每批已等 port save 排空，此处直接拉取最新 workflow 并同步到后端，无需再入队 post_save
        if (!isStandalone() && hasValidDbId && updateWorkflowSync) {
          await new Promise((r) => setTimeout(r, 300));
          const wf = getWorkflow?.() ?? workflowRef.current;
          if (wf) await updateWorkflowSync(wfId, wf);
        }
      }

      const j = runningJobsByJobIdRef.current.get(jobId);
      if (j?.pollIntervalId) {
        clearInterval(j.pollIntervalId);
        j.pollIntervalId = undefined;
      }
      if (j) {
        j.taskIdsByNodeId.forEach((taskId, nodeId) => {
          if (runningTaskIdsRef.current.get(nodeId) === taskId) {
            runningTaskIdsRef.current.delete(nodeId);
          }
        });
        runningJobsByJobIdRef.current.delete(jobId);
      }
    },
    [
      workflow,
      setWorkflow,
      isPausedRef,
      setIsPaused,
      runningTaskIdsRef,
      getLightX2VConfig,
      getDescendants,
      getTopologicalBatches,
      validateWorkflow,
      setValidationErrors,
      setGlobalError,
      updateNodeData,
      voiceList,
      saveInputNodesAndRefresh,
      getInputValueForPort,
      refreshWorkflowFromBackend,
      updateWorkflowSync,
      getWorkflow,
      waitForPortSaveQueueDrain,
    ]
  );

  const processQueue = useCallback(() => {
    const queue = executionQueueRef.current;
    while (queue.length > 0 && runningJobCountRef.current < MAX_CONCURRENT_JOBS) {
      const job = queue.shift()!;
      if (job.cancelled) {
        refreshPendingNodeIds();
        continue;
      }
      const jobId = job.id;
      const jobInfo = {
        affectedNodeIds: job.affectedNodeIds,
        taskIdsByNodeId: new Map<string, string>(),
        nodeIdToIsCloud: new Map<string, boolean>(),
        abortController: new AbortController(),
      };
      runningJobsByJobIdRef.current.set(jobId, jobInfo);
      runningJobCountRef.current += 1;
      executeOneRun(jobId, job.startNodeId, job.onlyOne).finally(() => {
        runningJobCountRef.current -= 1;
        refreshPendingNodeIds();
        processQueue();
      });
    }
    if (runningJobCountRef.current === 0 && executionQueueRef.current.length === 0) {
      setWorkflow((prev) => (prev ? { ...prev, isRunning: false } : null));
    }
    refreshPendingNodeIds();
  }, [executeOneRun, setWorkflow, refreshPendingNodeIds]);

  const runWorkflow = useCallback(
    async (startNodeId?: string, onlyOne?: boolean) => {
      if (!workflow) return;
      const isPreset = workflow.id?.startsWith('preset-');
      if ((workflow.isDirty || isPreset) && saveWorkflowBeforeRun) {
        try {
          const savedId = await saveWorkflowBeforeRun(workflow);
          setWorkflow((prev) =>
            prev
              ? {
                  ...prev,
                  isDirty: false,
                  ...(savedId && savedId !== prev.id ? { id: savedId } : {}),
                }
              : null
          );
          if (savedId && savedId !== workflow.id) {
            effectiveWorkflowIdRef.current = savedId;
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          setGlobalError({ message: '保存工作流失败，无法执行', details: msg });
          return;
        }
      }
      const affectedNodeIds = getAffectedNodeIds(startNodeId, onlyOne);
      const job: QueueJob = {
        id: `job-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        startNodeId,
        onlyOne,
        cancelled: false,
        affectedNodeIds,
      };
      executionQueueRef.current.push(job);
      setWorkflow((prev) => (prev ? { ...prev, isRunning: true } : null));
      refreshPendingNodeIds();
      processQueue();
    },
    [
      workflow,
      setWorkflow,
      processQueue,
      getAffectedNodeIds,
      refreshPendingNodeIds,
      saveWorkflowBeforeRun,
      setGlobalError,
    ]
  );

  const cancelNodeRun = useCallback(
    (nodeId: string) => {
      const queue = executionQueueRef.current;
      const queuedIndex = queue.findIndex((j) => !j.cancelled && j.affectedNodeIds.has(nodeId));
      if (queuedIndex !== -1) {
        queue[queuedIndex].cancelled = true;
        refreshPendingNodeIds();
        return;
      }
      const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
      const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
      for (const [jid, jobInfo] of runningJobsByJobIdRef.current) {
        if (jobInfo.affectedNodeIds.has(nodeId)) {
          jobInfo.taskIdsByNodeId.forEach((taskId, nid) => {
            const isCloud = jobInfo.nodeIdToIsCloud.get(nid) ?? false;
            (async () => {
              try {
                if (isStandalone() || isCloud) {
                  await lightX2VCancelTask(cloudUrl, cloudToken, taskId);
                } else {
                  await apiRequest(`/api/v1/task/cancel?task_id=${taskId}`, { method: 'GET' });
                }
              } catch (e) {
                console.warn('[WorkflowExecution] Cancel task failed:', taskId, e);
              }
            })();
          });
          jobInfo.abortController.abort();
          setWorkflow((prev) =>
            prev
              ? {
                  ...prev,
                  nodes: prev.nodes.map((n) =>
                    jobInfo.affectedNodeIds.has(n.id) &&
                    (n.status === NodeStatus.RUNNING || n.status === NodeStatus.PENDING)
                      ? { ...n, status: NodeStatus.IDLE, error: 'Cancelled' }
                      : n
                  ),
                }
              : null
          );
          refreshPendingNodeIds();
          break;
        }
      }
    },
    [refreshPendingNodeIds, setWorkflow]
  );

  const stopWorkflow = useCallback(async () => {
    if (!workflow) return;
    executionQueueRef.current = [];
    runningJobsByJobIdRef.current.forEach((jobInfo) => jobInfo.abortController.abort());
    const taskIds = Array.from(runningTaskIdsRef.current.values());
    const taskIdToNodeId = new Map<string, string>();
    runningTaskIdsRef.current.forEach((tid, nid) => taskIdToNodeId.set(tid, nid));
    const taskIdToIsCloud = new Map<string, boolean>();
    runningJobsByJobIdRef.current.forEach((j) => {
      j.taskIdsByNodeId.forEach((tid, nid) => {
        taskIdToIsCloud.set(tid, j.nodeIdToIsCloud.get(nid) ?? false);
      });
    });
    runningJobsByJobIdRef.current.clear();
    setPendingRunNodeIds([]);

    const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
    const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
    await Promise.all(
      taskIds.map(async (taskId) => {
        try {
          if (isStandalone() || taskIdToIsCloud.get(String(taskId))) {
            await lightX2VCancelTask(cloudUrl, cloudToken, String(taskId));
          } else {
            await apiRequest(`/api/v1/task/cancel?task_id=${taskId}`, { method: 'GET' });
          }
        } catch (_) {}
      })
    );

    const cancelledNodeIds = new Set<string>();
    await Promise.all(
      taskIds.map(async (taskId) => {
        try {
          let status: string;
          if (isStandalone() || taskIdToIsCloud.get(String(taskId))) {
            const info = await lightX2VTaskQuery(cloudUrl, cloudToken, String(taskId));
            status = info.status;
          } else {
            const res = await apiRequest(`/api/v1/task/query?task_id=${taskId}`, { method: 'GET' });
            const data = res.ok ? ((await res.json().catch(() => ({}))) as { status?: string }) : {};
            status = data.status || 'UNKNOWN';
          }
          if (status === 'CANCELLED') {
            const nid = taskIdToNodeId.get(String(taskId));
            if (nid) cancelledNodeIds.add(nid);
          }
        } catch (_) {}
      })
    );
    runningTaskIdsRef.current.clear();
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = new AbortController();
    }
    setWorkflow((prev) =>
      prev
        ? {
            ...prev,
            isRunning: false,
            nodes: prev.nodes.map((n) =>
              n.status === NodeStatus.RUNNING || n.status === NodeStatus.PENDING
                ? { ...n, status: NodeStatus.IDLE, error: cancelledNodeIds.has(n.id) ? 'Cancelled' : undefined }
                : n
            ),
          }
        : null
    );
  }, [workflow, runningTaskIdsRef, abortControllerRef, setWorkflow]);

  return {
    runWorkflow,
    stopWorkflow,
    cancelNodeRun,
    pendingRunNodeIds,
    resolveLightX2VResultRef,
    validateWorkflow,
    getDescendants,
  };
}

export const useWorkflowExecution = useWorkflowExecutionImpl;
