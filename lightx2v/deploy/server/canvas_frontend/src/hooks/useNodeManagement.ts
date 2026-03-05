import React, { useCallback, useRef } from 'react';
import { WorkflowState, WorkflowNode, ToolDefinition, DataType, Port, NodeStatus } from '../../types';
import { TOOLS } from '../../constants';
import { getOutputValueByPort, INPUT_PORT_IDS, isPortKeyedOutputValue, isValidOutputEntry } from '../utils/outputValuePort';
import { useTranslation, Language } from '../i18n/useTranslation';

interface UseNodeManagementProps {
  workflow: WorkflowState | null;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  selectedNodeId: string | null;
  setSelectedNodeId: (id: string | null) => void;
  setValidationErrors: (errors: { message: string; type: 'ENV' | 'INPUT' }[]) => void;
  canvasRef: React.RefObject<HTMLDivElement>;
  screenToWorldCoords: (x: number, y: number) => { x: number; y: number };
  view: { x: number; y: number; zoom: number };
  getNodeOutputs: (node: WorkflowNode) => Port[];
  lang: Language;
}

export const useNodeManagement = ({
  workflow,
  setWorkflow,
  selectedNodeId,
  setSelectedNodeId,
  setValidationErrors,
  canvasRef,
  screenToWorldCoords,
  view,
  getNodeOutputs,
  lang
}: UseNodeManagementProps) => {
  const { t } = useTranslation(lang);

  const addNode = useCallback((tool: ToolDefinition, x?: number, y?: number, dataOverride?: Record<string, any>, nodeId?: string, allowOverwrite?: boolean, _forceEditMode?: boolean) => {
    const defaultData: Record<string, any> = { ...dataOverride };

    // 先处理 TTS 节点的特殊逻辑（需要在通用模型选择之前）
    if (tool.id === 'tts') {
      if (!defaultData.model) {
        // 优先使用 'lightx2v' 作为默认模型
        const lightx2vModel = tool.models?.find(m => m.id === 'lightx2v');
        defaultData.model = lightx2vModel ? 'lightx2v' : (tool.models && tool.models.length > 0 ? tool.models[0].id : 'lightx2v');
      }
      if (defaultData.model === 'lightx2v' || defaultData.model?.startsWith('lightx2v')) {
        if (!defaultData.voiceType) defaultData.voiceType = 'zh_female_vv_uranus_bigtts';
        if (!defaultData.emotionScale) defaultData.emotionScale = 3;
        if (!defaultData.speechRate) defaultData.speechRate = 0;
        if (!defaultData.pitch) defaultData.pitch = 0;
        if (!defaultData.loudnessRate) defaultData.loudnessRate = 0;
        if (!defaultData.resourceId) {
          defaultData.resourceId = "seed-tts-2.0";
        }
      } else {
        if (!defaultData.voice) defaultData.voice = "Kore";
      }
    } else {
      // 对于其他工具，使用通用模型选择逻辑
      if (tool.models && tool.models.length > 0 && !defaultData.model) defaultData.model = tool.models[0].id;
    }

    if ((tool.id === 'text-to-image' || tool.id === 'image-to-image') && !defaultData.aspectRatio) defaultData.aspectRatio = "1:1";
    if (tool.id.includes('video-gen') && !defaultData.aspectRatio) defaultData.aspectRatio = "16:9";
    if (tool.id === 'text-input' && defaultData.value === undefined) defaultData.value = lang === 'zh' ? '在此输入文本' : '';
    if (tool.id === 'lightx2v-voice-clone') {
      if (!defaultData.style) defaultData.style = "正常";
      if (!defaultData.speed) defaultData.speed = 1.0;
      if (!defaultData.volume) defaultData.volume = 0;
      if (!defaultData.pitch) defaultData.pitch = 0;
      if (!defaultData.language) defaultData.language = "ZH_CN";
    }
    if (tool.id === 'text-generation') {
      if (!defaultData.model) {
        defaultData.model = 'deepseek-v3-2-251201';
      }
      if (!defaultData.mode) defaultData.mode = 'basic';
      if (!defaultData.custom_outputs) defaultData.custom_outputs = [{ id: 'out-text1', label: t('execution_results'), description: 'Main text response.' }];
    }
    const rect = canvasRef.current?.getBoundingClientRect();
    const worldPos = x !== undefined && y !== undefined ? { x, y } : screenToWorldCoords((rect?.width || 800) / 2, (rect?.height || 600) / 2);

    // 如果指定了nodeId，使用指定的ID，否则自动生成
    let newNodeId: string;
    if (nodeId) {
      // 检查ID是否已存在
      const existingNode = workflow?.nodes.find(n => n.id === nodeId);
      if (existingNode) {
        if (allowOverwrite) {
          // 如果允许覆盖，先删除旧节点，然后使用相同的ID创建新节点
          setWorkflow(prev => prev ? ({
            ...prev,
            nodes: prev.nodes.filter(n => n.id !== nodeId),
            connections: prev.connections.filter(c => c.source_node_id !== nodeId && c.target_node_id !== nodeId),
            isDirty: true
          }) : null);
          newNodeId = nodeId; // 使用指定的ID，不修改
        } else {
          // 如果不允许覆盖，添加后缀使其唯一
          newNodeId = `${nodeId}-${Date.now()}`;
        }
      } else {
        newNodeId = nodeId; // ID不存在，直接使用
      }
    } else {
      newNodeId = `node-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    }

    const newNode: WorkflowNode = { id: newNodeId, tool_id: tool.id, x: worldPos.x, y: worldPos.y, status: NodeStatus.IDLE, data: defaultData };
    setWorkflow(prev => prev ? ({ ...prev, nodes: [...prev.nodes, newNode], isDirty: true }) : null);
    setSelectedNodeId(newNodeId);
    return newNode;
  }, [screenToWorldCoords, t, canvasRef, setWorkflow, setSelectedNodeId, workflow]);

  const deleteNode = useCallback((nodeId: string) => {
    if (!nodeId) return;
    setWorkflow(prev => prev ? ({
      ...prev,
      nodes: prev.nodes.filter(n => n.id !== nodeId),
      connections: prev.connections.filter(c => c.source_node_id !== nodeId && c.target_node_id !== nodeId),
      isDirty: true
    }) : null);
    if (selectedNodeId === nodeId) {
      setSelectedNodeId(null);
    }
  }, [selectedNodeId, setWorkflow, setSelectedNodeId]);

  const updateNodeData = useCallback((nodeId: string, key: string, value: any) => {
    setValidationErrors([]);

    // 以下更新无论是否在运行快照视图都应用，保证 AI 等调用能真正改到工作流（如 custom_outputs）
    // 支持一次性合并多个 data 字段（避免多次调用时后一次覆盖前一次），由 AI 工作流等调用
    // 对数组（如 custom_outputs）做深拷贝，确保引用变化以便 React 与 getNodeOutputs 等能正确更新
    if (key === '__mergeData' && typeof value === 'object' && value !== null && !Array.isArray(value)) {
      const cloneForMerge = (v: any): any => {
        if (Array.isArray(v)) return v.map(item => (item && typeof item === 'object' && !Array.isArray(item) ? { ...item } : item));
        if (v && typeof v === 'object' && !Array.isArray(v)) return { ...v };
        return v;
      };
      const clonedValue = Object.fromEntries(Object.entries(value).map(([k, v]) => [k, cloneForMerge(v)]));
      setWorkflow(prev => {
        if (!prev) return null;
        return {
          ...prev,
          nodes: prev.nodes.map(n => n.id === nodeId ? { ...n, status: NodeStatus.IDLE, data: { ...n.data, ...clonedValue } } : n),
          isDirty: true
        };
      });
      return;
    }


    // output_value 仅允许来自：服务器查询、/output/{port_id}/save 返回、或 node_output_history 的 entry；不得在前端擅自改为 url 或非法格式。
    // 只接受 port-keyed 的 ref：entry 为 { kind: 'file' | 'task', ... }，禁止 url 字符串或 kind === 'url'。
    if (key === 'output_value') {
      const isRefOrPortKeyed = (v: any, toolId: string): boolean => {
        if (v == null) return true;
        if (typeof v === 'string') return false;
        if (Array.isArray(v)) return v.every((item: any) => isValidOutputEntry(item, toolId));
        if (typeof v === 'object' && isPortKeyedOutputValue(v)) {
          return Object.values(v).every((val: any) =>
            val == null || (Array.isArray(val) ? val.every((i: any) => isValidOutputEntry(i, toolId)) : isValidOutputEntry(val, toolId))
          );
        }
        return isValidOutputEntry(v, toolId);
      };

      setWorkflow(prev => {
        if (!prev) return null;
        const node = prev.nodes.find(n => n.id === nodeId);
        const tool = node ? TOOLS.find(t => t.id === node.tool_id) : undefined;
        const isInputWithValue = tool?.category === 'Input' && ['image-input', 'audio-input', 'video-input', 'text-input'].includes(node?.tool_id ?? '');
        const portId = node ? INPUT_PORT_IDS[node.tool_id] : undefined;
        const effectiveValue = isInputWithValue && portId && isPortKeyedOutputValue(value)
          ? value[portId]
          : value;
        return {
          ...prev,
          nodes: prev.nodes.map(n => {
            if (n.id !== nodeId) return n;
            if (!isRefOrPortKeyed(value, n.tool_id)) {
              console.warn('updateNodeData output_value: only allow ref (kind file|task), reject:', nodeId, value);
              return n;
            }
            const nextData = isInputWithValue ? { ...n.data, value: effectiveValue } : n.data;
            return { ...n, output_value: value, data: nextData, status: NodeStatus.IDLE };
          }),
          isDirty: true
        };
      });
      return;
    }

    setWorkflow(prev => {
      if (!prev) return null;
      return {
        ...prev,
        nodes: prev.nodes.map(n => n.id === nodeId ? { ...n, status: NodeStatus.IDLE, data: { ...n.data, [key]: value } } : n),
        isDirty: true
      };
    });
  }, [setValidationErrors, setWorkflow]);

  const updateNodeName = useCallback((nodeId: string, name: string) => {
    setWorkflow(prev => {
      if (!prev) return null;
      return {
        ...prev,
        isDirty: true,
        nodes: prev.nodes.map(n => n.id === nodeId ? { ...n, name: name.trim() || undefined } : n)
      };
    });
  }, [setWorkflow]);

  const getReplaceableTools = useCallback((nodeId: string): ToolDefinition[] => {
    if (!workflow) return [];
    const node = workflow.nodes.find(n => n.id === nodeId);
    if (!node) return [];

    const currentNodeOutputs = getNodeOutputs(node);
    const outputTypes = currentNodeOutputs.map(o => o.type);
    const outputCount = currentNodeOutputs.length;

    return TOOLS.filter(tool => {
      if (tool.id === node.tool_id) return false;

      if (tool.id === 'text-generation') {
        if (node.tool_id === 'text-generation') {
          return true;
        }
        return true;
      }

      if (node.tool_id === 'text-generation') {
        if (tool.outputs.length !== outputCount) return false;
        return tool.outputs.every((out, idx) => out.type === outputTypes[idx]);
      }

      if (tool.outputs.length !== outputCount) return false;
      return tool.outputs.every((out, idx) => out.type === outputTypes[idx]);
    });
  }, [workflow, getNodeOutputs]);

  const replaceNode = useCallback((nodeId: string, newToolId: string) => {
    if (!workflow) return;
    const node = workflow.nodes.find(n => n.id === nodeId);
    if (!node) return;

    const newTool = TOOLS.find(t => t.id === newToolId);
    if (!newTool) return;

    const currentNodeOutputs = getNodeOutputs(node);
    const newToolOutputs = newTool.outputs;

    // Validate compatibility
    if (node.tool_id !== 'text-generation' && newTool.id !== 'text-generation') {
      if (currentNodeOutputs.length !== newToolOutputs.length) {
        console.warn('Output count mismatch');
        return;
      }
      const compatible = currentNodeOutputs.every((out, idx) => out.type === newToolOutputs[idx].type);
      if (!compatible) {
        console.warn('Output types mismatch');
        return;
      }
    }

    // 合并默认参数：工具级别 -> 模型级别 -> 保留的兼容数据
    const newData: Record<string, any> = {};

    // 1. 先应用工具级别的默认参数
    if (newTool.defaultParams) {
      Object.assign(newData, newTool.defaultParams);
    }

    // 2. 确定使用的模型，并应用模型级别的默认参数
    let selectedModelId: string | undefined;

    // 如果新工具支持模型，尝试保留旧模型的兼容性，否则使用默认模型
    if (newTool.models && newTool.models.length > 0) {
      // 检查旧节点的模型是否在新工具中可用
      const oldModelId = node.data.model;
      const oldModelCompatible = oldModelId && newTool.models.some(m => m.id === oldModelId);

      if (oldModelCompatible && oldModelId) {
        selectedModelId = oldModelId;
      } else {
        selectedModelId = newTool.models[0].id;
      }

      newData.model = selectedModelId;

      // 应用模型级别的默认参数
      const modelDef = newTool.models.find(m => m.id === selectedModelId);
      if (modelDef?.defaultParams) {
        Object.assign(newData, modelDef.defaultParams);
      }
    }

    // 3. 保留兼容的数据（覆盖默认值）
    // Preserve custom_outputs for text-generation
    if (newTool.id === 'text-generation' && node.tool_id === 'text-generation') {
      if (node.data.custom_outputs) {
        newData.custom_outputs = node.data.custom_outputs;
      }
      if (node.data.mode) {
        newData.mode = node.data.mode;
      }
    }

    // Preserve aspectRatio for image/video tools
    if ((newTool.id === 'text-to-image' || newTool.id === 'image-to-image' || newTool.id.includes('video-gen')) &&
        (node.tool_id === 'text-to-image' || node.tool_id === 'image-to-image' || node.tool_id.includes('video-gen'))) {
      if (node.data.aspectRatio) {
        newData.aspectRatio = node.data.aspectRatio;
      }
    }

    // 4. 保留其他兼容的数据字段（如果新工具也支持这些字段）
    // 例如：保留 value 字段（如果新工具是输入类型）
    if (newTool.category === 'Input' && node.data.value !== undefined) {
      newData.value = node.data.value;
    }

    setWorkflow(prev => prev ? ({
      ...prev,
      nodes: prev.nodes.map(n => n.id === nodeId ? { ...n, tool_id: newToolId, data: newData } : n),
      isDirty: true
    }) : null);
  }, [workflow, getNodeOutputs, setWorkflow]);

  const quickAddInput = useCallback((node: WorkflowNode, port: Port) => {
    const toolIdMap: Record<DataType, string> = {
      [DataType.TEXT]: 'text-input',
      [DataType.IMAGE]: 'image-input',
      [DataType.AUDIO]: 'audio-input',
      [DataType.VIDEO]: 'video-input'
    };
    const tool = TOOLS.find(t => t.id === toolIdMap[port.type]);
    if (!tool) return;

    const worldPos = { x: node.x - 300, y: node.y };
    const newNodeId = `node-source-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

    const defaultData: Record<string, any> = {};
    if (tool.models && tool.models.length > 0) defaultData.model = tool.models[0].id;
    if (tool.id === 'text-input') defaultData.value = lang === 'zh' ? '在此输入文本' : '';

    const newNode: WorkflowNode = {
      id: newNodeId,
      tool_id: tool.id,
      x: worldPos.x,
      y: worldPos.y,
      status: NodeStatus.IDLE,
      data: defaultData
    };

    const newConn = {
      id: `conn-${Date.now()}`,
      source_node_id: newNodeId,
      source_port_id: tool.outputs[0].id,
      target_node_id: node.id,
      target_port_id: port.id
    };

    setWorkflow(prev => {
      if (!prev) return null;
      return {
        ...prev,
        nodes: [...prev.nodes, newNode],
        connections: [...prev.connections, newConn],
        isDirty: true
      };
    });
    setSelectedNodeId(newNodeId);
  }, [setWorkflow, setSelectedNodeId]);

  const quickAddOutput = useCallback((node: WorkflowNode, port: Port, toolId: string) => {
    const targetTool = TOOLS.find(t => t.id === toolId);
    if (!targetTool) return;

    const matchingInput = targetTool.inputs.find(input => input.type === port.type);
    if (!matchingInput) return;

    const worldPos = { x: node.x + 300, y: node.y };
    const newNodeId = `node-target-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

    const defaultData: Record<string, any> = {};
    if (targetTool.models && targetTool.models.length > 0) defaultData.model = targetTool.models[0].id;
    if (targetTool.id === 'text-generation') {
      defaultData.custom_outputs = [{ id: 'out-text1', label: t('execution_results'), description: 'Main text response.' }];
      defaultData.mode = 'basic';
    }

    const newNode: WorkflowNode = {
      id: newNodeId,
      tool_id: targetTool.id,
      x: worldPos.x,
      y: worldPos.y,
      status: NodeStatus.IDLE,
      data: defaultData
    };

    const newConn = {
      id: `conn-${Date.now()}`,
      source_node_id: node.id,
      source_port_id: port.id,
      target_node_id: newNodeId,
      target_port_id: matchingInput.id
    };

    setWorkflow(prev => {
      if (!prev) return null;
      return {
        ...prev,
        nodes: [...prev.nodes, newNode],
        connections: [...prev.connections, newConn],
        isDirty: true
      };
    });
    setSelectedNodeId(newNodeId);
  }, [t, setWorkflow, setSelectedNodeId]);

  const pinOutputToCanvas = useCallback((value: any, type: DataType) => {
    const toolIdMap: Record<DataType, string> = {
      [DataType.TEXT]: 'text-input',
      [DataType.IMAGE]: 'image-input',
      [DataType.AUDIO]: 'audio-input',
      [DataType.VIDEO]: 'video-input'
    };
    const tool = TOOLS.find(t => t.id === toolIdMap[type]);
    if (!tool) return;
    // image-input 要求 node.data.value 为数组，单张图需包装成 [url]
    const payload =
      type === DataType.IMAGE
        ? { value: Array.isArray(value) ? value : value != null ? [value] : [] }
        : { value };
    addNode(tool, 100, 100, payload);
  }, [addNode]);

  return {
    addNode,
    deleteNode,
    updateNodeData,
    updateNodeName,
    getReplaceableTools,
    replaceNode,
    quickAddInput,
    quickAddOutput,
    pinOutputToCanvas
  };
};
