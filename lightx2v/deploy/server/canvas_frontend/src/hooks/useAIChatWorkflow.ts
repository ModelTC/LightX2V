import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  WorkflowState,
  WorkflowNode,
  Connection,
  ToolDefinition,
  Port,
  DataType,
  ChatMessage,
  ChatImage,
  ChatSource,
  ChatOperation,
  ChatOperationResult
} from '../../types';
import { TOOLS } from '../../constants';
import { isStandalone } from '../config/runtimeMode';
import { getWorkflowChat, putWorkflowChat } from '../utils/workflowFileManager';
import { deepseekChat, deepseekChatStream, deepseekText, ppchatChatCompletions, ppchatChatCompletionsStream, ppchatGeminiText, doubaoText, lightX2VGetVoiceList, lightX2VGetCloneVoiceList } from '../../services/geminiService';
import { useTranslation, Language } from '../i18n/useTranslation';

// Re-export for backward compatibility (other components import from useAIChatWorkflow)
export type { ChatMessage, ChatImage, ChatSource };

type Operation = ChatOperation;
type OperationResult = ChatOperationResult;

/** 用于左侧「执行中」步骤列表的简短标签 */
function getOperationLabel(op: Operation, lang: 'zh' | 'en'): string {
  const zh: Record<string, string> = {
    add_node: '添加节点',
    delete_node: '删除节点',
    update_node: '修改节点',
    replace_node: '替换节点',
    add_connection: '添加连接',
    delete_connection: '删除连接',
    move_node: '移动节点'
  };
  const en: Record<string, string> = {
    add_node: 'Add node',
    delete_node: 'Delete node',
    update_node: 'Update node',
    replace_node: 'Replace node',
    add_connection: 'Add connection',
    delete_connection: 'Delete connection',
    move_node: 'Move node'
  };
  const t = lang === 'zh' ? zh : en;
  return t[op.type] ?? op.type;
}

export interface AIChatSendOptions {
  image?: ChatImage;
  useSearch?: boolean;
}

interface UseAIChatWorkflowProps {
  workflow: WorkflowState | null;
  setWorkflow: (workflow: WorkflowState | null) => void;
  /** 获取当前最新工作流（操作执行后、重试等场景下让 AI 看到实际画布状态，避免闭包旧状态） */
  getWorkflow?: () => WorkflowState | null;
  addNode: (tool: ToolDefinition, x?: number, y?: number, dataOverride?: Record<string, any>, nodeId?: string, allowOverwrite?: boolean, forceEditMode?: boolean) => WorkflowNode | null;
  deleteNode: (nodeId: string) => void;
  updateNodeData: (nodeId: string, key: string, value: any) => void;
  replaceNode: (nodeId: string, newToolId: string) => void;
  addConnection: (connection: {
    id: string;
    source_node_id: string;
    source_port_id: string;
    target_node_id: string;
    target_port_id: string;
  }) => void;
  deleteConnection: (connectionId: string) => void;
  getNodeOutputs: (node: WorkflowNode) => Port[];
  getReplaceableTools: (nodeId: string) => ToolDefinition[];
  screenToWorldCoords: (x: number, y: number) => { x: number; y: number };
  canvasRef: React.RefObject<HTMLDivElement>;
  lang: Language;
  lightX2VVoiceList?: { voices?: any[]; emotions?: string[]; languages?: any[] } | null;
  getLightX2VConfig: (workflow: WorkflowState | null) => { url: string; token: string };
  getCurrentHistoryIndex?: () => number;
  undoToIndex?: (index: number) => void;
  /** 当前各节点的输出（执行结果），用于在 prompt 中让 AI 看到节点已生成的内容 */
  activeOutputs?: Record<string, any>;
  /** 清除当前选中的运行快照，使画布进入编辑模式（AI 执行 add_node 等操作前调用） */
  clearSelectedRunId?: () => void;
  /** 工作流编辑：侧重工具/参数/连接；构思：侧重内容，多轮反问后再生成，每次 choices 含「直接生成工作流」 */
  chatMode?: 'edit' | 'ideation';
}

export const useAIChatWorkflow = ({
  workflow,
  setWorkflow,
  getWorkflow,
  addNode: originalAddNode,
  deleteNode: originalDeleteNode,
  updateNodeData: originalUpdateNodeData,
  replaceNode: originalReplaceNode,
  addConnection: originalAddConnection,
  deleteConnection: originalDeleteConnection,
  getNodeOutputs,
  getReplaceableTools,
  screenToWorldCoords,
  canvasRef,
  lang,
  lightX2VVoiceList,
  getLightX2VConfig,
  getCurrentHistoryIndex,
  undoToIndex,
  activeOutputs: activeOutputsProp = {},
  clearSelectedRunId,
  chatMode = 'edit'
}: UseAIChatWorkflowProps) => {
  const { t } = useTranslation(lang);
  // 使用 ref 存储完整的对话历史（用于发送请求）
  const chatHistoryRef = useRef<ChatMessage[]>(workflow?.chatHistory || []);
  // 使用 state 存储对话历史（用于 UI 显示）
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>(workflow?.chatHistory || []);
  const [isProcessing, setIsProcessing] = useState(false);
  /** AI 已返回 workflow 类型，正在解析并执行操作时的进度（用于左侧动效） */
  const [isExecutingOperations, setIsExecutingOperations] = useState(false);
  const [executingProgress, setExecutingProgress] = useState({ current: 0, total: 0 });
  /** 当前执行的操作步骤标签列表（用于方案 C 步骤列表） */
  const [executingStepLabels, setExecutingStepLabels] = useState<string[]>([]);
  const [aiModel, setAiModel] = useState<string>('deepseek-v3-2-251201'); // AI模型选择
  const [aiVoiceList, setAiVoiceList] = useState<{ voices?: any[]; emotions?: string[]; languages?: any[] } | null>(null);
  const [loadingAiVoiceList, setLoadingAiVoiceList] = useState(false);
  const aiVoiceListLoadedRef = useRef<string>(''); // Track which URL+token combo has been loaded
  const [aiCloneVoiceList, setAiCloneVoiceList] = useState<any[]>([]);
  const [loadingAiCloneVoiceList, setLoadingAiCloneVoiceList] = useState(false);
  const aiCloneVoiceListLoadedRef = useRef<string>('');
  /** 当前请求的 AbortController，用于用户点击「停止」取消生成 */
  const abortControllerRef = useRef<AbortController | null>(null);
  const [highlightedElements, setHighlightedElements] = useState<{
    nodeIds?: string[];
    connectionIds?: string[];
  }>({});
  const [chatContextNodes, setChatContextNodes] = useState<{ nodeId: string; name: string }[]>([]);

  // 使用ref跟踪最新的workflow状态
  const workflowRef = useRef<WorkflowState | null>(workflow);
  useEffect(() => {
    workflowRef.current = workflow;
  }, [workflow]);

  // 当工作流变化时，加载对话历史（独立存储 or 内嵌）
  useEffect(() => {
    const wfId = workflow?.id;
    if (!wfId) {
      chatHistoryRef.current = [];
      setChatHistory([]);
      return;
    }

    if (isStandalone()) {
      const fromWorkflow = workflow?.chatHistory;
      const list = Array.isArray(fromWorkflow) && fromWorkflow.length > 0 ? fromWorkflow : [];
      chatHistoryRef.current = list;
      setChatHistory(list);
      return;
    }

    let cancelled = false;
    getWorkflowChat(wfId).then((data) => {
      if (cancelled) return;
      const list = data?.messages ?? [];
      if (list.length > 0) {
        chatHistoryRef.current = list;
        setChatHistory(list);
        console.log('[AI Chat] 从独立存储加载对话历史:', list.length, '条消息');
      } else if (workflow?.chatHistory && workflow.chatHistory.length > 0) {
        chatHistoryRef.current = workflow.chatHistory;
        setChatHistory(workflow.chatHistory);
        console.log('[AI Chat] 从工作流（兼容）加载对话历史:', workflow.chatHistory.length, '条消息');
      } else {
        chatHistoryRef.current = [];
        setChatHistory([]);
      }
    }).catch(() => {
      if (!cancelled) {
        const fallback = workflow?.chatHistory ?? [];
        chatHistoryRef.current = fallback;
        setChatHistory(fallback);
      }
    });
    return () => { cancelled = true; };
  }, [workflow?.id]);

  // 添加消息到历史记录（同时更新 ref 和 state）
  const addMessageToHistory = useCallback((message: ChatMessage) => {
    chatHistoryRef.current = [...chatHistoryRef.current, message];
    setChatHistory(prev => {
      const newHistory = [...prev, message];
      console.log('[AI Chat] 添加消息到历史记录:', {
        messageId: message.id,
        role: message.role,
        totalHistory: newHistory.length,
        contentPreview: message.content.substring(0, 50) + '...'
      });
      return newHistory;
    });
  }, []);

  // 当对话历史更新时，同步到存储（独立 API 或 workflow）。仅依赖 chatHistory 与 workflow?.id，避免 workflow 引用频繁变化导致重复 PUT
  const syncChatDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    const wf = workflowRef.current;
    const wfId = wf?.id ?? workflow?.id;
    if (!wfId) return;

    const currentChatHistory = chatHistoryRef.current || [];
    const toPersist = currentChatHistory.map((m) => ({
      id: m.id,
      role: m.role,
      content: m.content,
      image: m.image,
      useSearch: m.useSearch,
      sources: m.sources,
      timestamp: m.timestamp,
      error: m.error
    }));

    if (isStandalone()) {
      const workflowChatHistory = (wf?.chatHistory ?? workflow?.chatHistory) || [];
      const needsUpdate =
        currentChatHistory.length !== workflowChatHistory.length ||
        JSON.stringify(currentChatHistory) !== JSON.stringify(workflowChatHistory);
      if (needsUpdate && wf) {
        setWorkflow({ ...wf, chatHistory: currentChatHistory, isDirty: true });
      }
      return;
    }

    if (syncChatDebounceRef.current) clearTimeout(syncChatDebounceRef.current);
    syncChatDebounceRef.current = setTimeout(() => {
      syncChatDebounceRef.current = null;
      putWorkflowChat(wfId, toPersist).then((ok) => {
        if (ok) console.log('[AI Chat] 对话历史已保存到独立存储');
      });
    }, 800);
    return () => {
      if (syncChatDebounceRef.current) clearTimeout(syncChatDebounceRef.current);
    };
  }, [chatHistory, workflow?.id]);

  // 获取前10个音色信息（必要时从后端/云端拉取）
  const getTopVoicesInfo = useCallback(async (): Promise<string> => {
    let voiceList = lightX2VVoiceList;
    if (!voiceList?.voices || voiceList.voices.length === 0) {
      voiceList = aiVoiceList;
    }
    if (!voiceList?.voices || voiceList.voices.length === 0) {
      const apiClient = (window as any).__API_CLIENT__;
      const hasApiClient = !!apiClient;
      const config = getLightX2VConfig(workflowRef.current || null);
      if (!hasApiClient && (!config.url || !config.token)) {
        console.warn('[LightX2V] Missing URL or token for AI chat voice list');
        return '';
      }

      const loadKey = `${config.url}:${config.token}`;
      if (aiVoiceListLoadedRef.current === loadKey && !loadingAiVoiceList && aiVoiceList) {
        voiceList = aiVoiceList;
      } else if (!loadingAiVoiceList) {
        setLoadingAiVoiceList(true);
        try {
          const voiceData = await lightX2VGetVoiceList(config.url, config.token);
          setAiVoiceList(voiceData);
          aiVoiceListLoadedRef.current = loadKey;
          voiceList = voiceData;
        } catch (error: any) {
          console.error('[LightX2V] Failed to load voice list for AI chat:', error);
          setAiVoiceList(null);
          aiVoiceListLoadedRef.current = '';
          return '';
        } finally {
          setLoadingAiVoiceList(false);
        }
      } else {
        return '';
      }
    }

    if (!voiceList?.voices || voiceList.voices.length === 0) return '';

    const topVoices = voiceList.voices.slice(0, 10).map((voice: any, index: number) => {
      const voiceId = voice.voice_type || voice.id || '';
      const voiceName = voice.name || voice.voice_name || voiceId;
      const gender = voice.gender || (voiceId.toLowerCase().includes('female') ? 'female' : voiceId.toLowerCase().includes('male') ? 'male' : 'unknown');
      return `${index + 1}. ${voiceName} (voiceType: "${voiceId}", gender: ${gender})`;
    }).join('\n');

    return `\n\n可用音色列表（前10个，用于TTS节点自动选择）：\n${topVoices}\n\n**TTS节点音色选择规则：**\n- 根据用户描述的需求（如"女声"、"男声"、"温柔"、"专业"等）自动选择合适的音色\n- 如果用户没有明确指定，默认使用第一个音色（通常是常用音色）\n- 在data中设置voiceType字段，例如: {"data": {"voiceType": "zh_female_vv_uranus_bigtts"}}\n- 注意：voiceType必须是上面列表中列出的voiceType值，不能使用不存在的音色ID`;
  }, [lightX2VVoiceList, aiVoiceList, loadingAiVoiceList, getLightX2VConfig]);

  // 获取前10个克隆音色信息（必要时从后端/云端拉取）
  const getTopCloneVoicesInfo = useCallback(async (): Promise<string> => {
    let cloneList = aiCloneVoiceList;
    if (!Array.isArray(cloneList) || cloneList.length === 0) {
      const apiClient = (window as any).__API_CLIENT__;
      const hasApiClient = !!apiClient;
      const config = getLightX2VConfig(workflowRef.current || null);
      if (!hasApiClient && (!config.url || !config.token)) {
        console.warn('[LightX2V] Missing URL or token for AI chat clone voice list');
        return '';
      }

      const loadKey = `${config.url}:${config.token}`;
      if (aiCloneVoiceListLoadedRef.current === loadKey && !loadingAiCloneVoiceList && aiCloneVoiceList.length > 0) {
        cloneList = aiCloneVoiceList;
      } else if (!loadingAiCloneVoiceList) {
        setLoadingAiCloneVoiceList(true);
        try {
          const cloneListResult = await lightX2VGetCloneVoiceList(config.url, config.token);
          const normalized = Array.isArray(cloneListResult) ? cloneListResult : [];
          setAiCloneVoiceList(normalized);
          aiCloneVoiceListLoadedRef.current = loadKey;
          cloneList = normalized;
        } catch (error: any) {
          console.error('[LightX2V] Failed to load clone voice list for AI chat:', error);
          setAiCloneVoiceList([]);
          aiCloneVoiceListLoadedRef.current = '';
          return '';
        } finally {
          setLoadingAiCloneVoiceList(false);
        }
      } else {
        return '';
      }
    }

    if (!Array.isArray(cloneList) || cloneList.length === 0) {
      return `\n\n**音色克隆提示：**\n- 当前没有可用的克隆音色列表\n- 如果用户需要使用克隆音色，先创建克隆音色（通过音色克隆节点生成）后再选择`;
    }

    const topClones = cloneList.slice(0, 10).map((voice: any, index: number) => {
      const speakerId = voice.speaker_id || voice.id || '';
      const voiceName = voice.name || voice.voice_name || speakerId;
      return `${index + 1}. ${voiceName} (speaker_id: "${speakerId}")`;
    }).join('\n');

    return `\n\n可用克隆音色列表（前10个，用于克隆音色节点选择）：\n${topClones}\n\n**音色克隆节点选择规则：**\n- 在data中设置speakerId字段，例如: {"data": {"speakerId": "your_speaker_id"}}\n- speakerId必须是上面列表中列出的speaker_id值，不能使用不存在的ID`;
  }, [aiCloneVoiceList, loadingAiCloneVoiceList, getLightX2VConfig]);

  // 生成工具描述
  const generateToolsDescription = useCallback(() => {
    const toolsInfo = TOOLS.map(tool => {
      const inputs = tool.inputs.map(inp => `${inp.label} (${inp.id}: ${inp.type})`).join(', ');
      const outputs = tool.outputs.map(out => `${out.label} (${out.id}: ${out.type})`).join(', ');
      const models = tool.models?.map(m => `${m.id}`).join(', ') || 'N/A';
      return `- ${tool.name} (${tool.id}): ${lang === 'zh' ? tool.description_zh : tool.description}
  Inputs: ${inputs || 'None'} ${inputs ? '(注意：这些是输入端口，只能接收匹配类型的数据)' : ''}
  Outputs: ${outputs || 'None'} ${outputs ? '(注意：这些是输出端口，只能连接到匹配类型的输入端口)' : ''}
  Models: ${models}
  Category: ${lang === 'zh' ? tool.category_zh : tool.category}`;
    }).join('\n\n');

    // 生成所有可用的toolId列表
    const allToolIds = TOOLS.map(t => t.id).join(', ');

    return `Available Tools:\n\n${toolsInfo}\n\n**重要：可用的toolId列表（只能使用这些toolId，不能创建不存在的工具）：**
**重要：模型选择规则（必须遵守）：** 禁止推荐或建议使用任何具体模型名称。每个工具的 model 参数**必须且仅能**从该工具上方「Models」列表中列出的**当前可使用的模型**里选择，不能使用列表外或臆造的模型ID。若不指定 model，系统将使用该工具的默认模型。
${allToolIds}

数据类型说明：
- TEXT: 文本类型，只能连接到TEXT类型的输入端口
- IMAGE: 图像类型，只能连接到IMAGE类型的输入端口
- AUDIO: 音频类型，只能连接到AUDIO类型的输入端口
- VIDEO: 视频类型，只能连接到VIDEO类型的输入端口

**Input 类型节点的使用规则（非常重要！）：**
**全局要求：** 无论用户需求多么简单或复杂，生成或修改的每一个工作流都必须至少包含一个 Input 类型节点（text-input, image-input, audio-input, video-input）作为最初始的用户输入层，方便用户后续直接修改输入内容。如果当前工作流缺少 Input 节点，应先添加与场景匹配的 Input 节点，再继续设置其他节点。
Input 类型的节点（text-input, image-input, audio-input, video-input）**仅作为最初始的第一层用户输入节点**，用于方便用户后续手动更改输入内容。这些节点应该：
1. **只在工作流的最开始使用**，作为用户可以直接编辑的输入源
2. **不要在工作流的中间层添加 Input 节点**，中间层应该直接通过输出端口（port）连接到下游节点的输入端口
3. **如果工作流中已经有相同类型的节点可以输出数据，应该直接连接这些节点的输出端口，而不是添加新的 Input 节点**
4. **示例：**
   - ✅ 正确：text-input → text-to-image → image-to-image（text-input 作为初始输入）
   - ✅ 正确：text-generation → text-to-image（直接连接，不需要添加 text-input）
   - ❌ 错误：text-generation → text-input → text-to-image（中间层不应该添加 Input 节点）
   - ❌ 错误：image-to-image → image-input → video-gen-image（中间层不应该添加 Input 节点）

连接规则：
1. 源节点的输出端口类型必须与目标节点的输入端口类型完全匹配
2. 例如：text-input的输出是TEXT类型，只能连接到接受TEXT类型输入的节点
3. 例如：text-to-image的输出是IMAGE类型，只能连接到接受IMAGE类型输入的节点
4. 类型不匹配的连接会导致错误，请务必检查类型兼容性
5. **优先使用已有节点的输出端口进行连接，而不是添加新的 Input 节点**`;
  }, [lang]);

  // 将节点输出内容整理为可放入 prompt 的简短形式，避免超长或 base64 撑爆上下文
  const sanitizeOutputForPrompt = useCallback((value: any): any => {
    if (value == null) return value;
    if (typeof value === 'string') {
      if (value.length > 2000) return value.slice(0, 2000) + '...[truncated]';
      if (/^data:(\w+\/\w+);base64,/.test(value)) return '[MEDIA: base64]';
      if (/^https?:\/\/.+/i.test(value) && /\.(mp4|webm|ogg|mp3|wav|png|jpg|jpeg|gif|webp)(\?|$)/i.test(value)) return '[MEDIA: URL]';
      return value;
    }
    if (Array.isArray(value)) return value.map(sanitizeOutputForPrompt);
    if (typeof value === 'object') {
      const out: Record<string, any> = {};
      for (const [k, v] of Object.entries(value)) out[k] = sanitizeOutputForPrompt(v);
      return out;
    }
    return value;
  }, []);

  // 构建AI Prompt（使用 getWorkflow 获取“当前”工作流，确保操作失败/重试后 AI 看到的是实际画布状态）
  const buildAIPrompt = useCallback(async (userInput: string) => {
    const currentWorkflow = getWorkflow?.() ?? workflow;
    if (!currentWorkflow) return '';

    const toolsDesc = generateToolsDescription();
    const voicesInfo = await getTopVoicesInfo();
    const cloneVoicesInfo = await getTopCloneVoicesInfo();
    console.log('voicesInfo', voicesInfo);
    const nodesInfo = currentWorkflow.nodes.map(n => {
      const rawOutput = activeOutputsProp[n.id] ?? (n as any).output_value ?? null;
      return {
        id: n.id,
        tool_id: n.tool_id,
        x: n.x,
        y: n.y,
        data: n.data,
        ...(rawOutput != null ? { output_value: sanitizeOutputForPrompt(rawOutput) } : {})
      };
    });
    const connectionsInfo = currentWorkflow.connections;

    const directGenerateLabel = t('generate_workflow_directly');
    const modeBlock = chatMode === 'ideation'
      ? `
**当前模式：构思模式**
- 你的首要任务是**内容构思**，而非直接操作工作流。通过多轮反问丰富用户意图（例如：想要几个分镜、每个分镜的画面偏好、风格、节奏等）。
- 每次回复末尾请提供 choices 引导用户选择或补充，且**必须包含「${directGenerateLabel}」**选项，让用户可随时跳过构思直接执行。
- 仅当用户明确确认信息充足、或用户点击/说出「${directGenerateLabel}」时，再根据已收集的信息返回 type: "workflow" 并执行操作。
- 构思阶段用 type: "conversation" 回复，用 choices 做多轮澄清（如分镜数、分镜一选项、分镜二选项等），最后再生成工作流。
`
      : `
**当前模式：工作流编辑模式**
- 侧重对当前工作流的**操作**（选择更合适的工具、参数、位置、连接等）。当用户意图明确时可直接生成并执行操作。
`;

    return `你是一个工作流编辑助手。用户会通过自然语言与你交流，可能是：
1. 想要修改工作流（添加节点、删除节点、修改节点、添加连接等）
2. 普通对话（打招呼、询问问题、闲聊等，与工作流无关）
${modeBlock}

当前工作流状态：
- 节点列表：${JSON.stringify(nodesInfo)}（其中 outputValue 表示该节点当前已生成/展示的输出内容，便于理解用户如「改这段文案」等请求）
- 连接列表：${JSON.stringify(connectionsInfo)}
${toolsDesc}${voicesInfo}${cloneVoicesInfo}

用户输入：${userInput}

**复杂工作流的构建策略（非常重要！）：**
- **对于复杂场景的工作流（需要很多个有关联性的提示词），总是可以在第一步使用 text-generation 节点作为规划器（planner）**
- **规划器的作用：** 生成多个字段的输出，每个字段作为后续节点的 prompt，确保所有提示词之间的关联性和一致性
- **工作流结构：** text-generation (planner) → 多个下游节点（每个节点使用 planner 的一个输出字段作为输入）
- **示例：** 如果用户要求创建复杂的多步骤视频生成工作流，应该：
  1. 第一步：创建 text-generation 节点作为规划器，设置 mode 为 "custom"，配置 customInstruction 和 custom_outputs
  2. customInstruction 应该详细说明需要生成的所有提示词字段及其用途（如：场景描述、角色描述、动作描述、运镜描述等）
  3. custom_outputs 应该定义所有需要的输出字段，每个字段对应一个后续节点的输入
  4. 后续步骤：将规划器的各个输出字段连接到对应的下游节点（如：场景描述 → text-to-image，角色描述 → text-to-image，动作描述 → video-gen-text 等）

请先判断用户的意图：
- 如果用户想要修改工作流（如"添加一个文本输入节点"、"删除这个节点"、"连接这两个节点"等），生成JSON格式的操作指令
- 如果用户只是普通对话（如"你好"、"谢谢"、"这个功能怎么用"等），返回对话类型的JSON格式，用自然语言正常回答用户问题

**输出格式：**

1. **如果用户想要修改工作流**，输出JSON格式：
{
  "type": "workflow",
  "operations": [
    {
      "type": "add_node",
      "details": {
        "toolId": "text-input",
        "x": 100,
        "y": 200,
        "data": { "value": "默认文本" },
        "nodeId": "text_input_1"
      }
    }
  ],
  "explanation": "对操作的解释说明"
}

2. **如果用户只是普通对话**（如"你好"、"谢谢"、"这个功能怎么用"等），输出JSON格式：
{
  "type": "conversation",
  "content": "你的回答内容（用自然语言，不要说明这是普通对话）"
}

3. **如果用户描述不明确，需要更多信息**，输出JSON格式：
{
  "type": "conversation",
  "content": "你想查询什么内容？请具体说明你想要修改工作流（比如添加节点、删除节点等）还是进行其他操作。"
}
   - **可选字段 choices**：当用户需求不明确时，用 choices 引导用户明确**他想要的内容**（如：动画场景、故事内容、角色形象、风格、时长等），**不要**把 choices 写成工具或工作流步骤。前端会把选项展示为可点击按钮，用户点击后该选项会作为下一条消息发送。例如用户说「想做动画」时，应引导用户明确场景/故事/角色，例如：
{"type": "conversation", "content": "好的，先确定一下您想要的动画内容：您更想先定下哪一块？", "choices": ["先定故事/剧本内容", "先定角色形象和风格", "先定场景和镜头"]}
   - **策略**：只有在信息足够、能明确执行工作流操作时才返回 type: "workflow" 并执行；信息不足时返回 type: "conversation"，用 choices 引导用户补充**想要的内容**（场景、故事、角色、风格等），待用户明确后再执行操作。

支持的操作类型：
1. add_node: 添加节点（toolId必需，x/y可选，data/model可选，nodeId可选，tempId可选）
   - **重要：toolId必须是Available Tools列表中列出的工具ID，不能使用不存在的工具**
   - **在生成add_node操作前，必须验证toolId是否在Available Tools列表中**
   - **不能创建或假设不存在的工具，只能使用现有的工具**
   - **Input 类型节点的使用规则：**
     * Input 类型节点（text-input, image-input, audio-input, video-input）**仅作为最初始的第一层用户输入节点**，用于方便用户后续手动更改输入内容
     * **不要在工作流的中间层添加 Input 节点**，中间层应该直接通过输出端口（port）连接到下游节点的输入端口
     * **如果工作流中已经有相同类型的节点可以输出数据，应该直接连接这些节点的输出端口，而不是添加新的 Input 节点**
     * **示例：**
       - ✅ 正确：text-input → text-to-image → image-to-image（text-input 作为初始输入）
       - ✅ 正确：text-generation → text-to-image（直接连接，不需要添加 text-input）
       - ❌ 错误：text-generation → text-input → text-to-image（中间层不应该添加 Input 节点）
       - ❌ 错误：image-to-image → image-input → video-gen-image（中间层不应该添加 Input 节点）
   - 常用toolId: "text-input"（文本输入）, "text-to-image"（文生图）, "image-to-image"（图生图）, "video-gen-image"（图生视频）, "video-gen-text"（文生视频）, "text-generation"（文本生成）, "video-input"（视频输入）, "image-input"（图像输入）, "audio-input"（音频输入）
   - **参数设置规则：**
     * 可以在data中指定任意参数，例如: {"data": {"value": "文本内容", "aspectRatio": "16:9", "voiceType": "zh_female_vv_uranus_bigtts"}}
     * 若需指定模型，可在顶层指定 model，且**必须**从该工具在 Available Tools 中的「Models」列表（当前可使用的模型）里选择，不能推荐或自造模型名，例如: {"model": "该工具Models列表中的某个id"}
     * **如果某个参数没有在data中指定，系统会自动使用该工具和模型的默认值**
     * 例如：文生图节点默认aspectRatio为"1:1"，TTS节点（lightx2v模型）默认voiceType为"zh_female_vv_uranus_bigtts"、emotionScale为3等
     * 你只需要指定需要修改的参数，其他参数会自动使用默认值
   - **text-generation节点（文本生成大模型）的特殊配置：**
     * **作为规划器（Planner）的重要作用：**
       - **对于复杂场景的工作流（需要很多个有关联性的提示词），总是可以在第一步使用 text-generation 节点作为规划器（planner）**
       - **这个规划器节点会生成多个字段的输出，每个字段作为后续节点的 prompt**
       - **使用模式：** 设置 mode 为 "custom"，配置 customInstruction 和 custom_outputs
       - **工作流结构：** text-generation (planner) → 多个下游节点（每个节点使用 planner 的一个输出字段作为输入）
       - **示例：** 如果用户要求创建复杂的多步骤视频生成工作流，应该：
         1. 第一步：创建 text-generation 节点作为规划器，设置 customInstruction 指导生成多个相关的提示词（如：场景描述、角色描述、动作描述、运镜描述等），设置 custom_outputs 定义这些输出字段
         2. 后续步骤：将规划器的各个输出字段连接到对应的下游节点（如：场景描述 → text-to-image，角色描述 → text-to-image，动作描述 → video-gen-text 等）
     * **mode**: 可选，默认为"basic"。可选值："basic"（基础模式）、"enhance"（提示词增强）、"summarize"（总结）、"polish"（润色）、"custom"（自定义）
     * **customInstruction**: 当mode为"custom"时，必须设置此字段。这是系统指令，用于指导AI输出指定的内容和字段。例如：
       {"data": {"mode": "custom", "customInstruction": "你是一位专业的数字人视频脚本编写者。根据用户输入，生成语音脚本、语调指令、肖像提示和数字人视频动作提示。所有输出字段必须使用与用户输入相同的语言。"}}
     * **custom_outputs**: 可选，用于定义结构化输出字段。这是一个数组，每个元素包含id、label、description。当设置了custom_outputs时，AI会以JSON格式输出，每个字段对应一个输出端口，可以连接到下游节点。例如：
       {"data": {"mode": "custom", "customInstruction": "根据用户输入生成TTS文本和语气指令", "custom_outputs": [{"id": "speech_text", "label": "语音脚本", "description": "人物对听众说的话，纯对话文本，不包含语气标记"}, {"id": "tone", "label": "语调指令", "description": "语音风格的提示，包含情感、节奏、重音点"}]}}
     * **使用场景示例：**
       - **复杂工作流规划器：** 对于需要多个有关联性提示词的复杂场景，第一步创建 text-generation 节点作为规划器，生成所有需要的提示词字段，然后连接到下游节点
       - 生成数字人视频脚本：设置customInstruction指导生成语音脚本、语调指令、肖像提示、动作提示，设置custom_outputs定义这些输出字段
       - 生成多分镜视频规划：设置customInstruction指导生成多个分镜的图像提示和运镜描述，设置custom_outputs定义每个分镜的输出字段
       - 生成TTS文本和语气：设置customInstruction指导同时生成TTS文本和语气指令，设置custom_outputs定义这两个输出字段
       - **TTS 文本撰写规则：** 当 text-generation 节点用于生成 TTS 文本（如 speech_text 字段）时，输出必须是纯粹的语音内容，禁止在括号中附加动作、舞台指令或其他系统说明。语气、情感、节奏等指令应放在 customInstruction、tone 或其他上下文字段里；动作、表情、镜头等内容则放在后续（如果有）的数字人视频提示词节点中。
     * **重要提示：**
       - **对于复杂场景，优先考虑使用 text-generation 节点作为第一步的规划器**
       - customInstruction应该详细说明每个输出字段的用途和要求
       - custom_outputs中的id会作为输出端口ID，用于后续连接操作
       - 如果用户要求生成多个相关字段（如TTS文本+语气指令），应该使用custom_outputs而不是创建多个节点
   - **nodeId: 可选，指定节点的ID，用于后续引用和修改。建议使用有意义的ID，如"video_merge_1"、"text_input_1"等。如果不指定，系统会自动生成**
   - tempId: 可选，用于后续连接操作时引用此节点，例如: {"tempId": "node_0"}。如果同时指定了nodeId，tempId可以省略（直接使用nodeId）
2. delete_node: 删除节点（支持nodeId、nodeIds数组、toolId、all）
3. update_node: 更新节点参数（nodeId必需，updates对象，支持嵌套路径如"data.value"）
   - 例如: {"updates": {"data.value": "新文本", "data.model": "新模型", "toolId": "image-to-image"}}
   - **可直接修改某节点的生成结果（输出值）**：updates 可包含 outputValue，用于覆盖该节点当前的输出（如文本生成节点已生成的文案）。例如用户不满意某文案时：{"updates": {"outputValue": "用户满意的新文案"}}。修改后会同步到画布展示与下游节点执行时的输入。多输出节点可传对象，如 {"outputValue": {"out-text": "文案", "out-tone": "语气"}}
   - **重要：如果更新会改变节点的 toolId（例如从 text-to-image 改为 image-to-image），必须在 add_connection 之前执行，确保目标节点有正确的输入端口**
   - 例如：如果要连接图像输入到某个节点，但该节点当前是 text-to-image，需要先执行 update_node 将其改为 image-to-image，然后再执行 add_connection
4. replace_node: 替换节点类型（nodeId和newToolId必需）
   - **重要：如果替换会改变节点的输入端口类型（例如从 text-to-image 改为 image-to-image），必须在 add_connection 之前执行，确保目标节点有正确的输入端口**
   - 例如：如果要连接图像输入到某个节点，但该节点当前是 text-to-image，需要先执行 replace_node 将其改为 image-to-image，然后再执行 add_connection
5. add_connection: 添加连接（sourceNodeId和targetNodeId必需，source_port_id和targetPortId可选）
   - **重要：连接必须建立在已有的节点基础上，不能连接不存在的节点**
   - **重要：如果目标节点需要改变类型才能接受某个输入，必须先执行 update_node 或 replace_node 改变目标节点的 toolId，然后再执行 add_connection**
   - **执行顺序示例：**
     * 如果要连接图像到某个节点，但该节点当前是 text-to-image（只有 in-text 输入）：
       - 错误顺序：先 add_connection（尝试连接到 in-image，但不存在），后 update_node（改为 image-to-image）
       - 正确顺序：先 update_node（改为 image-to-image，现在有 in-image 输入），后 add_connection（连接到 in-image）
   - sourceNodeId和targetNodeId可以是节点的实际ID（如果add_node时指定了nodeId，使用该nodeId），也可以是add_node操作中指定的tempId或数字索引（"0", "1"等）
   - 端口会自动匹配，但也可以明确指定，例如: {"source_port_id": "out-text", "targetPortId": "in-text"}
   - 在生成连接前，必须检查：
     * **目标节点当前的实际 toolId 是否有对应的输入端口**（例如：如果要连接图像，目标节点必须是 image-to-image，不能是 text-to-image）
     * 源节点的输出类型和目标节点的输入类型是否匹配
   - **数据类型匹配规则（必须严格遵守）：**
     * 可用的数据类型：TEXT（文本）、IMAGE（图像）、AUDIO（音频）、VIDEO（视频）
     * 源节点的输出端口类型必须与目标节点的输入端口类型完全匹配
     * 例如：out-text输出只能连接到in-text输入，out-image输出只能连接到in-image输入
     * 不能将out-text连接到in-image，不能将in-image连接到out-text，等等
6. delete_connection: 删除连接（支持connectionId或sourceNodeId+targetNodeId）
7. move_node: 移动节点（nodeId必需，x/y或deltaX/deltaY）

重要提示：
- **操作是按顺序执行的，后面的操作依赖于前面操作的结果**
- **如果目标节点需要改变类型才能接受某个输入，必须先改变节点类型（update_node/replace_node），再添加连接（add_connection）**
- **对于普通对话，必须返回 {"type": "conversation", "content": "回答内容"} 格式；可选增加 "choices" 引导用户明确想要的内容（如场景、故事、角色、风格等），choices 应是用户想选的内容描述，不是工具名或步骤名，前端会展示为可点击按钮，用户点击后该选项会作为下一条消息发送**
- 如果用户描述不明确（如只说「想做动画」），返回 conversation 并可用 choices 引导用户补充**内容**（例如：先定故事、先定角色、先定场景等）
- 验证操作的合法性（节点ID是否存在、连接是否有效等）
- 如果操作不合法，返回 {"type": "conversation", "content": "错误说明"} 而不是执行
- 对于批量操作，生成多个操作指令，并确保顺序正确
- 节点位置如果不指定，会自动计算合适的位置（建议x间距400，y间距200）

**add_node操作的关键规则（必须严格遵守）：**
1. **只能使用Available Tools列表中列出的toolId，不能使用不存在的toolId**
2. **在生成add_node操作前，必须检查toolId是否在Available Tools列表中**
3. **不能创建、假设或生成不存在的工具，例如：不能使用"video-merge"、"video-combine"等不存在的toolId**
4. **如果用户要求的功能没有对应的工具，返回错误说明该功能不可用，而不是创建不存在的工具**
5. **Input 类型节点的使用规则（非常重要！）：**
   - Input 类型节点（text-input, image-input, audio-input, video-input）**仅作为最初始的第一层用户输入节点**，用于方便用户后续手动更改输入内容
   - **不要在工作流的中间层添加 Input 节点**，中间层应该直接通过输出端口（port）连接到下游节点的输入端口
   - **如果工作流中已经有相同类型的节点可以输出数据，应该直接连接这些节点的输出端口，而不是添加新的 Input 节点**
   - **在生成 add_node 操作前，检查工作流中是否已经有可以输出相同类型数据的节点，如果有，应该使用 add_connection 连接，而不是添加新的 Input 节点**
   - **示例：**
     * ✅ 正确：text-input → text-to-image → image-to-image（text-input 作为初始输入）
     * ✅ 正确：text-generation → text-to-image（直接连接，不需要添加 text-input）
     * ❌ 错误：text-generation → text-input → text-to-image（中间层不应该添加 Input 节点）
     * ❌ 错误：image-to-image → image-input → video-gen-image（中间层不应该添加 Input 节点）
6. **参数设置：**
   - **模型必须从当前可使用的模型中选择：** 禁止推荐具体模型。若工具有模型选项（见 Available Tools 中该工具的「Models」列表），指定 model 时**必须**使用该列表中某一项，不能使用列表外或臆造的模型ID。不指定时系统使用该工具默认模型。
   - 可以在data中指定任意参数，例如: {"data": {"value": "文本", "aspectRatio": "16:9"}}
   - **不需要指定所有参数，系统会自动使用默认值**
   - 文生图/图生图节点默认aspectRatio为"1:1"，视频节点默认aspectRatio为"16:9"
   - **TTS节点（lightx2v模型）音色选择：**
     * 如果用户明确指定了音色需求（如"女声"、"男声"、"温柔"、"专业"等），根据可用音色列表自动选择最匹配的音色
     * 如果用户没有明确指定，默认使用第一个音色（通常是常用音色）
     * voiceType必须在可用音色列表中，不能使用不存在的音色ID
     * 其他默认参数：emotionScale为3、speechRate为0、pitch为0、loudnessRate为0、resourceId为"seed-tts-2.0"
   - **text-generation节点（文本生成）默认参数：**
     * mode默认为"basic"
     * custom_outputs默认为[{"id": "out-text", "label": "执行结果", "description": "Main text response."}]
     * **当用户需要生成多个相关字段时（如TTS文本+语气指令、多个分镜提示等），应该设置mode为"custom"，并配置customInstruction和custom_outputs**
   - 音色克隆节点默认style为"正常"、speed为1.0、volume为0、pitch为0、language为"ZH_CN"
6. **text-generation节点的使用建议（非常重要！）：**
   - **对于复杂场景的工作流（需要很多个有关联性的提示词），总是可以在第一步使用 text-generation 节点作为规划器（planner）**
   - **规划器的作用：** 生成多个字段的输出，每个字段作为后续节点的 prompt，确保所有提示词之间的关联性和一致性
   - **工作流构建策略：**
     * 如果用户要求创建复杂的工作流（涉及多个步骤、多个提示词、多个生成任务），第一步应该创建 text-generation 节点作为规划器
     * 规划器节点应该设置 mode 为 "custom"，配置 customInstruction 和 custom_outputs
     * customInstruction 应该详细说明需要生成的所有提示词字段及其用途
     * custom_outputs 应该定义所有需要的输出字段，每个字段对应一个后续节点的输入
     * 然后将规划器的各个输出字段连接到对应的下游节点
   - **示例场景：**
     * 复杂视频生成工作流：text-generation (planner) → 生成场景提示、角色提示、动作提示、运镜提示 → 分别连接到 text-to-image、video-gen-text 等节点
     * 多分镜视频工作流：text-generation (planner) → 生成多个分镜的提示词 → 分别连接到多个 text-to-image 节点
     * 数字人视频工作流：text-generation (planner) → 生成语音脚本、语调指令、肖像提示、动作提示 → 分别连接到 TTS、text-to-image、video-gen 等节点
   - 如果用户要求生成多个相关字段（如"生成TTS文本和语气指令"、"生成多个分镜的提示词"），应该创建一个text-generation节点，设置mode为"custom"，配置customInstruction和custom_outputs
   - customInstruction应该详细说明每个输出字段的用途、格式要求和语言要求
   - custom_outputs中的id会作为输出端口ID，后续可以通过add_connection连接到其他节点
   - 例如：如果用户说"生成数字人视频脚本"，应该创建一个text-generation节点，设置customInstruction指导生成语音脚本、语调指令、肖像提示、动作提示，设置custom_outputs定义这四个输出字段
7. **禁止推荐具体模型：** 不要推荐、建议或写出任何具体模型名称。需要指定 model 时，**必须**从 Available Tools 里该工具当前的「Models」列表（即当前可使用的模型）中选取其一；不指定则使用系统默认。

**操作执行顺序的关键规则（非常重要！）：**
1. **所有操作是按顺序执行的，后面的操作依赖于前面操作的结果**
2. **如果目标节点需要改变类型才能接受某个输入，必须先改变节点类型，再添加连接**
   - 例如：如果要连接图像输入到某个节点，但该节点当前是 text-to-image（只有 in-text 输入），需要先执行 update_node 或 replace_node 将其改为 image-to-image（有 in-image 输入），然后再执行 add_connection
   - 错误示例：先 add_connection（尝试连接到 text-to-image 的 in-image，但该端口不存在），后 update_node（改为 image-to-image）
   - 正确示例：先 update_node（将 text-to-image 改为 image-to-image），后 add_connection（连接到 image-to-image 的 in-image）
3. **操作顺序建议：**
   - 第一步：add_node（添加新节点）
   - 第二步：update_node / replace_node（修改现有节点的类型或参数，确保节点有正确的输入端口）
   - 第三步：delete_connection（删除不需要的连接）
   - 第四步：add_connection（添加新连接，此时目标节点已经有正确的输入端口）
   - 第五步：move_node（移动节点位置）
4. **在生成 add_connection 操作前，必须确保：**
   - 目标节点已经存在
   - 目标节点已经有正确的 toolId（如果目标节点需要是 image-to-image 才能接受图像输入，必须先执行 update_node 将其改为 image-to-image）
   - 目标节点有对应的输入端口（例如：如果要连接图像，目标节点必须是 image-to-image 或支持图像输入的工具）

**连接操作的关键规则：**
1. 连接必须建立在已有的节点基础上（sourceNodeId和targetNodeId必须存在于当前工作流或本次操作中已创建的节点）
2. 数据类型必须完全匹配：源节点的输出类型 = 目标节点的输入类型
3. 在生成add_connection操作前，必须检查：
   - 源节点是否存在
   - 目标节点是否存在
   - **目标节点是否已经有正确的 toolId（如果需要接受图像输入，必须是 image-to-image，不能是 text-to-image）**
   - 源节点的输出端口类型
   - 目标节点的输入端口类型（**必须检查目标节点当前的实际 toolId 对应的输入端口**）
   - 两者类型是否匹配
4. 如果类型不匹配，不要生成连接操作，而是：
   - 先执行 replace_node 替换目标节点的toolId
   - 然后再执行 add_connection
5. **常见错误示例：**
   - 错误：先 add_connection（连接到 text-to-image 的 in-image），后 replace_node（改为 image-to-image）
   - 正确：先 replace_node（改为 image-to-image），后 add_connection（连接到 image-to-image 的 in-image）

**在add_node时可以使用tempId，然后在add_connection时使用这个tempId来引用节点**

输出ONLY JSON，不要其他文本。`;
  }, [workflow, getWorkflow, generateToolsDescription, getTopVoicesInfo, getTopCloneVoicesInfo, activeOutputsProp, sanitizeOutputForPrompt, chatMode]);

  // 执行添加节点操作
  const executeAddNode = useCallback(async (
    details: {
      tool_id?: string;
      toolId?: string; // legacy
      x?: number;
      y?: number;
      data?: Record<string, any>;
      model?: string;
      tempId?: string; // AI生成的临时ID，用于后续连接操作
      nodeId?: string; // AI指定的节点ID，用于后续引用和修改
    }
  ): Promise<OperationResult> => {
    const toolId = details.tool_id ?? (details as any).toolId;
    const tool = TOOLS.find(t => t.id === toolId);
    if (!tool) {
      return {
        success: false,
        operation: { type: 'add_node', details },
        error: `Tool not found: ${toolId}`
      };
    }

    if (details.model && tool.models) {
      const modelExists = tool.models.some(m => m.id === details.model);
      if (!modelExists) {
        return {
          success: false,
          operation: { type: 'add_node', details },
          error: `Model not found: ${details.model} for tool ${details.tool_id}`
        };
      }
    }

    let nodeX = details.x;
    let nodeY = details.y;

    if (nodeX === undefined || nodeY === undefined) {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        const center = screenToWorldCoords(rect.width / 2, rect.height / 2);
        nodeX = center.x;
        nodeY = center.y;
      } else {
        const maxX = workflow && workflow.nodes.length > 0
          ? Math.max(...workflow.nodes.map(n => n.x))
          : 0;
        nodeX = maxX + 400;
        nodeY = 200;
      }
    }

    // 合并默认参数：工具级别 -> 模型级别 -> AI指定的值
    const nodeData: Record<string, any> = {};

    // 1. 先应用工具级别的默认参数
    if (tool.defaultParams) {
      Object.assign(nodeData, tool.defaultParams);
    }

    // 2. 确定使用的模型
    const selectedModel = details.model || (tool.models && tool.models.length > 0 ? tool.models[0].id : undefined);
    if (selectedModel) {
      nodeData.model = selectedModel;

      // 3. 应用模型级别的默认参数（会覆盖工具级别的相同参数）
      const modelDef = tool.models?.find(m => m.id === selectedModel);
      if (modelDef?.defaultParams) {
        Object.assign(nodeData, modelDef.defaultParams);
      }
    }

    // 4. 最后应用AI指定的参数（优先级最高，会覆盖所有默认值）
    if (details.data) {
      Object.assign(nodeData, details.data);
    }

    // 如果AI在顶层指定了model，也要应用
    if (details.model) {
      nodeData.model = details.model;
    }

    // 若当前处于运行快照视图，先清除快照选择使画布进入编辑模式，再添加节点（通过 forceEditMode 让 addNode 在当次调用中生效）
    clearSelectedRunId?.();
    // 如果AI指定了nodeId，使用指定的ID；否则使用tempId（如果存在）或自动生成
    const specifiedNodeId = details.nodeId || details.tempId;
    // 对于AI Chat场景，如果指定了nodeId，允许覆盖已存在的节点（使用相同的ID）；forceEditMode 为 true 时即使选中了运行快照也允许添加
    const newNode = originalAddNode(tool, nodeX, nodeY, nodeData, specifiedNodeId, !!specifiedNodeId, true);

    if (!newNode) {
      return {
        success: false,
        operation: { type: 'add_node', details },
        error: 'Failed to add node'
      };
    }

    return {
      success: true,
      operation: { type: 'add_node', details },
      result: {
        nodeId: newNode.id,
        node: newNode,
        tempId: details.tempId // 返回临时ID，用于后续映射
      },
      affectedElements: { nodeIds: [newNode.id] }
    };
  }, [workflow, originalAddNode, screenToWorldCoords, canvasRef, clearSelectedRunId]);

  // 执行删除节点操作
  const executeDeleteNode = useCallback(async (
    details: {
      nodeId?: string;
      nodeIds?: string[];
      tool_id?: string;
      toolId?: string; // legacy
      all?: boolean;
    }
  ): Promise<OperationResult> => {
    if (!workflow) {
      return {
        success: false,
        operation: { type: 'delete_node', details },
        error: 'No workflow loaded'
      };
    }

    let nodeIdsToDelete: string[] = [];

    if (details.all) {
      nodeIdsToDelete = workflow.nodes.map(n => n.id);
    } else if (details.tool_id ?? (details as any).toolId) {
      const byToolId = details.tool_id ?? (details as any).toolId;
      nodeIdsToDelete = workflow.nodes
        .filter(n => n.tool_id === byToolId)
        .map(n => n.id);
    } else if (details.nodeIds) {
      nodeIdsToDelete = details.nodeIds;
    } else if (details.nodeId) {
      nodeIdsToDelete = [details.nodeId];
    } else {
      return {
        success: false,
        operation: { type: 'delete_node', details },
        error: 'No node specified for deletion'
      };
    }

    const missingNodes = nodeIdsToDelete.filter(
      id => !workflow.nodes.some(n => n.id === id)
    );
    if (missingNodes.length > 0) {
      return {
        success: false,
        operation: { type: 'delete_node', details },
        error: `Nodes not found: ${missingNodes.join(', ')}`
      };
    }

    const deletedNodeIds: string[] = [];
    for (const nodeId of nodeIdsToDelete) {
      originalDeleteNode(nodeId);
      deletedNodeIds.push(nodeId);
    }

    return {
      success: true,
      operation: { type: 'delete_node', details },
      result: { deletedNodeIds },
      affectedElements: { nodeIds: deletedNodeIds }
    };
  }, [workflow, originalDeleteNode]);

  // 执行更新节点操作
  const executeUpdateNode = useCallback(async (
    details: {
      nodeId: string;
      updates: Record<string, any>;
    }
  ): Promise<OperationResult> => {
    if (!workflow) {
      return {
        success: false,
        operation: { type: 'update_node', details },
        error: 'No workflow loaded'
      };
    }

    const node = workflow.nodes.find(n => n.id === details.nodeId);
    if (!node) {
      return {
        success: false,
        operation: { type: 'update_node', details },
        error: `Node not found: ${details.nodeId}`
      };
    }

    // 切回编辑模式，否则 useNodeManagement.updateNodeData 会因 selectedRunId 直接 return，更新不会生效
    clearSelectedRunId?.();

    // 如果更新包含 tool_id，应该使用 replaceNode 而不是 updateNodeData
    const newToolIdFromUpdate = details.updates.tool_id ?? (details.updates as any).toolId;
    if (newToolIdFromUpdate) {
      const newToolId = newToolIdFromUpdate;
      const otherUpdates = { ...details.updates };
      delete otherUpdates.tool_id;
      delete (otherUpdates as any).toolId;

      // 先替换 toolId
      originalReplaceNode(details.nodeId, newToolId);

      // 然后更新其他字段：先收集所有 data 相关更新，一次性合并，避免多次 setWorkflow 互相覆盖（如 customInstruction + custom_outputs）
      if (Object.keys(otherUpdates).length > 0) {
        const tool = TOOLS.find(t => t.id === newToolId);
        if (otherUpdates.model && tool?.models) {
          const modelExists = tool.models.some(m => m.id === otherUpdates.model);
          if (!modelExists) {
            return {
              success: false,
              operation: { type: 'update_node', details },
              error: `Invalid model: ${otherUpdates.model} for tool ${newToolId}`
            };
          }
        }

        const dataMerge: Record<string, any> = {};
        if (typeof otherUpdates.data === 'object' && otherUpdates.data !== null && !Array.isArray(otherUpdates.data)) {
          Object.assign(dataMerge, otherUpdates.data);
        }
        for (const [key, value] of Object.entries(otherUpdates)) {
          if (key === 'data') continue;
          if (key.includes('.') && key.split('.')[0] === 'data') {
            dataMerge[key.split('.').slice(1).join('.')] = value;
          }
        }
        if (Object.keys(dataMerge).length > 0) {
          originalUpdateNodeData(details.nodeId, '__mergeData', dataMerge);
        }
        const norm = (k: string) => (k === 'outputValue' ? 'output_value' : k === 'toolId' ? 'tool_id' : k);
        for (const [key, value] of Object.entries(otherUpdates)) {
          if (key === 'data' || (key.includes('.') && key.split('.')[0] === 'data')) continue;
          originalUpdateNodeData(details.nodeId, norm(key), value);
        }
      }

      return {
        success: true,
        operation: { type: 'update_node', details },
        result: { nodeId: details.nodeId },
        affectedElements: { nodeIds: [details.nodeId] }
      };
    }

    // 如果没有 toolId 更新，正常处理其他更新：先收集所有 data 相关更新，一次性合并，避免多次 setWorkflow 互相覆盖（如 customInstruction + custom_outputs）
    const tool = TOOLS.find(t => t.id === node.tool_id);
    if (details.updates.model && tool?.models) {
      const modelExists = tool.models.some(m => m.id === details.updates.model);
      if (!modelExists) {
        return {
          success: false,
          operation: { type: 'update_node', details },
          error: `Invalid model: ${details.updates.model} for tool ${node.tool_id}`
        };
      }
    }

    const dataMerge: Record<string, any> = {};
    if (typeof details.updates.data === 'object' && details.updates.data !== null && !Array.isArray(details.updates.data)) {
      Object.assign(dataMerge, details.updates.data);
    }
    for (const [key, value] of Object.entries(details.updates)) {
      if (key === 'data') continue;
      if (key.includes('.')) {
        const parts = key.split('.');
        if (parts[0] === 'data') {
          dataMerge[parts.slice(1).join('.')] = value;
        } else {
          return {
            success: false,
            operation: { type: 'update_node', details },
            error: `Nested path update not supported: ${key}`
          };
        }
      }
    }
    if (Object.keys(dataMerge).length > 0) {
      originalUpdateNodeData(details.nodeId, '__mergeData', dataMerge);
    }
    const norm = (k: string) => (k === 'outputValue' ? 'output_value' : k === 'toolId' ? 'tool_id' : k);
    for (const [key, value] of Object.entries(details.updates)) {
      if (key === 'data' || key.includes('.')) continue;
      originalUpdateNodeData(details.nodeId, norm(key), value);
    }

    return {
      success: true,
      operation: { type: 'update_node', details },
      result: { nodeId: details.nodeId },
      affectedElements: { nodeIds: [details.nodeId] }
    };
  }, [workflow, originalUpdateNodeData, originalReplaceNode, clearSelectedRunId]);

  // 执行替换节点操作
  const executeReplaceNode = useCallback(async (
    details: {
      nodeId: string;
      newToolId: string;
    }
  ): Promise<OperationResult> => {
    if (!workflow) {
      return {
        success: false,
        operation: { type: 'replace_node', details },
        error: 'No workflow loaded'
      };
    }

    const node = workflow.nodes.find(n => n.id === details.nodeId);
    if (!node) {
      return {
        success: false,
        operation: { type: 'replace_node', details },
        error: `Node not found: ${details.nodeId}`
      };
    }

    const newTool = TOOLS.find(t => t.id === details.newToolId);
    if (!newTool) {
      return {
        success: false,
        operation: { type: 'replace_node', details },
        error: `Tool not found: ${details.newToolId}`
      };
    }

    const replaceableTools = getReplaceableTools(details.nodeId);
    const isCompatible = replaceableTools.some(t => t.id === details.newToolId);

    if (!isCompatible) {
      return {
        success: false,
        operation: { type: 'replace_node', details },
        error: `Tool ${details.newToolId} is not compatible with node ${details.nodeId}`
      };
    }

    originalReplaceNode(details.nodeId, details.newToolId);

    return {
      success: true,
      operation: { type: 'replace_node', details },
      result: { nodeId: details.nodeId, oldToolId: node.tool_id, newToolId: details.newToolId },
      affectedElements: { nodeIds: [details.nodeId] }
    };
  }, [workflow, originalReplaceNode, getReplaceableTools]);

  // 执行添加连接操作（workflowOverride：同一批操作中前面 update_node 等已修改的 workflow，用于查找新 port 如 custom_outputs 更新后的输出）
  const executeAddConnection = useCallback(async (
    details: {
      source_node_id?: string;
      sourceNodeId?: string;
      source_port_id?: string;
      sourcePortId?: string;
      target_node_id?: string;
      targetNodeId?: string;
      target_port_id?: string;
      targetPortId?: string;
    },
    createdNodes?: Map<string, WorkflowNode>,
    tempIdToNodeId?: Map<string, string>,
    createdNodesInOrder?: WorkflowNode[],
    workflowOverride?: WorkflowState | null
  ): Promise<OperationResult> => {
    // 优先使用调用方传入的 workflow（同一批操作中已应用前面 update_node 的中间状态），否则用 ref
    let currentWorkflow = workflowOverride ?? workflowRef.current;
    if (!currentWorkflow) {
      return {
        success: false,
        operation: { type: 'add_connection', details },
        error: 'No workflow loaded'
      };
    }

    // 将临时ID转换为实际节点ID（兼容 AI 传 sourceNodeId 或 source_node_id）
    let sourceNodeId = details.source_node_id ?? (details as any).sourceNodeId;
    let targetNodeId = details.target_node_id ?? (details as any).targetNodeId;

    if (tempIdToNodeId) {
      // 如果sourceNodeId是临时ID，转换为实际ID
      if (tempIdToNodeId.has(sourceNodeId)) {
        sourceNodeId = tempIdToNodeId.get(sourceNodeId)!;
      }
      // 如果targetNodeId是临时ID，转换为实际ID
      if (tempIdToNodeId.has(targetNodeId)) {
        targetNodeId = tempIdToNodeId.get(targetNodeId)!;
      }
    }

    // 先从workflow中查找节点，如果找不到，从刚创建的节点中查找
    let sourceNode = currentWorkflow.nodes.find(n => n.id === sourceNodeId);
    let targetNode = currentWorkflow.nodes.find(n => n.id === targetNodeId);

    // 如果节点不在workflow中，尝试从刚创建的节点中获取
    if (!sourceNode && createdNodes) {
      sourceNode = createdNodes.get(sourceNodeId) || undefined;
    }
    if (!targetNode && createdNodes) {
      targetNode = createdNodes.get(targetNodeId) || undefined;
    }

    // 如果仍然找不到，尝试通过数字索引匹配（AI可能使用"0", "1"等作为索引）
    if (!sourceNode && createdNodesInOrder && createdNodesInOrder.length > 0) {
      const index = parseInt(sourceNodeId, 10);
      if (!isNaN(index) && index >= 0 && index < createdNodesInOrder.length) {
        sourceNode = createdNodesInOrder[index];
        sourceNodeId = sourceNode.id;
      }
    }
    if (!targetNode && createdNodesInOrder && createdNodesInOrder.length > 0) {
      const index = parseInt(targetNodeId, 10);
      if (!isNaN(index) && index >= 0 && index < createdNodesInOrder.length) {
        targetNode = createdNodesInOrder[index];
        targetNodeId = targetNode.id;
      }
    }

    // 如果节点仍然找不到，等待一小段时间让React状态更新
    if (!sourceNode || !targetNode) {
      // 最多等待10次，每次50ms，总共最多500ms
      for (let i = 0; i < 10; i++) {
        await new Promise(resolve => setTimeout(resolve, 50));
        // 重新获取最新的workflow状态
        currentWorkflow = workflowRef.current;
        if (!currentWorkflow) break;

        if (!sourceNode) {
          sourceNode = currentWorkflow.nodes.find(n => n.id === (details.source_node_id ?? (details as any).sourceNodeId));
        }
        if (!targetNode) {
          targetNode = currentWorkflow.nodes.find(n => n.id === (details.target_node_id ?? (details as any).targetNodeId));
        }
        if (sourceNode && targetNode) break;
      }
    }

    if (!sourceNode) {
      return {
        success: false,
        operation: { type: 'add_connection', details },
        error: `Source node not found: ${details.source_node_id ?? (details as any).sourceNodeId} (resolved to: ${sourceNodeId})`
      };
    }
    if (!targetNode) {
      return {
        success: false,
        operation: { type: 'add_connection', details },
        error: `Target node not found: ${details.target_node_id ?? (details as any).targetNodeId} (resolved to: ${targetNodeId})`
      };
    }

    // 如果指定了 target_port_id，检查目标节点是否有该端口
    const rawTargetPortId = details.target_port_id ?? (details as any).targetPortId;
    if (rawTargetPortId) {
      let targetTool = TOOLS.find(t => t.id === targetNode.tool_id);
      let targetInputs = targetTool?.inputs || [];
      let targetPort = targetInputs.find(p => p.id === rawTargetPortId);

      // 如果找不到目标端口，可能是节点刚刚被替换，等待状态更新
      if (!targetPort) {
        for (let i = 0; i < 10; i++) {
          await new Promise(resolve => setTimeout(resolve, 50));
          currentWorkflow = workflowRef.current;
          if (!currentWorkflow) break;

          // 重新获取目标节点
          targetNode = currentWorkflow.nodes.find(n => n.id === targetNodeId);
          if (!targetNode) break;

          // 重新检查端口
          targetTool = TOOLS.find(t => t.id === targetNode.tool_id);
          targetInputs = targetTool?.inputs || [];
          targetPort = targetInputs.find(p => p.id === rawTargetPortId);

          if (targetPort) {
            // 找到了端口，退出循环
            break;
          }
        }
      }
    }

    let source_port_id = details.source_port_id ?? (details as any).sourcePortId;
    let targetPortId = details.target_port_id ?? (details as any).targetPortId;

    if (!source_port_id || !targetPortId) {
      const sourceOutputs = getNodeOutputs(sourceNode);
      const targetTool = TOOLS.find(t => t.id === targetNode.tool_id);
      const targetInputs = targetTool?.inputs || [];

      if (!source_port_id) {
        source_port_id = sourceOutputs[0]?.id;
        if (!source_port_id) {
          return {
            success: false,
            operation: { type: 'add_connection', details },
            error: `Source node has no output ports`
          };
        }
      }

      if (!targetPortId) {
        const sourceOutput = sourceOutputs.find(o => o.id === source_port_id);
        if (!sourceOutput) {
          return {
            success: false,
            operation: { type: 'add_connection', details },
            error: `Source port not found: ${source_port_id}`
          };
        }

        const matchingInput = targetInputs.find(
          inp => inp.type === sourceOutput.type
        );
        if (!matchingInput) {
          return {
            success: false,
            operation: { type: 'add_connection', details },
            error: `No compatible input port found on target node`
          };
        }
        targetPortId = matchingInput.id;
      }
    }

    const sourceOutputs = getNodeOutputs(sourceNode);
    const targetTool = TOOLS.find(t => t.id === targetNode.tool_id);
    const targetInputs = targetTool?.inputs || [];

    const sourcePort = sourceOutputs.find(p => p.id === source_port_id);
    const targetPort = targetInputs.find(p => p.id === targetPortId);

    if (!sourcePort) {
      return {
        success: false,
        operation: { type: 'add_connection', details },
        error: `Source port not found: ${source_port_id}`
      };
    }
    if (!targetPort) {
      return {
        success: false,
        operation: { type: 'add_connection', details },
        error: `Target port not found: ${targetPortId}`
      };
    }
    if (sourcePort.type !== targetPort.type) {
      return {
        success: false,
        operation: { type: 'add_connection', details },
        error: `Port type mismatch: ${sourcePort.type} != ${targetPort.type}`
      };
    }

    // 使用最新的workflow状态检查连接
    currentWorkflow = workflowRef.current;
    if (!currentWorkflow) {
      return {
        success: false,
        operation: { type: 'add_connection', details },
        error: 'No workflow loaded'
      };
    }

    const existingConnection = currentWorkflow.connections.find(
      c => c.source_node_id === sourceNodeId &&
          c.source_port_id === source_port_id &&
          c.target_node_id === targetNodeId &&
          c.target_port_id === targetPortId
    );
    if (existingConnection) {
      // 连接已存在，视为成功（幂等性）
      return {
        success: true,
        operation: { type: 'add_connection', details },
        result: { connectionId: existingConnection.id, connection: existingConnection },
        affectedElements: { connectionIds: [existingConnection.id] }
      };
    }

    const connectionId = `conn-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const connection = {
      id: connectionId,
      source_node_id: sourceNodeId,
      source_port_id: source_port_id!,
      target_node_id: targetNodeId,
      target_port_id: targetPortId!
    };

    originalAddConnection(connection);

    return {
      success: true,
      operation: { type: 'add_connection', details },
      result: { connectionId, connection },
      affectedElements: { connectionIds: [connectionId] }
    };
  }, [originalAddConnection, getNodeOutputs]);

  // 将 update_node 的 data 合并应用到给定 workflow（用于同一批操作中后续 add_connection 能读到新 port）
  const applyUpdateNodeToWorkflow = useCallback((wf: WorkflowState, nodeId: string, updates: Record<string, any>): WorkflowState => {
    const dataMerge: Record<string, any> = {};
    if (typeof updates.data === 'object' && updates.data !== null && !Array.isArray(updates.data)) {
      Object.assign(dataMerge, updates.data);
    }
    for (const [key, value] of Object.entries(updates)) {
      if (key === 'data') continue;
      if (key.includes('.') && key.split('.')[0] === 'data') {
        dataMerge[key.split('.').slice(1).join('.')] = value;
      }
    }
    if (Object.keys(dataMerge).length === 0) return wf;
    return {
      ...wf,
      nodes: wf.nodes.map(n => n.id === nodeId ? { ...n, data: { ...n.data, ...dataMerge } } : n)
    };
  }, []);

  // 执行删除连接操作
  const executeDeleteConnection = useCallback(async (
    details: {
      connectionId?: string;
      source_node_id?: string;
      sourceNodeId?: string;
      target_node_id?: string;
      targetNodeId?: string;
    }
  ): Promise<OperationResult> => {
    if (!workflow) {
      return {
        success: false,
        operation: { type: 'delete_connection', details },
        error: 'No workflow loaded'
      };
    }

    let connectionIdsToDelete: string[] = [];

    if (details.connectionId) {
      connectionIdsToDelete = [details.connectionId];
    } else if ((details.source_node_id ?? (details as any).sourceNodeId) && (details.target_node_id ?? (details as any).targetNodeId)) {
      const sid = details.source_node_id ?? (details as any).sourceNodeId;
      const tid = details.target_node_id ?? (details as any).targetNodeId;
      connectionIdsToDelete = workflow.connections
        .filter(
          c => c.source_node_id === sid && c.target_node_id === tid
        )
        .map(c => c.id);
    } else {
      return {
        success: false,
        operation: { type: 'delete_connection', details },
        error: 'No connection specified for deletion'
      };
    }

    const missingConnections = connectionIdsToDelete.filter(
      id => !workflow.connections.some(c => c.id === id)
    );
    if (missingConnections.length > 0) {
      return {
        success: false,
        operation: { type: 'delete_connection', details },
        error: `Connections not found: ${missingConnections.join(', ')}`
      };
    }

    for (const connectionId of connectionIdsToDelete) {
      originalDeleteConnection(connectionId);
    }

    return {
      success: true,
      operation: { type: 'delete_connection', details },
      result: { deletedConnectionIds: connectionIdsToDelete },
      affectedElements: { connectionIds: connectionIdsToDelete }
    };
  }, [workflow, originalDeleteConnection]);

  // 执行单个操作（runningWorkflow：同一批中前面已应用的 update_node 等中间状态，供 add_connection 查找新 port）
  const executeOperation = useCallback(async (
    operation: Operation,
    createdNodes?: Map<string, WorkflowNode>,
    tempIdToNodeId?: Map<string, string>,
    createdNodesInOrder?: WorkflowNode[],
    runningWorkflow?: WorkflowState | null
  ): Promise<OperationResult> => {
    const currentWorkflow = runningWorkflow ?? workflowRef.current;
    if (!currentWorkflow) {
      return {
        success: false,
        operation,
        error: 'No workflow loaded'
      };
    }

    try {
      switch (operation.type) {
        case 'add_node':
          return await executeAddNode(operation.details);
        case 'delete_node':
          return await executeDeleteNode(operation.details);
        case 'update_node':
          return await executeUpdateNode(operation.details);
        case 'replace_node':
          return await executeReplaceNode(operation.details);
        case 'add_connection':
          return await executeAddConnection(operation.details, createdNodes, tempIdToNodeId, createdNodesInOrder, runningWorkflow ?? undefined);
        case 'delete_connection':
          return await executeDeleteConnection(operation.details);
        default:
          return {
            success: false,
            operation,
            error: `Unknown operation type: ${operation.type}`
          };
      }
    } catch (error: any) {
      return {
        success: false,
        operation,
        error: error.message || String(error)
      };
    }
  }, [executeAddNode, executeDeleteNode, executeUpdateNode, executeReplaceNode, executeAddConnection, executeDeleteConnection]);

  // 批量执行操作；onStep(current, total) 用于左侧「执行中」动效
  const executeOperations = useCallback(async (
    operations: Operation[],
    onStep?: (current: number, total: number) => void
  ): Promise<{ success: boolean; results: OperationResult[]; errors: string[] }> => {
    const results: OperationResult[] = [];
    const errors: string[] = [];
    const allAffectedElements: { nodeIds?: Set<string>; connectionIds?: Set<string> } = {
      nodeIds: new Set(),
      connectionIds: new Set()
    };

    // 跟踪新创建的节点，用于后续连接操作
    const createdNodes = new Map<string, WorkflowNode>();
    // 临时ID到实际节点ID的映射（tempId -> actualNodeId）
    const tempIdToNodeId = new Map<string, string>();

    // 同一批操作中的“运行中”工作流：前面 update_node 等已应用到此副本，供后续 add_connection 查找新 port（如 custom_outputs 更新后的 shot1_image）
    let runningWorkflow: WorkflowState | null = getWorkflow?.() ?? workflow ?? null;

    const total = operations.length;
    onStep?.(0, total);
    for (let i = 0; i < operations.length; i++) {
      const operation = operations[i];
      const result = await executeOperation(operation, createdNodes, tempIdToNodeId, undefined, runningWorkflow ?? undefined);
      onStep?.(i + 1, total);

      // 如果执行了 replace_node 或 update_node（包含 toolId），等待一小段时间让状态更新
      // 这样后续的 add_connection 操作可以获取到更新后的节点状态
      if ((operation.type === 'replace_node' ||
           (operation.type === 'update_node' && (operation.details.updates?.tool_id ?? (operation.details.updates as any)?.toolId))) &&
          i < operations.length - 1) {
        // 检查下一个操作是否是 add_connection，且目标节点是刚刚替换的节点
        const nextOp = operations[i + 1];
        if (nextOp && nextOp.type === 'add_connection') {
          const replacedNodeId = operation.type === 'replace_node'
            ? operation.details.nodeId
            : operation.details.nodeId;
          if ((nextOp.details.target_node_id ?? (nextOp.details as any).targetNodeId) === replacedNodeId) {
            // 等待状态更新
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        }
      }
      results.push(result);

      if (!result.success) {
        errors.push(result.error || 'Unknown error');
        break; // 停止执行
      }

      // 同步 runningWorkflow：后续 add_connection 需要基于已更新的节点（如 update_node 后的新 custom_outputs）查找 port
      if (runningWorkflow) {
        if (operation.type === 'update_node' && operation.details.nodeId && operation.details.updates && !operation.details.updates.tool_id) {
          // 仅 data 更新（如 custom_outputs）时同步；含 toolId 时走 replaceNode，不在此模拟
          runningWorkflow = applyUpdateNodeToWorkflow(runningWorkflow, operation.details.nodeId, operation.details.updates);
        } else if (operation.type === 'add_node' && result.result?.node) {
          runningWorkflow = { ...runningWorkflow, nodes: [...runningWorkflow.nodes, result.result.node] };
        }
      }

      // 如果刚创建了节点，保存到createdNodes中，并建立tempId映射
      if (operation.type === 'add_node' && result.result?.node) {
        createdNodes.set(result.result.nodeId, result.result.node);
        // 如果AI指定了tempId，建立映射
        if (result.result.tempId) {
          tempIdToNodeId.set(result.result.tempId, result.result.nodeId);
        }
      }

      // 收集影响的元素
      if (result.affectedElements?.nodeIds) {
        result.affectedElements.nodeIds.forEach(id => allAffectedElements.nodeIds?.add(id));
      }
      if (result.affectedElements?.connectionIds) {
        result.affectedElements.connectionIds.forEach(id => allAffectedElements.connectionIds?.add(id));
      }
    }

    // 设置高亮
    if (allAffectedElements.nodeIds?.size || allAffectedElements.connectionIds?.size) {
      setHighlightedElements({
        nodeIds: Array.from(allAffectedElements.nodeIds || []),
        connectionIds: Array.from(allAffectedElements.connectionIds || [])
      });
      // 3秒后清除高亮
      setTimeout(() => setHighlightedElements({}), 3000);
    }

    return {
      success: errors.length === 0,
      results,
      errors
    };
  }, [executeOperation, getWorkflow, workflow, applyUpdateNodeToWorkflow]);

  const addNodeToChatContext = useCallback((nodeId: string, name: string) => {
    setChatContextNodes(prev => {
      if (prev.some(n => n.nodeId === nodeId)) return prev;
      return [...prev, { nodeId, name }];
    });
  }, []);
  const removeNodeFromChatContext = useCallback((nodeId: string) => {
    setChatContextNodes(prev => prev.filter(n => n.nodeId !== nodeId));
  }, []);
  const clearChatContextNodes = useCallback(() => {
    setChatContextNodes([]);
  }, []);

  // 处理用户输入
  const handleUserInput = useCallback(async (userInput: string, options: AIChatSendOptions = {}) => {
    const trimmedInput = userInput.trim();
    const hasImage = !!options.image;
    if (!workflow || isProcessing || (!trimmedInput && !hasImage)) return;

    setIsProcessing(true);
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;
    const useSearch = options.useSearch ?? false;
    const imageDataUrl = options.image
      ? `data:${options.image.mimeType};base64,${options.image.data}`
      : undefined;
    const effectiveInput = trimmedInput || (hasImage
      ? (lang === 'zh' ? '用户提供了一张参考图片。' : 'User provided a reference image.')
      : '');
    const contextPrefix =
      chatContextNodes.length > 0
        ? (lang === 'zh'
            ? '用户选中的节点（用于帮助理解意图；请结合对话上下文和用户指令判断修改范围，可仅修改这些节点、其前置/后置节点或整个工作流等）：\n'
            : 'User selected nodes (for intent reference; determine scope of changes from conversation context and user instruction—e.g. only these nodes, their predecessors/successors, or the entire workflow):\n') +
          chatContextNodes.map(n => `- [${n.nodeId}] ${n.name}`).join('\n') +
          '\n\n' +
          (lang === 'zh' ? '用户说：' : 'User said: ')
        : '';
    const effectiveUserMessage = contextPrefix + effectiveInput;

    // 添加用户消息到历史记录
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}-user`,
      role: 'user',
      content: trimmedInput,
      image: options.image,
      useSearch,
      timestamp: Date.now()
    };

    // 先添加到历史记录（ref 和 state）
    addMessageToHistory(userMessage);

    // 获取当前完整的历史记录（从 ref 获取，确保是最新的）
    const currentHistory = chatHistoryRef.current;

    let assistantMessageId: string | null = null;
    try {
      // 构建AI Prompt（带上选中节点上下文）
      const systemPrompt = await buildAIPrompt(effectiveUserMessage);

      // 构建消息历史（只保留最近的对话，避免token过多）
      // currentHistory 已经包含了刚添加的 userMessage，我们需要排除它，因为最后会单独添加
      const previousHistory = currentHistory.slice(0, -1); // 排除最后一条（刚添加的 userMessage）
      const recentHistory = previousHistory.slice(-30); // 保留最近50条历史消息

      console.log('[AI Chat] 对话历史:', {
        totalHistory: currentHistory.length,
        previousHistory: previousHistory.length,
        recentHistory: recentHistory.length,
        recentHistoryMessages: recentHistory.map(m => ({ role: m.role, content: m.content.substring(0, 50) + '...' }))
      });

      // 构建完整的消息列表（包含系统提示词、历史记录和当前用户消息）
      const messages = [
        {
          role: 'system' as const,
          content: systemPrompt
        },
        ...recentHistory.map(msg => {
          const historyText = msg.content || (msg.image ? (lang === 'zh' ? '用户提供了一张参考图片。' : 'User provided a reference image.') : '');
          return {
            role: msg.role,
            content: historyText
          };
        }).filter(msg => msg.content),
        {
          role: 'user' as const,
          content: effectiveUserMessage
        }
      ];

      console.log('[AI Chat] 发送给 AI 的消息数量:', messages.length, {
        system: 1,
        history: recentHistory.length,
        currentUser: 1
      });

      // 判断模型类型
      const isDeepSeekModel = aiModel.startsWith('deepseek-');
      const isGeminiModel = aiModel.startsWith('ppchat-gemini-') || aiModel.startsWith('gemini-');
      const isDoubaoModel = aiModel.startsWith('doubao-');

      // 创建初始的 assistant 消息用于流式更新
      assistantMessageId = `msg-${Date.now()}-assistant`;
      let assistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        thinking: '',
        isStreaming: true,
        useSearch,
        timestamp: Date.now()
      };
      addMessageToHistory(assistantMessage);

      // 使用流式 API
      let thinkingText = '';
      let contentText = '';
      let aiResponse = '';

      const extractSources = (raw: any): ChatSource[] => {
        const results: ChatSource[] = [];
        const seen = new Set<string>();
        if (!raw || !Array.isArray(raw.output)) return results;
        raw.output.forEach((item: any) => {
          if (item?.type !== 'message' || !Array.isArray(item.content)) return;
          item.content.forEach((part: any) => {
            const annotations = Array.isArray(part?.annotations) ? part.annotations : [];
            annotations.forEach((ann: any) => {
              if (ann?.type !== 'url_citation') return;
              const url = ann.url || ann.uri;
              if (!url || seen.has(url)) return;
              seen.add(url);
              results.push({
                url,
                title: ann.title,
                siteName: ann.site_name
              });
            });
          });
        });
        return results;
      };

      try {
      if (isDeepSeekModel) {
        if (useSearch) {
          const systemMessage = messages.find(m => m.role === 'system');
          const conversationMessages = messages.filter(m => m.role !== 'system');
          let fullPrompt = '';
          if (systemMessage) {
            fullPrompt += systemMessage.content + '\n\n';
          }
          const conversationText = conversationMessages.map(msg => {
            if (msg.role === 'user') {
              return `User: ${msg.content}`;
            } else {
              return `Assistant: ${msg.content}`;
            }
          }).join('\n\n');
          fullPrompt += conversationText;
          fullPrompt += '\n\n请以 JSON 格式返回，包含 "operations" 数组和 "explanation" 字段。';

          const response = await deepseekText(
            fullPrompt,
            'basic',
            undefined,
            aiModel,
            [{ id: 'operations', description: 'Array of operations' }, { id: 'explanation', description: 'Explanation of operations' }],
            true,
            true
          );

          let responseData: any = response;
          let rawResponse: any = null;
          if (response && typeof response === 'object' && 'raw_response' in response) {
            rawResponse = (response as any).raw_response;
            responseData = (response as any).data;
          }

          if (typeof responseData === 'string') {
            aiResponse = responseData;
          } else if (responseData && typeof responseData === 'object') {
            aiResponse = JSON.stringify(responseData);
          } else {
            throw new Error('Unexpected response format from deepseekText');
          }
          if (rawResponse) {
            assistantMessage.sources = extractSources(rawResponse);
          }
          assistantMessage.content = aiResponse;
        } else {
          // DeepSeek 模型使用流式 API
          for await (const chunk of deepseekChatStream(messages, aiModel, 'json_object', signal)) {
            if (chunk.type === 'thinking') {
              thinkingText += chunk.text;
              // 更新思考过程
              assistantMessage.thinking = thinkingText;
              assistantMessage.isStreaming = true;
              // 流式输出时，只显示思考过程，不显示 JSON 内容
              assistantMessage.content = '';
              setChatHistory(prev => {
                const updated = prev.map(msg =>
                  msg.id === assistantMessageId
                    ? { ...assistantMessage }
                    : msg
                );
                chatHistoryRef.current = updated;
                return updated;
              });
            } else if (chunk.type === 'content') {
              contentText += chunk.text;
              aiResponse += chunk.text;
              // 流式输出时，不实时显示 JSON 内容，等完成后解析并只显示 explanation
              assistantMessage.isStreaming = true;
              // 暂时不更新 content，等流式输出完成后再解析
              setChatHistory(prev => {
                const updated = prev.map(msg =>
                  msg.id === assistantMessageId
                    ? { ...assistantMessage }
                    : msg
                );
                chatHistoryRef.current = updated;
                return updated;
              });
            }
          }
        }
        } else if (isGeminiModel) {
          // Gemini 模型使用流式 API
          const geminiModelId = aiModel.replace('ppchat-', '');
          try {
            for await (const chunk of ppchatChatCompletionsStream(messages, geminiModelId, 'json_object', signal)) {
              if (chunk.type === 'content') {
                contentText += chunk.text;
                aiResponse += chunk.text;
                // 流式输出时，不实时显示 JSON 内容，等完成后解析并只显示 explanation
                assistantMessage.isStreaming = true;
                // 暂时不更新 content，等流式输出完成后再解析
                setChatHistory(prev => {
                  const updated = prev.map(msg =>
                    msg.id === assistantMessageId
                      ? { ...assistantMessage }
                      : msg
                  );
                  chatHistoryRef.current = updated;
                  return updated;
                });
              }
            }
          } catch (error) {
            // 如果流式 API 失败，回退到非流式
            console.warn('[AI Chat] Stream API failed, falling back to non-stream:', error);
      const systemMessage = messages.find(m => m.role === 'system');
      const conversationMessages = messages.filter(m => m.role !== 'system');
      let fullPrompt = '';
      if (systemMessage) {
        fullPrompt += systemMessage.content + '\n\n';
      }
            const conversationText = conversationMessages.map(msg => {
              if (msg.role === 'user') {
                return `User: ${msg.content}`;
              } else {
                return `Assistant: ${msg.content}`;
              }
            }).join('\n\n');
            fullPrompt += conversationText;
            fullPrompt += '\n\n请以 JSON 格式返回，包含 "operations" 数组和 "explanation" 字段。';

            const response = await ppchatGeminiText(
              fullPrompt,
              'basic',
              undefined,
              geminiModelId,
              [{ id: 'operations', description: 'Array of operations' }, { id: 'explanation', description: 'Explanation of operations' }],
              imageDataUrl
            );

            if (typeof response === 'string') {
              aiResponse = response;
            } else if (response && typeof response === 'object') {
              aiResponse = JSON.stringify(response);
            } else {
              throw new Error('Unexpected response format from ppchatGeminiText');
            }
            assistantMessage.content = aiResponse;
          }
        } else if (isDoubaoModel) {
          // Doubao 模型暂不支持流式，使用非流式
          const systemMessage = messages.find(m => m.role === 'system');
          const conversationMessages = messages.filter(m => m.role !== 'system');
          let fullPrompt = '';
          if (systemMessage) {
            fullPrompt += systemMessage.content + '\n\n';
          }
      const conversationText = conversationMessages.map(msg => {
        if (msg.role === 'user') {
          return `User: ${msg.content}`;
        } else {
          return `Assistant: ${msg.content}`;
        }
      }).join('\n\n');
      fullPrompt += conversationText;
          fullPrompt += '\n\n请以 JSON 格式返回，包含 "operations" 数组和 "explanation" 字段。';

          const response = await doubaoText(
            fullPrompt,
            'basic',
            undefined,
            aiModel,
            [{ id: 'operations', description: 'Array of operations' }, { id: 'explanation', description: 'Explanation of operations' }],
            imageDataUrl,
            useSearch,
            true
          );

          let responseData: any = response;
          let rawResponse: any = null;
          if (response && typeof response === 'object' && 'raw_response' in response) {
            rawResponse = (response as any).raw_response;
            responseData = (response as any).data;
          }

          if (typeof responseData === 'string') {
            aiResponse = responseData;
          } else if (responseData && typeof responseData === 'object') {
            aiResponse = JSON.stringify(responseData);
          } else {
            throw new Error('Unexpected response format from doubaoText');
          }
          if (rawResponse) {
            assistantMessage.sources = extractSources(rawResponse);
          }
          assistantMessage.content = aiResponse;
        } else {
          // 默认使用 DeepSeek 流式 API
          for await (const chunk of deepseekChatStream(messages, aiModel, 'json_object', signal)) {
            if (chunk.type === 'thinking') {
              thinkingText += chunk.text;
              assistantMessage.thinking = thinkingText;
              assistantMessage.isStreaming = true;
              // 流式输出时，只显示思考过程，不显示 JSON 内容
              assistantMessage.content = '';
              setChatHistory(prev => {
                const updated = prev.map(msg =>
                  msg.id === assistantMessageId
                    ? { ...assistantMessage }
                    : msg
                );
                chatHistoryRef.current = updated;
                return updated;
              });
            } else if (chunk.type === 'content') {
              contentText += chunk.text;
              aiResponse += chunk.text;
              // 流式输出时，不实时显示 JSON 内容，等完成后解析并只显示 explanation
              assistantMessage.isStreaming = true;
              // 暂时不更新 content，等流式输出完成后再解析
              setChatHistory(prev => {
                const updated = prev.map(msg =>
                  msg.id === assistantMessageId
                    ? { ...assistantMessage }
                    : msg
                );
                chatHistoryRef.current = updated;
                return updated;
              });
            }
          }
        }

        // 流式输出完成，更新最终状态
        assistantMessage.isStreaming = false;

        // 解析流式输出的 JSON，提取 explanation 和 operations
        let parsed: any;
        try {
          // 尝试解析 JSON
          if (typeof aiResponse === 'string') {
            // 如果响应包含代码块，提取 JSON
            if (aiResponse.includes('```json')) {
              const jsonMatch = aiResponse.match(/```json\s*([\s\S]*?)\s*```/);
              if (jsonMatch && jsonMatch[1]) {
                parsed = JSON.parse(jsonMatch[1].trim());
              } else {
                try {
                  parsed = JSON.parse(aiResponse);
                } catch {
                  parsed = { content: aiResponse };
                }
              }
            } else {
              try {
                parsed = JSON.parse(aiResponse);
              } catch {
                parsed = { content: aiResponse };
              }
            }
          } else {
            parsed = aiResponse;
          }
        } catch (error) {
          parsed = { content: aiResponse };
        }

        // 如果是工作流操作，只显示 explanation，不显示 JSON
        const responseType = parsed.type || (parsed.operations ? 'workflow' : 'conversation');
        if (responseType === 'workflow' && parsed.operations) {
          // 只显示 explanation，不显示 JSON 操作字段
          assistantMessage.content = parsed.explanation || (lang === 'zh' ? '操作已生成' : 'Operations generated');
        } else if (responseType === 'conversation') {
          // 普通对话，显示内容
          assistantMessage.content = parsed.content || parsed.explanation || aiResponse;
        } else {
          // 其他情况，保持原内容
          assistantMessage.content = parsed.content || parsed.explanation || aiResponse;
        }

        if (thinkingText) {
          assistantMessage.thinking = thinkingText;
        }

        setChatHistory(prev => {
          const updated = prev.map(msg =>
            msg.id === assistantMessageId
              ? { ...assistantMessage }
              : msg
          );
          chatHistoryRef.current = updated;
          return updated;
        });
      } catch (streamError: any) {
        // 流式输出出错，更新错误状态
        assistantMessage.isStreaming = false;
        assistantMessage.error = streamError.message || String(streamError);
        assistantMessage.content = streamError.message || String(streamError);
        setChatHistory(prev => {
          const updated = prev.map(msg =>
            msg.id === assistantMessageId
              ? { ...assistantMessage }
              : msg
          );
          chatHistoryRef.current = updated;
          return updated;
        });
        setIsProcessing(false);
        return;
      }

      // 解析AI返回
      let parsed: any;
      let isWorkflowRelated = false;

      try {
        // 尝试解析 JSON
        if (typeof aiResponse === 'string') {
          // 如果响应包含代码块，提取 JSON
          if (aiResponse.includes('```json')) {
            const jsonMatch = aiResponse.match(/```json\s*([\s\S]*?)\s*```/);
            if (jsonMatch && jsonMatch[1]) {
              parsed = JSON.parse(jsonMatch[1].trim());
              isWorkflowRelated = true;
            } else {
              // 尝试直接解析
              try {
                parsed = JSON.parse(aiResponse);
                isWorkflowRelated = true;
              } catch {
                // 不是JSON，是普通文本回答
                parsed = { content: aiResponse };
                isWorkflowRelated = false;
              }
            }
          } else {
            // 尝试解析为JSON
            try {
              parsed = JSON.parse(aiResponse);
              isWorkflowRelated = true;
            } catch {
              // 不是JSON，是普通文本回答
              parsed = { content: aiResponse };
              isWorkflowRelated = false;
            }
          }
        } else {
          parsed = aiResponse;
          isWorkflowRelated = true;
        }
      } catch (error) {
        // 解析失败，可能是普通文本回答
        parsed = { content: aiResponse };
        isWorkflowRelated = false;
      }

      // 检查响应类型
      const responseType = parsed.type || (parsed.operations ? 'workflow' : 'conversation');

      // 如果是普通对话类型，更新已存在的消息
      if (responseType === 'conversation') {
        assistantMessage.isStreaming = false;
        assistantMessage.content = parsed.content || parsed.explanation || aiResponse;
        assistantMessage.choices = Array.isArray(parsed.choices) ? parsed.choices : undefined;
        if (thinkingText) {
          assistantMessage.thinking = thinkingText;
        }
        setChatHistory(prev => {
          const updated = prev.map(msg =>
            msg.id === assistantMessageId
              ? { ...assistantMessage }
              : msg
          );
          chatHistoryRef.current = updated;
          return updated;
        });
        setIsProcessing(false);
        return;
      }

      // 向后兼容：如果没有 type 字段，但有 error 字段，当作普通对话处理
      if (parsed.error && !parsed.operations) {
        assistantMessage.isStreaming = false;
        assistantMessage.content = parsed.question || parsed.error || parsed.content || aiResponse;
        assistantMessage.choices = Array.isArray(parsed.choices) ? parsed.choices : undefined;
        if (thinkingText) {
          assistantMessage.thinking = thinkingText;
        }
        setChatHistory(prev => {
          const updated = prev.map(msg =>
            msg.id === assistantMessageId
              ? { ...assistantMessage }
              : msg
          );
          chatHistoryRef.current = updated;
          return updated;
        });
        setIsProcessing(false);
        return;
      }

      // 如果与工作流无关（没有 operations），当作普通对话处理
      if (!parsed.operations || (Array.isArray(parsed.operations) && parsed.operations.length === 0)) {
        assistantMessage.isStreaming = false;
        assistantMessage.content = parsed.content || parsed.explanation || aiResponse;
        assistantMessage.choices = Array.isArray(parsed.choices) ? parsed.choices : undefined;
        if (thinkingText) {
          assistantMessage.thinking = thinkingText;
        }
        setChatHistory(prev => {
          const updated = prev.map(msg =>
            msg.id === assistantMessageId
              ? { ...assistantMessage }
              : msg
          );
          chatHistoryRef.current = updated;
          return updated;
        });
        setIsProcessing(false);
        return;
      }

      // 获取操作列表
      let operations: Operation[] = [];
      if (parsed.operations) {
        operations = parsed.operations;
      } else if (Array.isArray(parsed)) {
        operations = parsed;
      } else {
        operations = [parsed];
      }

      // 在执行操作前，记录当前的历史索引作为检查点
      const historyCheckpoint = getCurrentHistoryIndex ? getCurrentHistoryIndex() : undefined;
      assistantMessage.historyCheckpoint = historyCheckpoint;

      // 执行操作（汇报进度供左侧动效）
      const stepLabels = operations.map(op => getOperationLabel(op, lang));
      setIsExecutingOperations(true);
      setExecutingProgress({ current: 0, total: operations.length });
      setExecutingStepLabels(stepLabels);
      const executionResult = await executeOperations(operations, (current, total) => {
        setExecutingProgress({ current, total });
      });
      setIsExecutingOperations(false);
      setExecutingStepLabels([]);

      // 更新已存在的AI回复消息
      const errorDetails = executionResult.errors.length > 0
        ? executionResult.errors.join('; ')
        : executionResult.results
            .filter(r => !r.success)
            .map(r => r.error || 'Unknown error')
            .join('; ');

      assistantMessage.isStreaming = false;
      assistantMessage.content = parsed.explanation || (executionResult.success
        ? (lang === 'zh' ? '操作已执行' : 'Operations executed successfully')
        : (lang === 'zh' ? `操作执行失败: ${errorDetails}` : `Operation failed: ${errorDetails}`));
      assistantMessage.operations = operations;
      assistantMessage.operationResults = executionResult.results;
      assistantMessage.error = executionResult.success ? undefined : (errorDetails || 'Unknown error');
      // historyCheckpoint 已经在执行操作前设置
      if (thinkingText) {
        assistantMessage.thinking = thinkingText;
      }

      setChatHistory(prev => {
        const updated = prev.map(msg =>
          msg.id === assistantMessageId
            ? { ...assistantMessage }
            : msg
        );
        chatHistoryRef.current = updated;
        return updated;
      });

      // 输出详细错误信息到控制台，方便调试
      if (!executionResult.success) {
        console.error('[AI Chat Workflow] Operation execution failed:', {
          errors: executionResult.errors,
          results: executionResult.results,
          operations,
          workflow: workflowRef.current
        });
      }

      // 发送完成后清空对话上下文节点，下次需重新选择
      clearChatContextNodes();
    } catch (error: any) {
      const isAbort = error?.name === 'AbortError' || (typeof error?.message === 'string' && error.message.toLowerCase().includes('abort'));
      if (isAbort) {
        if (assistantMessageId) {
          setChatHistory(prev => {
            const updated = prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, content: lang === 'zh' ? '已停止生成' : 'Generation stopped', isStreaming: false }
                : msg
            );
            chatHistoryRef.current = updated;
            return updated;
          });
        } else {
          addMessageToHistory({
            id: `msg-${Date.now()}-stop`,
            role: 'assistant',
            content: lang === 'zh' ? '已停止生成' : 'Generation stopped',
            timestamp: Date.now()
          });
        }
      } else {
        const raw = error.message || String(error);
        const displayMsg = raw === 'Fetch is aborted' ? t('fetch_aborted') : raw;
        const errorMessage: ChatMessage = {
          id: `msg-${Date.now()}-error`,
          role: 'assistant',
          content: (lang === 'zh' ? '错误: ' : 'Error: ') + displayMsg,
          timestamp: Date.now(),
          error: displayMsg
        };
        addMessageToHistory(errorMessage);
      }
    } finally {
      abortControllerRef.current = null;
      setIsExecutingOperations(false);
      setExecutingStepLabels([]);
      setIsProcessing(false);
    }
  }, [workflow, isProcessing, buildAIPrompt, executeOperations, addMessageToHistory, chatContextNodes, lang, clearChatContextNodes]);

  /** 停止当前 AI 生成（仅对流式请求有效） */
  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  // 清除对话历史
  const clearChatHistory = useCallback(() => {
    chatHistoryRef.current = [];
    setChatHistory([]);
    setChatContextNodes([]);
    console.log('[AI Chat] 已清除对话历史');
  }, []);

  // 撤销指定消息的所有操作
  const undoMessageOperations = useCallback((messageId: string) => {
    const message = chatHistory.find(m => m.id === messageId);
    if (message && message.historyCheckpoint !== undefined && undoToIndex) {
      // 回退到执行操作前的历史索引
      undoToIndex(message.historyCheckpoint);
    } else if (message && !message.historyCheckpoint && undoToIndex) {
      // 如果没有检查点，尝试回退一步（向后兼容）
      const currentIndex = getCurrentHistoryIndex ? getCurrentHistoryIndex() : -1;
      if (currentIndex > 0) {
        undoToIndex(currentIndex - 1);
      }
    }
  }, [chatHistory, undoToIndex, getCurrentHistoryIndex]);

  return {
    chatHistory,
    isProcessing,
    /** AI 已生成并正在执行工作流操作时为 true，用于左侧「执行中」动效 */
    isExecutingOperations,
    /** 执行进度 { current, total }，配合 isExecutingOperations 显示步骤 */
    executingProgress,
    /** 当前执行步骤的标签列表（添加节点、添加连接等），用于步骤列表逐条打勾 */
    executingStepLabels,
    highlightedElements,
    handleUserInput,
    stopGeneration,
    clearChatHistory,
    setHighlightedElements,
    aiModel,
    setAiModel,
    undoMessageOperations,
    chatContextNodes,
    addNodeToChatContext,
    removeNodeFromChatContext,
    clearChatContextNodes
  };
};
