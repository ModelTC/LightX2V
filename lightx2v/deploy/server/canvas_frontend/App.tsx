
import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import {
  Plus, Play, Pause, Save, Trash2, Search, Settings,
  Layers, ChevronRight, AlertCircle, CheckCircle2,
  X, Type, Image as ImageIcon, Volume2, Video as VideoIcon,
  Cpu, Sparkles, AlignLeft, Download, RefreshCw,
  Terminal, MousePointer2, Wand2, Globe, Palette, Clapperboard, UserCircle, UserCog,
  Maximize, ZoomIn, ZoomOut, Zap, MessageSquare, PenTool, FileText, Star, Edit3, Boxes,
  Camera, Mic, Wand, ListPlus, Hash, Info, PlayCircle, FastForward, ArrowUpRight,
  Target, Activity, History, Clock, Maximize2, DownloadCloud, BookOpen, ChevronLeft, ChevronDown, ChevronUp,
  Calendar, LayoutGrid, Sparkle, ToggleLeft, ToggleRight, Timer, PlayCircle as PlayIcon,
  Key, Globe2, Upload, Languages, ShieldCheck, TriangleAlert, SaveAll, Eraser
} from 'lucide-react';
import { TOOLS, updateLightX2VModels } from './constants';
import {
  WorkflowNode, Connection, WorkflowState,
  NodeStatus, DataType, Port, ToolDefinition
} from './types';
import { geminiText, geminiImage, geminiSpeech, geminiVideo, lightX2VTask, lightX2VTTS, lightX2VVoiceClone, lightX2VVoiceCloneTTS, lightX2VGetVoiceList, lightX2VGetCloneVoiceList, lightX2VGetModelList, deepseekText, doubaoText, ppchatGeminiText } from './services/geminiService';
import { removeGeminiWatermark } from './services/watermarkRemover';
import { PRESET_WORKFLOWS } from './preset_workflow';
import { Editor } from './src/components/editor/Editor';
import { Dashboard } from './src/components/dashboard/Dashboard';
import { ExpandedOutputModal } from './src/components/modals/ExpandedOutputModal';
import { CloneVoiceModal } from './src/components/modals/CloneVoiceModal';
import { AudioEditorModal } from './src/components/modals/AudioEditorModal';
import { VideoEditorModal } from './src/components/modals/VideoEditorModal';
import { useWorkflow } from './src/hooks/useWorkflow';
import { useCanvas } from './src/hooks/useCanvas';
import { useVoiceList } from './src/hooks/useVoiceList';
import { useTranslation } from './src/i18n/useTranslation';
import { useNodeManagement } from './src/hooks/useNodeManagement';
import { useConnectionManagement } from './src/hooks/useConnectionManagement';
import { useModalState } from './src/hooks/useModalState';
import { useResultManagement } from './src/hooks/useResultManagement';
import { useShowIntermediateResults } from './src/hooks/useShowIntermediateResults';
import { useWorkflowExecution } from './src/hooks/useWorkflowExecution';
import { useUndoRedo } from './src/hooks/useUndoRedo';
import { useAIChatWorkflow } from './src/hooks/useAIChatWorkflow';
import { useWorkflowAutoSave } from './src/hooks/useWorkflowAutoSave';
import { getAccessToken, initLightX2VToken, apiRequest } from './src/utils/apiClient';
import { checkWorkflowOwnership, getCurrentUserId } from './src/utils/workflowUtils';
import { isStandalone } from './src/config/runtimeMode';
import { collectLightX2VResultRefs } from './src/utils/resultRef';
import { getNodeOutputUrl, saveInputFileViaOutputSave } from './src/utils/workflowFileManager';
import { getOutputValueByPort, setOutputValueByPort, toFileRefForOutputValue } from './src/utils/outputValuePort';

// --- Main App ---

const App: React.FC = () => {
  const [lang, setLang] = useState<'en' | 'zh'>(() => {
    const saved = localStorage.getItem('omniflow_lang');
    return (saved as any) || 'zh';
  });
  const [currentView, setCurrentView] = useState<'DASHBOARD' | 'EDITOR'>('DASHBOARD');
  const [activeTab, setActiveTab] = useState<'MY' | 'COMMUNITY' | 'PRESET'>(() => (isStandalone() ? 'PRESET' : 'MY'));
  const [communityWorkflows, setCommunityWorkflows] = useState<WorkflowState[]>([]);
  const nameSaveTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Use useWorkflow Hook
  const {
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
    deleteWorkflow: deleteWorkflowFromHook,
    updateWorkflowVisibility,
    ensureWorkflowOwned,
    isLoading: isLoadingWorkflows,
    isSaving: isSavingWorkflow
  } = useWorkflow();
  const [showIntermediateResults, setShowIntermediateResults] = useShowIntermediateResults();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
  const [activeOutputs, setActiveOutputs] = useState<Record<string, any>>({});

  // Canvas ref
  const canvasRef = useRef<HTMLDivElement>(null);
  const nodeHeightsRef = useRef<Map<string, number>>(new Map());

  // Use useCanvas Hook
  const {
    view,
    setView,
    isPanning,
    isOverNode,
    setIsOverNode,
    draggingNode,
    connecting,
    setConnecting,
    mousePos,
    clientMousePos,
    zoomIn,
    zoomOut,
    resetView,
    handleMouseMove: canvasHandleMouseMove,
    handleMouseDown: canvasHandleMouseDown,
    handleMouseUp: canvasHandleMouseUp,
    handleMouseLeave: canvasHandleMouseLeave,
    handleWheel: canvasHandleWheel,
    handleNodeDragStart: canvasHandleNodeDragStart,
    handleNodeDrag: canvasHandleNodeDrag,
    handleNodeDragEnd: canvasHandleNodeDragEnd,
    screenToWorldCoords
  } = useCanvas(workflow, canvasRef);

  // Use useModalState Hook
  const modalState = useModalState();

  // Use useUndoRedo Hook
  const undoRedo = useUndoRedo({
    workflow,
    setWorkflow
  });

  const [ticker, setTicker] = useState(0);
  /** 模型列表更新后递增，用于触发重渲染使节点/配置面板重新读取 TOOLS（含 cloud 模型） */
  const [lightX2VModelsVersion, setLightX2VModelsVersion] = useState(0);
  const [validationErrors, setValidationErrors] = useState<{ message: string; type: 'ENV' | 'INPUT' }[]>([]);
  const [globalError, setGlobalError] = useState<{ message: string; details?: string } | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [sidebarDefaultTab, setSidebarDefaultTab] = useState<'tools' | 'chat'>('tools');
  const [focusAIChatInput, setFocusAIChatInput] = useState(false);
  const [aiChatMode, setAiChatMode] = useState<'edit' | 'ideation'>('edit');
  const isPausedRef = useRef(false);
  const runningTaskIdsRef = useRef<Map<string, string>>(new Map()); // Map<nodeId, taskId> for tracking LightX2V tasks
  const abortControllerRef = useRef<AbortController | null>(null); // AbortController for cancelling tasks

  // Use translation hook from components
  const { t } = useTranslation(lang);

  // 初始化 LIGHTX2V_TOKEN（确保在组件挂载时也初始化）
  useEffect(() => {
    initLightX2VToken();
  }, []);

  // 缓存配置，避免重复计算
  const configCacheRef = useRef<{ config: { url: string; token: string }; key: string } | null>(null);
  const lightX2VConfigRef = useRef<{ url: string; token: string } | null>(null);
  const configKeyRef = useRef<string>('');

  // Helper function to get LightX2V config from shared store and API client
  // 优先使用环境变量 LIGHTX2V_URL，如果没有则根据是否有 apiClient 决定
  const getLightX2VConfig = useCallback((workflow: WorkflowState | null) => {
    // 从主应用获取 sharedStore 和 apiClient
    const sharedStore = (window as any).__SHARED_STORE__;
    const apiClient = (window as any).__API_CLIENT__;

    // 优先检查环境变量 LIGHTX2V_URL
    const DEFAULT_LIGHTX2V_URL = 'https://x2v.light-ai.top';
    const envUrl = process.env.LIGHTX2V_URL;

    // 创建缓存 key（基于配置的关键因素）
    const cacheKey = `${envUrl || 'empty'}:${!!apiClient}`;

    // 如果配置没有变化，返回缓存的配置
    if (configCacheRef.current && configCacheRef.current.key === cacheKey) {
      return configCacheRef.current.config;
    }

    // 获取 token（只在配置变化时调用）
    const token = getAccessToken();

    let config: { url: string; token: string };

    // 如果环境变量明确设置了（不为空且不等于空字符串），优先使用环境变量
    // 即使存在 apiClient，也应该使用环境变量的 URL，因为用户明确指定了要直接访问该 URL
    if (envUrl && envUrl.trim()) {
      const url = envUrl.trim();
      config = {
        url: url,
        token: token
      };
    } else if (apiClient) {
      // 如果环境变量未设置，优先使用 apiClient（如果存在）
      // 这样可以避免跨域问题，因为请求会通过主应用的代理
      config = {
        url: '', // 空字符串表示使用相对路径和主应用的 apiClient
        token: token
      };
    } else {
      // 只有在独立运行模式且没有 apiClient 时，才使用默认值
      // 如果是在主应用中运行（qiankun），应该总是有 apiClient
      // 如果没有 apiClient，说明可能是独立运行，使用默认 URL
      const url = DEFAULT_LIGHTX2V_URL;
      (process.env as any).LIGHTX2V_URL = url;
      config = {
        url: url,
        token: token
      };
    }

    // 缓存配置
    configCacheRef.current = { config, key: cacheKey };

    return config;
  }, []);

  // 获取并更新 LightX2V 模型列表（只在配置真正变化时调用，而不是每次 workflow 变化时）
  // 同时加载本地和云端模型列表
  useEffect(() => {
    const loadLightX2VModels = async () => {
      try {
        const config = getLightX2VConfig(workflow);
        const configKey = `${config.url || 'empty'}:${config.token ? 'hasToken' : 'noToken'}`;

        // 检查云端配置
        const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
        const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();
        const cloudConfigKey = `${cloudUrl}:${cloudToken ? 'hasToken' : 'noToken'}`;
        const fullConfigKey = `${configKey}:${cloudConfigKey}`;

        // 如果配置没有变化，跳过（这是主要的优化点）
        if (configKeyRef.current === fullConfigKey && lightX2VConfigRef.current) {
          return;
        }

        // 更新缓存的配置
        lightX2VConfigRef.current = config;
        configKeyRef.current = fullConfigKey;

        // 并行加载本地和云端模型列表
        const loadPromises: Promise<Array<{ task: string; model_cls: string; stage: string }>>[] = [];

        // 加载本地模型列表
        if (config.url || config.token) {
          loadPromises.push(
            lightX2VGetModelList(config.url, config.token).catch(err => {
              console.warn('[LightX2V] 获取本地模型列表失败:', err);
              return [];
            })
          );
        } else {
          loadPromises.push(Promise.resolve([]));
        }

        // 加载云端模型列表
        if (cloudUrl && cloudToken) {
          loadPromises.push(
            lightX2VGetModelList(cloudUrl, cloudToken).catch(err => {
              console.warn('[LightX2V] 获取云端模型列表失败:', err);
              return [];
            })
          );
        } else {
          loadPromises.push(Promise.resolve([]));
        }

        const [localModels, cloudModels] = await Promise.all(loadPromises);

        // 更新模型列表（本地模型 + 云端模型带 -cloud 后缀）
        if (localModels.length > 0 || cloudModels.length > 0) {
          updateLightX2VModels(localModels, cloudModels);
          setLightX2VModelsVersion((v) => v + 1); // 触发重渲染，让节点/配置面板重新读取 TOOLS
          console.log('[LightX2V] 模型列表已更新:', {
            local: localModels.length,
            cloud: cloudModels.length
          });
        }
      } catch (error) {
        console.error('[LightX2V] 获取模型列表失败:', error);
        // 不抛出错误，允许应用继续运行
      }
    };

    loadLightX2VModels();
    // 注意：虽然依赖 workflow，但内部有配置检查，所以实际只在配置变化时才会调用 API
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [getLightX2VConfig]);

  // Use useVoiceList Hook (after getLightX2VConfig is defined)
  const voiceList = useVoiceList(workflow, selectedNodeId, getLightX2VConfig);

  // Use useWorkflowAutoSave Hook
  const autoSaveHook = useWorkflowAutoSave({
    workflow,
    ensureWorkflowOwned,
    onSave: async (w) => {
      try {
        await saveWorkflowToDatabase(w, { name: w.name });
      } catch (e) {
        // 仅部署前端时后端不可用，回退到本地保存
        console.warn('[App] Autosave backend failed, saving locally:', e);
        await saveWorkflowToLocalOnly(w, { name: w.name });
      }
      setWorkflow(prev => prev && prev.id === w.id ? { ...prev, isDirty: false } : prev);
    }
  });

  // Helper function to get node outputs (needed by hooks)
  const getNodeOutputs = (node: WorkflowNode): Port[] => {
    const tool = TOOLS.find(t => t.id === node.tool_id);
    if (node.tool_id === 'text-generation' && node.data.custom_outputs) {
      return node.data.custom_outputs.map((o: any, i: number) => ({
        id: o.id ?? `out-text${i + 1}`,
        label: o.label ?? `Output ${i + 1}`,
        description: o.description,
        type: DataType.TEXT
      }));
    }
    return tool?.outputs || [];
  };

  // Use useNodeManagement Hook
  const nodeManagement = useNodeManagement({
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
  });

  // Use useConnectionManagement Hook
  const connectionManagement = useConnectionManagement({
    workflow,
    setWorkflow,
    selectedConnectionId,
    setSelectedConnectionId,
    setConnecting
  });

  // Use useResultManagement Hook
  const resultManagement = useResultManagement({
    workflow,
    showIntermediateResults,
    expandedOutput: modalState.expandedOutput,
    tempEditValue: modalState.tempEditValue,
    setWorkflow,
    setExpandedOutput: modalState.setExpandedOutput,
    setIsEditingResult: modalState.setIsEditingResult,
    lang
  });

  // Use useAIChatWorkflow Hook
  const aiChatWorkflow = useAIChatWorkflow({
    workflow,
    setWorkflow,
    getWorkflow,
    addNode: nodeManagement.addNode,
    deleteNode: nodeManagement.deleteNode,
    updateNodeData: nodeManagement.updateNodeData,
    replaceNode: nodeManagement.replaceNode,
    addConnection: connectionManagement.addConnection,
    deleteConnection: connectionManagement.deleteConnection,
    getNodeOutputs,
    getReplaceableTools: nodeManagement.getReplaceableTools,
    screenToWorldCoords,
    canvasRef,
    lang,
    lightX2VVoiceList: voiceList.lightX2VVoiceList,
    getLightX2VConfig,
    getCurrentHistoryIndex: undoRedo.getCurrentHistoryIndex,
    undoToIndex: undoRedo.undoToIndex,
    chatMode: aiChatMode
  });

  // Use useWorkflowExecution Hook
  const workflowExecution = useWorkflowExecution({
    workflow,
    setWorkflow,
    isPausedRef,
    setIsPaused,
    runningTaskIdsRef,
    abortControllerRef,
    getLightX2VConfig,
    setValidationErrors,
    setGlobalError,
    updateNodeData: nodeManagement.updateNodeData,
    voiceList,
    lang,
    onSaveExecutionToLocal: isStandalone() ? async (w) => { await saveWorkflowToLocalOnly(w); } : undefined,
    saveWorkflowBeforeRun: !isStandalone() && saveWorkflowToDatabase
      ? async (w) => await saveWorkflowToDatabase(w, { name: w.name })
      : undefined,
    refreshWorkflowFromBackend: !isStandalone() ? refreshWorkflowFromBackend : undefined,
    getWorkflow: !isStandalone() ? getWorkflow : undefined,
    // 批量/单节点执行前后各调用一次，将 workflow（含 node 参数）同步到后端
    updateWorkflowSync: !isStandalone() && saveWorkflowToDatabase
      ? async (_workflowId, w) => { await saveWorkflowToDatabase(w, { name: w.name }); }
      : undefined,
  });

  // Destructure updateNodeData from nodeManagement (for use in other places)
  const updateNodeData = nodeManagement.updateNodeData;

  // runWorkflow, validateWorkflow, stopWorkflow, cancelNodeRun, pendingRunNodeIds, getDescendants, resolveLightX2VResultRef from useWorkflowExecution
  const runWorkflow = workflowExecution.runWorkflow;
  const stopWorkflow = workflowExecution.stopWorkflow;
  const cancelNodeRun = workflowExecution.cancelNodeRun;
  const pendingRunNodeIds = workflowExecution.pendingRunNodeIds;
  const validateWorkflow = workflowExecution.validateWorkflow;
  const getDescendants = workflowExecution.getDescendants;
  const resolveLightX2VResultRef = workflowExecution.resolveLightX2VResultRef;

  // Pre-resolve lightx2v result refs when entering workflow editor (warm cache for node previews)
  useEffect(() => {
    if (currentView !== 'EDITOR' || !workflow || !resolveLightX2VResultRef) return;
    const refs = workflow.nodes.flatMap((n) => collectLightX2VResultRefs(n.output_value));
    refs.forEach((ref) => resolveLightX2VResultRef(ref).catch(() => {}));
  }, [currentView, workflow?.id, workflow?.nodes, resolveLightX2VResultRef]);

  // Helper functions for voice selection
  const getLanguageDisplayName = useCallback((langCode: string) => {
    const languageMap: Record<string, string> = {
      'chinese': '中文',
      'en_us': '美式英语',
      'en_gb': '英式英语',
      'en_au': '澳洲英语',
      'es': '西语',
      'ja': '日语'
    };
    return languageMap[langCode] || langCode;
  }, []);

  // filteredVoices, isFemaleVoice are provided by useVoiceList Hook

  const toggleLang = () => {
    const next = lang === 'en' ? 'zh' : 'en';
    setLang(next);
    localStorage.setItem('omniflow_lang', next);
  };

  // Workflow loading is handled by useWorkflow Hook

  // Load workflow from URL hash on mount
  useEffect(() => {
    const loadWorkflowFromHash = async () => {
      // Check if URL hash contains workflow ID
      const hash = window.location.hash;
      const workflowMatch = hash.match(/^#?workflow\/([a-f0-9-]+)$/i);

      if (workflowMatch) {
        const workflowId = workflowMatch[1];
        console.log('[App] Loading workflow from URL hash:', workflowId);

        try {
          if (isStandalone()) {
            const loaded = await loadWorkflow(workflowId);
            if (loaded) {
              setWorkflow(loaded.workflow);
              setCurrentView('EDITOR');
            }
            return;
          }

          const currentUserId = getCurrentUserId();
          const { isPreset, workflow: existingWorkflow } = await checkWorkflowOwnership(workflowId, currentUserId);

          if (existingWorkflow && !isPreset) {
              // Only load user's own workflows on hash change (preset workflows should be copied first)
              const loaded = await loadWorkflow(workflowId);
              if (loaded) {
                setWorkflow(loaded.workflow);
                setCurrentView('EDITOR');
              }
          } else {
            console.warn('[App] Workflow not found in database:', workflowId);
          }
        } catch (error) {
          console.error('[App] Failed to load workflow from URL hash:', error);
        }
      }
    };

    // Load workflow from hash on mount
    loadWorkflowFromHash();
  }, []); // Only run on mount - eslint-disable-line react-hooks/exhaustive-deps

  // Also listen for hash changes (when user navigates via browser back/forward)
  useEffect(() => {
    const handleHashChange = async () => {
      const hash = window.location.hash;
      const workflowMatch = hash.match(/^#?workflow\/([a-f0-9-]+)$/i);

      if (workflowMatch) {
        const workflowId = workflowMatch[1];

        // 同一工作流且当前已有状态（含后台运行/执行结果）：仅切回编辑页，保留状态
        if (workflow?.id === workflowId) {
          setCurrentView('EDITOR');
          return;
        }

        // Only load if it's a different workflow
        if (!workflow || workflow.id !== workflowId) {
          console.log('[App] Hash changed, loading workflow:', workflowId);

          try {
            if (isStandalone()) {
              const loaded = await loadWorkflow(workflowId);
              if (loaded) {
                setWorkflow(loaded.workflow);
                setCurrentView('EDITOR');
              }
              return;
            }
            const currentUserId = getCurrentUserId();
            const { isPreset, workflow: existingWorkflow } = await checkWorkflowOwnership(workflowId, currentUserId);

            if (existingWorkflow && !isPreset) {
              // Only load user's own workflows on hash change (preset workflows should be copied first)
              const loaded = await loadWorkflow(workflowId);
              if (loaded) {
                setWorkflow(loaded.workflow);
                setCurrentView('EDITOR');
              }
            } else if (isPreset) {
              // For preset workflows, redirect to dashboard
              setCurrentView('DASHBOARD');
              // Clear hash or set to empty
              if (window.history && window.history.replaceState) {
                window.history.replaceState(null, '', '#');
              }
            }
          } catch (error) {
            console.error('[App] Failed to load workflow from hash change:', error);
          }
        }
      } else if (hash === '' || hash === '#') {
        // Hash cleared - go back to dashboard
        if (currentView === 'EDITOR') {
          setCurrentView('DASHBOARD');
        }
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [workflow, currentView, loadWorkflow, getLightX2VConfig, setWorkflow, setCurrentView]);

  useEffect(() => {
    let interval: any;
    if (workflow?.isRunning) {
      interval = setInterval(() => setTicker(t => t + 1), 100);
    }
    return () => clearInterval(interval);
  }, [workflow?.isRunning]);

  // Voice list loading is handled by useVoiceList Hook
  // Auto-match resource_id when voice list is loaded
  useEffect(() => {
    if (!voiceList.lightX2VVoiceList?.voices || !workflow) return;

    const voiceData = voiceList.lightX2VVoiceList;
    if (voiceData.voices && voiceData.voices.length > 0) {
          workflow.nodes.forEach(node => {
            const isLightX2V = node.data.model === 'lightx2v' || node.data.model?.startsWith('lightx2v');
            if (node.tool_id === 'tts' && isLightX2V && node.data.voiceType) {
              const matchingVoice = voiceData.voices.find((v: any) => v.voice_type === node.data.voiceType);
              if (matchingVoice?.resource_id) {
                // Only update if resourceId is empty, wrong, or missing
                if (!node.data.resourceId || node.data.resourceId !== matchingVoice.resource_id) {
                  updateNodeData(node.id, 'resourceId', matchingVoice.resource_id);
                  console.log(`[LightX2V] Auto-matched resource_id: ${matchingVoice.resource_id} for voice: ${node.data.voiceType}`);
                }
              }
            }
          });
        }
  }, [voiceList.lightX2VVoiceList, workflow]);

  const loadCommunityWorkflows = useCallback(async () => {
    if (isStandalone()) {
      setCommunityWorkflows([]);
      return;
    }
    try {
      const response = await apiRequest('/api/v1/workflow/list?public=true&page=1&page_size=100');
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }
      const data = await response.json();
      const workflows: WorkflowState[] = (data.workflows || []).map((wf: any) => ({
        id: wf.workflow_id,
        name: wf.name,
        nodes: wf.nodes || [],
        connections: wf.connections || [],
        isDirty: false,
        isRunning: false,
        globalInputs: wf.global_inputs || {},
        updatedAt: wf.update_t ? wf.update_t * 1000 : Date.now(),
        visibility: wf.visibility || 'public',
        thumsupCount: wf.thumsup_count ?? 0,
        thumsupLiked: wf.thumsup_liked ?? false,
        authorName: wf.author_name || wf.user_id || 'Anonymous',
        authorId: wf.user_id
      }));
      setCommunityWorkflows(workflows);
    } catch (error) {
      console.warn('[Workflow] Failed to load community workflows:', error);
      setCommunityWorkflows([]);
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'COMMUNITY') {
      loadCommunityWorkflows();
    }
  }, [activeTab, loadCommunityWorkflows]);

  // saveWorkflowToLocal and deleteWorkflow are provided by useWorkflow Hook

  const deleteWorkflow = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(t('confirm_delete'))) return;
    deleteWorkflowFromHook(id);
  }, [t, deleteWorkflowFromHook]);

  const handleEditorVisibilityChange = useCallback(async (isPublic: boolean) => {
    if (!workflow) return;
    const visibility = isPublic ? 'public' : 'private';
    const nextWorkflow = { ...workflow, visibility, isDirty: true };
    setWorkflow(nextWorkflow);

    const isSavedWorkflow = workflow.id && workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
    const isTemporaryId = workflow.id && (workflow.id.startsWith('temp-') || workflow.id.startsWith('flow-'));

    try {
      if (isSavedWorkflow && !isTemporaryId) {
        await updateWorkflowVisibility(workflow.id, visibility);
      }

      const savedId = await saveWorkflowToDatabase(nextWorkflow, { name: nextWorkflow.name });
      await saveWorkflowToLocal({ ...nextWorkflow, id: savedId || nextWorkflow.id });
      setWorkflow(prev => prev ? ({ ...prev, id: savedId || prev.id, isDirty: false, visibility }) : prev);
      autoSaveHook?.resetAutoSaveTimer();
    } catch (error: any) {
      setGlobalError({
        message: t('execution_failed'),
        details: error?.message || String(error)
      });
    }
  }, [workflow, updateWorkflowVisibility, saveWorkflowToDatabase, saveWorkflowToLocal, autoSaveHook, setWorkflow, setGlobalError, t]);

  const handleToggleWorkflowVisibility = useCallback(async (workflowId: string, visibility: 'private' | 'public') => {
    try {
      await updateWorkflowVisibility(workflowId, visibility);

      if (workflow && workflow.id === workflowId) {
        const nextWorkflow = { ...workflow, visibility, isDirty: true };
        const savedId = await saveWorkflowToDatabase(nextWorkflow, { name: nextWorkflow.name });
        await saveWorkflowToLocal({ ...nextWorkflow, id: savedId || nextWorkflow.id });
        setWorkflow(prev => prev ? ({ ...prev, id: savedId || prev.id, isDirty: false, visibility }) : prev);
        autoSaveHook?.resetAutoSaveTimer();
      }
    } catch (error: any) {
      setGlobalError({
        message: t('execution_failed'),
        details: error?.message || String(error)
      });
    }
  }, [updateWorkflowVisibility, workflow, saveWorkflowToDatabase, saveWorkflowToLocal, setWorkflow, autoSaveHook, setGlobalError, t]);

  const handleToggleWorkflowThumsup = useCallback(async (workflowId: string) => {
    if (isStandalone()) {
      setMyWorkflows(prev => prev.map(w => {
        if (w.id !== workflowId) return w;
        const nextLiked = !w.thumsupLiked;
        const nextCount = Math.max(0, (w.thumsupCount ?? 0) + (nextLiked ? 1 : -1));
        return { ...w, thumsupLiked: nextLiked, thumsupCount: nextCount };
      }));
      setCommunityWorkflows(prev => prev.map(w => {
        if (w.id !== workflowId) return w;
        const nextLiked = !w.thumsupLiked;
        const nextCount = Math.max(0, (w.thumsupCount ?? 0) + (nextLiked ? 1 : -1));
        return { ...w, thumsupLiked: nextLiked, thumsupCount: nextCount };
      }));
      setWorkflow(prev => prev && prev.id === workflowId
        ? { ...prev, thumsupLiked: !prev.thumsupLiked, thumsupCount: Math.max(0, (prev.thumsupCount ?? 0) + (prev.thumsupLiked ? -1 : 1)) }
        : prev);
      return;
    }
    try {
      const response = await apiRequest(`/api/v1/workflow/${workflowId}/thumsup`, {
        method: 'POST'
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }
      const data = await response.json();
      const nextCount = data.thumsup_count ?? 0;
      const nextLiked = data.thumsup_liked ?? false;

      setMyWorkflows(prev => prev.map(w => w.id === workflowId ? { ...w, thumsupCount: nextCount, thumsupLiked: nextLiked } : w));
      setCommunityWorkflows(prev => prev.map(w => w.id === workflowId ? { ...w, thumsupCount: nextCount, thumsupLiked: nextLiked } : w));
      setWorkflow(prev => prev ? (prev.id === workflowId ? { ...prev, thumsupCount: nextCount, thumsupLiked: nextLiked } : prev) : prev);
    } catch (error: any) {
      setGlobalError({
        message: t('execution_failed'),
        details: error?.message || String(error)
      });
    }
  }, [setMyWorkflows, setCommunityWorkflows, setWorkflow, setGlobalError, t]);

  const scheduleNameSave = useCallback((nextWorkflow: WorkflowState) => {
    if (!nextWorkflow?.id) return;
    const isUuid = nextWorkflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
    const isSavedWorkflow = isUuid || isStandalone();
    if (!isSavedWorkflow) return;

    if (nameSaveTimerRef.current) {
      clearTimeout(nameSaveTimerRef.current);
    }

    nameSaveTimerRef.current = setTimeout(async () => {
      try {
        const savedId = await saveWorkflowToDatabase(nextWorkflow, { name: nextWorkflow.name });
        await saveWorkflowToLocal({ ...nextWorkflow, id: savedId || nextWorkflow.id });
        setWorkflow(prev => prev && prev.name === nextWorkflow.name ? { ...prev, id: savedId || prev.id, isDirty: false } : prev);
        autoSaveHook?.resetAutoSaveTimer();
      } catch (error: any) {
        setGlobalError({
          message: t('execution_failed'),
          details: error?.message || String(error)
        });
      }
    }, 600);
  }, [saveWorkflowToDatabase, saveWorkflowToLocal, setWorkflow, autoSaveHook, setGlobalError, t]);

  const openWorkflow = useCallback(async (w: WorkflowState) => {
    // 若正在重新进入的是当前已在后台运行（或刚跑完）的同一工作流，保留当前状态，仅切回编辑页，便于“退出后继续跟踪、再进入仍是之前样子”
    if (workflow?.id === w.id && (
      workflow.isRunning ||
      workflow.nodes?.some((n) => n.status === NodeStatus.RUNNING || n.status === NodeStatus.SUCCESS || n.status === NodeStatus.ERROR)
    )) {
      setCurrentView('EDITOR');
      if (typeof window !== 'undefined' && window.history?.replaceState) {
        window.history.replaceState(null, '', `#workflow/${w.id}`);
      }
      return;
    }

    setSelectedNodeId(null);
    setSelectedConnectionId(null);
    setValidationErrors([]);
    voiceList.resetVoiceList(); // Reset voice list when switching workflows
    voiceList.resetCloneVoiceList(); // Reset clone voice list

    // Try to load from database if workflow has a database ID
    let workflowToOpen = w;
    let workflowIdToOpen = w.id;

    // 先保证 workflow 属于当前用户
    // 若为预设工作流，则调用/create 创建用户自己的 workflow
    // 若为公共工作流，则调用/copy 复制公共工作流
    if (!isStandalone() && workflowIdToOpen) {
      try {
        const newId = await ensureWorkflowOwned(workflowToOpen);
        if (newId && newId !== workflowIdToOpen) {
          const loaded = await loadWorkflow(newId);
          if (loaded) {
            workflowToOpen = loaded.workflow;
            workflowIdToOpen = newId;
          }
        }
      } catch (error) {
        console.error('[App] Failed to ensure workflow owned:', error);
      }
    }

    const newWorkflow = {
      ...workflowToOpen,
      id: workflowIdToOpen, // Use the correct workflow ID (original or copied)
      isDirty: false,
      isRunning: false
    };
    setWorkflow(newWorkflow);
    setCurrentView('EDITOR');

    // Update URL to show workflow_id
    if (typeof window !== 'undefined' && window.history && window.history.replaceState) {
      window.history.replaceState(null, '', `#workflow/${workflowIdToOpen}`);
    }

    // History will be automatically initialized by useUndoRedo when workflow changes
  }, [workflow, loadWorkflow, voiceList, setWorkflow, setCurrentView, setSelectedNodeId, setSelectedConnectionId, setValidationErrors, ensureWorkflowOwned]);

  const createNewWorkflow = useCallback(async () => {
    setSelectedNodeId(null);
    setSelectedConnectionId(null);
    setValidationErrors([]);
    voiceList.resetVoiceList(); // Reset voice list when creating new workflow
    voiceList.resetCloneVoiceList(); // Reset clone voice list

    // 纯前端部署使用 UUID；有后端时用 temp- 由后端分配正式 ID
    const newWorkflowId = isStandalone()
      ? (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `workflow-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`)
      : `temp-${Date.now()}`;

    // Create workflow in database
    try {
      const tempFlow: WorkflowState = {
        id: newWorkflowId,
        name: t('untitled'),
        nodes: [],
        connections: [],
        isDirty: false,
        isRunning: false,
        globalInputs: {},
        updatedAt: Date.now(),
        createAt: Date.now(),
        visibility: 'private'
      };
      // Create workflow in database (backend will generate workflow_id)
      const response = await apiRequest('/api/v1/workflow/create', {
          method: 'POST',
          body: JSON.stringify(tempFlow)
      });

      let finalWorkflowId = null;
      if (response.ok) {
          const data = await response.json();
          finalWorkflowId = data.workflow_id;
          console.log('[Workflow] Workflow created in database with ID:', finalWorkflowId);
      } else {
          const errorText = await response.text();
          throw new Error(errorText);
      }

      // Update workflow with backend-generated ID
      const finalFlow = { ...tempFlow, id: finalWorkflowId, isDirty: false };
      setWorkflow(finalFlow);
      setCurrentView('EDITOR');
      // Update URL to show workflow_id
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, '', `#workflow/${finalWorkflowId}`);
      }
    } catch (error) {
      console.error('[App] Failed to create workflow in database, creating local workflow:', error);
      // Fallback to local workflow (use same ID: UUID in standalone, temp- when backend failed)
      const newFlow: WorkflowState = {
        id: newWorkflowId,
        name: t('untitled'),
        nodes: [],
        connections: [],
        isDirty: true,
        isRunning: false,
        globalInputs: {},
        updatedAt: Date.now(),
        createAt: Date.now()
      };
      setWorkflow(newFlow);
      setCurrentView('EDITOR');
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, '', `#workflow/${newWorkflowId}`);
      }
    }
    // History will be automatically initialized by useUndoRedo when workflow changes
  }, [saveWorkflowToDatabase, loadWorkflow, t, voiceList, setWorkflow, setCurrentView, setSelectedNodeId, setSelectedConnectionId, setValidationErrors]);

  const handleCreateWorkflow = useCallback(() => {
    setSidebarDefaultTab('tools');
    void createNewWorkflow();
  }, [createNewWorkflow]);

  const handleOpenWorkflow = useCallback((workflowToOpen: WorkflowState) => {
    setSidebarDefaultTab('tools');
    void openWorkflow(workflowToOpen);
  }, [openWorkflow]);

  const handleBackToDashboard = useCallback(() => {
    if (typeof window !== 'undefined' && window.history?.replaceState) {
      window.history.replaceState(null, '', '#');
    }
    setCurrentView('DASHBOARD');
    void loadWorkflows();
  }, [loadWorkflows, setCurrentView]);

  const handleAIGenerateClick = useCallback(() => {
    setSidebarDefaultTab('chat');
    setFocusAIChatInput(true);
    void createNewWorkflow();
  }, [createNewWorkflow]);

  const selectedNode = useMemo(() => workflow?.nodes.find(n => n.id === selectedNodeId), [workflow, selectedNodeId]);

  // expandedResultData, activeResultsList, and handleManualResultEdit are now provided by useResultManagement Hook
  const expandedResultData = resultManagement.expandedResultData;
  const resultEntries = resultManagement.resultEntries;
  const handleManualResultEdit = resultManagement.handleManualResultEdit;

  // 放大 modal 打开时默认进入编辑模式并初始化 tempEditValue（仅 TEXT 类型）；content 为 string 时才写入，避免 file ref 未解析时写入 JSON
  const expandedOutputOpenKeyRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (!modalState.expandedOutput) {
      expandedOutputOpenKeyRef.current = null;
      return;
    }
    if (!expandedResultData) return;
    const key = `${modalState.expandedOutput.nodeId}:${modalState.expandedOutput.fieldId ?? ''}`;
    const keyChanged = expandedOutputOpenKeyRef.current !== key;
    if (keyChanged) {
      expandedOutputOpenKeyRef.current = key;
      modalState.setIsEditingResult(true);
    }
    if (expandedResultData.type === DataType.TEXT) {
      const content = expandedResultData.content;
      const isStableText = typeof content === 'string';
      const initial = isStableText ? content : (typeof content === 'object' ? JSON.stringify(content, null, 2) : String(content ?? ''));
      if (keyChanged) {
        modalState.setTempEditValue(initial);
      } else if (isStableText && (modalState.tempEditValue === '' || modalState.tempEditValue === (lang === 'zh' ? '加载中…' : 'Loading…'))) {
        modalState.setTempEditValue(content);
      } else if (isStableText && typeof modalState.tempEditValue === 'string' && modalState.tempEditValue.trim().startsWith('{') && modalState.tempEditValue.includes('file_id')) {
        modalState.setTempEditValue(content);
      }
    }
  }, [modalState.expandedOutput, expandedResultData, modalState.tempEditValue, lang]);


  const handleGlobalInputChange = useCallback((nodeId: string, portId: string, value: any) => {
    setValidationErrors([]);
    setWorkflow(prev => prev ? ({ ...prev, globalInputs: { ...prev.globalInputs, [`${nodeId}-${portId}`]: value }, isDirty: true }) : null);
  }, []);

  const handleDescriptionChange = useCallback((description: string) => {
    setWorkflow(prev => prev ? ({ ...prev, description, isDirty: true }) : null);
  }, []);

  const handleTagsChange = useCallback((tags: string[]) => {
    setWorkflow(prev => prev ? ({ ...prev, tags, isDirty: true }) : null);
  }, []);

  // addNode and pinOutputToCanvas are now provided by useNodeManagement Hook
  const addNode = nodeManagement.addNode;
  const pinOutputToCanvas = nodeManagement.pinOutputToCanvas;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Undo/Redo shortcuts (Ctrl+Z / Ctrl+Y or Ctrl+Shift+Z)
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key === 'z') {
        e.preventDefault();
        if (currentView === 'EDITOR' && workflow) {
          undoRedo.undo();
        }
        return;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.shiftKey && e.key === 'z'))) {
        e.preventDefault();
        if (currentView === 'EDITOR' && workflow) {
          undoRedo.redo();
        }
        return;
      }

      // Delete/Backspace for nodes and connections
      if ((e.key === 'Delete' || e.key === 'Backspace') && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '')) {
        if (selectedNodeId) nodeManagement.deleteNode(selectedNodeId);
        if (selectedConnectionId) connectionManagement.deleteConnection(selectedConnectionId);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNodeId, selectedConnectionId, nodeManagement, connectionManagement, currentView, workflow, undoRedo]);

  // Close replace menu, output quick add menu, model select, and voice select when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (modalState.showReplaceMenu && !target.closest('.replace-menu-container')) {
        modalState.setShowReplaceMenu(null);
      }
      if (modalState.showOutputQuickAdd && !target.closest('.output-quick-add-menu')) {
        modalState.setShowOutputQuickAdd(null);
      }
      if (modalState.showModelSelect && !target.closest('.model-select-container')) {
        modalState.setShowModelSelect(null);
      }
      if (modalState.showVoiceSelect && !target.closest('.voice-select-container')) {
        modalState.setShowVoiceSelect(null);
      }
    };
    window.addEventListener('click', handleClickOutside);
    return () => window.removeEventListener('click', handleClickOutside);
  }, [modalState.showReplaceMenu, modalState.showOutputQuickAdd, modalState.showModelSelect, modalState.showVoiceSelect]);

  // deleteSelectedNode is now provided by useNodeManagement Hook
  const deleteSelectedNode = useCallback(() => {
    if (!selectedNodeId) return;
    nodeManagement.deleteNode(selectedNodeId);
  }, [selectedNodeId, nodeManagement]);

  // getReplaceableTools is now provided by useNodeManagement Hook
  const getReplaceableTools = nodeManagement.getReplaceableTools;

  // Replace a node with another compatible tool (complex logic with connection mapping)
  const replaceNode = useCallback((nodeId: string, newToolId: string) => {
    if (!workflow) return;
    const node = workflow.nodes.find(n => n.id === nodeId);
    if (!node) return;

    const newTool = TOOLS.find(t => t.id === newToolId);
    if (!newTool) return;

    // Get current node outputs
    const currentNodeOutputs = getNodeOutputs(node);

    // Handle text-generation special case (dynamic outputs)
    let newToolOutputs: Port[] = [];
    let newCustomOutputs: any[] | undefined = undefined;
    if (newToolId === 'text-generation') {
      // If replacing with text-generation, preserve customOutputs if they exist (including custom port ids)
      if (node.tool_id === 'text-generation' && node.data.custom_outputs) {
        newCustomOutputs = node.data.custom_outputs;
        newToolOutputs = node.data.custom_outputs.map((o: any) => ({
          id: o.id ?? `out-text1`,
          label: o.label ?? 'Output',
          description: o.description,
          type: DataType.TEXT
        }));
      } else {
        // When replacing another node with text-generation, create customOutputs (default ids out-text1, out-text2...)
        newCustomOutputs = currentNodeOutputs.map((out, idx) => ({
          id: out.id ?? `out-text${idx + 1}`,
          label: out.label || `Output ${idx + 1}`,
          description: out.label || `Output ${idx + 1}`
        }));
        newToolOutputs = newCustomOutputs.map((o: any) => ({ id: o.id, label: o.label, description: o.description, type: DataType.TEXT }));
      }
    } else {
      newToolOutputs = newTool.outputs;
    }

    // Check if outputs are compatible (for text-generation, we've already created matching outputs)
    if (currentNodeOutputs.length !== newToolOutputs.length) return;
    const isCompatible = currentNodeOutputs.every((out, idx) => {
      if (idx >= newToolOutputs.length) return false;
      return out.type === newToolOutputs[idx].type;
    });
    if (!isCompatible) return;

    // Create a mapping of old output port IDs to new ones
    const outputPortMap: Record<string, string> = {};
    currentNodeOutputs.forEach((oldOut, idx) => {
      if (idx < newToolOutputs.length) {
        outputPortMap[oldOut.id] = newToolOutputs[idx].id;
      }
    });

    // Update the node
    setWorkflow(prev => {
      if (!prev) return null;

      // Create new node with new tool
      const newNode: WorkflowNode = {
        ...node,
        tool_id: newToolId,
        data: {
          ...node.data,
          // Reset model if the new tool doesn't have models
          model: newTool.models && newTool.models.length > 0 ? (newTool.models[0].id || node.data.model) : undefined,
          // Preserve custom_outputs if replacing with text-generation and current node has them
          custom_outputs: newToolId === 'text-generation' ? (newCustomOutputs || node.data.custom_outputs) : (newToolId !== 'text-generation' ? node.data.custom_outputs : undefined)
        },
        status: NodeStatus.IDLE,
        error: undefined,
        execution_time: undefined,
        start_time: undefined
      };

      // Update connections: map old output port IDs to new ones
      // Special handling for TTS -> Voice Clone replacement
      const isTTSToVoiceClone = node.tool_id === 'tts' && newToolId === 'lightx2v-voice-clone';

      const updatedConnections = prev.connections.map(conn => {
        // Handle output connections (source is the replaced node)
        if (conn.source_node_id === nodeId) {
          const newSourcePortId = outputPortMap[conn.source_port_id];
          if (newSourcePortId) {
            return { ...conn, source_port_id: newSourcePortId };
          }
          // If no mapping found, remove the connection
          return null;
        }

        // Handle input connections (target is the replaced node)
        if (conn.target_node_id === nodeId) {
          // Special case: TTS -> Voice Clone
          if (isTTSToVoiceClone) {
            // Map in-text to in-tts-text
            if (conn.target_port_id === 'in-text') {
              return { ...conn, target_port_id: 'in-tts-text' };
            }
            // Remove in-context-tone connection
            if (conn.target_port_id === 'in-context-tone') {
              return null;
            }
          }

          // For other replacements, try to map input ports
          const oldTool = TOOLS.find(t => t.id === node.tool_id);
          const oldInputs = oldTool?.inputs || [];
          const newInputs = newTool.inputs || [];

          // Try to find matching input port by type and position
          const oldInputIndex = oldInputs.findIndex(inp => inp.id === conn.target_port_id);
          if (oldInputIndex >= 0 && oldInputIndex < newInputs.length) {
            const oldInput = oldInputs[oldInputIndex];
            const newInput = newInputs[oldInputIndex];
            // Only map if types match
            if (oldInput.type === newInput.type) {
              return { ...conn, target_port_id: newInput.id };
            }
          }

          // If no mapping found, check if port ID exists in new tool
          const portExists = newInputs.some(inp => inp.id === conn.target_port_id);
          if (portExists) {
            return conn; // Keep connection if port exists
          }

          // Remove connection if port doesn't exist
          return null;
        }

        return conn;
      }).filter((c): c is Connection => c !== null);

      return {
        ...prev,
        nodes: prev.nodes.map(n => n.id === nodeId ? newNode : n),
        connections: updatedConnections,
        isDirty: true
      };
    });

    modalState.setShowReplaceMenu(null);
  }, [workflow]);

  // deleteSelectedConnection and addConnection are now provided by useConnectionManagement Hook
  const deleteSelectedConnection = useCallback(() => {
    if (!selectedConnectionId) return;
    connectionManagement.deleteConnection(selectedConnectionId);
  }, [selectedConnectionId, connectionManagement]);

  const addConnection = connectionManagement.addConnection;

  // quickAddInput is now provided by useNodeManagement Hook
  const quickAddInput = nodeManagement.quickAddInput;

  // Get tools that can accept a specific output type
  const getCompatibleToolsForOutput = useCallback((outputType: DataType): ToolDefinition[] => {
    return TOOLS.filter(tool => {
      // Skip input nodes
      if (tool.category === 'Input') return false;
      // Find tools that have at least one input port matching the output type
      return tool.inputs.some(input => input.type === outputType);
    });
  }, []);

  // quickAddOutput is now provided by useNodeManagement Hook
  const quickAddOutput = nodeManagement.quickAddOutput;

  const disconnectedInputs = useMemo(() => {
    if (!workflow) return [];
    const list: { nodeId: string; port: Port; toolName: string; isSourceNode?: boolean; dataType: DataType }[] = [];

    workflow.nodes.forEach(node => {
      const tool = TOOLS.find(t => t.id === node.tool_id);
      if (!tool || tool.category === 'Input') return;
      tool.inputs.forEach(port => {
        const isConnected = workflow.connections.some(c => c.target_node_id === node.id && c.target_port_id === port.id);
        if (!isConnected) {
           list.push({ nodeId: node.id, port, toolName: (lang === 'zh' ? tool.name_zh : tool.name), dataType: port.type });
        }
      });
    });

    workflow.nodes.forEach(node => {
      const tool = TOOLS.find(t => t.id === node.tool_id);
      if (!tool || tool.category !== 'Input') return;

      const val = node.data.value;
      const isEmpty = (Array.isArray(val) && val.length === 0) || !val;
      if (isEmpty) {
        list.push({
          nodeId: node.id,
          port: tool.outputs[0],
          toolName: (lang === 'zh' ? tool.name_zh : tool.name),
          isSourceNode: true,
          dataType: tool.outputs[0].type
        });
      }
    });

    return list;
  }, [workflow?.nodes, workflow?.connections, lang]);

  // validateWorkflow and runWorkflow are now provided by useWorkflowExecution Hook

  // activeResultsList is now provided by useResultManagement Hook

  // 节点拖拽时用 window 监听 mousemove/mouseup，保证鼠标移出画布或经过其他节点时位置仍跟随（与连接线拖拽一致）；并禁止拖拽期间文本选中
  useEffect(() => {
    if (!draggingNode) return;
    document.body.style.userSelect = 'none';
    const onMove = (e: MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const world = screenToWorldCoords(x, y);
      setWorkflow(prev =>
        prev
          ? {
              ...prev,
              nodes: prev.nodes.map(n =>
                n.id === draggingNode.id
                  ? { ...n, x: world.x - draggingNode.offsetX, y: world.y - draggingNode.offsetY }
                  : n
              ),
              isDirty: true
            }
          : null
      );
    };
    const onUp = () => {
      document.body.style.userSelect = '';
      canvasHandleNodeDragEnd();
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [draggingNode, canvasHandleNodeDragEnd, screenToWorldCoords]);

  // Enhanced mouse handlers that combine canvas Hook with workflow updates（节点拖拽位置更新已移到上面 window 监听）
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      // Check if mouse is over a node or port
      const target = e.target as HTMLElement;
      const isOverNodeElement =
        target.closest('.node-element') || target.closest('.port') || target.closest('.connection-path');
      setIsOverNode(!!isOverNodeElement);

      // Call canvas Hook's handleMouseMove for panning and connecting line
      canvasHandleMouseMove(e);
    },
    [canvasHandleMouseMove]
  );

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    // Call canvas Hook's handleMouseDown for panning
    canvasHandleMouseDown(e);
    // Additional logic: clear selection when clicking on canvas
    if (!(e.target as HTMLElement).closest('.node-element') &&
        !(e.target as HTMLElement).closest('.port') &&
        !(e.target as HTMLElement).closest('.connection-path')) {
      setSelectedNodeId(null);
      setSelectedConnectionId(null);
    }
  }, [canvasHandleMouseDown]);

  const handleNodeDragStart = useCallback((nodeId: string, offsetX: number, offsetY: number) => {
    canvasHandleNodeDragStart(nodeId, offsetX, offsetY);
  }, [canvasHandleNodeDragStart]);

  const handleNodeDrag = useCallback((nodeId: string, x: number, y: number) => {
    if (!draggingNode || draggingNode.id !== nodeId) return;
    // Call canvas Hook's handleNodeDrag
    canvasHandleNodeDrag(nodeId, x, y);
    // Update workflow with new node position
    setWorkflow(prev => prev ? ({
      ...prev,
      nodes: prev.nodes.map(n => n.id === nodeId ? {
        ...n,
        x: x - draggingNode.offsetX,
        y: y - draggingNode.offsetY
      } : n),
      isDirty: true
    }) : null);
  }, [draggingNode, canvasHandleNodeDrag]);

  const handleNodeDragEnd = useCallback(() => {
    canvasHandleNodeDragEnd();
  }, [canvasHandleNodeDragEnd]);

  const clearSnapshot = useCallback(() => {}, []);

  const getNodeOutputUrlForWorkflow = useCallback(
    (nodeId: string, portId: string, fileId?: string, runId?: string) => (workflow?.id ? getNodeOutputUrl(workflow.id, nodeId, portId, fileId, runId) : Promise.resolve(null)),
    [workflow?.id]
  );

  const [audioEditorDisplayUrl, setAudioEditorDisplayUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!modalState.showAudioEditor || !workflow?.id) {
      setAudioEditorDisplayUrl(null);
      return;
    }
    const node = workflow.nodes.find(n => n.id === modalState.showAudioEditor);
    if (!node || node.tool_id !== 'audio-input') {
      setAudioEditorDisplayUrl(null);
      return;
    }
    const audioSrc = node.data.value ?? (node.output_value && getOutputValueByPort(node, 'out-audio'));
    if (!audioSrc) {
      setAudioEditorDisplayUrl(null);
      return;
    }
    if (typeof audioSrc === 'string') {
      setAudioEditorDisplayUrl(audioSrc);
      return;
    }
    const ref = audioSrc as { file_id?: string; run_id?: string };
    if (ref.file_id && getNodeOutputUrl) {
      getNodeOutputUrl(workflow.id, modalState.showAudioEditor, 'out-audio', ref.file_id, ref.run_id).then((url) => setAudioEditorDisplayUrl(url ?? ''));
    } else {
      setAudioEditorDisplayUrl('');
    }
  }, [modalState.showAudioEditor, workflow?.id, workflow?.nodes, getNodeOutputUrl]);

  const sourceNodes = React.useMemo(() => (workflow?.nodes ?? []), [workflow]);

  const sourceOutputs = React.useMemo(() => {
    if (!workflow) return {};
    return Object.fromEntries(
      workflow.nodes
        .filter((n) => n.output_value !== undefined && n.output_value !== null)
        .map((n) => [n.id, n.output_value])
    );
  }, [workflow]);

  if (currentView === 'DASHBOARD') {
    return (
      <div className="flex flex-col h-full">
        <Dashboard
          lang={lang}
          myWorkflows={myWorkflows}
          communityWorkflows={communityWorkflows}
          hideCommunityTab={isStandalone()}
          activeTab={activeTab}
          onToggleLang={toggleLang}
          onCreateWorkflow={handleCreateWorkflow}
          onAIGenerate={handleAIGenerateClick}
          onOpenWorkflow={handleOpenWorkflow}
          onDeleteWorkflow={deleteWorkflow}
          onToggleThumbsup={handleToggleWorkflowThumsup}
          onToggleWorkflowVisibility={handleToggleWorkflowVisibility}
          onSetActiveTab={setActiveTab}
          resolveLightX2VResultRef={resolveLightX2VResultRef}
        />
      </div>
    );
  }

  if (!workflow) return null;

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-200 selection:bg-indigo-500/30 font-sans overflow-hidden">
      <ExpandedOutputModal
        lang={lang}
        expandedOutput={modalState.expandedOutput}
        expandedResultData={expandedResultData}
        resolveLightX2VResultRef={resolveLightX2VResultRef}
        workflowId={workflow?.id}
        workflow={workflow ?? undefined}
        sourceOutputs={sourceOutputs}
        resultEntryOutputForNode={modalState.expandedOutput ? (resultEntries.find(e => e.nodeId === modalState.expandedOutput?.nodeId)?.output) : undefined}
        getNodeOutputUrl={workflow?.id ? getNodeOutputUrlForWorkflow : undefined}
        isEditingResult={modalState.isEditingResult}
        tempEditValue={modalState.tempEditValue}
          onClose={() => {
            modalState.setExpandedOutput(null);
            modalState.setIsEditingResult(false);
          }}
          onEditToggle={() => {
            if (!modalState.isEditingResult) {
              modalState.setTempEditValue(
                typeof expandedResultData?.content === 'object'
                  ? JSON.stringify(expandedResultData.content, null, 2)
                  : expandedResultData?.content || ''
              );
            }
            modalState.setIsEditingResult(!modalState.isEditingResult);
          }}
          onSaveEdit={handleManualResultEdit}
          onTempEditValueChange={modalState.setTempEditValue}
      />

      <Editor
        lang={lang}
        workflow={workflow}
        view={view}
        selectedNodeId={selectedNodeId}
        selectedConnectionId={selectedConnectionId}
        connecting={connecting}
        mousePos={mousePos}
        clientMousePos={clientMousePos}
        nodeHeights={nodeHeightsRef.current}
        sourceNodes={sourceNodes}
        sourceOutputs={sourceOutputs}
        isPaused={isPaused}
        isRunning={workflow.isRunning}
        isSaving={isSavingWorkflow}
        sidebarCollapsed={modalState.sidebarCollapsed}
        sidebarDefaultTab={sidebarDefaultTab}
        focusAIChatInput={focusAIChatInput}
        onAIChatInputFocused={() => setFocusAIChatInput(false)}
        validationErrors={validationErrors}
        globalError={globalError}
        canvasRef={canvasRef}
        onBack={handleBackToDashboard}
        onWorkflowNameChange={(name) => {
          setWorkflow(p => {
            if (!p) return null;
            const next = { ...p, name, isDirty: true };
            scheduleNameSave(next);
            return next;
          });
        }}
        onZoomIn={zoomIn}
        onZoomOut={zoomOut}
        onResetView={resetView}
        onToggleLang={toggleLang}
        onClearSnapshot={clearSnapshot}
        onSave={async () => {
          if (!workflow || isSavingWorkflow) return; // Prevent multiple saves
          try {
            await saveWorkflowToDatabase(workflow, { name: workflow.name });
            // Also save to localStorage as backup
            await saveWorkflowToLocal(workflow);
            setWorkflow(prev => prev ? { ...prev, isDirty: false } : null);
            // 手动保存后重置自动保存计时器
            if (autoSaveHook) {
              autoSaveHook.resetAutoSaveTimer();
            }
          } catch (error) {
            console.error('[App] Failed to save workflow to database:', error);
            // Fallback to localStorage
            await saveWorkflowToLocal(workflow);
          }
        }}
        onPause={() => {
                const newPausedState = !isPaused;
                setIsPaused(newPausedState);
                isPausedRef.current = newPausedState;
              }}
        onRun={() => runWorkflow()}
        onStop={async () => {
          if (stopWorkflow) {
            await stopWorkflow();
          }
        }}
          onToggleSidebar={() => modalState.setSidebarCollapsed(!modalState.sidebarCollapsed)}
        canUndo={undoRedo.canUndo}
        canRedo={undoRedo.canRedo}
        onUndo={undoRedo.undo}
        onRedo={undoRedo.redo}
        onVisibilityChange={handleEditorVisibilityChange}
        onAddNode={addNode}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
        onMouseUp={canvasHandleMouseUp}
        onMouseLeave={canvasHandleMouseLeave}
        onWheel={canvasHandleWheel}
        onNodeSelect={setSelectedNodeId}
        onConnectionSelect={setSelectedConnectionId}
        onDeleteConnection={connectionManagement.deleteConnection}
        onNodeDragStart={handleNodeDragStart}
        onNodeDrag={handleNodeDrag}
        onNodeDragEnd={handleNodeDragEnd}
        getNodeOutputs={getNodeOutputs}
        isOverNode={isOverNode}
        isPanning={isPanning}
        draggingNode={draggingNode}
        screenToWorldCoords={screenToWorldCoords}
        onCloseValidation={() => setValidationErrors([])}
        onCloseError={() => setGlobalError(null)}
        showReplaceMenu={modalState.showReplaceMenu}
        showOutputQuickAdd={modalState.showOutputQuickAdd}
        showModelSelect={modalState.showModelSelect}
        showVoiceSelect={modalState.showVoiceSelect}
        lightX2VVoiceList={voiceList.lightX2VVoiceList}
        cloneVoiceList={voiceList.cloneVoiceList}
        onUpdateNodeData={updateNodeData}
        onUpdateNodeName={nodeManagement.updateNodeName}
        onDeleteNode={deleteSelectedNode}
        onReplaceNode={replaceNode}
        onRunWorkflow={runWorkflow}
        onCancelNodeRun={cancelNodeRun}
        pendingRunNodeIds={pendingRunNodeIds}
        resolveLightX2VResultRef={resolveLightX2VResultRef}
        getNodeOutputUrl={workflow?.id ? getNodeOutputUrlForWorkflow : undefined}
        refreshWorkflowFromBackend={!isStandalone() ? refreshWorkflowFromBackend : undefined}
          onSetReplaceMenu={modalState.setShowReplaceMenu}
          onSetOutputQuickAdd={modalState.setShowOutputQuickAdd}
          onSetModelSelect={modalState.setShowModelSelect}
          onSetVoiceSelect={modalState.setShowVoiceSelect}
          onSetExpandedOutput={modalState.setExpandedOutput}
          onSetShowAudioEditor={modalState.setShowAudioEditor}
          onSetShowVideoEditor={modalState.setShowVideoEditor}
        onSetConnecting={setConnecting}
        onAddConnection={addConnection}
        getReplaceableTools={getReplaceableTools}
        getCompatibleToolsForOutput={getCompatibleToolsForOutput}
        quickAddInput={quickAddInput}
        quickAddOutput={quickAddOutput}
        onNodeHeightChange={(nodeId, height) => {
          nodeHeightsRef.current.set(nodeId, height);
          // Force re-render to update connections
          setTicker(prev => prev + 1);
        }}
        disconnectedInputs={disconnectedInputs}
        loadingVoiceList={voiceList.loadingVoiceList}
        voiceSearchQuery={voiceList.voiceSearchQuery}
        setVoiceSearchQuery={voiceList.setVoiceSearchQuery}
        showVoiceFilter={voiceList.showVoiceFilter}
        setShowVoiceFilter={voiceList.setShowVoiceFilter}
        voiceFilterGender={voiceList.voiceFilterGender}
        setVoiceFilterGender={voiceList.setVoiceFilterGender}
        filteredVoices={voiceList.filteredVoices}
        isFemaleVoice={voiceList.isFemaleVoice}
        loadingCloneVoiceList={voiceList.loadingCloneVoiceList}
        onGlobalInputChange={handleGlobalInputChange}
          onDescriptionChange={handleDescriptionChange}
          onTagsChange={handleTagsChange}
          onShowCloneVoiceModal={() => modalState.setShowCloneVoiceModal(true)}
        resultsCollapsed={modalState.resultsCollapsed}
        resultEntries={resultEntries}
        onToggleResultsCollapsed={() => modalState.setResultsCollapsed(!modalState.resultsCollapsed)}
        showIntermediateResults={showIntermediateResults}
        onToggleShowIntermediate={() => setShowIntermediateResults(v => !v)}
        onExpandOutput={(nodeId, fieldId, runId) => modalState.setExpandedOutput({ nodeId, fieldId, historyEntryId: runId === 'current' ? undefined : runId })}
        onPinOutputToCanvas={pinOutputToCanvas}
        isAIChatOpen={modalState.isAIChatOpen}
        isAIChatCollapsed={modalState.isAIChatCollapsed}
        onToggleAIChat={() => {
          // 如果面板已打开但被折叠，则展开它
          if (modalState.isAIChatOpen && modalState.isAIChatCollapsed) {
            modalState.setIsAIChatCollapsed(false);
          } else {
            // 否则切换打开/关闭状态
            const newState = !modalState.isAIChatOpen;
            modalState.setIsAIChatOpen(newState);
            // 如果打开，同时展开面板
            if (newState) {
              modalState.setIsAIChatCollapsed(false);
            }
          }
        }}
        onToggleAIChatCollapse={() => modalState.setIsAIChatCollapsed(!modalState.isAIChatCollapsed)}
        aiChatMode={aiChatMode}
        onAIChatModeChange={setAiChatMode}
        aiChatHistory={aiChatWorkflow.chatHistory}
        isAIProcessing={aiChatWorkflow.isProcessing}
        isAIExecutingOperations={aiChatWorkflow.isExecutingOperations}
        executingProgress={aiChatWorkflow.executingProgress}
        executingStepLabels={aiChatWorkflow.executingStepLabels}
        onAIStopGeneration={aiChatWorkflow.stopGeneration}
        onAISendMessage={aiChatWorkflow.handleUserInput}
        onAIClearHistory={aiChatWorkflow.clearChatHistory}
        chatContextNodes={aiChatWorkflow.chatContextNodes}
        onAddNodeToChatContext={(nodeId, name) => {
          aiChatWorkflow.addNodeToChatContext(nodeId, name);
          setSidebarDefaultTab('chat');
        }}
        onRemoveNodeFromChatContext={aiChatWorkflow.removeNodeFromChatContext}
        aiModel={aiChatWorkflow.aiModel}
        onAIModelChange={aiChatWorkflow.setAiModel}
        onAIUndo={(messageId) => {
          // 撤销这次AI对话的所有操作
          aiChatWorkflow.undoMessageOperations(messageId);
        }}
            onAIRetry={(messageId) => {
              // 重试：找到对应的用户消息并重新发送，并把上次失败的报错信息附加进去供 AI 参考
              const history = aiChatWorkflow.chatHistory;
              const assistantMessageIndex = history.findIndex(m => m.id === messageId);

              if (assistantMessageIndex >= 0) {
                const failedMsg = history[assistantMessageIndex];
                const errorParts: string[] = [];
                if (failedMsg.error) errorParts.push(failedMsg.error);
                if (failedMsg.operationResults?.length) {
                  failedMsg.operationResults
                    .filter((r: { success?: boolean; error?: string }) => !r.success && r.error)
                    .forEach((r: { error?: string }) => errorParts.push(r.error!));
                }
                const errorText = errorParts.length > 0 ? errorParts.join('; ') : '';

                let userMessage = null;
                if (assistantMessageIndex > 0) {
                  const prevMessage = history[assistantMessageIndex - 1];
                  if (prevMessage && prevMessage.role === 'user') userMessage = prevMessage;
                }
                if (!userMessage) {
                  for (let i = assistantMessageIndex - 1; i >= 0; i--) {
                    if (history[i].role === 'user') {
                      userMessage = history[i];
                      break;
                    }
                  }
                }

                if (userMessage) {
                  const contentToSend = errorText
                    ? `${userMessage.content}\n\n[上次执行失败，错误信息：${errorText}。请根据错误修正后重试。]`
                    : userMessage.content;
                  aiChatWorkflow.handleUserInput(contentToSend, {
                    image: userMessage.image,
                    useSearch: userMessage.useSearch
                  });
                }
              }
            }}
            nodeConfigPanelCollapsed={modalState.nodeConfigPanelCollapsed}
            onToggleNodeConfigPanel={() => modalState.setNodeConfigPanelCollapsed(!modalState.nodeConfigPanelCollapsed)}
            rightPanelSplitRatio={modalState.rightPanelSplitRatio}
            onRightPanelResize={(deltaY: number) => {
              const container = document.querySelector('.flex.flex-col.w-80.border-l');
              if (container) {
                const containerHeight = container.clientHeight;
                const currentRatio = modalState.rightPanelSplitRatio;
                const deltaRatio = deltaY / containerHeight;
                const newRatio = Math.max(0.2, Math.min(0.8, currentRatio + deltaRatio));
                modalState.setRightPanelSplitRatio(newRatio);
                localStorage.setItem('omniflow_right_panel_split_ratio', newRatio.toString());
              }
            }}
            aiChatPanelPosition={modalState.aiChatPanelPosition}
            aiChatPanelSize={modalState.aiChatPanelSize}
            onAiChatPanelPositionChange={(position) => {
              modalState.setAiChatPanelPosition(position);
              localStorage.setItem('omniflow_ai_chat_panel_position', JSON.stringify(position));
            }}
            onAiChatPanelSizeChange={(size) => {
              modalState.setAiChatPanelSize(size);
              localStorage.setItem('omniflow_ai_chat_panel_size', JSON.stringify(size));
            }}
          />

      <style>{`
        @keyframes marching-ants { from { stroke-dashoffset: 40; } to { stroke-dashoffset: 0; } }
        .animate-marching-ants { animation: marching-ants 1.5s linear infinite; }
        .animate-spin-slow { animation: spin 10s linear infinite; }
        .custom-scrollbar::-webkit-scrollbar { width: 5px; height: 5px; }
        .custom-range-thumb::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 12px;
          height: 12px;
          border-radius: 999px;
          background: #90dce1;
          border: 2px solid #0f172a;
          box-shadow: 0 0 10px rgba(144, 220, 225, 0.35);
          cursor: pointer;
          pointer-events: auto;
        }
        .custom-range-thumb { pointer-events: none; }
        .custom-range-thumb::-moz-range-thumb {
          width: 12px;
          height: 12px;
          border-radius: 999px;
          background: #90dce1;
          border: 2px solid #0f172a;
          box-shadow: 0 0 10px rgba(144, 220, 225, 0.35);
          cursor: pointer;
          pointer-events: auto;
        }
        .custom-range-thumb::-webkit-slider-runnable-track,
        .custom-range-thumb::-moz-range-track {
          background: transparent;
        }
        .canvas-grid { background-image: radial-gradient(rgba(51, 65, 85, 0.4) 1px, transparent 1px); }
        .connection-path:hover { stroke-opacity: 0.8; stroke-width: 5px; }
      `}</style>

        <CloneVoiceModal
          isOpen={modalState.showCloneVoiceModal}
          lightX2VConfig={getLightX2VConfig(workflow)}
                onClose={(newSpeakerId?: string) => {
            modalState.setShowCloneVoiceModal(false);
                  // Reload clone voice list after closing
                  const config = getLightX2VConfig(workflow);
                  if (config.url && config.token) {
            lightX2VGetCloneVoiceList(config.url, config.token)
              .then(voices => {
                voiceList.resetCloneVoiceList();
                      // Auto-select the newly created voice in the current node if applicable
                      if (newSpeakerId && selectedNodeId && workflow) {
                        const node = workflow.nodes.find(n => n.id === selectedNodeId);
                        if (node && node.tool_id === 'lightx2v-voice-clone') {
                          updateNodeData(selectedNodeId, 'speakerId', newSpeakerId);
                        }
                      }
              })
              .catch(err => console.error('[LightX2V] Failed to reload clone voice list:', err));
                  }
                }}
              />

      {modalState.showAudioEditor && workflow && (() => {
        const node = workflow.nodes.find(n => n.id === modalState.showAudioEditor);
        if (!node || node.tool_id !== 'audio-input') return null;
        const audioSrc = node.data.value ?? (node.output_value && getOutputValueByPort(node, 'out-audio'));
        if (!audioSrc) return null;
        const isRef = typeof audioSrc === 'object' && audioSrc !== null && (audioSrc as any).file_id;
        const displayUrl = audioEditorDisplayUrl ?? (typeof audioSrc === 'string' ? audioSrc : '');
        if (!displayUrl && isRef) return null;
        return (
          <AudioEditorModal
            nodeId={modalState.showAudioEditor}
            audioData={displayUrl}
            audioRange={node.data.audio_range}
            onRangeChange={(range) => {
              updateNodeData(modalState.showAudioEditor!, 'audio_range', range);
            }}
            onClose={() => modalState.setShowAudioEditor(null)}
            onSave={async (trimmedAudio) => {
              if (!workflow?.id) return;
              try {
                const saveRef = await saveInputFileViaOutputSave(workflow.id, node.id, 'out-audio', trimmedAudio);
                const ref = toFileRefForOutputValue(saveRef);
                if (ref) {
                  const nextOutput = setOutputValueByPort(node.output_value, node.tool_id, 'out-audio', ref);
                  updateNodeData(modalState.showAudioEditor!, 'output_value', nextOutput);
                  updateNodeData(modalState.showAudioEditor!, 'audio_range', node.data.audio_range || { start: 0, end: 100 });
                }
              } catch (e) {
                console.warn('[App] saveInputFileViaOutputSave failed:', e);
              }
              modalState.setShowAudioEditor(null);
            }}
            lang={lang}
          />
        );
      })()}

      {modalState.showVideoEditor && workflow && (() => {
        const node = workflow.nodes.find(n => n.id === modalState.showVideoEditor);
        if (!node || node.tool_id !== 'video-input' || !node.data.value) return null;
        const videoSource = node.data.videoOriginal || node.data.value;
        return (
          <VideoEditorModal
            nodeId={modalState.showVideoEditor}
            videoData={videoSource}
            trimStart={node.data.trimStart}
            trimEnd={node.data.trimEnd}
            onRangeChange={(start, end) => {
              updateNodeData(modalState.showVideoEditor!, 'trimStart', start);
              updateNodeData(modalState.showVideoEditor!, 'trimEnd', end);
            }}
            onClose={() => modalState.setShowVideoEditor(null)}
            onUpdate={(start, end, trimmedUrl) => {
              updateNodeData(modalState.showVideoEditor!, 'trimStart', start);
              updateNodeData(modalState.showVideoEditor!, 'trimEnd', end);
              if (!node.data.videoOriginal) {
                updateNodeData(modalState.showVideoEditor!, 'videoOriginal', videoSource);
              }
              if (trimmedUrl) {
                updateNodeData(modalState.showVideoEditor!, 'value', trimmedUrl);
              }
            }}
            lang={lang}
          />
        );
      })()}
    </div>
  );
};

export default App;
