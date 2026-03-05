import React from 'react';
import { WorkflowState, WorkflowNode, Port, ToolDefinition, DataType } from '../../../types';
import { ViewState } from '../../utils/canvas';
import { EditorHeader } from './EditorHeader';
import { ToolPalette } from './ToolPalette';
import { Canvas } from '../canvas/Canvas';
import { NodeConfigPanel } from './NodeConfigPanel';
import { ResultsPanel } from './ResultsPanel';
import { AIChatPanel } from './AIChatPanel';
import { DraggableAIChatPanel } from './DraggableAIChatPanel';
import { ResizableDivider } from './ResizableDivider';
import { ErrorModal } from '../modals/ErrorModal';
import { ValidationModal } from '../modals/ValidationModal';
import { Language } from '../../i18n/useTranslation';
import { Bot, Sparkle, X } from 'lucide-react';

interface EditorProps {
  lang: Language;
  workflow: WorkflowState;
  view: ViewState;
  selectedNodeId: string | null;
  selectedConnectionId: string | null;
  connecting: {
    nodeId: string;
    portId: string;
    type: any;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null;
  mousePos: { x: number; y: number };
  activeOutputs: Record<string, any>;
  nodeHeights: Map<string, number>;
  sourceNodes: any[];
  sourceOutputs: Record<string, any>;
  isPaused: boolean;
  isRunning: boolean;
  sidebarCollapsed: boolean;
  validationErrors: { message: string; type: 'ENV' | 'INPUT' }[];
  globalError: { message: string; details?: string } | null;
  canvasRef: React.RefObject<HTMLDivElement>;
  onBack: () => void;
  onWorkflowNameChange: (name: string) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetView: () => void;
  onToggleLang: () => void;
  onClearSnapshot: () => void;
  onSave: () => void;
  onPause: () => void;
  onRun: () => void;
  onToggleSidebar: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onVisibilityChange: (isPublic: boolean) => void;
  onAddNode: (tool: any) => void;
  sidebarDefaultTab?: 'tools' | 'chat';
  onSidebarTabChange?: (tab: 'tools' | 'chat') => void;
  focusAIChatInput?: boolean;
  onAIChatInputFocused?: () => void;
  onMouseMove: (e: React.MouseEvent) => void;
  onMouseDown: (e: React.MouseEvent) => void;
  onMouseUp: () => void;
  onMouseLeave: () => void;
  onWheel: (e: React.WheelEvent) => void;
  onNodeSelect: (nodeId: string) => void;
  onConnectionSelect: (connectionId: string) => void;
  onDeleteConnection: (connectionId: string) => void;
  onNodeDragStart: (nodeId: string, offsetX: number, offsetY: number) => void;
  onNodeDrag: (nodeId: string, x: number, y: number) => void;
  onNodeDragEnd: () => void;
  getNodeOutputs: (node: any) => any[];
  isOverNode: boolean;
  isPanning: boolean;
  onCloseValidation: () => void;
  onCloseError: () => void;
  // Node component props
  showReplaceMenu?: string | null;
  showOutputQuickAdd?: { nodeId: string; portId: string } | null;
  showModelSelect?: string | null;
  showVoiceSelect?: string | null;
  lightX2VVoiceList?: { voices?: any[]; emotions?: string[]; languages?: any[] } | null;
  cloneVoiceList?: any[];
  onUpdateNodeData?: (nodeId: string, key: string, value: any) => void;
  onUpdateNodeName?: (nodeId: string, name: string) => void;
  onDeleteNode?: (nodeId: string) => void;
  onReplaceNode?: (nodeId: string, newToolId: string) => void;
  onRunWorkflow?: (nodeId?: string, runThisOnly?: boolean) => void;
  onCancelNodeRun?: (nodeId: string) => void;
  pendingRunNodeIds?: string[];
  onSetReplaceMenu?: (nodeId: string | null) => void;
  onSetOutputQuickAdd?: (value: { nodeId: string; portId: string } | null) => void;
  onSetModelSelect?: (nodeId: string | null) => void;
  onSetVoiceSelect?: (nodeId: string | null) => void;
  onSetExpandedOutput?: (value: { nodeId: string; fieldId?: string } | null) => void;
  onSetShowAudioEditor?: (nodeId: string | null) => void;
  onSetShowVideoEditor?: (nodeId: string | null) => void;
  onSetConnecting?: (value: {
    nodeId: string;
    portId: string;
    type: DataType;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null) => void;
  onAddConnection?: (connection: {
    id: string;
    sourceNodeId: string;
    sourcePortId: string;
    targetNodeId: string;
    targetPortId: string;
  }) => void;
  getReplaceableTools?: (nodeId: string) => ToolDefinition[];
  getCompatibleToolsForOutput?: (outputType: DataType) => ToolDefinition[];
  quickAddInput?: (node: WorkflowNode, port: Port) => void;
  quickAddOutput?: (node: WorkflowNode, port: Port, toolId: string) => void;
  // NodeConfigPanel props
  disconnectedInputs?: Array<{
    nodeId: string;
    port: Port;
    toolName: string;
    isSourceNode?: boolean;
    dataType: DataType;
  }>;
  loadingVoiceList?: boolean;
  voiceSearchQuery?: string;
  setVoiceSearchQuery?: (query: string) => void;
  showVoiceFilter?: boolean;
  setShowVoiceFilter?: (show: boolean) => void;
  voiceFilterGender?: string;
  setVoiceFilterGender?: (gender: string) => void;
  filteredVoices?: any[];
  isFemaleVoice?: (voiceType: string) => boolean;
  loadingCloneVoiceList?: boolean;
  onGlobalInputChange?: (nodeId: string, portId: string, value: any) => void;
  onDescriptionChange?: (description: string) => void;
  onTagsChange?: (tags: string[]) => void;
  onShowCloneVoiceModal?: () => void;
  // ResultsPanel props
  resultsCollapsed?: boolean;
  showIntermediateResults?: boolean;
  resultEntries?: import('../../hooks/useResultManagement').ResultEntry[];
  onToggleResultsCollapsed?: () => void;
  onToggleShowIntermediate?: () => void;
  onExpandOutput?: (nodeId: string, fieldId?: string, runId?: string) => void;
  onPinOutputToCanvas?: (content: any, type: DataType) => void;
  resolveLightX2VResultRef?: (ref: import('../../hooks/useWorkflowExecution').LightX2VResultRef) => Promise<string>;
  getNodeOutputUrl?: (nodeId: string, portId: string, fileId?: string, runId?: string) => Promise<string | null>;
  refreshWorkflowFromBackend?: (workflowId: string) => Promise<void>;
  onNodeHeightChange?: (nodeId: string, height: number) => void;
  // AI Chat props
  isAIChatOpen?: boolean;
  isAIChatCollapsed?: boolean;
  onToggleAIChat?: () => void;
  onToggleAIChatCollapse?: () => void;
  aiChatHistory?: any[];
  isAIProcessing?: boolean;
  isAIExecutingOperations?: boolean;
  executingProgress?: { current: number; total: number };
  executingStepLabels?: string[];
  onAIStopGeneration?: () => void;
  onAISendMessage?: (message: string, options?: { image?: { data: string; mimeType: string }; useSearch?: boolean }) => void;
  onAIClearHistory?: () => void;
  chatContextNodes?: { nodeId: string; name: string }[];
  onAddNodeToChatContext?: (nodeId: string, name: string) => void;
  onRemoveNodeFromChatContext?: (nodeId: string) => void;
  onAIUndo?: (messageId: string) => void;
  onAIRetry?: (messageId: string) => void;
  aiModel?: string;
  onAIModelChange?: (model: string) => void;
  // NodeConfigPanel collapse
  nodeConfigPanelCollapsed?: boolean;
  onToggleNodeConfigPanel?: () => void;
  // Right panel resize
  rightPanelSplitRatio?: number;
  onRightPanelResize?: (deltaY: number) => void;
  // AI Chat Panel position and size (for floating window)
  aiChatPanelPosition?: { x: number; y: number };
  aiChatPanelSize?: { width: number; height: number };
  onAiChatPanelPositionChange?: (position: { x: number; y: number }) => void;
  onAiChatPanelSizeChange?: (size: { width: number; height: number }) => void;
  aiChatMode?: 'edit' | 'ideation';
  onAIChatModeChange?: (mode: 'edit' | 'ideation') => void;
}

export const Editor: React.FC<EditorProps> = ({
  lang,
  workflow,
  view,
  selectedNodeId,
  selectedConnectionId,
  connecting,
  mousePos,
  clientMousePos,
  nodeHeights,
  sourceNodes,
  sourceOutputs,
  isPaused,
  isRunning,
  sidebarCollapsed,
  validationErrors,
  globalError,
  canvasRef,
  onBack,
  onWorkflowNameChange,
  onZoomIn,
  onZoomOut,
  onResetView,
  onToggleLang,
  onClearSnapshot,
  onSave,
  onPause,
  onRun,
  onStop,
  onToggleSidebar,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onVisibilityChange,
  onAddNode,
  sidebarDefaultTab,
  onSidebarTabChange,
  focusAIChatInput,
  onAIChatInputFocused,
  onMouseMove,
  onMouseDown,
  onMouseUp,
  onMouseLeave,
  onWheel,
  onNodeSelect,
  onConnectionSelect,
  onDeleteConnection,
  onNodeDragStart,
  onNodeDrag,
  onNodeDragEnd,
  getNodeOutputs,
  isOverNode,
  isPanning,
  draggingNode = null,
  screenToWorldCoords,
  onCloseValidation,
  onCloseError,
  // Node component props
  showReplaceMenu,
  showOutputQuickAdd,
  showModelSelect,
  showVoiceSelect,
  lightX2VVoiceList,
  cloneVoiceList = [],
  onUpdateNodeData = () => {},
  onUpdateNodeName = () => {},
  onDeleteNode = () => {},
  onReplaceNode = () => {},
  onRunWorkflow = () => {},
  onCancelNodeRun = () => {},
  pendingRunNodeIds = [],
  onSetReplaceMenu = () => {},
  onSetOutputQuickAdd = () => {},
  onSetModelSelect = () => {},
  onSetVoiceSelect = () => {},
  onSetExpandedOutput = () => {},
  onSetShowAudioEditor = () => {},
  onSetShowVideoEditor = () => {},
  onSetConnecting = () => {},
  onAddConnection = () => {},
  getReplaceableTools = () => [],
  getCompatibleToolsForOutput = () => [],
  quickAddInput = () => {},
  quickAddOutput = () => {},
  onNodeHeightChange,
  // NodeConfigPanel props
  disconnectedInputs = [],
  loadingVoiceList = false,
  voiceSearchQuery = '',
  setVoiceSearchQuery = () => {},
  showVoiceFilter = false,
  setShowVoiceFilter = () => {},
  voiceFilterGender = 'all',
  setVoiceFilterGender = () => {},
  filteredVoices = [],
  isFemaleVoice = () => false,
  loadingCloneVoiceList = false,
  onGlobalInputChange = () => {},
  onDescriptionChange = () => {},
  onTagsChange = () => {},
  onShowCloneVoiceModal = () => {},
  // ResultsPanel props
  resultsCollapsed = true,
  showIntermediateResults = true,
  resultEntries = [],
  onToggleResultsCollapsed = () => {},
  onToggleShowIntermediate = () => {},
  onExpandOutput = () => {},
  onPinOutputToCanvas = () => {},
  resolveLightX2VResultRef,
  getNodeOutputUrl,
  refreshWorkflowFromBackend,
  // AI Chat props
  isAIChatOpen = false,
  isAIChatCollapsed = false,
  onToggleAIChat = () => {},
  onToggleAIChatCollapse = () => {},
  aiChatHistory = [],
  isAIProcessing = false,
  isAIExecutingOperations = false,
  executingProgress = { current: 0, total: 0 },
  executingStepLabels = [],
  onAIStopGeneration = () => {},
  onAISendMessage = () => {},
  onAIClearHistory = () => {},
  chatContextNodes = [],
  onAddNodeToChatContext = () => {},
  onRemoveNodeFromChatContext = () => {},
  onAIUndo = () => {},
  onAIRetry = () => {},
  aiModel = 'deepseek-v3-2-251201',
  onAIModelChange = () => {},
  // NodeConfigPanel collapse
  nodeConfigPanelCollapsed = false,
  onToggleNodeConfigPanel = () => {},
  // Right panel split ratio
  rightPanelSplitRatio = 0.5,
  onRightPanelResize = () => {},
  // AI Chat Panel position and size
  aiChatPanelPosition,
  aiChatPanelSize,
  onAiChatPanelPositionChange = () => {},
  onAiChatPanelSizeChange = () => {},
  aiChatMode = 'edit',
  onAIChatModeChange = () => {}
}) => {
  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-200 selection:bg-[#90dce1]/30 font-sans overflow-hidden">
      <ErrorModal error={globalError} lang={lang} onClose={onCloseError} />
      <ValidationModal errors={validationErrors} lang={lang} onClose={onCloseValidation} />

      <EditorHeader
        lang={lang}
        workflow={workflow}
        view={view}
        isPaused={isPaused}
        isRunning={isRunning}
        canvasRef={canvasRef}
        onBack={onBack}
        onWorkflowNameChange={onWorkflowNameChange}
        onZoomIn={onZoomIn}
        onZoomOut={onZoomOut}
        onResetView={onResetView}
        onToggleLang={onToggleLang}
        onClearSnapshot={onClearSnapshot}
        onSave={onSave}
        onPause={onPause}
        onRun={onRun}
        onStop={onStop}
        canUndo={canUndo}
        canRedo={canRedo}
        onUndo={onUndo}
        onRedo={onRedo}
        onVisibilityChange={onVisibilityChange}
      />

      <div className="flex-1 flex overflow-hidden relative">
        <ToolPalette
          lang={lang}
          collapsed={sidebarCollapsed}
          onToggleCollapse={onToggleSidebar}
          onAddNode={onAddNode}
          defaultTab={sidebarDefaultTab}
          onTabChange={onSidebarTabChange}
          focusChatInput={focusAIChatInput}
          onChatInputFocused={onAIChatInputFocused}
          chatHistory={aiChatHistory}
          isProcessing={isAIProcessing}
          isExecutingOperations={isAIExecutingOperations}
          executingProgress={executingProgress}
          executingStepLabels={executingStepLabels}
          onStopGeneration={onAIStopGeneration}
          onSendMessage={onAISendMessage}
          onClearHistory={onAIClearHistory}
          chatContextNodes={chatContextNodes}
          onRemoveNodeFromChatContext={onRemoveNodeFromChatContext}
          onUndo={onAIUndo}
          onRetry={onAIRetry}
          aiModel={aiModel}
          onModelChange={onAIModelChange}
          chatMode={aiChatMode}
          onChatModeChange={onAIChatModeChange}
        />

        <Canvas
          workflow={workflow}
          view={view}
          selectedNodeId={selectedNodeId}
          selectedConnectionId={selectedConnectionId}
          connecting={connecting}
          mousePos={mousePos}
          clientMousePos={clientMousePos}
          nodeHeights={nodeHeights}
          sourceNodes={sourceNodes}
          sourceOutputs={sourceOutputs}
          isOverNode={isOverNode}
          isPanning={isPanning}
          canvasRef={canvasRef}
          onMouseMove={onMouseMove}
          onMouseDown={onMouseDown}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseLeave}
          onWheel={onWheel}
          onNodeSelect={onNodeSelect}
          onConnectionSelect={onConnectionSelect}
          onDeleteConnection={onDeleteConnection}
          onNodeDragStart={onNodeDragStart}
          onNodeDrag={onNodeDrag}
          onNodeDragEnd={onNodeDragEnd}
          getNodeOutputs={getNodeOutputs}
          draggingNode={draggingNode}
          screenToWorldCoords={screenToWorldCoords}
          lang={lang}
          showReplaceMenu={showReplaceMenu}
          showOutputQuickAdd={showOutputQuickAdd}
          showModelSelect={showModelSelect}
          showVoiceSelect={showVoiceSelect}
          lightX2VVoiceList={lightX2VVoiceList}
          cloneVoiceList={cloneVoiceList}
          onUpdateNodeData={onUpdateNodeData}
          onDeleteNode={onDeleteNode}
          onReplaceNode={onReplaceNode}
          onRunWorkflow={onRunWorkflow}
          onCancelNodeRun={onCancelNodeRun}
          pendingRunNodeIds={pendingRunNodeIds}
          onSetReplaceMenu={onSetReplaceMenu}
          onSetOutputQuickAdd={onSetOutputQuickAdd}
          onSetModelSelect={onSetModelSelect}
          onSetVoiceSelect={onSetVoiceSelect}
          onSetExpandedOutput={onSetExpandedOutput}
          onSetShowAudioEditor={onSetShowAudioEditor}
          onSetShowVideoEditor={onSetShowVideoEditor}
          onSetConnecting={onSetConnecting}
          onAddConnection={onAddConnection}
          getReplaceableTools={getReplaceableTools}
          getCompatibleToolsForOutput={getCompatibleToolsForOutput}
          quickAddInput={quickAddInput}
          quickAddOutput={quickAddOutput}
          onNodeHeightChange={onNodeHeightChange}
          onAddNodeToChat={onAddNodeToChatContext}
          resolveLightX2VResultRef={resolveLightX2VResultRef}
          getNodeOutputUrl={getNodeOutputUrl}
          refreshWorkflowFromBackend={refreshWorkflowFromBackend}
        />

        {/* 右侧面板容器：NodeConfigPanel */}
        <div className="flex flex-col min-h-0 w-80 border-l border-slate-800/60 bg-slate-900/40 backdrop-blur-xl z-30 relative overflow-hidden">
          <NodeConfigPanel
            lang={lang}
            workflow={workflow}
            selectedNodeId={selectedNodeId}
            sourceOutputs={sourceOutputs}
            disconnectedInputs={disconnectedInputs}
            lightX2VVoiceList={lightX2VVoiceList}
            loadingVoiceList={loadingVoiceList}
            voiceSearchQuery={voiceSearchQuery}
            setVoiceSearchQuery={setVoiceSearchQuery}
            showVoiceFilter={showVoiceFilter}
            setShowVoiceFilter={setShowVoiceFilter}
            voiceFilterGender={voiceFilterGender}
            setVoiceFilterGender={setVoiceFilterGender}
            filteredVoices={filteredVoices}
            isFemaleVoice={isFemaleVoice}
            cloneVoiceList={cloneVoiceList}
            loadingCloneVoiceList={loadingCloneVoiceList}
            onUpdateNodeData={onUpdateNodeData}
            onUpdateNodeName={onUpdateNodeName}
            onDeleteNode={onDeleteNode}
            onGlobalInputChange={onGlobalInputChange}
            onDescriptionChange={onDescriptionChange}
            onTagsChange={onTagsChange}
            onShowCloneVoiceModal={onShowCloneVoiceModal}
            collapsed={nodeConfigPanelCollapsed}
            resolveLightX2VResultRef={resolveLightX2VResultRef}
            getNodeOutputUrl={getNodeOutputUrl}
          />
        </div>

        {/* 可拖拽的 AI Chat Panel 浮动窗口 */}
        {isAIChatOpen && !isAIChatCollapsed && (
          <DraggableAIChatPanel
            isOpen={isAIChatOpen}
            onClose={onToggleAIChat}
            chatHistory={aiChatHistory}
            isProcessing={isAIProcessing}
            isExecutingOperations={isAIExecutingOperations}
            executingProgress={executingProgress}
            executingStepLabels={executingStepLabels}
            onStopGeneration={onAIStopGeneration}
            onSendMessage={onAISendMessage}
            onClearHistory={onAIClearHistory}
            chatContextNodes={chatContextNodes}
            onRemoveNodeFromChatContext={onRemoveNodeFromChatContext}
            onUndo={onAIUndo}
            onRetry={onAIRetry}
            lang={lang}
            aiModel={aiModel}
            onModelChange={onAIModelChange}
            position={aiChatPanelPosition}
            size={aiChatPanelSize}
            onPositionChange={onAiChatPanelPositionChange}
            onSizeChange={onAiChatPanelSizeChange}
            chatMode={aiChatMode}
            onChatModeChange={onAIChatModeChange}
          />
        )}

        {/* 右下角 AI Bot 浮动按钮 */}
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-3">
          {/* 文字气泡提示 - 仅在未展开时显示 */}
          {(!isAIChatOpen || isAIChatCollapsed) && (
            <div className="relative bg-slate-900/90 backdrop-blur-xl border border-slate-800 px-4 py-2.5 rounded-2xl shadow-2xl animate-in slide-in-from-right-4 fade-in duration-500 pointer-events-none whitespace-nowrap hidden lg:block">
              <div className="flex items-center gap-2.5">
                <div className="p-1 rounded-lg bg-[#90dce1]/10">
                  <Sparkle size={14} className="text-[#90dce1] animate-pulse" />
                </div>
                <span className="text-[11px] font-bold text-slate-200 tracking-wide">
                  {lang === 'zh' ? '需要协助构建工作流吗？' : 'Need help with workflows?'}
                </span>
              </div>
              <div className="absolute right-[-6px] top-1/2 -translate-y-1/2 w-3 h-3 bg-slate-900/90 border-r border-t border-slate-800 rotate-45" />
            </div>
          )}
          <button
            onClick={onToggleAIChat}
            className={`w-16 h-16 rounded-full shadow-2xl transition-all flex items-center justify-center ${
              isAIChatOpen && !isAIChatCollapsed
                ? 'bg-slate-800 rotate-90 text-slate-300'
                : 'bg-gradient-to-br from-[#90dce1] to-[#68b1b5] text-slate-950 hover:shadow-[#90dce1]/30'
            } hover:scale-110 active:scale-95`}
            title={lang === 'zh' ? 'AI助手' : 'AI Assistant'}
          >
            {isAIChatOpen && !isAIChatCollapsed ? <X size={24} /> : <Bot size={28} />}
          </button>
        </div>
      </div>

      <ResultsPanel
        lang={lang}
        workflow={workflow}
        showIntermediateResults={showIntermediateResults}
        resultsCollapsed={resultsCollapsed}
        onToggleCollapsed={onToggleResultsCollapsed}
        resultEntries={resultEntries}
        onToggleShowIntermediate={onToggleShowIntermediate}
        onExpandOutput={onExpandOutput}
        onPinOutputToCanvas={onPinOutputToCanvas}
        resolveLightX2VResultRef={resolveLightX2VResultRef}
        getNodeOutputUrl={getNodeOutputUrl}
      />
    </div>
  );
};
