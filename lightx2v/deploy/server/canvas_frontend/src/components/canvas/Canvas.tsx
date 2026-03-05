import React, { useRef, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Trash2 } from 'lucide-react';
import { WorkflowState, WorkflowNode, Connection, NodeStatus, DataType } from '../../../types';
import { ViewState } from '../../utils/canvas';
import { Connection as ConnectionComponent } from './Connection';
import { Node } from './Node';
import { TOOLS } from '../../../constants';
import { ToolDefinition, Port } from '../../../types';

interface CanvasProps {
  workflow: WorkflowState;
  view: ViewState;
  selectedNodeId: string | null;
  selectedConnectionId: string | null;
  connecting: {
    nodeId: string;
    portId: string;
    type: DataType;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null;
  mousePos: { x: number; y: number };
  clientMousePos?: { clientX: number; clientY: number };
  nodeHeights: Map<string, number>;
  sourceNodes: WorkflowNode[];
  sourceOutputs: Record<string, any>;
  isOverNode: boolean;
  isPanning: boolean;
  draggingNode: { id: string; offsetX: number; offsetY: number } | null;
  canvasRef: React.RefObject<HTMLDivElement>;
  screenToWorldCoords: (x: number, y: number) => { x: number; y: number };
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
  getNodeOutputs: (node: WorkflowNode) => any[];
  lang: 'en' | 'zh';
  // Node component props
  showReplaceMenu?: string | null;
  showOutputQuickAdd?: { nodeId: string; portId: string } | null;
  showModelSelect?: string | null;
  showVoiceSelect?: string | null;
  lightX2VVoiceList?: { voices?: any[]; emotions?: string[]; languages?: any[] } | null;
  cloneVoiceList?: any[];
  onUpdateNodeData?: (nodeId: string, key: string, value: any) => void;
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
    source_node_id: string;
    source_port_id: string;
    target_node_id: string;
    target_port_id: string;
  }) => void;
  getReplaceableTools?: (nodeId: string) => ToolDefinition[];
  getCompatibleToolsForOutput?: (outputType: DataType) => ToolDefinition[];
  quickAddInput?: (node: WorkflowNode, port: Port) => void;
  quickAddOutput?: (node: WorkflowNode, port: Port, toolId: string) => void;
  onNodeHeightChange?: (nodeId: string, height: number) => void;
  onAddNodeToChat?: (nodeId: string, name: string) => void;
  resolveLightX2VResultRef?: (ref: import('../../hooks/useWorkflowExecution').LightX2VResultRef) => Promise<string>;
  getNodeOutputUrl?: (nodeId: string, portId: string, fileId?: string, runId?: string) => Promise<string | null>;
  refreshWorkflowFromBackend?: (workflowId: string) => Promise<void>;
}

export const Canvas: React.FC<CanvasProps> = ({
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
  isOverNode,
  isPanning,
  draggingNode = null,
  canvasRef,
  screenToWorldCoords,
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
  lang,
  showReplaceMenu,
  showOutputQuickAdd,
  showModelSelect,
  showVoiceSelect,
  lightX2VVoiceList,
  cloneVoiceList = [],
  onUpdateNodeData = () => {},
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
  onAddNodeToChat,
  resolveLightX2VResultRef,
  getNodeOutputUrl,
  refreshWorkflowFromBackend,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  // 连接线终点：用逻辑像素 mousePos 转世界坐标（mousePos 在 useCanvas 中已按父级 scale 修正，兼容主应用 #app scale(0.8)）
  const connectingEndWorld = useMemo(() => {
    if (!connecting) return null;
    return {
      x: (mousePos.x - view.x) / view.zoom,
      y: (mousePos.y - view.y) / view.zoom
    };
  }, [connecting, mousePos.x, mousePos.y, view.x, view.y, view.zoom]);

  // 选中连线时用 Portal 在画布上渲染删除按钮（不依赖 SVG foreignObject，保证可点击）
  const selectedConnectionDeleteButton = useMemo(() => {
    if (!selectedConnectionId || !canvasRef.current) return null;
    const c = workflow.connections.find((conn) => conn.id === selectedConnectionId);
    if (!c) return null;
    const sNode = sourceNodes.find((n) => n.id === c.source_node_id);
    const tNode = sourceNodes.find((n) => n.id === c.target_node_id);
    if (!sNode || !tNode) return null;
    const sOutputs = getNodeOutputs(sNode);
    const tTool = TOOLS.find((t) => t.id === tNode.tool_id);
    const tInputs = tTool?.inputs || [];
    const outputPortIndex = sOutputs.findIndex((p) => p.id === c.source_port_id);
    const inputPortIndex = tInputs.findIndex((p) => p.id === c.target_port_id);
    const sourceTool = TOOLS.find((t) => t.id === sNode.tool_id);
    const isSourceInput = sourceTool?.category === 'Input';
    const sourceNodeWidth = isSourceInput ? 320 : 224;
    const sourceNodeHeight = nodeHeights.get(sNode.id) || Math.max(140, 48 + 16 + Math.max(0, (sOutputs.length - 1) * 30) + 30 + 16);
    const nodeBottomY = sNode.y + sourceNodeHeight;
    const x1 = sNode.x + sourceNodeWidth;
    const y1 = nodeBottomY - ((sOutputs.length - 1 - outputPortIndex) * 30) - 24;
    const x2 = tNode.x;
    const y2 = tNode.y + 71 + (inputPortIndex * 30);
    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    const left = view.x + midX * view.zoom - 15;
    const top = view.y + midY * view.zoom - 15;
    const btn = (
      <div
        className="connection-delete-portal"
        style={{
          position: 'absolute',
          left,
          top,
          width: 30,
          height: 30,
          zIndex: 20,
          pointerEvents: 'auto'
        }}
      >
        <button
          type="button"
          onMouseDown={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation();
            onDeleteConnection(selectedConnectionId);
          }}
          className="p-1.5 bg-red-500 text-white rounded-full shadow-lg hover:bg-red-600 transition-all active:scale-90 pointer-events-auto"
        >
          <Trash2 size={12} />
        </button>
      </div>
    );
    return createPortal(btn, canvasRef.current);
  }, [selectedConnectionId, workflow.connections, sourceNodes, nodeHeights, view, getNodeOutputs, onDeleteConnection]);

  return (
    <main
      ref={canvasRef}
      className="flex-1 h-full relative overflow-hidden canvas-grid bg-[#0a0f1e]"
      style={{
        cursor: isOverNode ? 'default' : isPanning ? 'grabbing' : 'grab',
        userSelect: connecting || draggingNode ? 'none' : undefined
      }}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseDown={onMouseDown}
      onMouseLeave={onMouseLeave}
      onWheel={onWheel}
    >
      {selectedConnectionDeleteButton}
      <div
        style={{
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
          transformOrigin: '0 0',
          width: '100%',
          height: '100%'
        }}
      >
        {/* SVG for connections - no pointer-events-none so connection hit areas and delete button receive clicks; nodes wrapper has pointer-events-none so empty area clicks reach here */}
        <svg ref={svgRef} className="absolute inset-0 w-full h-full z-0 overflow-visible">
          {workflow.connections.map((c) => {
            const sNode = sourceNodes.find((n) => n.id === c.source_node_id);
            const tNode = sourceNodes.find((n) => n.id === c.target_node_id);
            if (!sNode || !tNode) return null;

            const sOutputs = getNodeOutputs(sNode);
            const tTool = TOOLS.find((t) => t.id === tNode.tool_id);
            const tInputs = tTool?.inputs || [];

            // Calculate node height based on output count if not available from nodeHeights
            // Header (48px) + top padding (16px) + outputs area ((n-1)*30 + 30) + bottom padding (16px)
            const sourceNodeHeight = nodeHeights.get(sNode.id) || Math.max(140, 48 + 16 + Math.max(0, (sOutputs.length - 1) * 30) + 30 + 16);

            return (
              <ConnectionComponent
                key={c.id}
                connection={c}
                sourceNode={sNode}
                targetNode={tNode}
                sourceOutputs={sOutputs}
                targetInputs={tInputs}
                sourceNodeHeight={sourceNodeHeight}
                isSelected={selectedConnectionId === c.id}
                view={view}
                onSelect={() => {
                  onConnectionSelect(c.id);
                  onNodeSelect('');
                }}
                onDelete={() => {
                  onDeleteConnection(c.id);
                }}
                isConnecting={!!connecting}
              />
            );
          })}
          {connecting && (
            <path
              d={`M ${connecting.startX} ${connecting.startY} C ${
                connecting.startX + 100 / view.zoom
              } ${connecting.startY}, ${((mousePos.x - view.x) / view.zoom) - 100 / view.zoom} ${
                (mousePos.y - view.y) / view.zoom
              }, ${(mousePos.x - view.x) / view.zoom} ${(mousePos.y - view.y) / view.zoom}`}
              stroke="#4f46e5"
              strokeWidth={3 / view.zoom}
              strokeDasharray={`${6 / view.zoom},${6 / view.zoom}`}
              fill="none"
              className="animate-marching-ants"
            />
          )}
        </svg>

        {/* Nodes - wrapper z-[1] above SVG; pointer-events-none so clicks on empty area pass through to connection layer; each Node still receives clicks (pointer-events auto by default) */}
        <div className="absolute inset-0 z-[1] pointer-events-none">
        {sourceNodes.map((node) => (
          <Node
            key={node.id}
            node={node}
            workflow={workflow}
            isSelected={selectedNodeId === node.id}
            sourceOutputs={sourceOutputs}
            nodeHeight={nodeHeights.get(node.id) || 140}
            onSelect={onNodeSelect}
            onDragStart={onNodeDragStart}
            onDrag={onNodeDrag}
            onDragEnd={onNodeDragEnd}
            getNodeOutputs={getNodeOutputs}
            canvasRef={canvasRef}
            view={view}
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
            connecting={connecting}
            onNodeHeightChange={onNodeHeightChange}
            onAddNodeToChat={onAddNodeToChat}
            resolveLightX2VResultRef={resolveLightX2VResultRef}
            getNodeOutputUrl={getNodeOutputUrl}
            screenToWorldCoords={screenToWorldCoords}
            refreshWorkflowFromBackend={refreshWorkflowFromBackend}
          />
        ))}
        </div>
      </div>
    </main>
  );
};
