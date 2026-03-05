import React from 'react';
import { Connection as ConnectionType, WorkflowNode, NodeStatus } from '../../../types';
import { ToolDefinition } from '../../../types';
import { TOOLS } from '../../../constants';

interface ConnectionProps {
  connection: ConnectionType;
  sourceNode: WorkflowNode;
  targetNode: WorkflowNode;
  sourceOutputs: any[];
  targetInputs: any[];
  sourceNodeHeight: number;
  isSelected: boolean;
  view: { zoom: number };
  onSelect: () => void;
  onDelete: () => void;
  /** 正在拖拽连线时为 true，此时不响应点击选中，避免误选 */
  isConnecting?: boolean;
}

export const Connection: React.FC<ConnectionProps> = ({
  connection,
  sourceNode,
  targetNode,
  sourceOutputs,
  targetInputs,
  sourceNodeHeight,
  isSelected,
  view,
  onSelect,
  onDelete,
  isConnecting = false,
  getConnectionClickWorldCoords
}) => {
  // Calculate port positions
  const outputPortIndex = sourceOutputs.findIndex((p) => p.id === connection.source_port_id);
  const inputPortIndex = targetInputs.findIndex((p) => p.id === connection.target_port_id);

  const sourceTool = TOOLS.find((t) => t.id === sourceNode.tool_id);
  const isSourceInput = sourceTool?.category === 'Input';
  const sourceNodeWidth = isSourceInput ? 320 : 224;
  const portOffset = 18;
  const x1 = sourceNode.x + sourceNodeWidth; // Output port center X
  const nodeBottomY = sourceNode.y + sourceNodeHeight;
  const y1 = nodeBottomY - ((sourceOutputs.length - 1 - outputPortIndex) * 30) - 24; // Output port center Y
  const x2 = targetNode.x; // Input port center X
  const y2 = targetNode.y + 71 + (inputPortIndex * 30); // Input port center Y

  const path = `M ${x1} ${y1} C ${x1 + 100} ${y1}, ${x2 - 100} ${y2}, ${x2} ${y2}`;
  const isTargetRunning = targetNode.status === NodeStatus.RUNNING || targetNode.status === NodeStatus.PENDING;

  return (
    <g style={{ pointerEvents: isConnecting ? 'none' : 'auto' }}>
      <path
        d={path}
        stroke="transparent"
        strokeWidth="15"
        fill="none"
        className={isConnecting ? 'pointer-events-none' : 'pointer-events-auto cursor-pointer'}
        onClick={(e) => {
          if (isConnecting) return;
          e.stopPropagation();
          onSelect();
        }}
      />
      <path
        d={path}
        stroke={isSelected ? '#90dce1' : '#5fb3b9'}
        strokeWidth={isSelected ? 4 : 3}
        fill="none"
        className="connection-path transition-all"
      />
      {isTargetRunning && (
        <circle r="4" fill="#90dce1" className="shadow-lg shadow-[#90dce1]/50">
          <animateMotion path={path} dur="1.5s" repeatCount="indefinite" />
        </circle>
      )}
    </g>
  );
};
