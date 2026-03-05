import { useCallback } from 'react';
import { WorkflowState, Connection } from '../../types';

interface UseConnectionManagementProps {
  workflow: WorkflowState | null;
  setWorkflow: React.Dispatch<React.SetStateAction<WorkflowState | null>>;
  selectedConnectionId: string | null;
  setSelectedConnectionId: (id: string | null) => void;
  setConnecting: (connecting: { nodeId: string; portId: string; isOutput: boolean } | null) => void;
}

export const useConnectionManagement = ({
  workflow,
  setWorkflow,
  selectedConnectionId,
  setSelectedConnectionId,
  setConnecting
}: UseConnectionManagementProps) => {
  const addConnection = useCallback((connection: {
    id: string;
    source_node_id: string;
    source_port_id: string;
    target_node_id: string;
    target_port_id: string;
  }) => {
    setWorkflow(prev => prev ? ({ ...prev, connections: [...prev.connections, connection], isDirty: true }) : null);
    setConnecting(null);
  }, [setWorkflow, setConnecting]);

  const deleteConnection = useCallback((connectionId: string) => {
    if (!connectionId) return;
    setWorkflow(prev => prev ? ({ ...prev, connections: prev.connections.filter(c => c.id !== connectionId), isDirty: true }) : null);
    if (selectedConnectionId === connectionId) {
      setSelectedConnectionId(null);
    }
  }, [selectedConnectionId, setWorkflow, setSelectedConnectionId]);

  return {
    addConnection,
    deleteConnection
  };
};
