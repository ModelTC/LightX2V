import { useState, useCallback, useRef, useEffect } from 'react';
import { WorkflowState } from '../../types';

interface UseUndoRedoProps {
  workflow: WorkflowState | null;
  setWorkflow: (workflow: WorkflowState | null) => void;
  maxHistorySize?: number;
}

export const useUndoRedo = ({
  workflow,
  setWorkflow,
  maxHistorySize = 50
}: UseUndoRedoProps) => {
  const [history, setHistory] = useState<WorkflowState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const isUndoRedoRef = useRef(false);
  const lastWorkflowRef = useRef<WorkflowState | null>(null);
  const historyIndexRef = useRef(-1);

  // Sync ref with state
  useEffect(() => {
    historyIndexRef.current = historyIndex;
  }, [historyIndex]);

  // Initialize history when workflow changes (but not during undo/redo)
  useEffect(() => {
    if (isUndoRedoRef.current) {
      isUndoRedoRef.current = false;
      return;
    }

    if (!workflow) {
      setHistory([]);
      setHistoryIndex(-1);
      historyIndexRef.current = -1;
      lastWorkflowRef.current = null;
      return;
    }

    // Only track structural changes (not runtime state)
    // Create a normalized version of workflow that excludes runtime state
    const normalizeWorkflow = (w: WorkflowState) => {
      return {
        id: w.id,
        name: w.name,
        nodes: w.nodes.map(n => ({
          id: n.id,
          tool_id: n.tool_id,
          x: n.x,
          y: n.y,
          data: n.data
          // Exclude: status, error, executionTime, startTime, outputValue
        })),
        connections: w.connections,
        globalInputs: w.globalInputs
        // Exclude: isDirty, isRunning, showIntermediateResults (localStorage), updatedAt
      };
    };

    const normalizedWorkflow = normalizeWorkflow(workflow);
    const workflowStr = JSON.stringify(normalizedWorkflow);
    const lastWorkflowStr = lastWorkflowRef.current ? JSON.stringify(normalizeWorkflow(lastWorkflowRef.current)) : null;

    // Check if this is a new workflow (ID changed or history is empty)
    setHistory(prev => {
      const isNewWorkflow = prev.length === 0 || (prev.length > 0 && prev[0].id !== workflow.id);

      // If new workflow or workflow structure changed, update history
      if (isNewWorkflow || workflowStr !== lastWorkflowStr) {
        const newState = JSON.parse(JSON.stringify(workflow)) as WorkflowState;

        if (isNewWorkflow) {
          // Initialize history with current state as the first entry
          setHistoryIndex(0);
          historyIndexRef.current = 0;
          lastWorkflowRef.current = workflow;
          return [newState];
        } else {
          // Workflow structure changed, add new state to history
          const currentIndex = historyIndexRef.current;
          const newHistory = prev.slice(0, currentIndex + 1);
          const updatedHistory = [...newHistory, newState];

          // Limit history size
          if (updatedHistory.length > maxHistorySize) {
            const trimmed = updatedHistory.slice(-maxHistorySize);
            setHistoryIndex(maxHistorySize - 1);
            historyIndexRef.current = maxHistorySize - 1;
            lastWorkflowRef.current = workflow;
            return trimmed;
          }

          const newIndex = newHistory.length;
          setHistoryIndex(newIndex);
          historyIndexRef.current = newIndex;
          lastWorkflowRef.current = workflow;
          return updatedHistory;
        }
      }

      // No change, return previous history
      return prev;
    });
  }, [workflow, maxHistorySize]);

  const canUndo = historyIndex > 0;
  const canRedo = historyIndex < history.length - 1;

  const undo = useCallback(() => {
    if (!canUndo || history.length === 0) return;

    isUndoRedoRef.current = true;
    const newIndex = historyIndex - 1;
    setHistoryIndex(newIndex);
    historyIndexRef.current = newIndex;
    setWorkflow(history[newIndex]);
  }, [canUndo, historyIndex, history, setWorkflow]);

  const redo = useCallback(() => {
    if (!canRedo || history.length === 0) return;

    isUndoRedoRef.current = true;
    const newIndex = historyIndex + 1;
    setHistoryIndex(newIndex);
    historyIndexRef.current = newIndex;
    setWorkflow(history[newIndex]);
  }, [canRedo, historyIndex, history, setWorkflow]);

  // 获取当前历史索引
  const getCurrentHistoryIndex = useCallback(() => {
    return historyIndexRef.current;
  }, []);

  // 回退到指定的历史索引
  const undoToIndex = useCallback((targetIndex: number) => {
    if (targetIndex < 0 || targetIndex >= history.length) return;
    if (targetIndex === historyIndexRef.current) return;

    isUndoRedoRef.current = true;
    setHistoryIndex(targetIndex);
    historyIndexRef.current = targetIndex;
    setWorkflow(history[targetIndex]);
  }, [history, setWorkflow]);

  // Reset history when workflow is replaced (e.g., opening a different workflow)
  const resetHistory = useCallback(() => {
    if (workflow) {
      setHistory([workflow]);
      setHistoryIndex(0);
      historyIndexRef.current = 0;
      lastWorkflowRef.current = workflow;
    } else {
      setHistory([]);
      setHistoryIndex(-1);
      historyIndexRef.current = -1;
      lastWorkflowRef.current = null;
    }
    isUndoRedoRef.current = false;
  }, [workflow]);

  return {
    canUndo,
    canRedo,
    undo,
    redo,
    resetHistory,
    getCurrentHistoryIndex,
    undoToIndex
  };
};
