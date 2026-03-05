import { useEffect, useRef, useCallback } from 'react';
import { WorkflowState } from '../../types';
import { apiRequest } from '../utils/apiClient';
import { workflowSaveQueue } from '../utils/workflowSaveQueue';
import { isStandalone } from '../config/runtimeMode';

const AUTO_SAVE_INTERVAL = 30000; // 30 seconds
const MAX_NODE_HISTORY = 20;

function trimNodeOutputHistory(hist: Record<string, unknown[]> | undefined): Record<string, unknown[]> {
  if (!hist || typeof hist !== 'object') return {};
  const out: Record<string, unknown[]> = {};
  for (const [nodeId, entries] of Object.entries(hist)) {
    if (Array.isArray(entries)) {
      out[nodeId] = entries.slice(0, MAX_NODE_HISTORY);
    }
  }
  return out;
}

interface UseWorkflowAutoSaveProps {
  workflow: WorkflowState | null;
  onSave?: (workflow: WorkflowState) => Promise<void>;
  /** 在请求后端前确保工作流属于当前用户；若不拥有则先创建新 UUID 并更新状态，返回实际应使用的 id */
  ensureWorkflowOwned?: (workflow: WorkflowState) => Promise<string>;
}

export interface UseWorkflowAutoSaveReturn {
  resetAutoSaveTimer: () => void; // 重置自动保存计时器（手动保存后调用）
}

/**
 * Hook for auto-saving workflows
 * Automatically saves workflow structure (nodes, connections, global_inputs) every 30 seconds
 * if the workflow is dirty (has unsaved changes)
 * Now uses save queue to avoid conflicts with manual saves
 */
export const useWorkflowAutoSave = ({
  workflow,
  onSave,
  ensureWorkflowOwned
}: UseWorkflowAutoSaveProps): UseWorkflowAutoSaveReturn => {
  const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastSavedRef = useRef<number>(0);

  // 重置自动保存计时器（手动保存后调用）
  const resetAutoSaveTimer = useCallback(() => {
    // 清除当前计时器
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }
    // 更新最后保存时间，防止立即触发自动保存
    lastSavedRef.current = Date.now();
    console.log('[Workflow AutoSave] Timer reset after manual save');
  }, []);

  useEffect(() => {
    if (!workflow || !workflow.isDirty || !workflow.id) {
      // Clear timer if workflow is not dirty or doesn't have an ID
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
      return;
    }

    // Clear previous timer
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
    }

    // Set up auto-save timer
    autoSaveTimerRef.current = setTimeout(async () => {
      try {
        // Only auto-save if workflow hasn't been saved in the last 30 seconds
        const now = Date.now();
        if (now - lastSavedRef.current < AUTO_SAVE_INTERVAL) {
          return;
        }

        // Use save queue for auto-save (lower priority, but still queued)
        await workflowSaveQueue.enqueue(workflow.id, async () => {
          if (isStandalone()) {
            lastSavedRef.current = now;
            if (onSave) await onSave(workflow);
            console.log('[Workflow AutoSave] [Standalone] Workflow autosaved locally:', workflow.id);
            return;
          }
          try {
            const idToUse = ensureWorkflowOwned ? await ensureWorkflowOwned(workflow) : workflow.id;
            const response = await apiRequest(`/api/v1/workflow/${idToUse}/update`, {
              method: 'POST',
              body: JSON.stringify({
                name: workflow.name,
                description: workflow.description ?? '',
                nodes: workflow.nodes,
                connections: workflow.connections,
                global_inputs: workflow.globalInputs,
                tags: workflow.tags ?? [],
                node_output_history: trimNodeOutputHistory(workflow.nodeOutputHistory)
              })
            });

            if (response.ok) {
              const data = await response.json();
              lastSavedRef.current = now;
              console.log('[Workflow AutoSave] Workflow autosaved:', workflow.id, data);
              if (onSave) await onSave(workflow);
            } else {
              console.warn('[Workflow AutoSave] Backend autosave failed, falling back to local save:', workflow.id);
              lastSavedRef.current = now;
              if (onSave) await onSave(workflow);
            }
          } catch (err) {
            console.warn('[Workflow AutoSave] Backend unavailable, falling back to local save:', err);
            lastSavedRef.current = now;
            if (onSave) await onSave(workflow);
          }
        });
      } catch (error) {
        console.error('[Workflow AutoSave] Error autosaving workflow:', error);
      }
    }, AUTO_SAVE_INTERVAL);

    // Cleanup on unmount or when dependencies change
    return () => {
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
    };
  }, [workflow?.isDirty, workflow?.id, workflow?.name, workflow?.description, workflow?.nodes, workflow?.connections, workflow?.globalInputs, workflow?.tags, workflow?.nodeOutputHistory, onSave, ensureWorkflowOwned]);

  return {
    resetAutoSaveTimer
  };
};
