import { apiRequest } from './apiClient';
import { isStandalone } from '../config/runtimeMode';

/** 仅前端模式下的默认用户 id，用于归属判断语义一致 */
const DEFAULT_USER_ID_STANDALONE = 'default-user';

/**
 * 检查工作流是否属于当前用户（是否可更新/保存/自动保存）
 * 返回 owned：当前用户是否拥有该工作流；!owned 时需先创建新 UUID 再操作。
 * 纯前端模式：preset-* 视为不拥有，其他视为拥有（本地）。
 */
export async function checkWorkflowOwnership(
  workflowId: string,
  currentUserId: string | null
): Promise<{ owned: boolean; isPreset: boolean; workflow: any | null }> {
  if (!workflowId) {
    return { owned: false, isPreset: false, workflow: null };
  }
  const isPreset = workflowId.startsWith('preset-');
  if (isStandalone() || isPreset) {
    return { owned: !isPreset, isPreset, workflow: null };
  }
  const userId = currentUserId ?? DEFAULT_USER_ID_STANDALONE;

  try {
    const checkResponse = await apiRequest(`/api/v1/workflow/${workflowId}`);

    if (checkResponse.ok) {
      const existingWorkflow = await checkResponse.json();
      const belongsToUser = existingWorkflow.user_id && existingWorkflow.user_id === userId;
      return { owned: !!belongsToUser, isPreset, workflow: belongsToUser ? existingWorkflow : null };
    } else {
      // 工作流不存在（404 等）→ 不拥有，需先创建新 UUID
      return { owned: false, isPreset: true, workflow: null };
    }
  } catch (error) {
    console.warn('[WorkflowUtils] Failed to check workflow ownership:', error);
    return { owned: false, isPreset: true, workflow: null };
  }
}

/**
 * 获取当前用户ID
 * 仅前端模式返回默认用户 id，便于归属判断一致。
 * @returns 当前用户ID或null（非 standalone 时）；standalone 时为默认用户 id
 */
export function getCurrentUserId(): string | null {
  try {
    if (isStandalone()) {
      return DEFAULT_USER_ID_STANDALONE;
    }
    const sharedStore = (window as any).__SHARED_STORE__;
    if (sharedStore) {
      const user = sharedStore.getState('user');
      return user?.user_id || null;
    }

    // Fallback to localStorage
    try {
      const userStr = localStorage.getItem('currentUser');
      if (userStr) {
        const user = JSON.parse(userStr);
        return user?.user_id || null;
      }
    } catch (e) {
      // Ignore
    }

    return null;
  } catch (error) {
    console.warn('[WorkflowUtils] Failed to get current user ID:', error);
    return null;
  }
}
