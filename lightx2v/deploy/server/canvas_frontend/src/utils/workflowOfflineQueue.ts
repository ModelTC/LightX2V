/**
 * 工作流离线保存队列
 * 当网络不可用时，将保存操作存储到 localStorage
 * 网络恢复后自动同步
 */

import { apiRequest } from './apiClient';

interface OfflineSaveTask {
  workflowId: string;
  workflowData: any;
  timestamp: number;
  retryCount: number;
}

const OFFLINE_QUEUE_KEY = 'workflow_offline_save_queue';
const MAX_RETRY_COUNT = 5;
const SYNC_INTERVAL = 5000; // 5 seconds
const MAX_QUEUE_SIZE = 10; // 最多保存10个离线任务，避免超出localStorage限制
const MAX_QUEUE_SIZE_BYTES = 4 * 1024 * 1024; // 4MB，为localStorage留出安全空间

class WorkflowOfflineQueue {
  private syncInterval: NodeJS.Timeout | null = null;

  /**
   * 估算数据大小（字节）
   */
  private estimateSize(data: any): number {
    try {
      return new Blob([JSON.stringify(data)]).size;
    } catch {
      return JSON.stringify(data).length * 2; // 粗略估算，UTF-16编码
    }
  }

  /**
   * 压缩工作流数据（移除不必要的字段以减小体积）
   */
  private compressWorkflowData(workflowData: any): any {
    const compressed = { ...workflowData };
    // 移除节点中的大字段（如base64数据）
    if (compressed.nodes) {
      compressed.nodes = compressed.nodes.map((node: any) => {
        const { output_value, ...rest } = node;
        return rest;
      });
    }
    return compressed;
  }

  /**
   * 添加离线保存任务
   */
  addTask(workflowId: string, workflowData: any): void {
    try {
      const queue = this.getQueue();
      // 如果队列中已有相同工作流的任务，移除旧任务（保留最新的）
      const filteredQueue = queue.filter(task => task.workflowId !== workflowId);

      // 压缩工作流数据以减小体积
      const compressedData = this.compressWorkflowData(workflowData);
      const newTask: OfflineSaveTask = {
        workflowId,
        workflowData: compressedData,
        timestamp: Date.now(),
        retryCount: 0
      };

      // 检查队列大小限制
      const newQueue = [...filteredQueue, newTask];
      const queueSize = this.estimateSize(newQueue);

      // 如果超出大小限制，移除最旧的任务
      let finalQueue = newQueue;
      if (queueSize > MAX_QUEUE_SIZE_BYTES || newQueue.length > MAX_QUEUE_SIZE) {
        // 按时间戳排序，保留最新的任务
        finalQueue = newQueue
          .sort((a, b) => b.timestamp - a.timestamp)
          .slice(0, MAX_QUEUE_SIZE);

        // 如果仍然太大，继续移除直到满足大小限制
        while (this.estimateSize(finalQueue) > MAX_QUEUE_SIZE_BYTES && finalQueue.length > 1) {
          finalQueue.pop();
        }

        const removedCount = newQueue.length - finalQueue.length;
        if (removedCount > 0) {
          console.warn(`[WorkflowOfflineQueue] Removed ${removedCount} old task(s) due to size limit`);
        }
      }

      localStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(finalQueue));
      console.log('[WorkflowOfflineQueue] Task added to offline queue:', workflowId, `(${finalQueue.length}/${MAX_QUEUE_SIZE} tasks)`);

      // 启动同步检查
      this.startSync();
    } catch (error) {
      // 如果存储失败，可能是localStorage已满
      if (error instanceof Error && (error.name === 'QuotaExceededError' || error.message.includes('quota'))) {
        console.error('[WorkflowOfflineQueue] localStorage quota exceeded, removing oldest tasks');
        // 尝试移除最旧的任务后重试
        const queue = this.getQueue();
        if (queue.length > 1) {
          const sortedQueue = queue.sort((a, b) => b.timestamp - a.timestamp);
          const reducedQueue = sortedQueue.slice(0, Math.floor(queue.length / 2));
          try {
            localStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(reducedQueue));
            console.log('[WorkflowOfflineQueue] Reduced queue size and retried');
          } catch (retryError) {
            console.error('[WorkflowOfflineQueue] Failed to save even after reducing queue:', retryError);
          }
        }
      } else {
        console.error('[WorkflowOfflineQueue] Failed to add task:', error);
      }
    }
  }

  /**
   * 移除指定工作流的离线任务
   */
  removeTask(workflowId: string): void {
    try {
      const queue = this.getQueue();
      const filteredQueue = queue.filter(task => task.workflowId !== workflowId);

      if (filteredQueue.length < queue.length) {
        this.saveQueue(filteredQueue);
        console.log(`[WorkflowOfflineQueue] Removed task for deleted workflow: ${workflowId}`);
      }
    } catch (error) {
      console.error('[WorkflowOfflineQueue] Failed to remove task:', error);
    }
  }

  /**
   * 获取离线队列
   */
  private getQueue(): OfflineSaveTask[] {
    try {
      const queueStr = localStorage.getItem(OFFLINE_QUEUE_KEY);
      if (!queueStr) return [];
      return JSON.parse(queueStr);
    } catch (error) {
      console.error('[WorkflowOfflineQueue] Failed to get queue:', error);
      return [];
    }
  }

  /**
   * 保存队列
   */
  private saveQueue(queue: OfflineSaveTask[]): void {
    try {
      localStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(queue));
    } catch (error) {
      console.error('[WorkflowOfflineQueue] Failed to save queue:', error);
    }
  }

  /**
   * 检查网络是否可用
   */
  private isOnline(): boolean {
    return navigator.onLine;
  }

  /**
   * 获取同步锁（使用 BroadcastChannel 或 localStorage）
   */
  private async acquireSyncLock(): Promise<boolean> {
    try {
      // 使用 BroadcastChannel 实现跨标签页通信
      const channel = new BroadcastChannel('workflow_offline_sync');
      const lockKey = 'workflow_offline_sync_lock';
      const lockTimeout = 30000; // 30秒超时

      // 检查是否有其他标签页正在同步
      const existingLock = localStorage.getItem(lockKey);
      if (existingLock) {
        const lockTime = parseInt(existingLock, 10);
        if (Date.now() - lockTime < lockTimeout) {
          // 其他标签页正在同步
          return false;
        }
        // 锁已过期，清除
        localStorage.removeItem(lockKey);
      }

      // 获取锁
      this.syncLock = `${Date.now()}-${Math.random()}`;
      localStorage.setItem(lockKey, Date.now().toString());

      // 监听其他标签页的同步完成消息
      channel.onmessage = (event) => {
        if (event.data === 'sync_complete' && this.syncLock === event.data.lockId) {
          this.syncLock = null;
          localStorage.removeItem(lockKey);
        }
      };

      return true;
    } catch (error) {
      // BroadcastChannel 不支持时，使用简单锁
      if (this.isSyncing) {
        return false;
      }
      this.isSyncing = true;
      return true;
    }
  }

  /**
   * 释放同步锁
   */
  private releaseSyncLock(): void {
    try {
      const channel = new BroadcastChannel('workflow_offline_sync');
      const lockKey = 'workflow_offline_sync_lock';

      if (this.syncLock) {
        channel.postMessage({ type: 'sync_complete', lockId: this.syncLock });
        this.syncLock = null;
      }
      localStorage.removeItem(lockKey);
    } catch (error) {
      // 简单锁模式
      this.isSyncing = false;
    }
  }

  /**
   * 同步离线队列
   */
  async syncQueue(): Promise<void> {
    if (!this.isOnline()) {
      return;
    }

    // 获取同步锁，防止多个标签页同时同步
    const hasLock = await this.acquireSyncLock();
    if (!hasLock) {
      return; // 其他标签页正在同步
    }

    try {
      const queue = this.getQueue();
      if (queue.length === 0) {
        this.releaseSyncLock();
        return;
      }

      console.log(`[WorkflowOfflineQueue] Syncing ${queue.length} offline tasks...`);

      // 触发同步开始事件（用于UI提示）
      this.notifySyncStart(queue.length);

    const remainingTasks: OfflineSaveTask[] = [];

    for (const task of queue) {
      try {
        // 尝试更新工作流
        const updateResponse = await apiRequest(`/api/v1/workflow/${task.workflowId}/update`, {
          method: 'POST',
          body: JSON.stringify(task.workflowData)
        });

        if (updateResponse.ok) {
          console.log(`[WorkflowOfflineQueue] Successfully synced workflow: ${task.workflowId}`);
          // 任务成功，不加入剩余队列
          this.notifySyncProgress(queue.length - remainingTasks.length, queue.length);
        } else if (updateResponse.status === 404) {
          // 工作流不存在，可能是被删除了，直接移除任务
          console.log(`[WorkflowOfflineQueue] Workflow ${task.workflowId} not found (may have been deleted), removing from queue`);
          // 不重试，直接移除
        } else {
          // 其他错误，重试
          this.handleTaskRetry(task, remainingTasks);
        }
      } catch (error) {
        console.error(`[WorkflowOfflineQueue] Failed to sync workflow ${task.workflowId}:`, error);
        // 网络错误，重试
        this.handleTaskRetry(task, remainingTasks);
      }
    }

      // 保存剩余任务
      if (remainingTasks.length > 0) {
        this.saveQueue(remainingTasks);
        console.log(`[WorkflowOfflineQueue] ${remainingTasks.length} tasks remaining in queue`);
        // 触发同步部分完成事件
        this.notifySyncPartial(remainingTasks.length, queue.length);
      } else {
        // 清空队列
        localStorage.removeItem(OFFLINE_QUEUE_KEY);
        console.log('[WorkflowOfflineQueue] All tasks synced successfully');
        this.stopSync();
        // 触发同步完成事件
        this.notifySyncComplete();
      }
    } catch (error) {
      console.error('[WorkflowOfflineQueue] Error during sync:', error);
    } finally {
      // 释放同步锁
      this.releaseSyncLock();
    }
  }

  /**
   * 通知同步开始
   */
  private notifySyncStart(totalTasks: number): void {
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('workflow-offline-sync-start', {
        detail: { totalTasks }
      }));
    }
  }

  /**
   * 通知同步进度
   */
  private notifySyncProgress(completed: number, total: number): void {
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('workflow-offline-sync-progress', {
        detail: { completed, total }
      }));
    }
  }

  /**
   * 通知同步部分完成
   */
  private notifySyncPartial(remaining: number, total: number): void {
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('workflow-offline-sync-partial', {
        detail: { remaining, total }
      }));
    }
  }

  /**
   * 通知同步完成
   */
  private notifySyncComplete(): void {
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('workflow-offline-sync-complete'));
    }
  }

  /**
   * 处理任务重试
   */
  private handleTaskRetry(task: OfflineSaveTask, remainingTasks: OfflineSaveTask[]): void {
    if (task.retryCount < MAX_RETRY_COUNT) {
      task.retryCount++;
      remainingTasks.push(task);
      console.log(`[WorkflowOfflineQueue] Task ${task.workflowId} will retry (${task.retryCount}/${MAX_RETRY_COUNT})`);
    } else {
      console.warn(`[WorkflowOfflineQueue] Task ${task.workflowId} exceeded max retry count, removing from queue`);
    }
  }

  /**
   * 启动同步检查
   */
  startSync(): void {
    if (this.syncInterval) {
      return; // 已经在运行
    }

    // 立即尝试同步一次
    this.syncQueue();

    // 定期检查并同步
    this.syncInterval = setInterval(() => {
      this.syncQueue();
    }, SYNC_INTERVAL);

    // 监听网络状态变化
    window.addEventListener('online', () => {
      console.log('[WorkflowOfflineQueue] Network online, syncing queue...');
      this.syncQueue();
    });

    // 监听页面可见性变化，页面隐藏时尝试同步
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        this.syncQueue();
      }
    });

    // 监听页面卸载，尝试同步离线队列
    window.addEventListener('beforeunload', () => {
      // 同步队列（同步执行，不等待）
      if (this.isOnline() && this.getQueue().length > 0) {
        // 使用 sendBeacon 或同步请求尝试保存
        const queue = this.getQueue();
        if (queue.length > 0) {
          // 保存到 localStorage（已经保存了，这里只是确保）
          try {
            localStorage.setItem(OFFLINE_QUEUE_KEY, JSON.stringify(queue));
          } catch (error) {
            console.error('[WorkflowOfflineQueue] Failed to save queue on unload:', error);
          }
        }
      }
    });
  }

  /**
   * 停止同步检查
   */
  stopSync(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
    }
  }

  /**
   * 获取队列中的任务数量
   */
  getQueueSize(): number {
    return this.getQueue().length;
  }

  /**
   * 清空队列
   */
  clearQueue(): void {
    localStorage.removeItem(OFFLINE_QUEUE_KEY);
    this.stopSync();
  }
}

// 单例实例
export const workflowOfflineQueue = new WorkflowOfflineQueue();

// 在应用启动时启动同步
if (typeof window !== 'undefined') {
  workflowOfflineQueue.startSync();
}
