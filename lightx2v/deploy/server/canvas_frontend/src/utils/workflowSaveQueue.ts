/**
 * 工作流保存队列
 * 确保保存操作按顺序执行，避免并发冲突
 */

import { workflowOfflineQueue } from './workflowOfflineQueue';

type SaveTask = {
  workflowId: string;
  saveFn: () => Promise<any>;
  resolve: (value: any) => void;
  reject: (error: any) => void;
  createdAt: number; // 任务创建时间，用于超时检测
};

const SAVE_TIMEOUT = 30000; // 30秒超时
const MAX_QUEUE_SIZE = 20; // 最大队列大小，防止内存泄漏

class WorkflowSaveQueue {
  private queue: SaveTask[] = [];
  private processing = false;
  private currentWorkflowId: string | null = null;

  /**
   * 添加保存任务到队列
   * @param workflowId 工作流ID
   * @param saveFn 保存函数
   * @returns Promise
   */
  async enqueue(workflowId: string, saveFn: () => Promise<any>): Promise<any> {
    return new Promise((resolve, reject) => {
      // 如果队列中已有相同工作流的任务，移除旧任务（保留最新的）
      this.queue = this.queue.filter(task => task.workflowId !== workflowId);

      // 限制队列大小，防止内存泄漏
      if (this.queue.length >= MAX_QUEUE_SIZE) {
        const oldestTask = this.queue.shift();
        if (oldestTask) {
          oldestTask.reject(new Error('Save queue is full, oldest task removed'));
        }
      }

      this.queue.push({
        workflowId,
        saveFn,
        resolve,
        reject,
        createdAt: Date.now()
      });

      // 如果当前没有在处理，开始处理队列
      if (!this.processing) {
        this.processQueue();
      }
    });
  }

  /**
   * 处理队列中的保存任务
   */
  private async processQueue() {
    if (this.processing || this.queue.length === 0) {
      return;
    }

    this.processing = true;

    while (this.queue.length > 0) {
      const task = this.queue.shift();
      if (!task) break;

      this.currentWorkflowId = task.workflowId;

      // 检查任务是否超时
      const taskAge = Date.now() - task.createdAt;
      if (taskAge > SAVE_TIMEOUT) {
        console.warn(`[WorkflowSaveQueue] Task for workflow ${task.workflowId} timed out (${taskAge}ms old)`);
        task.reject(new Error(`Save task timed out after ${SAVE_TIMEOUT}ms`));
        this.currentWorkflowId = null;
        continue;
      }

      try {
        console.log(`[WorkflowSaveQueue] Processing save for workflow: ${task.workflowId}`);

        // 使用 Promise.race 实现超时控制
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error(`Save operation timed out after ${SAVE_TIMEOUT}ms`)), SAVE_TIMEOUT);
        });

        const result = await Promise.race([task.saveFn(), timeoutPromise]);
        task.resolve(result);
        console.log(`[WorkflowSaveQueue] Save completed for workflow: ${task.workflowId}`);
      } catch (error) {
        // 如果错误是因为工作流被删除，静默处理
        if (error instanceof Error && error.message.includes('deleted')) {
          console.log(`[WorkflowSaveQueue] Save cancelled for deleted workflow: ${task.workflowId}`);
        } else if (error instanceof Error && error.message.includes('timed out')) {
          console.error(`[WorkflowSaveQueue] Save timed out for workflow: ${task.workflowId}`);
          // 超时错误，可能需要添加到离线队列
          this.handleSaveTimeout(task.workflowId, error);
        } else {
          console.error(`[WorkflowSaveQueue] Save failed for workflow: ${task.workflowId}`, error);
        }
        task.reject(error);
      } finally {
        this.currentWorkflowId = null;
        // 添加小延迟，避免连续请求过快
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    }

    this.processing = false;
  }

  /**
   * 检查指定工作流是否正在保存
   */
  isSaving(workflowId: string): boolean {
    return this.currentWorkflowId === workflowId ||
           this.queue.some(task => task.workflowId === workflowId);
  }

  /**
   * 移除指定工作流的保存任务
   * @param workflowId 工作流ID
   */
  removeTask(workflowId: string): void {
    const removedTasks = this.queue.filter(task => task.workflowId === workflowId);
    this.queue = this.queue.filter(task => task.workflowId !== workflowId);

    // 如果正在处理的工作流被删除，取消当前任务
    if (this.currentWorkflowId === workflowId) {
      this.currentWorkflowId = null;
      // 注意：当前正在执行的任务无法取消，但会在执行时检查工作流是否存在
    }

    // 拒绝被移除的任务
    removedTasks.forEach(task => {
      task.reject(new Error(`Workflow ${workflowId} was deleted, save task cancelled`));
    });

    if (removedTasks.length > 0) {
      console.log(`[WorkflowSaveQueue] Removed ${removedTasks.length} save task(s) for deleted workflow: ${workflowId}`);
    }
  }

  /**
   * 处理保存超时
   */
  private async handleSaveTimeout(workflowId: string, error: Error): Promise<void> {
    // 尝试将超时的保存任务添加到离线队列
    try {
      // 注意：这里无法获取 workflowData，所以只能记录错误
      console.warn(`[WorkflowSaveQueue] Save timeout for ${workflowId}, consider adding to offline queue manually`);
    } catch (err) {
      console.error('[WorkflowSaveQueue] Failed to handle save timeout:', err);
    }
  }

  /**
   * 清理超时任务
   */
  cleanupTimeoutTasks(): void {
    const now = Date.now();
    const timeoutTasks = this.queue.filter(task => now - task.createdAt > SAVE_TIMEOUT);

    if (timeoutTasks.length > 0) {
      timeoutTasks.forEach(task => {
        task.reject(new Error(`Save task timed out after ${SAVE_TIMEOUT}ms`));
      });
      this.queue = this.queue.filter(task => !timeoutTasks.includes(task));
      console.log(`[WorkflowSaveQueue] Cleaned up ${timeoutTasks.length} timeout task(s)`);
    }
  }

  /**
   * 清空队列
   */
  clear() {
    this.queue.forEach(task => {
      task.reject(new Error('Save queue cleared'));
    });
    this.queue = [];
    this.processing = false;
    this.currentWorkflowId = null;
  }
}

// 单例实例
export const workflowSaveQueue = new WorkflowSaveQueue();

// 定期清理超时任务（每30秒）
if (typeof window !== 'undefined') {
  setInterval(() => {
    workflowSaveQueue.cleanupTimeoutTasks();
  }, 30000);
}
