
export interface ChatImage {
  data: string;
  mimeType: string;
}

export interface ChatSource {
  title?: string;
  url: string;
  siteName?: string;
}

export interface ChatOperation {
  type: 'add_node' | 'delete_node' | 'update_node' | 'replace_node' |
        'add_connection' | 'delete_connection' | 'move_node';
  details: any;
}

export interface ChatOperationResult {
  success: boolean;
  operation: ChatOperation;
  result?: any;
  error?: string;
  affectedElements?: { nodeIds?: string[]; connectionIds?: string[] };
}

export interface ChatMessagePersisted {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  image?: ChatImage;
  useSearch?: boolean;
  sources?: ChatSource[];
  timestamp: number;
  error?: string;
}

export interface ChatMessage extends ChatMessagePersisted {
  operations?: ChatOperation[];
  operationResults?: ChatOperationResult[];
  historyCheckpoint?: number;
  thinking?: string;
  isStreaming?: boolean;
  choices?: string[];
}

export interface ChatHistoryList {
  workflowId: string;
  messages: ChatMessagePersisted[];
  updatedAt: number;
}

export type WorkflowChatHistoryMap = Record<string, ChatHistoryList>;

export interface ChatHistoryListResponse {
  workflow_id: string;
  messages: ChatMessagePersisted[];
  updated_at: number;
}

export enum DataType {
  TEXT = 'TEXT',
  IMAGE = 'IMAGE',
  AUDIO = 'AUDIO',
  VIDEO = 'VIDEO'
}

export enum NodeStatus {
  IDLE = 'IDLE',
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  SUCCESS = 'SUCCESS',
  ERROR = 'ERROR'
}

export interface Port {
  id: string;
  type: DataType;
  label: string;
  label_zh?: string;
}

export interface ModelDefaultParams {
  [key: string]: any; // 模型特定的默认参数
}

export interface ModelDefinition {
  id: string;
  name: string;
  defaultParams?: ModelDefaultParams;
}

export interface ToolDefinition {
  id: string;
  name: string;
  name_zh: string;
  category: string;
  category_zh: string;
  description: string;
  description_zh: string;
  inputs: Port[];
  outputs: Port[];
  icon: string;
  models?: ModelDefinition[];
  defaultParams?: ModelDefaultParams;
}

/** 从 /api/v1/task/query 同步的任务状态，用于节点显示排队位置、进度、运行时间（与主应用 TaskDetails 一致） */
export interface NodeRunState {
  status: string; // CREATED | PENDING | RUNNING | SUCCEED | FAILED | CANCEL
  subtasks?: Array<{
    status: string;
    estimated_pending_order?: number;
    estimated_pending_secs?: number;
    estimated_running_secs?: number;
    elapses?: Record<string, number>;
  }>;
}

export interface WorkflowNode {
  id: string;
  name?: string;
  tool_id: string;
  x: number;
  y: number;
  status: NodeStatus;
  data: Record<string, any>; // For internal settings (e.g. image_edits)
  output_value?: any; // For multi-output nodes, this is a Record<portId, value>
  error?: string;
  execution_time?: number;
  start_time?: number;
  completed_at?: number;
  /** 与主应用一致的任务状态（排队/进度/运行时间），由 task/query 轮询更新 */
  run_state?: NodeRunState;
}

export interface Connection {
  id: string;
  source_node_id: string;
  source_port_id: string;
  target_node_id: string;
  target_port_id: string;
}

export type NodeHistoryEntryKind = 'text' | 'file' | 'task'; // 兼容旧数据 kind: 'lightx2v_result' 视为 task；不再使用 kind: 'json'，多端口为 port_id 键字典

/** 节点输出中“已存文件”的引用，使用 kind（兼容旧 type） */
export interface FileReference {
  kind: 'file';
  file_id: string;
  mime_type?: string;
  /** @deprecated 使用 mime_type 代替 */
  ext?: string;
}

/** 任务结果引用，使用 kind（兼容旧 type / __type） */
export interface TaskReference {
  kind: 'task';
  task_id: string;
  output_name: string;
  is_cloud?: boolean;
  workflow_id?: string;
  node_id?: string;
  port_id?: string;
}

export type NodeHistoryValue =
  | { text: string }
  | { json: any }
  | {
      file_id?: string;
      url?: string;
      dataUrl?: string;
      mime_type?: string;
      /** @deprecated 使用 mime_type 代替 */
      ext?: string;
    }
  | {
      task_id: string;
      output_name: string;
      is_cloud: boolean;
      workflow_id?: string;
      node_id?: string;
      port_id?: string;
    };

/**
 * 节点历史条目。
 * - output_value 统一为以 port_id 为键的字典（out-text/out-audio/out-video/out-text1…）；每端口值为文本(string)、file ref 或 task ref，不再使用 kind: 'json'。
 */
export interface NodeHistoryEntry {
  id: string;
  timestamp: number;
  execution_time?: number;
  /** 以 port_id 为键的字典，值为 string | { kind: 'file', ... } | { kind: 'task', ... }（或兼容旧版 kind: 'text' 等） */
  output_value: Record<string, any>;
  params?: Record<string, any>;
  /** @deprecated 仅兼容旧数据读取，新数据统一存 output_value */
  output_value_port_keyed?: Record<string, any>;
  metadata?: { port_id?: string; [k: string]: any };
}

export interface WorkflowState {
  id: string;
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  connections: Connection[];
  isDirty: boolean;
  isRunning: boolean;
  globalInputs: Record<string, any>;
  nodeOutputHistory?: Record<string, NodeHistoryEntry[]>;
  updatedAt: number;
  createAt?: number;
  visibility?: 'private' | 'public';
  thumsupCount?: number;
  thumsupLiked?: boolean;
  authorName?: string;
  authorId?: string;
  userId?: string;
  tags?: string[];
  /**
   * AI 对话历史（运行时，从独立 API 加载）
   * 关联：通过 workflow.id 查询 GET /api/v1/workflow/{id}/chat
   */
  chatHistory?: ChatMessage[];
}
