import React, { useRef, useEffect } from 'react';
import { X, Minimize2, Maximize2, Bot, ChevronUp, ChevronDown } from 'lucide-react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { Language } from '../../i18n/useTranslation';
import { ChatMessage as ChatMessageType } from '../../hooks/useAIChatWorkflow';
import { TOOLS } from '../../../constants';

interface AIChatPanelProps {
  isOpen: boolean;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  onClose: () => void;
  chatHistory: ChatMessageType[];
  isProcessing: boolean;
  onSendMessage: (message: string, options?: { image?: { data: string; mimeType: string }; useSearch?: boolean }) => void;
  onUndo?: (messageId: string) => void;
  onRetry?: (messageId: string) => void;
  lang: Language;
  nodeConfigPanelCollapsed?: boolean;
  onToggleNodeConfigPanel?: () => void;
  aiModel?: string;
  onModelChange?: (model: string) => void;
  style?: React.CSSProperties;
}

export const AIChatPanel: React.FC<AIChatPanelProps> = ({
  isOpen,
  isCollapsed,
  onToggleCollapse,
  onClose,
  chatHistory,
  isProcessing,
  onSendMessage,
  onUndo,
  onRetry,
  lang,
  nodeConfigPanelCollapsed = false,
  onToggleNodeConfigPanel,
  aiModel = 'deepseek-v3-2-251201',
  onModelChange,
  style
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // 自动滚动到底部（只滚动容器内部，不影响页面）
  useEffect(() => {
    if (messagesEndRef.current && scrollContainerRef.current) {
      // 使用 requestAnimationFrame 确保 DOM 更新完成后再滚动
      requestAnimationFrame(() => {
        if (scrollContainerRef.current && messagesEndRef.current) {
          // 直接设置 scrollTop，避免 scrollIntoView 导致页面整体滚动
          scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
      });
    }
  }, [chatHistory, isProcessing]);

  if (!isOpen) return null;

  return (
    <div
      className={`flex flex-col transition-all ${
        isCollapsed ? 'h-0 overflow-hidden' : 'flex-1 min-h-0'
      }`}
      style={!isCollapsed ? (style || { flexBasis: 'auto', minHeight: '200px' }) : undefined}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/60 bg-slate-900/60">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <div className="w-6 h-6 rounded-full bg-[#90dce1] flex items-center justify-center shrink-0">
            <Bot size={14} className="text-white" />
          </div>
          <span className="text-sm font-bold text-slate-200 shrink-0">
            {lang === 'zh' ? 'AI助手' : 'AI Assistant'}
          </span>
          {/* 模型选择器 */}
          {onModelChange && (() => {
            const geminiTextTool = TOOLS.find(t => t.id === 'text-generation');
            const availableModels = geminiTextTool?.models || [];

            return (
              <select
                value={aiModel}
                onChange={(e) => onModelChange(e.target.value)}
                className="ml-2 px-3 py-1.5 text-xs bg-slate-800/80 hover:bg-slate-800 border border-slate-700/60 hover:border-slate-600 rounded-lg text-slate-200 focus:outline-none focus:ring-2 focus:ring-#90dce1/50 focus:border-[#90dce1] transition-all cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 fill=%27none%27 viewBox=%270 0 20 20%27%3E%3Cpath stroke=%27%23cbd5e1%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 stroke-width=%271.5%27 d=%27M6 8l4 4 4-4%27/%3E%3C/svg%3E')] bg-[length:16px_16px] bg-[right_8px_center] bg-no-repeat pr-8 shadow-sm"
                disabled={isProcessing}
                title={lang === 'zh' ? '选择AI模型' : 'Select AI Model'}
              >
                {availableModels.length > 0 ? (
                  availableModels.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))
                ) : (
                  <option value="deepseek-v3-2-251201">DeepSeek V3.2</option>
                )}
              </select>
            );
          })()}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {/* 向上展开按钮（折叠NodeConfigPanel） */}
          {onToggleNodeConfigPanel && (
            <button
              onClick={onToggleNodeConfigPanel}
              className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
              title={lang === 'zh' ? (nodeConfigPanelCollapsed ? '展开配置面板' : '折叠配置面板') : (nodeConfigPanelCollapsed ? 'Expand Config Panel' : 'Collapse Config Panel')}
            >
              <ChevronUp size={14} />
            </button>
          )}
          {/* 向下折叠按钮（折叠AIChatPanel） */}
          <button
            onClick={onToggleCollapse}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
            title={lang === 'zh' ? (isCollapsed ? '展开' : '向下折叠') : (isCollapsed ? 'Expand' : 'Collapse Down')}
          >
            {isCollapsed ? <Maximize2 size={14} /> : <ChevronDown size={14} />}
          </button>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
            title={lang === 'zh' ? '关闭' : 'Close'}
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {!isCollapsed && (
        <>
          {/* Messages */}
          <div
            ref={scrollContainerRef}
            className="flex-1 overflow-y-auto p-4 custom-scrollbar"
            style={{ minHeight: 0 }}
          >
            {chatHistory.length === 0 ? (
              <div className="flex flex-col items-center justify-center min-h-full text-center px-4">
                <Bot size={32} className="text-[#90dce1] mb-3" />
                <p className="text-sm text-slate-400 mb-1">
                  {lang === 'zh' ? '你好！我可以帮你修改工作流' : 'Hello! I can help you modify the workflow'}
                </p>
                <p className="text-xs text-slate-500">
                  {lang === 'zh' ? '试试说："做一个文字生成数字人视频的工作流"' : 'Try saying: "Create a workflow that generates digital human video from text"'}
                </p>
              </div>
            ) : (
              <>
                {chatHistory.map((message) => (
                  <ChatMessage
                    key={message.id}
                    {...message}
                    lang={lang}
                    onUndo={onUndo ? () => onUndo(message.id) : undefined}
                    onRetry={onRetry ? () => onRetry(message.id) : undefined}
                    thinking={message.thinking}
                    isStreaming={message.isStreaming}
                    choices={message.choices}
                    onChoiceClick={message.choices?.length ? (choice) => onSendMessage(choice) : undefined}
                  />
                ))}
                {isProcessing && (
                  <div className="flex gap-3 mb-4">
                    <div className="w-8 h-8 rounded-full bg-[#90dce1] flex items-center justify-center shrink-0">
                      <Bot size={16} className="text-white" />
                    </div>
                    <div className="flex-1 bg-slate-800/50 border border-slate-700 rounded-2xl px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="flex gap-1">
                          <div className="w-1.5 h-1.5 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                          <div className="w-1.5 h-1.5 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                          <div className="w-1.5 h-1.5 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                        </div>
                        <span className="text-sm text-slate-400">
                          {lang === 'zh' ? '正在思考...' : 'Thinking...'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Input */}
          <ChatInput
            onSend={onSendMessage}
            isProcessing={isProcessing}
            lang={lang}
          />
        </>
      )}
    </div>
  );
};
