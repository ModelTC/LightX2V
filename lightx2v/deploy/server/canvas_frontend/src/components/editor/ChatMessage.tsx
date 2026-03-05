import React, { useMemo, useState, useEffect, useRef } from 'react';
import { CheckCircle2, XCircle, ChevronDown, ChevronUp, Lightbulb, Globe, ExternalLink } from 'lucide-react';
import { Language } from '../../i18n/useTranslation';

const MarkdownRenderer: React.FC<{
  text: string;
  className?: string;
  textClass?: string;
  boldClass?: string;
  codeClass?: string;
  headerClass?: string;
}> = ({
  text,
  className = 'space-y-2',
  textClass = 'text-[13px] leading-relaxed text-slate-900',
  boldClass = 'text-white font-black bg-[#90dce1]/5 px-0.5 rounded',
  codeClass = 'bg-black/50 px-1 py-0.5 rounded text-[#90dce1] font-mono text-[11px]',
  headerClass = 'text-[12px] font-black text-slate-200 mt-3 mb-2 first:mt-0'
}) => {
  const lines = text.split('\n');

  return (
    <div className={className}>
      {lines.map((line, idx) => {
        if (line.trim().startsWith('###') || line.trim().startsWith('##') || line.trim().startsWith('#')) {
          const headerText = line.replace(/^#+\s*/, '');
          return <h4 key={idx} className={headerClass}>{headerText}</h4>;
        }

        if (!line.trim()) return <div key={idx} className="h-1" />;

        const parts = line.split(/(\*\*.*?\*\*|`.*?`)/g);

        return (
          <p key={idx} className={textClass}>
            {parts.map((part, pIdx) => {
              if (part.startsWith('**') && part.endsWith('**')) {
                return <strong key={pIdx} className={boldClass}>{part.slice(2, -2)}</strong>;
              }
              if (part.startsWith('`') && part.endsWith('`')) {
                return (
                  <code key={pIdx} className={codeClass}>
                    {part.slice(1, -1)}
                  </code>
                );
              }
              return part;
            })}
          </p>
        );
      })}
    </div>
  );
};

interface ChatMessageProps {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  image?: {
    data: string;
    mimeType: string;
  };
  useSearch?: boolean;
  sources?: {
    title?: string;
    url: string;
    siteName?: string;
  }[];
  timestamp: number;
  modelLabel?: string;
  latencyLabel?: string;
  operations?: any[];
  operationResults?: any[];
  error?: string;
  lang: Language;
  onUndo?: () => void;
  onRetry?: () => void;
  thinking?: string;
  isStreaming?: boolean;
  /** AI 返回的选项，用户点击后作为下一条消息发送 */
  choices?: string[];
  onChoiceClick?: (choice: string) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  content,
  image,
  useSearch,
  sources,
  timestamp,
  modelLabel,
  latencyLabel,
  operations,
  operationResults,
  error,
  lang,
  onUndo,
  onRetry,
  thinking,
  isStreaming,
  choices,
  onChoiceClick
}) => {
  const isUser = role === 'user';
  const hasOperations = operations && operations.length > 0;
  const hasError = !!error;
  const allSuccess = operationResults?.every(r => r.success) ?? false;
  const hasThinking = !!thinking && thinking.trim().length > 0;
  const sourceLinks = useMemo(() => {
    if (!useSearch) return [];
    if (sources && sources.length > 0) return sources;
    if (!content) return [];
    const matches = content.match(/https?:\/\/[^\s)]+/g) || [];
    return Array.from(new Set(matches)).map((url) => ({ url }));
  }, [useSearch, sources, content]);

  // 当有最终答案时，自动折叠思考过程
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(false);
  const thinkingScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // 如果有思考过程且有最终答案（不是流式输出中），默认折叠
    if (hasThinking && !isStreaming && content.trim().length > 0) {
      setIsThinkingExpanded(false);
    } else if (hasThinking && isStreaming) {
      // 流式输出中，展开思考过程
      setIsThinkingExpanded(true);
    }
  }, [hasThinking, isStreaming, content]);

  useEffect(() => {
    if (!isStreaming || !isThinkingExpanded || !thinkingScrollRef.current) return;
    thinkingScrollRef.current.scrollTop = thinkingScrollRef.current.scrollHeight;
  }, [isStreaming, isThinkingExpanded, thinking]);

  // 统计操作类型和数量
  const operationSummary = useMemo(() => {
    if (!operations || operations.length === 0) return null;

    const summary: Record<string, number> = {};
    operations.forEach(op => {
      const type = op.type;
      if (type === 'add_node') {
        summary['add_node'] = (summary['add_node'] || 0) + 1;
      } else if (type === 'delete_node') {
        summary['delete_node'] = (summary['delete_node'] || 0) + 1;
      } else if (type === 'add_connection') {
        summary['add_connection'] = (summary['add_connection'] || 0) + 1;
      } else if (type === 'delete_connection') {
        summary['delete_connection'] = (summary['delete_connection'] || 0) + 1;
      } else if (type === 'update_node' || type === 'update_node_data') {
        summary['update_node'] = (summary['update_node'] || 0) + 1;
      } else if (type === 'replace_node') {
        summary['replace_node'] = (summary['replace_node'] || 0) + 1;
      } else if (type === 'move_node') {
        summary['move_node'] = (summary['move_node'] || 0) + 1;
      }
    });

    const summaryTexts: string[] = [];
    if (summary['add_node']) {
      summaryTexts.push(lang === 'zh' ? `添加${summary['add_node']}个节点` : `Add ${summary['add_node']} node${summary['add_node'] > 1 ? 's' : ''}`);
    }
    if (summary['delete_node']) {
      summaryTexts.push(lang === 'zh' ? `删除${summary['delete_node']}个节点` : `Delete ${summary['delete_node']} node${summary['delete_node'] > 1 ? 's' : ''}`);
    }
    if (summary['add_connection']) {
      summaryTexts.push(lang === 'zh' ? `添加${summary['add_connection']}个连接` : `Add ${summary['add_connection']} connection${summary['add_connection'] > 1 ? 's' : ''}`);
    }
    if (summary['delete_connection']) {
      summaryTexts.push(lang === 'zh' ? `删除${summary['delete_connection']}个连接` : `Delete ${summary['delete_connection']} connection${summary['delete_connection'] > 1 ? 's' : ''}`);
    }
    if (summary['update_node']) {
      summaryTexts.push(lang === 'zh' ? `修改${summary['update_node']}个节点` : `Modify ${summary['update_node']} node${summary['update_node'] > 1 ? 's' : ''}`);
    }
    if (summary['replace_node']) {
      summaryTexts.push(lang === 'zh' ? `替换${summary['replace_node']}个节点` : `Replace ${summary['replace_node']} node${summary['replace_node'] > 1 ? 's' : ''}`);
    }
    if (summary['move_node']) {
      summaryTexts.push(lang === 'zh' ? `移动${summary['move_node']}个节点` : `Move ${summary['move_node']} node${summary['move_node'] > 1 ? 's' : ''}`);
    }

    return summaryTexts.length > 0 ? summaryTexts : null;
  }, [operations, lang]);

  return (
    <div className={`flex flex-col mb-3 ${isUser ? 'items-end' : 'items-start'}`}>
      {!isUser && modelLabel && (
        <div className="mb-2 text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">
          <span className="text-slate-400">{modelLabel}</span>
          {latencyLabel && <span className="ml-2 text-slate-600">• {latencyLabel}</span>}
        </div>
      )}
      {/* Message Bubble：流式且有思考过程时，生成完成前不显示气泡 */}
      {(isUser || !isStreaming || !hasThinking) && (
      <div
        className={`max-w-[90%] px-3.5 py-2.5 rounded-2xl text-[12px] leading-snug ${
          isUser
            ? 'bg-[#90dce1] text-slate-900 rounded-tr-md shadow-xl shadow-[#90dce1]/10'
            : 'bg-slate-900/80 text-slate-200 rounded-tl-md border border-slate-800/70 shadow-sm'
        }`}
      >
        <div className="max-h-80 overflow-y-auto custom-scrollbar pr-1">
          {/* 图片预览 */}
          {image && (
            <div className="mb-3 overflow-hidden rounded-xl border border-black/10">
              <img
                src={`data:${image.mimeType};base64,${image.data}`}
                alt="Sent"
                className="max-w-full h-auto object-cover max-h-64"
              />
            </div>
          )}

          {/* 最终答案 */}
          <div>
            {content.trim().length > 0 ? (
              <>
                <MarkdownRenderer
                  text={content}
                  textClass={isUser ? 'text-[13px] leading-relaxed text-slate-900' : 'text-[13px] leading-relaxed text-white'}
                  headerClass={isUser ? 'text-[12px] font-black text-slate-800 mt-3 mb-2 first:mt-0' : 'text-[12px] font-black text-slate-100 mt-3 mb-2 first:mt-0'}
                />
                {isStreaming && (
                  <span className="inline-block w-2 h-4 bg-[#90dce1] ml-1 animate-pulse" />
                )}
              </>
            ) : isStreaming ? (
              <span className="inline-block w-2 h-4 bg-[#90dce1] ml-1 animate-pulse" />
            ) : null}
          </div>

          {/* Sources */}
          {!isUser && useSearch && sourceLinks.length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-800/50">
              <span className="text-[9px] font-black uppercase text-[#90dce1] flex items-center gap-1 mb-2">
                <Globe size={10} /> {lang === 'zh' ? '参考来源' : 'Sources'}
              </span>
              <div className="flex flex-wrap gap-2">
                {sourceLinks.map((link: any) => (
                  <a
                    key={link.url}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 px-2 py-1 bg-slate-800/50 hover:bg-[#90dce1]/10 border border-slate-800 hover:border-[#90dce1]/30 rounded-lg text-[9px] text-slate-400 hover:text-[#90dce1] transition-all max-w-[220px]"
                  >
                    <ExternalLink size={8} />
                    <span className="truncate">{link.title || link.siteName || link.url}</span>
                  </a>
                ))}
              </div>
            </div>
          )}

          {/* Operation Status */}
          {hasOperations && (
            <div className="mt-2 pt-2 border-t border-slate-700/50 space-y-1">
              <div className="flex items-center gap-2 text-xs">
                {hasError ? (
                  <>
                    <XCircle size={12} className="text-red-400" />
                    <span className="text-red-400">
                      {lang === 'zh' ? '操作失败' : 'Operation failed'}
                    </span>
                  </>
                ) : allSuccess ? (
                  <>
                    <CheckCircle2 size={12} className="text-emerald-400" />
                    <span className="text-emerald-400">
                      {lang === 'zh' ? '操作成功' : 'Operation successful'}
                    </span>
                  </>
                ) : (
                  <span className="text-yellow-400">
                    {lang === 'zh' ? '部分操作成功' : 'Partially successful'}
                  </span>
                )}
              </div>
              {/* Operation Summary */}
              {operationSummary && operationSummary.length > 0 && (
                <div className="flex flex-col gap-0.5 text-[10px] text-slate-400">
                  {operationSummary.map((text, index) => (
                    <span key={index}>{text}</span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      )}

      {/* 选项按钮：AI 返回 choices 时展示，点击后作为下一条用户消息发送 */}
      {!isUser && choices && choices.length > 0 && onChoiceClick && !isStreaming && (
        <div className="flex flex-wrap gap-2 mt-2">
          {choices.map((choice, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => onChoiceClick(choice)}
              className="px-3 py-1.5 text-xs font-medium bg-[#90dce1]/12 hover:bg-[#90dce1]/22 border border-[#90dce1]/25 hover:border-[#90dce1]/45 rounded-xl text-slate-200 hover:text-[#90dce1] transition-all"
            >
              {choice}
            </button>
          ))}
        </div>
      )}

      {/* 思考过程（外置块） */}
      {hasThinking && !isUser && (
        <div className="mt-4 w-full max-w-[92%]">
          <div
            onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
            className="flex items-center justify-between p-3 bg-slate-900/50 border border-slate-800/80 rounded-xl cursor-pointer hover:bg-slate-800 transition-all group"
          >
            <div className="flex items-center gap-3">
              <Lightbulb size={14} className="text-slate-500 group-hover:text-[#90dce1] transition-colors" />
              <span className="text-[10px] font-black uppercase tracking-widest text-slate-400 group-hover:text-slate-200 transition-colors">
                {lang === 'zh' ? '思考过程' : 'Thought Process'}
              </span>
              {isStreaming && (
                <span className="text-[10px] font-black uppercase tracking-widest text-[#90dce1] animate-pulse">
                  {lang === 'zh' ? '思考中...' : 'Thinking...'}
                </span>
              )}
            </div>
            {isThinkingExpanded ? <ChevronUp size={14} className="text-slate-600" /> : <ChevronDown size={14} className="text-slate-600" />}
          </div>
          {isThinkingExpanded && (
            <div className="mt-2 p-4 bg-slate-900/30 border-l-2 border-slate-800/50 ml-3 animate-in fade-in slide-in-from-top-2 duration-300">
              <div ref={thinkingScrollRef} className="max-h-80 overflow-y-auto pr-2 custom-scrollbar">
                <MarkdownRenderer text={thinking} textClass="text-[12px] leading-relaxed text-slate-400" />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Action Buttons (for assistant messages) */}
      {!isUser && hasOperations && (
        <div className="flex gap-2 mt-2">
          {!hasError && onUndo && (
            <button
              onClick={onUndo}
              className="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
            >
              {lang === 'zh' ? '撤销' : 'Undo'}
            </button>
          )}
          {hasError && onRetry && (
            <button
              onClick={onRetry}
              className="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
            >
              {lang === 'zh' ? '重试' : 'Retry'}
            </button>
          )}
        </div>
      )}
    </div>
  );
};
