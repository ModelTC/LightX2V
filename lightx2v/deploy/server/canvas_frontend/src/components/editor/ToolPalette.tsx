import React, { useEffect, useState } from 'react';
import { ChevronLeft, Plus, Bot, Workflow } from 'lucide-react';
import { ToolDefinition } from '../../../types';
import { TOOLS } from '../../../constants';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { getIcon } from '../../utils/icons';
import { DraggableAIChatPanel } from './DraggableAIChatPanel';
import { ChatMessage as ChatMessageType } from '../../hooks/useAIChatWorkflow';

interface ToolPaletteProps {
  lang: Language;
  collapsed: boolean;
  onToggleCollapse: () => void;
  onAddNode: (tool: ToolDefinition) => void;
  defaultTab?: 'tools' | 'chat';
  onTabChange?: (tab: 'tools' | 'chat') => void;
  focusChatInput?: boolean;
  onChatInputFocused?: () => void;
  chatHistory?: ChatMessageType[];
  isProcessing?: boolean;
  isExecutingOperations?: boolean;
  executingProgress?: { current: number; total: number };
  executingStepLabels?: string[];
  onStopGeneration?: () => void;
  onSendMessage?: (message: string, options?: { image?: { data: string; mimeType: string }; useSearch?: boolean }) => void;
  onClearHistory?: () => void;
  chatContextNodes?: { nodeId: string; name: string }[];
  onRemoveNodeFromChatContext?: (nodeId: string) => void;
  onUndo?: (messageId: string) => void;
  onRetry?: (messageId: string) => void;
  aiModel?: string;
  onModelChange?: (model: string) => void;
  chatMode?: 'edit' | 'ideation';
  onChatModeChange?: (mode: 'edit' | 'ideation') => void;
}

export const ToolPalette: React.FC<ToolPaletteProps> = ({
  lang,
  collapsed,
  onToggleCollapse,
  onAddNode,
  defaultTab = 'tools',
  onTabChange,
  focusChatInput,
  onChatInputFocused,
  chatHistory = [],
  isProcessing = false,
  isExecutingOperations = false,
  executingProgress = { current: 0, total: 0 },
  executingStepLabels = [],
  onStopGeneration = () => {},
  onSendMessage = () => {},
  onClearHistory,
  chatContextNodes = [],
  onRemoveNodeFromChatContext,
  onUndo,
  onRetry,
  aiModel,
  onModelChange,
  chatMode = 'edit',
  onChatModeChange = () => {}
}) => {
  const { t } = useTranslation(lang);
  const [activeSidebarTab, setActiveSidebarTab] = useState<'tools' | 'chat'>(defaultTab);

  useEffect(() => {
    setActiveSidebarTab(defaultTab);
  }, [defaultTab]);

  const handleTabChange = (tab: 'tools' | 'chat') => {
    setActiveSidebarTab(tab);
    onTabChange?.(tab);
  };

  const categories = ['Input', 'AI Model', 'Image Processing'];
  const categoryMap: Record<string, string> = {
    Input: lang === 'zh' ? '输入' : 'Input',
    'AI Model': lang === 'zh' ? 'AI 模型' : 'AI Model',
    'Image Processing': lang === 'zh' ? '图像处理' : 'Image Processing'
  };
  const categoryStyleMap: Record<string, { color: string; hoverBorder: string; hoverIconBg: string; hoverIconText: string }> = {
    Input: {
      color: '#f59e0b',
      hoverBorder: 'hover:border-amber-400/40',
      hoverIconBg: 'group-hover:bg-amber-400',
      hoverIconText: 'group-hover:text-slate-950'
    },
    'AI Model': {
      color: '#a78bfa',
      hoverBorder: 'hover:border-violet-400/40',
      hoverIconBg: 'group-hover:bg-violet-400',
      hoverIconText: 'group-hover:text-slate-950'
    },
    'Image Processing': {
      color: '#90dce1',
      hoverBorder: 'hover:border-[#90dce1]/40',
      hoverIconBg: 'group-hover:bg-[#90dce1]',
      hoverIconText: 'group-hover:text-slate-950'
    }
  };

  return (
    <>
      {collapsed && (
        <button
          onClick={onToggleCollapse}
          className="absolute left-4 top-4 z-40 flex items-center gap-2 px-6 py-2.5 bg-[#90dce1] text-slate-900 rounded-2xl text-[11px] font-black uppercase tracking-widest transition-all shadow-[0_10px_30px_rgba(144,220,225,0.35)] hover:bg-[#7dd3da] hover:shadow-[0_0_28px_rgba(144,220,225,0.5)] hover:-translate-y-0.5"
          title={lang === 'zh' ? '展开工具面板' : 'Expand Tool Palette'}
        >
          <Plus size={16} />
          <span>
            {lang === 'zh' ? '添加节点' : 'Add Node'}
          </span>
        </button>
      )}
      <aside
        className={`${
          collapsed ? 'w-0' : 'w-[360px]'
        } border-r border-slate-800/60 bg-slate-900/40 backdrop-blur-3xl flex flex-col z-30 overflow-hidden transition-all duration-500 relative ring-1 ring-white/5 shadow-2xl min-h-0`}
      >
        <div className={`flex flex-col flex-1 min-h-0 ${collapsed ? 'hidden' : ''}`}>
          <div className="p-2 bg-slate-950/40 border-b border-slate-800/40 flex gap-1 shrink-0">
            <button
              onClick={() => handleTabChange('tools')}
              className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-[10px] font-black uppercase tracking-[0.1em] transition-all ${
                activeSidebarTab === 'tools'
                  ? 'bg-[#90dce1] text-slate-950 shadow-lg shadow-[#90dce1]/10'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
              }`}
            >
              <Workflow size={14} /> {t('tool_palette')}
            </button>
            <button
              onClick={() => handleTabChange('chat')}
              className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-[10px] font-black uppercase tracking-[0.1em] transition-all ${
                activeSidebarTab === 'chat'
                  ? 'bg-[#90dce1] text-slate-950 shadow-lg shadow-[#90dce1]/10'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
              }`}
            >
              <Bot size={14} /> LightX2V AI
            </button>
            <button
              onClick={onToggleCollapse}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl transition-all"
              title={lang === 'zh' ? '收起' : 'Collapse'}
            >
              <ChevronLeft size={14} />
            </button>
          </div>
          <div className="flex-1 min-h-0 overflow-y-scroll custom-scrollbar relative pointer-events-auto">
            {activeSidebarTab === 'tools' ? (
              <div className="p-4 space-y-8 animate-in slide-in-from-left-4 duration-300">
                {categories.map((cat) => (
                  <div key={cat} className="space-y-2.5">
                    <span
                      className="text-[11px] font-black uppercase tracking-[0.2em]"
                      style={{ color: `${categoryStyleMap[cat]?.color || '#64748b'}cc` }}
                    >
                      {categoryMap[cat]}
                    </span>
                    {TOOLS.filter((t) => t.category === cat).map((tool) => (
                      <div
                        key={tool.id}
                        onClick={() => onAddNode(tool)}
                        className={`flex items-center gap-4 p-4 rounded-3xl bg-slate-800/20 border border-slate-800/60 hover:bg-slate-800/40 cursor-pointer transition-all active:scale-95 group ${categoryStyleMap[cat]?.hoverBorder || ''}`}
                      >
                        <div
                          className={`w-12 h-12 rounded-2xl bg-slate-900 flex items-center justify-center transition-all shadow-xl border border-white/5 text-slate-400 ${categoryStyleMap[cat]?.hoverIconBg || ''} ${categoryStyleMap[cat]?.hoverIconText || ''}`}
                        >
                          {React.createElement(getIcon(tool.icon), { size: 20 })}
                        </div>
                        <div className="flex flex-col">
                          <span className="text-xs font-black text-slate-200 uppercase tracking-wide group-hover:text-white">
                            {lang === 'zh' ? tool.name_zh : tool.name}
                          </span>
                          <span className="text-[10px] text-slate-500 font-medium line-clamp-1 mt-0.5">
                            {lang === 'zh' ? tool.description_zh : tool.description}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-full animate-in slide-in-from-right-4 duration-300 p-3">
                <DraggableAIChatPanel
                  embedded
                  isOpen={true}
                  onClose={() => {}}
                  chatHistory={chatHistory}
                  isProcessing={isProcessing}
                  isExecutingOperations={isExecutingOperations}
                  executingProgress={executingProgress}
                  executingStepLabels={executingStepLabels}
                  onStopGeneration={onStopGeneration}
                  onSendMessage={onSendMessage}
                  onClearHistory={onClearHistory}
                  chatContextNodes={chatContextNodes}
                  onRemoveNodeFromChatContext={onRemoveNodeFromChatContext}
                  onUndo={onUndo}
                  onRetry={onRetry}
                  lang={lang}
                  aiModel={aiModel}
                  onModelChange={onModelChange}
                  chatMode={chatMode}
                  onChatModeChange={onChatModeChange}
                  autoFocusInput={focusChatInput}
                  onAutoFocusDone={onChatInputFocused}
                />
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
};
