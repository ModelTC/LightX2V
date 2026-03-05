import React, { useRef, useEffect, useCallback, useState } from 'react';
import { X, Bot, GripVertical, RotateCcw, ImagePlus, Globe, Send, Sparkle, Workflow, VideoIcon, Pencil, Lightbulb, Shuffle, CheckCircle2, Circle, Square } from 'lucide-react';
import { ChatMessage } from './ChatMessage';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { ChatMessage as ChatMessageType } from '../../hooks/useAIChatWorkflow';
import { TOOLS } from '../../../constants';

interface DraggableAIChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  chatHistory: ChatMessageType[];
  isProcessing: boolean;
  isExecutingOperations?: boolean;
  executingProgress?: { current: number; total: number };
  executingStepLabels?: string[];
  onStopGeneration?: () => void;
  onSendMessage: (message: string, options?: { image?: { data: string; mimeType: string }; useSearch?: boolean }) => void;
  onClearHistory?: () => void;
  chatContextNodes?: { nodeId: string; name: string }[];
  onRemoveNodeFromChatContext?: (nodeId: string) => void;
  onUndo?: (messageId: string) => void;
  onRetry?: (messageId: string) => void;
  lang: Language;
  aiModel?: string;
  onModelChange?: (model: string) => void;
  position?: { x: number; y: number };
  size?: { width: number; height: number };
  onPositionChange?: (position: { x: number; y: number }) => void;
  onSizeChange?: (size: { width: number; height: number }) => void;
  embedded?: boolean;
  autoFocusInput?: boolean;
  onAutoFocusDone?: () => void;
  chatMode?: 'edit' | 'ideation';
  onChatModeChange?: (mode: 'edit' | 'ideation') => void;
}

const DEFAULT_WIDTH = 400;
const DEFAULT_HEIGHT = 800;
const MIN_WIDTH = 300;
const MIN_HEIGHT = 300;
const AUTO_SCROLL_THRESHOLD = 160;
const QUICK_TIP_DISPLAY_COUNT = 3;
/** IME 刚结束组合后的一小段时间内，Enter 视为选字确认，不发送 */
const IME_ENTER_IGNORE_MS = 150;

function shuffleArray<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function pickRandomTips<T>(tips: T[], n: number): T[] {
  return shuffleArray(tips).slice(0, n);
}

type TipEntry = { key: string; labelZh: string; labelEn: string; icon: 'Bot' | 'Workflow' | 'VideoIcon' | 'Sparkle' | 'Pencil' | 'Lightbulb' };

const EDIT_MODE_TIPS: TipEntry[] = [
  { key: 'edit_1', labelZh: '做一个AI数字人工作流', labelEn: 'Create an AI digital human workflow', icon: 'Bot' },
  { key: 'edit_2', labelZh: '视频都改为9:16', labelEn: 'Change all video to 9:16', icon: 'VideoIcon' },
  { key: 'edit_3', labelZh: '支持哪些工具？', labelEn: 'What tools are supported?', icon: 'Workflow' },
  { key: 'edit_4', labelZh: '描述目前工作流', labelEn: 'Describe the current workflow', icon: 'Workflow' },
  { key: 'edit_5', labelZh: 'LightX2V 是什么？', labelEn: 'What is LightX2V?', icon: 'Sparkle' },
  { key: 'edit_6', labelZh: '支持什么样的AI视频生成？', labelEn: 'What AI video generation is supported?', icon: 'VideoIcon' },
  { key: 'edit_7', labelZh: '添加一个文本输入节点', labelEn: 'Add a text input node', icon: 'Pencil' },
  { key: 'edit_8', labelZh: '把这两个节点连起来', labelEn: 'Connect these two nodes', icon: 'Workflow' },
  { key: 'edit_9', labelZh: '帮我加一个图像生成节点', labelEn: 'Add an image generation node', icon: 'Workflow' },
  { key: 'edit_10', labelZh: '当前画布上有哪些节点？', labelEn: 'What nodes are on the canvas?', icon: 'Workflow' },
  { key: 'edit_11', labelZh: '把所有图片尺寸统一为 16:9', labelEn: 'Make all image dimensions 16:9', icon: 'VideoIcon' },
  { key: 'edit_12', labelZh: '添加语音克隆节点', labelEn: 'Add a voice clone node', icon: 'Bot' },
  { key: 'edit_13', labelZh: '把所有视频尺寸统一为 16:9', labelEn: 'Make all video dimensions 16:9', icon: 'VideoIcon' },
];

const IDEATION_MODE_TIPS: TipEntry[] = [
  { key: 'ideation_1', labelZh: '我想做宫崎骏风格动画', labelEn: 'I want to make a Studio Ghibli style animation', icon: 'Sparkle' },
  { key: 'ideation_2', labelZh: '我想做一个酷炫的科幻风转场动画', labelEn: 'I want a cool sci-fi transition animation', icon: 'VideoIcon' },
  { key: 'ideation_3', labelZh: '我想做一个虚拟数字人女友', labelEn: 'I want to create a virtual digital human girlfriend', icon: 'Bot' },
  { key: 'ideation_4', labelZh: '帮我构思一个产品宣传片', labelEn: 'Help me brainstorm a product promo video', icon: 'Lightbulb' },
  { key: 'ideation_5', labelZh: '我想做治愈系风景动画', labelEn: 'I want to make a healing landscape animation', icon: 'Sparkle' },
  { key: 'ideation_6', labelZh: '构思一个赛博朋克风格的短片', labelEn: 'Brainstorm a cyberpunk style short', icon: 'VideoIcon' },
  { key: 'ideation_7', labelZh: '我想做儿童向的科普动画', labelEn: 'I want to make a kids\' science animation', icon: 'Lightbulb' },
  { key: 'ideation_8', labelZh: '帮我想想一个品牌故事短片', labelEn: 'Help me think of a brand story short', icon: 'Lightbulb' },
  { key: 'ideation_9', labelZh: '我想做电影感的故事预告', labelEn: 'I want a cinematic story teaser', icon: 'VideoIcon' },
  { key: 'ideation_10', labelZh: '构思一个古风国潮的动画', labelEn: 'Brainstorm a traditional Chinese style animation', icon: 'Sparkle' },
  { key: 'ideation_11', labelZh: '我想做赛博朋克城市漫游', labelEn: 'I want a cyberpunk city walkthrough', icon: 'VideoIcon' },
  { key: 'ideation_12', labelZh: '帮我构思一个MV分镜', labelEn: 'Help me brainstorm an MV storyboard', icon: 'Lightbulb' },
  { key: 'ideation_13', labelZh: '我想做蒸汽朋克风格', labelEn: 'I want a steampunk style', icon: 'Sparkle' },
];

export const DraggableAIChatPanel: React.FC<DraggableAIChatPanelProps> = ({
  isOpen,
  onClose,
  chatHistory,
  isProcessing,
  isExecutingOperations = false,
  executingProgress = { current: 0, total: 0 },
  executingStepLabels = [],
  onStopGeneration = () => {},
  onSendMessage,
  onClearHistory,
  chatContextNodes = [],
  onRemoveNodeFromChatContext,
  onUndo,
  onRetry,
  lang,
  aiModel = 'deepseek-v3-2-251201',
  onModelChange,
  position: initialPosition,
  size: initialSize,
  onPositionChange,
  onSizeChange,
  embedded = false,
  autoFocusInput,
  onAutoFocusDone,
  chatMode = 'edit',
  onChatModeChange = (_m: 'edit' | 'ideation') => {}
}) => {
  const { t } = useTranslation(lang);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatInputRef = useRef<HTMLInputElement>(null);
  const isComposingRef = useRef(false);
  const lastCompositionEndRef = useRef(0);
  const shouldAutoScrollRef = useRef(true);

  // 位置和大小状态
  const [position, setPosition] = useState<{ x: number; y: number }>(() => {
    if (initialPosition) return initialPosition;
    // 初始位置：右下角按钮的左边（按钮在 right-6 bottom-6，即距离右边和底部24px，按钮宽度56px）
    // 对话框应该在按钮左边，距离右边 24 + 56 + 24 = 104px
    // 使用视口单位（vw 和 vh）
    if (typeof window !== 'undefined') {
      const viewportWidth = window.innerWidth * 1.25; // 100vw 的像素值
      const viewportHeight = window.innerHeight * 1.25; // 100vh 的像素值
      return {
        x: viewportWidth - DEFAULT_WIDTH - 104,
        y: viewportHeight - DEFAULT_HEIGHT - 24
      };
    }
    return { x: 0, y: 0 };
  });

  const [size, setSize] = useState<{ width: number; height: number }>(() => {
    if (initialSize) return initialSize;
    return { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT };
  });

  const [chatInput, setChatInput] = useState('');
  const [selectedImage, setSelectedImage] = useState<{ data: string; mimeType: string } | null>(null);
  const [isSearchEnabled, setIsSearchEnabled] = useState(false);
  const [showModelSelect, setShowModelSelect] = useState(false);

  const iconMap = useRef({ Bot, Workflow, VideoIcon, Sparkle, Pencil, Lightbulb }).current;
  const currentModeTipPool = chatMode === 'edit' ? EDIT_MODE_TIPS : IDEATION_MODE_TIPS;
  const [visibleQuickTipEntries, setVisibleQuickTipEntries] = useState<TipEntry[]>(() =>
    pickRandomTips(EDIT_MODE_TIPS, QUICK_TIP_DISPLAY_COUNT)
  );

  useEffect(() => {
    setVisibleQuickTipEntries(pickRandomTips(currentModeTipPool, QUICK_TIP_DISPLAY_COUNT));
  }, [chatMode]);

  const handleShuffleTips = useCallback(() => {
    setVisibleQuickTipEntries(pickRandomTips(currentModeTipPool, QUICK_TIP_DISPLAY_COUNT));
  }, [currentModeTipPool]);

  // 拖拽状态
  const isDraggingRef = useRef(false);
  const isResizingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });
  const resizeStartRef = useRef({ x: 0, y: 0, width: 0, height: 0 });
  const resizeDirectionRef = useRef<'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw' | null>(null);

  const isNearBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return true;
    const distanceToBottom = container.scrollHeight - (container.scrollTop + container.clientHeight);
    return distanceToBottom <= AUTO_SCROLL_THRESHOLD;
  }, []);

  const scrollToBottom = useCallback((force = false) => {
    if (!scrollContainerRef.current || !messagesEndRef.current) return;
    if (!force && !shouldAutoScrollRef.current) return;
    requestAnimationFrame(() => {
      if (!scrollContainerRef.current) return;
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    });
  }, []);

  // 自动滚动到底部（只滚动容器内部，不影响页面）
  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isProcessing, scrollToBottom]);

  // 处理思考/流式更新时内容变化导致的未滚动
  useEffect(() => {
    if (!scrollContainerRef.current) return;
    const handleScroll = () => {
      shouldAutoScrollRef.current = isNearBottom();
    };
    const container = scrollContainerRef.current;
    container.addEventListener('scroll', handleScroll);
    handleScroll();
    return () => container.removeEventListener('scroll', handleScroll);
  }, [isNearBottom]);

  useEffect(() => {
    if (!isProcessing) return;
    const intervalId = window.setInterval(() => {
      scrollToBottom();
    }, 300);
    return () => window.clearInterval(intervalId);
  }, [isProcessing, scrollToBottom]);

  useEffect(() => {
    if (!autoFocusInput || !isOpen) return;
    requestAnimationFrame(() => {
      chatInputRef.current?.focus();
      onAutoFocusDone?.();
    });
  }, [autoFocusInput, isOpen, onAutoFocusDone]);

  // 更新位置和大小到父组件
  useEffect(() => {
    if (onPositionChange) {
      onPositionChange(position);
    }
  }, [position, onPositionChange]);

  useEffect(() => {
    if (onSizeChange) {
      onSizeChange(size);
    }
  }, [size, onSizeChange]);

  // 处理窗口拖拽
  const handleHeaderMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button, select')) return;
    e.preventDefault();
    isDraggingRef.current = true;
    dragStartRef.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y
    };
    document.body.style.cursor = 'move';
    document.body.style.userSelect = 'none';
  }, [position]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDraggingRef.current) {
      const newX = e.clientX - dragStartRef.current.x;
      const newY = e.clientY - dragStartRef.current.y;

      // 限制在视口内（使用视口单位 vw 和 vh）
      const viewportWidth = window.innerWidth * 1.25; // 100vw 的像素值
      const viewportHeight = window.innerHeight * 1.25; // 100vh 的像素值
      const maxX = viewportWidth - size.width;
      const maxY = viewportHeight - size.height;

      setPosition({
        x: Math.max(0, Math.min(newX, maxX)),
        y: Math.max(0, Math.min(newY, maxY))
      });
    } else if (isResizingRef.current && resizeDirectionRef.current) {
      const direction = resizeDirectionRef.current;
      const deltaX = e.clientX - resizeStartRef.current.x;
      const deltaY = e.clientY - resizeStartRef.current.y;

      let newWidth = resizeStartRef.current.width;
      let newHeight = resizeStartRef.current.height;
      let newX = position.x;
      let newY = position.y;

      if (direction.includes('e')) {
        newWidth = Math.max(MIN_WIDTH, resizeStartRef.current.width + deltaX);
      }
      if (direction.includes('w')) {
        newWidth = Math.max(MIN_WIDTH, resizeStartRef.current.width - deltaX);
        newX = position.x + (resizeStartRef.current.width - newWidth);
      }
      if (direction.includes('s')) {
        newHeight = Math.max(MIN_HEIGHT, resizeStartRef.current.height + deltaY);
      }
      if (direction.includes('n')) {
        newHeight = Math.max(MIN_HEIGHT, resizeStartRef.current.height - deltaY);
        newY = position.y + (resizeStartRef.current.height - newHeight);
      }

      // 限制在视口内（使用视口单位 vw 和 vh）
      const viewportWidth = window.innerWidth * 1.25; // 100vw 的像素值
      const viewportHeight = window.innerHeight * 1.25; // 100vh 的像素值
      const maxX = viewportWidth - newWidth;
      const maxY = viewportHeight - newHeight;

      const clampedX = Math.max(0, Math.min(newX, maxX));
      const clampedY = Math.max(0, Math.min(newY, maxY));

      setSize({ width: newWidth, height: newHeight });
      setPosition({ x: clampedX, y: clampedY });
    }
  }, [position, size]);

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
    isResizingRef.current = false;
    resizeDirectionRef.current = null;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  const handleSend = useCallback((overrideMessage?: string) => {
    const message = overrideMessage ?? chatInput;
    if (!message.trim() && !selectedImage) return;
    onSendMessage(message, { image: selectedImage || undefined, useSearch: isSearchEnabled });
    setChatInput('');
    setSelectedImage(null);
  }, [chatInput, selectedImage, onSendMessage, isSearchEnabled]);

  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (aiModel.startsWith('deepseek-')) return;
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || '');
      const matches = result.match(/^data:([^;]+);base64,(.+)$/);
      if (matches) {
        setSelectedImage({ mimeType: matches[1], data: matches[2] });
      }
    };
    reader.readAsDataURL(file);
  }, [aiModel]);

  useEffect(() => {
    // 始终添加事件监听器，在handleMouseMove中检查isDraggingRef和isResizingRef
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  // 处理调整大小
  const handleResizeMouseDown = useCallback((direction: 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw') => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    isResizingRef.current = true;
    resizeDirectionRef.current = direction;
    resizeStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      width: size.width,
      height: size.height
    };
    document.body.style.userSelect = 'none';
  }, [size]);

  if (!isOpen) return null;

  const availableModels = TOOLS.find(t => t.id === 'text-generation')?.models || [];
  const selectedModel = availableModels.find(model => model.id === aiModel) || availableModels[0];
  const isDeepseekModel = aiModel.startsWith('deepseek-');
  const isGeminiModel = aiModel.startsWith('ppchat-gemini-') || aiModel.startsWith('gemini-');

  useEffect(() => {
    if (isDeepseekModel && selectedImage) {
      setSelectedImage(null);
    }
  }, [isDeepseekModel, selectedImage]);

  useEffect(() => {
    if (isGeminiModel && isSearchEnabled) {
      setIsSearchEnabled(false);
    }
  }, [isGeminiModel, isSearchEnabled]);

  return (
    <div
      ref={panelRef}
      className={`${embedded ? 'relative w-full h-full' : 'fixed'} z-50 flex flex-col bg-slate-950/95 backdrop-blur-2xl border border-slate-800/70 rounded-[2rem] shadow-[0_0_40px_rgba(0,0,0,0.6)] overflow-hidden`}
      style={embedded ? undefined : {
        left: `${position.x}px`,
        top: `${position.y}px`,
        width: `${size.width}px`,
        height: `${size.height}px`,
        minWidth: `${MIN_WIDTH}px`,
        minHeight: `${MIN_HEIGHT}px`
      }}
    >
      {/* Header - 可拖拽 */}
      <div
        ref={headerRef}
        onMouseDown={embedded ? undefined : handleHeaderMouseDown}
        className={`flex items-center justify-between px-5 py-4 border-b border-slate-800/60 bg-slate-950/60 ${embedded ? '' : 'cursor-move select-none'}`}
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {!embedded && <GripVertical size={14} className="text-slate-400 shrink-0" />}
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-[#90dce1]/15 text-[#90dce1] flex items-center justify-center">
              <Bot size={12} />
            </div>
            <span className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-300">
              {lang === 'zh' ? 'OmniFlow Assistant' : 'OmniFlow Assistant'}
            </span>
          </div>
        </div>
        {/* 模型选择器（居中） */}
        {onModelChange && (
          <div className="relative flex-1 flex justify-center">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowModelSelect((prev) => !prev);
              }}
              onMouseDown={(e) => e.stopPropagation()}
              className="px-4 py-2 rounded-2xl bg-slate-900/80 border border-slate-800 text-[10px] font-black uppercase tracking-widest text-slate-200 hover:border-[#90dce1]/40 transition-all shadow-sm"
              title={lang === 'zh' ? '选择AI模型' : 'Select AI Model'}
              disabled={isProcessing}
            >
              {selectedModel?.name || 'Model'}
            </button>
            {showModelSelect && (
              <div
                className="absolute top-full mt-2 w-56 bg-slate-900/95 border border-slate-800 rounded-2xl shadow-2xl z-40 overflow-hidden"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="max-h-64 overflow-y-auto custom-scrollbar py-2">
                  {availableModels.map(model => (
                    <button
                      key={model.id}
                      onClick={() => {
                        onModelChange(model.id);
                        setShowModelSelect(false);
                      }}
                      className={`w-full text-left px-4 py-2 text-[10px] font-bold uppercase tracking-widest transition-all ${
                        model.id === aiModel
                          ? 'bg-[#90dce1]/10 text-[#90dce1]'
                          : 'text-slate-300 hover:bg-[#90dce1]/10 hover:text-[#90dce1]'
                      }`}
                    >
                      {model.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        <div className="flex items-center gap-1 shrink-0">
          {onClearHistory && (
            <button
              onClick={() => {
                const confirmed = window.confirm(lang === 'zh'
                  ? '确定要清空所有聊天记录吗？'
                  : 'Are you sure you want to clear chat history?');
                if (confirmed) onClearHistory();
              }}
              className="p-2 text-slate-500 hover:text-red-400 hover:bg-slate-800 rounded-xl transition-all"
              title={lang === 'zh' ? '重置对话' : 'Reset History'}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <RotateCcw size={16} />
            </button>
          )}
          {!embedded && (
            <button
              onClick={onClose}
              className="p-2 text-slate-500 hover:text-white hover:bg-red-500/80 rounded-xl transition-all"
              title={lang === 'zh' ? '关闭' : 'Close'}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-scroll px-5 py-4 custom-scrollbar pointer-events-auto"
        style={{ minHeight: 0 }}
      >
        {chatHistory.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-full text-center px-4">
            <Bot size={32} className="text-[#90dce1] mb-3" />
            <p className="text-sm text-slate-400 mb-1">
              {lang === 'zh' ? '你想要做什么样的工作流呢？' : 'Hello! I can help you modify the workflow'}
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
                modelLabel={selectedModel?.name?.toUpperCase()}
                onUndo={onUndo ? () => onUndo(message.id) : undefined}
                onRetry={onRetry ? () => onRetry(message.id) : undefined}
                thinking={message.thinking}
                isStreaming={message.isStreaming}
                choices={message.choices}
                onChoiceClick={message.choices?.length ? (choice) => onSendMessage(choice) : undefined}
              />
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-slate-800 bg-slate-950 flex flex-col gap-3 shrink-0">
        {/* 方案 B + C：Workflow 图标 + 进度条 + 步骤列表逐条打勾 */}
        {isExecutingOperations && executingProgress.total > 0 && (
          <div className="rounded-xl bg-[#90dce1]/10 border border-[#90dce1]/30 text-[#90dce1] overflow-hidden">
            <div className="flex items-center gap-3 px-3 py-2">
              <Workflow className="w-5 h-5 shrink-0 animate-pulse" aria-hidden />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold">
                  {lang === 'zh' ? '正在应用操作' : 'Applying operations'}
                  {' '}
                  <span className="text-slate-300 font-mono">
                    ({executingProgress.current}/{executingProgress.total})
                  </span>
                </p>
                <div className="mt-1.5 h-1 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-[#90dce1] transition-all duration-300"
                    style={{ width: `${executingProgress.total ? (100 * executingProgress.current / executingProgress.total) : 0}%` }}
                  />
                </div>
              </div>
            </div>
            {executingStepLabels.length > 0 && (
              <ul className="px-3 pb-2.5 pt-0 space-y-1 max-h-32 overflow-y-auto custom-scrollbar">
                {executingStepLabels.map((label, i) => {
                  const done = i < executingProgress.current;
                  const current = i === executingProgress.current;
                  return (
                    <li
                      key={i}
                      className={`flex items-center gap-2 text-[11px] transition-colors ${
                        done ? 'text-emerald-400' : current ? 'text-[#90dce1] font-semibold' : 'text-slate-500'
                      }`}
                    >
                      {done ? (
                        <CheckCircle2 className="w-3.5 h-3.5 shrink-0 text-emerald-400" aria-hidden />
                      ) : (
                        <Circle className={`w-3.5 h-3.5 shrink-0 ${current ? 'text-[#90dce1]' : 'text-slate-600'}`} aria-hidden />
                      )}
                      <span className="truncate">{label}</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}
        {!isProcessing && (
          <div className="flex flex-wrap items-center gap-2">
            {visibleQuickTipEntries.map((entry) => {
              const label = lang === 'zh' ? entry.labelZh : entry.labelEn;
              const IconComponent = iconMap[entry.icon];
              return (
                <button
                  key={entry.key}
                  onClick={() => handleSend(label)}
                  className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/80 hover:bg-slate-800 border border-slate-800 hover:border-[#90dce1]/30 rounded-full text-[9px] font-bold uppercase tracking-wide text-slate-500 hover:text-[#90dce1] transition-all group"
                >
                  <span className="text-slate-500 group-hover:text-[#90dce1] transition-colors shrink-0">
                    {IconComponent ? <IconComponent size={10} /> : null}
                  </span>
                  {label}
                </button>
              );
            })}
            <button
              type="button"
              onClick={handleShuffleTips}
              className="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-900/80 hover:bg-slate-800 border border-slate-800 hover:border-[#90dce1]/30 rounded-full text-[9px] font-bold uppercase tracking-wide text-slate-500 hover:text-[#90dce1] transition-all"
              title={lang === 'zh' ? '换一批提示' : 'Shuffle tips'}
            >
              <Shuffle size={10} className="shrink-0" />
              {lang === 'zh' ? '换一批' : 'Shuffle'}
            </button>
          </div>
        )}
        <div className="space-y-4">
          {chatContextNodes.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 shrink-0">
                {lang === 'zh' ? '针对节点：' : 'Context: '}
              </span>
              {chatContextNodes.map((n) => (
                <span
                  key={n.nodeId}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-slate-800/80 border border-slate-700 rounded-lg text-[11px] text-slate-200"
                >
                  <span className="truncate max-w-[120px]">{n.name}</span>
                  {onRemoveNodeFromChatContext && (
                    <button
                      type="button"
                      onClick={() => onRemoveNodeFromChatContext(n.nodeId)}
                      className="p-0.5 rounded hover:bg-slate-600 text-slate-400 hover:text-white transition-colors"
                      title={lang === 'zh' ? '从对话中移除' : 'Remove from context'}
                    >
                      <X size={12} />
                    </button>
                  )}
                </span>
              ))}
            </div>
          )}
          {selectedImage && (
            <div className="relative inline-block group">
              <img
                src={`data:${selectedImage.mimeType};base64,${selectedImage.data}`}
                alt="Preview"
                className="w-20 h-20 object-cover rounded-2xl border-2 border-[#90dce1] shadow-2xl"
              />
              <button
                onClick={() => setSelectedImage(null)}
                className="absolute -top-2 -right-2 p-1.5 bg-red-500 text-white rounded-full shadow-xl hover:bg-red-600 transition-colors"
                title={lang === 'zh' ? '移除图片' : 'Remove image'}
              >
                <X size={12} />
              </button>
            </div>
          )}
          <div className="relative flex items-center gap-2">
            <input type="file" ref={fileInputRef} onChange={handleImageSelect} accept="image/*" className="hidden" />
            <div className="flex gap-2 shrink-0">
              <button
                onClick={() => {
                  if (isDeepseekModel) return;
                  fileInputRef.current?.click();
                }}
                className={`p-2.5 rounded-xl transition-all border ${
                  selectedImage
                    ? 'bg-[#90dce1]/20 border-[#90dce1] text-[#90dce1]'
                    : 'bg-slate-900 border-slate-800 text-slate-500 hover:text-slate-200 hover:border-slate-700'
                } ${isDeepseekModel ? 'opacity-40 cursor-not-allowed' : ''}`}
                title={
                  isDeepseekModel
                    ? (lang === 'zh' ? 'DeepSeek 模型不支持图片输入' : 'DeepSeek does not support image input')
                    : (lang === 'zh' ? '上传图片' : 'Upload image')
                }
                disabled={isProcessing || isDeepseekModel}
              >
                <ImagePlus size={18} />
              </button>
            </div>
            <div className="relative flex-1">
              <input
                ref={chatInputRef}
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onCompositionStart={() => { isComposingRef.current = true; }}
                onCompositionEnd={() => {
                  isComposingRef.current = false;
                  lastCompositionEndRef.current = Date.now();
                }}
                onKeyDown={(e) => {
                  if (e.key !== 'Enter' || e.shiftKey) return;
                  if (e.nativeEvent.isComposing || isComposingRef.current) return;
                  if (Date.now() - lastCompositionEndRef.current < IME_ENTER_IGNORE_MS) {
                    e.preventDefault();
                    return;
                  }
                  e.preventDefault();
                  handleSend();
                }}
                placeholder={lang === 'zh' ? '有什么我可以帮您的？' : 'Ask me anything...'}
                className="w-full bg-slate-900/70 border border-slate-800 rounded-2xl py-3 pl-4 pr-20 text-[12px] text-slate-200 focus:ring-2 focus:ring-[#90dce1]/20 transition-all outline-none placeholder:text-slate-600 shadow-inner"
                disabled={isProcessing}
              />
              <div className="absolute right-2 top-2 flex items-center gap-1.5">
                {!isProcessing && (
                  <button
                    onClick={() => {
                      if (isGeminiModel) return;
                      setIsSearchEnabled(!isSearchEnabled);
                    }}
                    className={`p-1.5 rounded-xl transition-all border ${
                      isSearchEnabled
                        ? 'bg-[#90dce1]/10 border-[#90dce1]/50 text-[#90dce1]'
                        : 'bg-transparent border-transparent text-slate-600 hover:text-slate-400'
                    } ${isGeminiModel ? 'opacity-40 cursor-not-allowed' : ''}`}
                    title={
                      isGeminiModel
                        ? (lang === 'zh' ? 'Gemini 模型不支持联网' : 'Gemini does not support web search')
                        : (lang === 'zh' ? '联网搜索' : 'Web Search')
                    }
                    disabled={isGeminiModel}
                  >
                    <Globe size={16} />
                  </button>
                )}
                {isProcessing ? (
                  <button
                    type="button"
                    onClick={onStopGeneration}
                    className="p-1.5 bg-red-500/90 hover:bg-red-500 text-white rounded-xl transition-all shadow-lg active:scale-95"
                    title={lang === 'zh' ? '停止生成' : 'Stop generating'}
                  >
                    <Square size={14} className="fill-current" />
                  </button>
                ) : (
                  <button
                    onClick={() => handleSend()}
                    disabled={!chatInput.trim() && !selectedImage}
                    className="p-1.5 bg-[#90dce1] text-slate-950 rounded-xl hover:bg-[#a6e4e8] transition-all disabled:opacity-20 disabled:grayscale shadow-lg active:scale-95"
                  >
                    <Send size={16} />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
        {/* 模式切换：工作流编辑 / 构思模式 */}
        {onChatModeChange && (
          <div className="flex rounded-xl bg-slate-900/80 border border-slate-800 p-0.5 w-full">
            <button
              type="button"
              onClick={() => onChatModeChange('edit')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all ${
                chatMode === 'edit'
                  ? 'bg-[#90dce1]/15 text-[#90dce1] border border-[#90dce1]/30'
                  : 'text-slate-500 hover:text-slate-300 border border-transparent'
              }`}
            >
              <Pencil size={12} className="shrink-0" />
              {t('chat_mode_edit')}
            </button>
            <button
              type="button"
              onClick={() => onChatModeChange('ideation')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all ${
                chatMode === 'ideation'
                  ? 'bg-[#90dce1]/15 text-[#90dce1] border border-[#90dce1]/30'
                  : 'text-slate-500 hover:text-slate-300 border border-transparent'
              }`}
            >
              <Lightbulb size={12} className="shrink-0" />
              {t('chat_mode_ideation')}
            </button>
          </div>
        )}
      </div>

      {!embedded && (
        <>
          {/* 调整大小的手柄 */}
          {/* 边缘 */}
          <div
            className="absolute top-0 left-0 right-0 h-1 cursor-ns-resize z-10 pointer-events-none"
            onMouseDown={handleResizeMouseDown('n')}
          />
          <div
            className="absolute bottom-0 left-0 right-0 h-1 cursor-ns-resize z-10 pointer-events-none"
            onMouseDown={handleResizeMouseDown('s')}
          />
          <div
            className="absolute top-0 bottom-0 left-0 w-1 cursor-ew-resize z-10 pointer-events-none"
            onMouseDown={handleResizeMouseDown('w')}
          />
          <div
            className="absolute top-0 bottom-0 right-0 w-1 cursor-ew-resize z-10 pointer-events-none"
            onMouseDown={handleResizeMouseDown('e')}
          />
          {/* 角落 */}
          <div
            className="absolute top-0 left-0 w-3 h-3 cursor-nwse-resize z-10"
            onMouseDown={handleResizeMouseDown('nw')}
          />
          <div
            className="absolute top-0 right-0 w-3 h-3 cursor-nesw-resize z-10"
            onMouseDown={handleResizeMouseDown('ne')}
          />
          <div
            className="absolute bottom-0 left-0 w-3 h-3 cursor-nesw-resize z-10"
            onMouseDown={handleResizeMouseDown('sw')}
          />
          <div
            className="absolute bottom-0 right-0 w-3 h-3 cursor-nwse-resize z-10"
            onMouseDown={handleResizeMouseDown('se')}
          />
        </>
      )}
    </div>
  );
};
