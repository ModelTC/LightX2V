import React, { useState, useRef, KeyboardEvent } from 'react';
import { Send } from 'lucide-react';
import { Language } from '../../i18n/useTranslation';

/** IME 刚结束组合后的一小段时间内，Enter 视为选字确认，不发送 */
const IME_ENTER_IGNORE_MS = 150;

interface ChatInputProps {
  onSend: (message: string) => void;
  isProcessing: boolean;
  lang: Language;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  isProcessing,
  lang
}) => {
  const [input, setInput] = useState('');
  const isComposingRef = useRef(false);
  const lastCompositionEndRef = useRef(0);

  const handleSend = () => {
    if (input.trim() && !isProcessing) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // 中文等 IME：组合中或刚结束组合后按 Enter 不发送，仅用于选字/确认
    if (e.key !== 'Enter' || e.shiftKey) return;
    if (e.nativeEvent.isComposing || isComposingRef.current) {
      return;
    }
    if (Date.now() - lastCompositionEndRef.current < IME_ENTER_IGNORE_MS) {
      e.preventDefault();
      return;
    }
    e.preventDefault();
    handleSend();
  };

  return (
    <div className="border-t border-slate-800 p-4 bg-slate-900/40">
      <div className="flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onCompositionStart={() => { isComposingRef.current = true; }}
          onCompositionEnd={() => {
            isComposingRef.current = false;
            lastCompositionEndRef.current = Date.now();
          }}
          onKeyDown={handleKeyDown}
          placeholder={lang === 'zh' ? '描述你想要对工作流做的修改...' : 'Describe the changes you want to make to the workflow...'}
          className="flex-1 bg-slate-900/70 border border-slate-700/60 rounded-2xl px-4 py-3 text-sm text-slate-200 placeholder-slate-500 resize-none focus:outline-none focus:ring-2 focus:ring-[#90dce1]/40 focus:border-[#90dce1]/60 transition-all"
          rows={2}
          disabled={isProcessing}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isProcessing}
          className={`px-4 py-3 rounded-2xl transition-all ${
            !input.trim() || isProcessing
              ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
              : 'bg-[#90dce1] hover:bg-[#7dd3da] text-slate-900 hover:scale-105'
          }`}
        >
          <Send size={18} />
        </button>
      </div>
      {isProcessing && (
        <div className="mt-2 text-xs text-slate-500 flex items-center gap-2">
          <div className="flex gap-1">
            <div className="w-1 h-1 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-1 h-1 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-1 h-1 bg-[#90dce1] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
          <span>{lang === 'zh' ? '正在处理...' : 'Processing...'}</span>
        </div>
      )}
    </div>
  );
};
