import React from 'react';
import { XCircle } from 'lucide-react';

interface TextNodePreviewProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  charsLabel?: string;
}

export const TextNodePreview: React.FC<TextNodePreviewProps> = ({
  value,
  onChange,
  placeholder = 'Enter input text here...',
  charsLabel = 'Characters'
}) => {
  return (
    <div className="space-y-2">
      <div className="relative group">
        <textarea
          value={value}
          onChange={e => onChange(e.target.value)}
          className="w-full h-32 bg-slate-950/50 border border-slate-800 rounded-xl p-3 text-[11px] leading-relaxed text-slate-200 resize-none focus:outline-none focus:border-[#90dce1]/50 focus:ring-1 focus:ring-[#90dce1]/20 transition-all custom-scrollbar"
          placeholder={placeholder}
        />
        {value && (
          <button
            onClick={() => onChange('')}
            className="absolute top-2 right-2 p-1 text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <XCircle size={14} />
          </button>
        )}
      </div>
      <div className="flex justify-end px-1">
        <span className="text-[8px] font-black text-slate-600 uppercase tracking-widest">
          {value.length} {charsLabel}
        </span>
      </div>
    </div>
  );
};
