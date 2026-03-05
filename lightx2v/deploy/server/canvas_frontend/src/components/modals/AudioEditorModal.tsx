import React from 'react';
import { X, Volume2 } from 'lucide-react';
import { Language } from '../../i18n/useTranslation';
import { AudioNodePreview } from '../previews/AudioNodePreview';

interface AudioEditorModalProps {
  nodeId: string;
  audioData: string;
  audioRange?: { start: number; end: number };
  onRangeChange?: (range: { start: number; end: number }) => void;
  onClose: () => void;
  onSave: (trimmedAudio: string) => void;
  lang: Language;
}

export const AudioEditorModal: React.FC<AudioEditorModalProps> = ({
  audioData,
  audioRange,
  onRangeChange,
  onClose,
  onSave,
  lang
}) => {
  const modalData = {
    original: audioData,
    trimmed: audioData,
    range: audioRange || { start: 0, end: 100 }
  };

  const title = lang === 'zh' ? '音频输入' : 'Audio Input';
  const subtitle = lang === 'zh' ? '源输入 - 可编辑' : 'Source Input - Editable';

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-lg z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-slate-900/95 border border-slate-800 rounded-[2.5rem] shadow-2xl w-full max-w-6xl min-h-[75vh] overflow-hidden" onClick={e => e.stopPropagation()}>
        <div className="px-8 py-6 border-b border-slate-800/80">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-slate-800 flex items-center justify-center text-[#90dce1]">
                <Volume2 size={18} />
              </div>
              <div className="flex flex-col">
                <span className="text-xs font-black uppercase tracking-widest text-slate-300">{title}</span>
                <span className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-600">{subtitle}</span>
              </div>
            </div>
            <button onClick={onClose} className="p-2 text-slate-400 hover:text-white transition-colors rounded-full hover:bg-slate-800/80">
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="px-8 py-10">
          <div className="max-w-4xl mx-auto">
            <AudioNodePreview
              audioData={modalData}
              onUpdate={(trimmed) => {
                onSave(trimmed);
                onClose();
              }}
              onRangeChange={onRangeChange}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
