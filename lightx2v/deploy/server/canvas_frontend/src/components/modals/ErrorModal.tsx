import React from 'react';
import { X, TriangleAlert } from 'lucide-react';
import { useTranslation, Language } from '../../i18n/useTranslation';

interface ErrorModalProps {
  error: { message: string; details?: string } | null;
  lang: Language;
  onClose: () => void;
}

export const ErrorModal: React.FC<ErrorModalProps> = ({ error, lang, onClose }) => {
  const { t } = useTranslation(lang);
  if (!error) return null;

  const message = (error.message === 'Fetch is aborted' ? t('fetch_aborted') : error.message);
  const details = (error.details === 'Fetch is aborted' ? t('fetch_aborted') : error.details);

  return (
    <div
      className="fixed inset-0 z-[120] flex items-center justify-center p-6 bg-slate-950/60 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 border border-red-500/30 rounded-[32px] shadow-2xl shadow-red-500/10 max-w-2xl w-full overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6 bg-red-500/10 border-b border-red-500/20 flex items-center gap-4">
          <div className="p-3 bg-red-500 rounded-2xl text-white shadow-lg">
            <TriangleAlert size={20} />
          </div>
          <div className="flex-1">
            <h2 className="text-sm font-black uppercase tracking-widest text-white">
              {lang === 'zh' ? '发生错误' : 'Error Occurred'}
            </h2>
            <p className="text-[10px] text-red-400/80 font-bold uppercase mt-1">
              {lang === 'zh'
                ? '应用遇到了一个未处理的错误'
                : 'An unhandled error occurred in the application'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>
        <div className="flex-1 p-6 overflow-y-auto max-h-[400px] custom-scrollbar space-y-4">
          <div className="space-y-2">
            <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest">
              {lang === 'zh' ? '错误信息' : 'Error Message'}
            </span>
            <div className="p-3 bg-red-500/5 border border-red-500/10 rounded-xl text-sm text-red-400 leading-relaxed font-medium">
              {message}
            </div>
          </div>
          {details && (
            <div className="space-y-2">
              <details>
                <summary className="text-[10px] text-slate-500 font-black uppercase tracking-widest cursor-pointer hover:text-slate-400">
                  {lang === 'zh' ? '详细信息' : 'Details'}
                </summary>
                <pre className="mt-2 p-3 bg-slate-950/50 border border-slate-800 rounded-xl text-xs text-slate-400 overflow-auto max-h-[200px] custom-scrollbar">
                  {details}
                </pre>
              </details>
            </div>
          )}
        </div>
        <div className="p-4 bg-slate-800/20 border-t border-slate-800 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-black uppercase tracking-widest transition-all"
          >
            {lang === 'zh' ? '关闭' : 'Close'}
          </button>
        </div>
      </div>
    </div>
  );
};
