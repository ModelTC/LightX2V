import React from 'react';
import { TriangleAlert } from 'lucide-react';
import { useTranslation, Language } from '../../i18n/useTranslation';

interface ValidationError {
  message: string;
  type: 'ENV' | 'INPUT';
}

interface ValidationModalProps {
  errors: ValidationError[];
  lang: Language;
  onClose: () => void;
}

export const ValidationModal: React.FC<ValidationModalProps> = ({
  errors,
  lang,
  onClose
}) => {
  const { t } = useTranslation(lang);

  if (errors.length === 0) return null;

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center p-6 bg-slate-950/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-red-500/30 rounded-[32px] shadow-2xl shadow-red-500/10 max-w-md w-full overflow-hidden flex flex-col">
        <div className="p-6 bg-red-500/10 border-b border-red-500/20 flex items-center gap-4">
          <div className="p-3 bg-red-500 rounded-2xl text-white shadow-lg">
            <TriangleAlert size={20} />
          </div>
          <div>
            <h2 className="text-sm font-black uppercase tracking-widest text-white">
              {t('validation_failed')}
            </h2>
            <p className="text-[10px] text-red-400/80 font-bold uppercase">
              {t('fix_validation')}
            </p>
          </div>
        </div>
        <div className="flex-1 p-6 overflow-y-auto max-h-[400px] custom-scrollbar space-y-4">
          {errors.some((e) => e.type === 'ENV') && (
            <div className="space-y-2">
              <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest">
                {t('env_vars')}
              </span>
              <div className="p-3 bg-red-500/5 border border-red-500/10 rounded-xl text-xs text-red-400 leading-relaxed font-medium">
                {errors.find((e) => e.type === 'ENV')?.message}
              </div>
            </div>
          )}
          {errors.filter((e) => e.type === 'INPUT').length > 0 && (
            <div className="space-y-2">
              <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest">
                {t('missing_inputs_msg')}
              </span>
              <div className="space-y-1.5">
                {errors
                  .filter((e) => e.type === 'INPUT')
                  .map((err, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 text-xs text-slate-400 font-medium"
                    >
                      <div className="w-1.5 h-1.5 rounded-full bg-red-500"></div>
                      {err.message}
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
        <div className="p-4 bg-slate-800/20 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-black uppercase tracking-widest transition-all"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
