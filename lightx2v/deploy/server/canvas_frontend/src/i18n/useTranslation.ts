import { useCallback } from 'react';
import { TRANSLATIONS } from './translations';

export type Language = 'en' | 'zh';

export const useTranslation = (lang: Language) => {
  const t = useCallback((key: string): string => {
    return TRANSLATIONS[lang]?.[key] || key;
  }, [lang]);

  return { t };
};
