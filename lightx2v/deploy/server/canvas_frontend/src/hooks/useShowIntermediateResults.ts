import { useState, useCallback } from 'react';

const STORAGE_KEY = 'canvas_show_intermediate_results';

function getStored(): boolean {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === 'false') return false;
    if (v === 'true') return true;
  } catch {
    // ignore
  }
  return true;
}

/** 是否显示中间节点结果，持久化在 localStorage，不依赖 workflow */
export function useShowIntermediateResults(): [
  boolean,
  (value: boolean | ((prev: boolean) => boolean)) => void
] {
  const [value, setValueState] = useState(getStored);
  const setValue = useCallback((v: boolean | ((prev: boolean) => boolean)) => {
    setValueState(prev => {
      const next = typeof v === 'function' ? v(prev) : v;
      try {
        localStorage.setItem(STORAGE_KEY, String(next));
      } catch {
        // ignore
      }
      return next;
    });
  }, []);
  return [value, setValue];
}
