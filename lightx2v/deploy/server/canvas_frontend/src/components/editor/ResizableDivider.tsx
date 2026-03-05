import React, { useRef, useCallback } from 'react';
import { GripVertical } from 'lucide-react';

interface ResizableDividerProps {
  onResize: (deltaY: number) => void;
  lang: 'en' | 'zh';
}

export const ResizableDivider: React.FC<ResizableDividerProps> = ({ onResize, lang }) => {
  const dividerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);
  const startYRef = useRef(0);
  const startHeightRef = useRef(0);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDraggingRef.current) return;

    const deltaY = e.clientY - startYRef.current;
    if (Math.abs(deltaY) > 0) {
      onResize(deltaY);
      startYRef.current = e.clientY;
    }
  }, [onResize]);

  const handleMouseUp = useCallback(() => {
    if (!isDraggingRef.current) return;

    isDraggingRef.current = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    isDraggingRef.current = true;
    startYRef.current = e.clientY;
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove, handleMouseUp]);

  return (
    <div
      ref={dividerRef}
      onMouseDown={handleMouseDown}
      className="flex items-center justify-center h-1 bg-slate-800/60 hover:bg-slate-700/80 cursor-row-resize transition-colors group relative z-40"
      title={lang === 'zh' ? '拖拽调整高度' : 'Drag to resize'}
    >
      <div className="absolute inset-0 flex items-center justify-center">
        <GripVertical
          size={12}
          className="text-slate-500 group-hover:text-slate-300 transition-colors"
        />
      </div>
    </div>
  );
};
