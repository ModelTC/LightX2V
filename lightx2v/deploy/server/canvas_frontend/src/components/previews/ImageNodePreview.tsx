import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Crop, Check, X, Plus, Trash2 } from 'lucide-react';

interface ImageEntry {
  source: string;
  original: string;
  cropped: string;
  crop_box: { x: number; y: number; w: number; h: number };
}

interface ImageNodePreviewProps {
  images: ImageEntry[];
  onUpdate: (newImages: ImageEntry[]) => void;
  onAddMore: () => void;
}

type DragMode = 'move' | 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se' | null;

export const ImageNodePreview: React.FC<ImageNodePreviewProps> = ({ images, onUpdate, onAddMore }) => {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isCropMode, setIsCropMode] = useState(false);
  const [crop_box, setCropBox] = useState({ x: 10, y: 10, w: 80, h: 80 });
  const [dragMode, setDragMode] = useState<DragMode>(null);

  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const startPosRef = useRef({ x: 0, y: 0, box: { x: 0, y: 0, w: 0, h: 0 } });

  // Sync internal crop_box state when images prop changes
  useEffect(() => {
    const current = images[selectedIndex];
    if (current && current.crop_box) {
      setCropBox(current.crop_box);
    }
  }, [selectedIndex, images]);

  const currentEntry = images[selectedIndex];

  const handleApplyCrop = () => {
    if (!imgRef.current || !currentEntry) return;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = imgRef.current;
    const realX = (crop_box.x / 100) * img.naturalWidth;
    const realY = (crop_box.y / 100) * img.naturalHeight;
    const realW = (crop_box.w / 100) * img.naturalWidth;
    const realH = (crop_box.h / 100) * img.naturalHeight;

    canvas.width = realW;
    canvas.height = realH;
    ctx.drawImage(img, realX, realY, realW, realH, 0, 0, realW, realH);

    const croppedData = canvas.toDataURL('image/png');
    const nextImages = [...images];
    nextImages[selectedIndex] = {
      ...currentEntry,
      cropped: croppedData,
      crop_box: { ...crop_box }
    };
    onUpdate(nextImages);
    setIsCropMode(false);
  };

  const removeImage = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const nextImages = images.filter((_, i) => i !== index);
    onUpdate(nextImages);
    if (selectedIndex >= nextImages.length) {
      setSelectedIndex(Math.max(0, nextImages.length - 1));
    }
  };

  const onMouseDown = (e: React.MouseEvent, mode: DragMode) => {
    e.stopPropagation();
    e.preventDefault();
    startPosRef.current = {
      x: e.clientX,
      y: e.clientY,
      box: { ...crop_box }
    };
    setDragMode(mode);
  };

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragMode || !containerRef.current) return;

    const start = startPosRef.current;
    const rect = containerRef.current.getBoundingClientRect();
    const dx = ((e.clientX - start.x) / rect.width) * 100;
    const dy = ((e.clientY - start.y) / rect.height) * 100;

    setCropBox(prev => {
      let { x, y, w, h } = start.box;
      const minSize = 5;

      switch (dragMode) {
        case 'move':
          x = Math.max(0, Math.min(100 - w, x + dx));
          y = Math.max(0, Math.min(100 - h, y + dy));
          break;
        case 'n':
          const nDiff = Math.min(dy, h - minSize);
          y = Math.max(0, y + nDiff);
          h = h - (y - start.box.y);
          break;
        case 's':
          h = Math.max(minSize, Math.min(100 - y, h + dy));
          break;
        case 'w':
          const wDiff = Math.min(dx, w - minSize);
          x = Math.max(0, x + wDiff);
          w = w - (x - start.box.x);
          break;
        case 'e':
          w = Math.max(minSize, Math.min(100 - x, w + dx));
          break;
        case 'nw':
          const nwYDiff = Math.min(dy, h - minSize);
          const nwXDiff = Math.min(dx, w - minSize);
          y = Math.max(0, y + nwYDiff);
          x = Math.max(0, x + nwXDiff);
          h = h - (y - start.box.y);
          w = w - (x - start.box.x);
          break;
        case 'ne':
          const neYDiff = Math.min(dy, h - minSize);
          y = Math.max(0, y + neYDiff);
          h = h - (y - start.box.y);
          w = Math.max(minSize, Math.min(100 - x, w + dx));
          break;
        case 'sw':
          const swXDiff = Math.min(dx, w - minSize);
          x = Math.max(0, x + swXDiff);
          w = w - (x - start.box.x);
          h = Math.max(minSize, Math.min(100 - y, h + dy));
          break;
        case 'se':
          w = Math.max(minSize, Math.min(100 - x, w + dx));
          h = Math.max(minSize, Math.min(100 - y, h + dy));
          break;
      }

      return { x, y, w, h };
    });
  }, [dragMode]);

  const onMouseUp = useCallback(() => {
    setDragMode(null);
  }, []);

  useEffect(() => {
    if (dragMode) {
      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [dragMode, onMouseMove, onMouseUp]);

  if (!currentEntry) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 overflow-x-auto pb-2 custom-scrollbar">
        {images.map((entry, idx) => (
          <div
            key={`${entry.source}-${idx}`}
            onClick={() => { setSelectedIndex(idx); setIsCropMode(false); }}
            className={`relative group shrink-0 w-12 h-12 rounded-lg border-2 overflow-hidden cursor-pointer transition-all ${selectedIndex === idx ? 'border-[#90dce1] ring-2 ring-[#90dce1]/20' : 'border-slate-800 opacity-60 hover:opacity-100'}`}
          >
            {entry.cropped ? <img src={entry.cropped} className="w-full h-full object-cover" alt="" /> : <div className="w-full h-full bg-slate-800" />}
            <button
              onClick={(e) => removeImage(idx, e)}
              className="absolute top-0 right-0 p-0.5 bg-red-500/80 text-white opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <Trash2 size={10} />
            </button>
          </div>
        ))}
        <button
          onClick={onAddMore}
          className="shrink-0 w-12 h-12 rounded-lg border-2 border-dashed border-slate-800 flex items-center justify-center text-slate-600 hover:text-[#90dce1] hover:border-[#90dce1]/50 transition-all"
        >
          <Plus size={16} />
        </button>
      </div>

      <div
        ref={containerRef}
        className="relative rounded-xl overflow-hidden border border-slate-800 bg-slate-900/50 select-none group"
      >
        {(isCropMode ? currentEntry.original : currentEntry.cropped) ? (
        <>
        <img
          ref={imgRef}
          src={isCropMode ? currentEntry.original : currentEntry.cropped}
          alt="Preview"
          className="w-full h-auto block pointer-events-none"
        />

        {isCropMode && (
          <div className="absolute inset-0 bg-black/40">
            <div
              className="absolute border border-[#90dce1] shadow-[0_0_0_9999px_rgba(0,0,0,0.5)] cursor-move group/crop"
              style={{
                left: `${crop_box.x}%`,
                top: `${crop_box.y}%`,
                width: `${crop_box.w}%`,
                height: `${crop_box.h}%`
              }}
              onMouseDown={(e) => onMouseDown(e, 'move')}
            >
              <div className="absolute -top-1.5 -left-1.5 w-3 h-3 bg-white border border-[#90dce1] rounded-full cursor-nw-resize z-10" onMouseDown={(e) => onMouseDown(e, 'nw')} />
              <div className="absolute -top-1.5 -right-1.5 w-3 h-3 bg-white border border-[#90dce1] rounded-full cursor-ne-resize z-10" onMouseDown={(e) => onMouseDown(e, 'ne')} />
              <div className="absolute -bottom-1.5 -left-1.5 w-3 h-3 bg-white border border-[#90dce1] rounded-full cursor-sw-resize z-10" onMouseDown={(e) => onMouseDown(e, 'sw')} />
              <div className="absolute -bottom-1.5 -right-1.5 w-3 h-3 bg-white border border-[#90dce1] rounded-full cursor-se-resize z-10" onMouseDown={(e) => onMouseDown(e, 'se')} />

              <div className="absolute top-0 left-0 right-0 h-1 cursor-n-resize" onMouseDown={(e) => onMouseDown(e, 'n')} />
              <div className="absolute bottom-0 left-0 right-0 h-1 cursor-s-resize" onMouseDown={(e) => onMouseDown(e, 's')} />
              <div className="absolute top-0 bottom-0 left-0 w-1 cursor-w-resize" onMouseDown={(e) => onMouseDown(e, 'w')} />
              <div className="absolute top-0 bottom-0 right-0 w-1 cursor-e-resize" onMouseDown={(e) => onMouseDown(e, 'e')} />

              <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 opacity-30 pointer-events-none">
                <div className="border-r border-b border-white/40" />
                <div className="border-r border-b border-white/40" />
                <div className="border-b border-white/40" />
                <div className="border-r border-b border-white/40" />
                <div className="border-r border-b border-white/40" />
                <div className="border-b border-white/40" />
                <div className="border-r border-white/40" />
                <div className="border-r border-white/40" />
                <div />
              </div>
            </div>
          </div>
        )}

        {!isCropMode && (
          <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => setIsCropMode(true)}
              className="p-1.5 bg-slate-900/80 backdrop-blur-md rounded-lg text-white hover:text-[#90dce1] border border-white/10"
            >
              <Crop size={14} />
            </button>
          </div>
        )}
        </>
        ) : (
          <div className="w-full min-h-24 bg-slate-800 flex items-center justify-center text-slate-500 text-sm">No image</div>
        )}
      </div>

      {isCropMode ? (
        <div className="flex gap-2 animate-in slide-in-from-top-1">
          <button
            onClick={handleApplyCrop}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-[#90dce1] text-slate-950 rounded-xl text-[10px] font-black uppercase transition-transform active:scale-95"
          >
            <Check size={14} /> Save Crop
          </button>
          <button
            onClick={() => setIsCropMode(false)}
            className="px-4 bg-slate-800 text-white rounded-xl hover:bg-slate-700 transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      ) : (
        <div className="flex justify-between items-center px-1">
          <span className="text-[8px] font-black text-slate-600 uppercase tracking-widest">
            {images.length} Image{images.length !== 1 ? 's' : ''} Loaded
          </span>
          <button
            onClick={() => setIsCropMode(true)}
            className="text-[8px] font-black text-[#90dce1] uppercase hover:text-white transition-colors"
          >
            Adjust Mask
          </button>
        </div>
      )}
    </div>
  );
};
