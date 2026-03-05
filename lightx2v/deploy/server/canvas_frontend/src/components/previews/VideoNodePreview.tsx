import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Scissors, Check, RotateCcw, X, Timer } from 'lucide-react';

interface VideoNodePreviewProps {
  videoUrl: string;
  onUpdate: (start: number, end: number, trimmedUrl?: string) => void;
  onRangeChange?: (start: number, end: number) => void;
  initialStart?: number;
  initialEnd?: number;
}

export const VideoNodePreview: React.FC<VideoNodePreviewProps> = ({ videoUrl, onUpdate, onRangeChange, initialStart, initialEnd }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [isTrimMode, setIsTrimMode] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [range, setRange] = useState({ start: 0, end: 100 });
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (duration > 0) {
      const s = initialStart !== undefined ? (initialStart / duration) * 100 : 0;
      const e = initialEnd !== undefined ? (initialEnd / duration) * 100 : 100;
      setRange({ start: s, end: e });
    }
  }, [initialStart, initialEnd, duration]);

  const notifyRangeChange = (nextRange: { start: number; end: number }) => {
    if (!onRangeChange || duration <= 0) return;
    const startSeconds = (nextRange.start / 100) * duration;
    const endSeconds = (nextRange.end / 100) * duration;
    onRangeChange(startSeconds, endSeconds);
  };

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.onloadedmetadata = () => {
        const d = videoRef.current?.duration || 0;
        setDuration(d);
      };
    }
  }, [videoUrl]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) videoRef.current.pause();
    else {
      const startTime = (range.start / 100) * duration;
      const endTime = (range.end / 100) * duration;
      if (videoRef.current.currentTime < startTime || videoRef.current.currentTime >= endTime) {
        videoRef.current.currentTime = startTime;
      }
      videoRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const seekTo = (percent: number) => {
    if (videoRef.current && duration) {
      videoRef.current.currentTime = (percent / 100) * duration;
    }
  };

  const handleApplyTrim = async () => {
    const startSeconds = (range.start / 100) * duration;
    const endSeconds = (range.end / 100) * duration;
    if (!videoRef.current || duration <= 0) {
      onUpdate(startSeconds, endSeconds);
      setIsTrimMode(false);
      return;
    }

    const canRecord = typeof MediaRecorder !== 'undefined' && typeof (videoRef.current as any).captureStream === 'function';
    if (!canRecord) {
      onUpdate(startSeconds, endSeconds);
      setIsTrimMode(false);
      return;
    }

    try {
      setIsProcessing(true);
      const stream = (videoRef.current as any).captureStream();
      const preferredMime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : MediaRecorder.isTypeSupported('video/webm')
        ? 'video/webm'
        : '';
      const recorder = preferredMime ? new MediaRecorder(stream, { mimeType: preferredMime }) : new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
        const reader = new FileReader();
        reader.onloadend = () => {
          onUpdate(startSeconds, endSeconds, reader.result as string);
          setIsProcessing(false);
          setIsTrimMode(false);
        };
        reader.readAsDataURL(blob);
      };

      const stopAt = endSeconds;
      const onTimeUpdate = () => {
        if (!videoRef.current) return;
        if (videoRef.current.currentTime >= stopAt) {
          videoRef.current.pause();
          videoRef.current.removeEventListener('timeupdate', onTimeUpdate);
          if (recorder.state !== 'inactive') {
            recorder.stop();
          }
        }
      };

      videoRef.current.currentTime = startSeconds;
      videoRef.current.addEventListener('timeupdate', onTimeUpdate);
      recorder.start();
      await videoRef.current.play();
    } catch (err) {
      console.error('Video trim failed', err);
      onUpdate(startSeconds, endSeconds);
      setIsProcessing(false);
      setIsTrimMode(false);
    }
  };

  return (
    <div className={`p-4 rounded-[2rem] border transition-all duration-300 ${isTrimMode ? 'bg-[#90dce1]/5 border-[#90dce1]/30' : 'bg-slate-950/40 border-slate-800'}`}>
      <div className="space-y-4">
        <div className="relative group rounded-2xl overflow-hidden border border-slate-800 bg-black aspect-video shadow-2xl">
          <video
            ref={videoRef}
            src={videoUrl}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onTimeUpdate={() => {
              if (videoRef.current && videoRef.current.currentTime > (range.end / 100) * duration) {
                videoRef.current.pause();
                setIsPlaying(false);
              }
            }}
            className="w-full h-full object-contain"
          />

          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all bg-black/40">
            <button
              onClick={togglePlay}
              className="w-14 h-14 rounded-full bg-white/10 backdrop-blur-xl border border-white/20 text-white flex items-center justify-center hover:scale-110 transition-transform shadow-2xl"
            >
              {isPlaying ? <Pause size={28} fill="white" /> : <Play size={28} fill="white" className="ml-1" />}
            </button>
          </div>

          <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 backdrop-blur-md rounded-lg text-[8px] font-black text-slate-300 uppercase tracking-widest">
            {((videoRef.current?.currentTime || 0)).toFixed(1)}s / {duration.toFixed(1)}s
          </div>
        </div>

        {!isTrimMode ? (
          <div className="bg-slate-900/40 rounded-2xl p-3 border border-slate-800/50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-[#90dce1]/10 text-[#90dce1]">
                <Timer size={14} />
              </div>
              <div className="flex flex-col">
                <span className="text-[8px] font-black text-slate-500 uppercase tracking-widest">Clip Length</span>
                <span className="text-[11px] font-bold text-slate-200">
                  {(((range.end - range.start) / 100) * duration).toFixed(1)}s
                </span>
              </div>
            </div>
            <button
              onClick={() => setIsTrimMode(true)}
              className="p-2 text-slate-500 hover:text-[#90dce1] transition-colors"
              title="Edit Trim"
            >
              <Scissors size={18} />
            </button>
          </div>
        ) : (
          <div className="space-y-4 animate-in fade-in slide-in-from-top-2 duration-300">
            <div className="flex items-center justify-between">
              <div className="flex-1" />
              <button
                onClick={togglePlay}
                className="w-12 h-12 flex items-center justify-center bg-slate-800 text-[#90dce1] border border-[#90dce1]/30 rounded-full hover:bg-slate-700 transition-all shadow-xl"
                title="Preview Selection"
              >
                {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className="ml-0.5" />}
              </button>
              <div className="flex-1 flex flex-col items-end">
                <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Selection</span>
                <span className="text-[10px] font-bold text-[#90dce1]">
                  {(((range.end - range.start) / 100) * duration).toFixed(1)}s
                </span>
              </div>
            </div>

            <div className="relative h-14 flex flex-col justify-end px-1 pb-2">
              <div
                className="absolute -top-1 pointer-events-none transition-all duration-75"
                style={{ left: `calc(${range.start}% + 4px)`, transform: 'translateX(-50%)' }}
              >
                <div className="bg-[#90dce1] text-slate-950 text-[9px] font-black px-1.5 py-0.5 rounded-md shadow-lg whitespace-nowrap">
                  {((range.start / 100) * duration).toFixed(1)}s
                </div>
                <div className="w-1.5 h-1.5 bg-[#90dce1] rotate-45 mx-auto -mt-1" />
              </div>

              <div
                className="absolute -top-1 pointer-events-none transition-all duration-75"
                style={{ left: `calc(${range.end}% + 4px)`, transform: 'translateX(-50%)' }}
              >
                <div className="bg-slate-200 text-slate-950 text-[9px] font-black px-1.5 py-0.5 rounded-md shadow-lg whitespace-nowrap">
                  {((range.end / 100) * duration).toFixed(1)}s
                </div>
                <div className="w-1.5 h-1.5 bg-slate-200 rotate-45 mx-auto -mt-1" />
              </div>

              <div className="relative h-2 flex items-center">
                <div className="absolute w-full h-2 bg-slate-900 border border-slate-800 rounded-full" />
                <div
                  className="absolute h-2 bg-[#90dce1] rounded-full shadow-[0_0_15px_rgba(144,220,225,0.3)]"
                  style={{ left: `${range.start}%`, width: `${range.end - range.start}%` }}
                />
                <input
                  type="range" value={range.start}
                  onChange={e => {
                    const val = parseInt(e.target.value);
                    setRange(r => {
                      const next = { ...r, start: Math.min(r.end - 5, val) };
                      notifyRangeChange(next);
                      return next;
                    });
                    seekTo(val);
                  }}
                  className="absolute w-full appearance-none bg-transparent custom-range-thumb z-10"
                />
                <input
                  type="range" value={range.end}
                  onChange={e => {
                    const val = parseInt(e.target.value);
                    setRange(r => {
                      const next = { ...r, end: Math.max(r.start + 5, val) };
                      notifyRangeChange(next);
                      return next;
                    });
                    seekTo(val);
                  }}
                  className="absolute w-full appearance-none bg-transparent custom-range-thumb z-10"
                />
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleApplyTrim}
                disabled={isProcessing}
                className="flex-1 py-3 bg-[#90dce1] text-slate-950 text-[10px] font-black uppercase rounded-xl flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-95 shadow-xl shadow-[#90dce1]/10 disabled:opacity-60"
              >
                <Check size={14} /> {isProcessing ? 'Processing...' : 'Apply Trim'}
              </button>
              <button
                onClick={() => setIsTrimMode(false)}
                className="px-4 py-3 bg-slate-800 text-slate-300 hover:text-white rounded-xl transition-colors"
              >
                <X size={16} />
              </button>
              <button
                onClick={() => { setRange({ start: 0, end: 100 }); seekTo(0); }}
                className="p-3 bg-slate-900 border border-slate-800 text-slate-500 hover:text-white rounded-xl transition-colors"
              >
                <RotateCcw size={16} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
