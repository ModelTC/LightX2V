import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Play, Pause, Scissors, RotateCcw, RefreshCw, Timer, Check, X } from 'lucide-react';

interface AudioDataEntry {
  original: string;
  trimmed: string;
  range: { start: number; end: number };
}

interface AudioNodePreviewProps {
  audioData: AudioDataEntry;
  onUpdate?: (trimmedBase64: string, range: { start: number; end: number }) => void;
  onRangeChange?: (range: { start: number; end: number }) => void;
  /** 只读播放（波形 + 进度），不显示裁剪等编辑入口 */
  readOnly?: boolean;
}

export const AudioNodePreview: React.FC<AudioNodePreviewProps> = ({ audioData, onUpdate, onRangeChange, readOnly = false }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isTrimMode, setIsTrimMode] = useState(false);
  const [range, setRange] = useState(audioData?.range || { start: 0, end: 100 });
  const [waveform, setWaveform] = useState<number[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Sync internal range state when prop changes (from focus mode updates)
  useEffect(() => {
    if (audioData?.range) {
      setRange(audioData.range);
    }
  }, [audioData?.range]);

  useEffect(() => {
    if (onRangeChange) {
      onRangeChange(range);
    }
  }, [range, onRangeChange]);

  useEffect(() => {
    const analyzeAudio = async () => {
      const src = audioData?.original;
      if (!src || typeof src !== 'string') {
        setDuration(0);
        setWaveform([]);
        return;
      }
      try {
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const response = await fetch(src, { credentials: 'same-origin' });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        if (arrayBuffer.byteLength === 0) {
          throw new Error('Empty response');
        }
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
        setDuration(audioBuffer.duration);

        const rawData = audioBuffer.getChannelData(0);
        const samples = 60;
        const blockSize = Math.floor(rawData.length / samples);
        const filteredData = [];

        for (let i = 0; i < samples; i++) {
          let blockStart = blockSize * i;
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum = sum + Math.abs(rawData[blockStart + j]);
          }
          filteredData.push(sum / (blockSize || 1));
        }

        const maxVal = Math.max(...filteredData);
        const multiplier = maxVal > 0 ? (1 / maxVal) : 1;
        setWaveform(filteredData.map(n => n * multiplier));

        audioCtx.close();
      } catch (e) {
        // 载入工作流时文件可能尚未解析、格式不被 decodeAudioData 支持（如 video/mp4）或需鉴权，降级为不显示波形
        console.warn("Audio analysis skipped or failed", e);
        setDuration(0);
        setWaveform([]);
      }
    };
    analyzeAudio();
  }, [audioData?.original]);

  const trimmedDuration = useMemo(() => {
    if (!duration) return 0;
    return ((range.end - range.start) / 100) * duration;
  }, [range, duration]);

  const [currentTime, setCurrentTime] = useState(0);
  const togglePlay = () => {
    if (!audioRef.current || !duration) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      const startTime = (range.start / 100) * duration;
      const endTime = (range.end / 100) * duration;
      if (audioRef.current.currentTime < startTime || audioRef.current.currentTime >= endTime) {
        audioRef.current.currentTime = startTime;
      }
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const bufferToWav = (buffer: AudioBuffer): Blob => {
    const numOfChan = buffer.numberOfChannels;
    const length = buffer.length * numOfChan * 2 + 44;
    const outBuffer = new ArrayBuffer(length);
    const view = new DataView(outBuffer);
    let pos = 0;
    const setUint32 = (data: number) => { view.setUint32(pos, data, true); pos += 4; };
    const setUint16 = (data: number) => { view.setUint16(pos, data, true); pos += 2; };
    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8);
    setUint32(0x45564157); // "WAVE"
    setUint32(0x20746d66); // "fmt "
    setUint32(16);
    setUint16(1);
    setUint16(numOfChan);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * numOfChan);
    setUint16(numOfChan * 2);
    setUint16(16);
    setUint32(0x61746164); // "data"
    setUint32(length - pos - 4);
    for (let i = 0; i < buffer.length; i++) {
      for (let channel = 0; channel < numOfChan; channel++) {
        let sample = buffer.getChannelData(channel)[i];
        sample = Math.max(-1, Math.min(1, sample));
        view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        pos += 2;
      }
    }
    return new Blob([view], { type: 'audio/wav' });
  };

  const handleApplyTrim = async () => {
    const src = audioData?.original;
    if (!src || typeof src !== 'string' || !duration) return;
    setIsProcessing(true);
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const response = await fetch(src, { credentials: 'same-origin' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
      const startOffset = Math.floor((range.start / 100) * audioBuffer.length);
      const endOffset = Math.floor((range.end / 100) * audioBuffer.length);
      const frameCount = endOffset - startOffset;
      const trimmedBuffer = audioCtx.createBuffer(audioBuffer.numberOfChannels, frameCount, audioBuffer.sampleRate);
      for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
        trimmedBuffer.getChannelData(i).set(audioBuffer.getChannelData(i).subarray(startOffset, endOffset));
      }
      const wavBlob = bufferToWav(trimmedBuffer);
      const reader = new FileReader();
      reader.onloadend = () => {
        onUpdate?.(reader.result as string, range);
        setIsProcessing(false);
        setIsTrimMode(false);
      };
      reader.readAsDataURL(wavBlob);
      audioCtx.close();
    } catch (e) {
      console.warn("Trimming failed", e);
      setIsProcessing(false);
    }
  };

  if (!audioData) return null;

  return (
    <div className={`p-4 rounded-[2rem] border transition-all duration-300 ${isTrimMode ? 'bg-[#90dce1]/5 border-[#90dce1]/30 shadow-[0_0_30px_rgba(144,220,225,0.05)]' : 'bg-slate-950/40 border-slate-800'}`}>
      {!isTrimMode ? (
        <div className="space-y-5">
          <div className="flex items-center gap-4">
            <button
              onClick={togglePlay}
              disabled={!waveform.length}
              className="w-12 h-12 rounded-full bg-[#90dce1] text-slate-950 flex items-center justify-center hover:scale-105 active:scale-95 transition-all shadow-lg shadow-[#90dce1]/20 disabled:opacity-30"
            >
              {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className="ml-1" />}
            </button>
            <div className="flex-1 flex flex-col justify-center">
              <div className="h-10 flex items-center gap-[2px] px-1 overflow-hidden">
                {waveform.length > 0 ? waveform.map((peak, i) => {
                  const pos = (i / waveform.length) * 100;
                  const isInRange = pos >= range.start && pos <= range.end;
                  return (
                    <div
                      key={i}
                      className={`flex-1 rounded-full transition-all duration-300 ${isInRange ? 'bg-[#90dce1]' : 'bg-slate-800'}`}
                      style={{ height: `${Math.max(10, peak * 100)}%` }}
                    />
                  );
                }) : <div className="w-full h-1 bg-slate-800 rounded-full animate-pulse" />}
              </div>
            </div>
          </div>
          <div className="bg-slate-900/40 rounded-2xl p-3 border border-slate-800/50 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-[#90dce1]/10 text-[#90dce1]">
                <Timer size={14} />
              </div>
              <div className="flex flex-col">
                <span className="text-[8px] font-black text-slate-500 uppercase tracking-widest">{readOnly ? 'Progress' : 'Active Duration'}</span>
                <span className="text-[11px] font-bold text-slate-200">
                  {readOnly ? `${currentTime.toFixed(1)}s / ${duration.toFixed(1)}s` : `${trimmedDuration.toFixed(2)}s`}
                </span>
              </div>
            </div>
            {!readOnly && (
              <button
                onClick={() => setIsTrimMode(true)}
                className="p-2 text-slate-500 hover:text-[#90dce1] transition-colors"
                title="Edit Trim"
              >
                <Scissors size={18} />
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-5 animate-in fade-in zoom-in-95 duration-300">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <Scissors size={14} className="text-[#90dce1]" />
              <span className="text-[10px] font-black uppercase text-[#90dce1] tracking-widest">Trim Mode</span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[11px] font-bold text-[#90dce1]">{trimmedDuration.toFixed(2)}s Selection</span>
            </div>
          </div>

          <div className="flex flex-col items-center gap-4">
            <button
              onClick={togglePlay}
              className="w-12 h-12 rounded-full bg-slate-800 text-[#90dce1] flex items-center justify-center hover:bg-slate-700 transition-all border border-[#90dce1]/20 shadow-xl"
            >
              {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className="ml-1" />}
            </button>
            <div className="w-full h-12 flex items-center gap-[2px] px-1 overflow-hidden">
              {waveform.map((peak, i) => {
                const pos = (i / waveform.length) * 100;
                const isInRange = pos >= range.start && pos <= range.end;
                return (
                  <div
                    key={i}
                    className={`flex-1 rounded-full transition-all duration-300 ${isInRange ? 'bg-[#90dce1]' : 'bg-slate-800'}`}
                    style={{ height: `${Math.max(10, peak * 100)}%` }}
                  />
                );
              })}
            </div>
          </div>

          <div className="relative h-14 flex flex-col justify-end px-1 pb-2 mt-4">
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

            <div className="relative h-1.5 flex items-center">
              <div className="absolute left-0 right-0 h-1.5 bg-slate-800 rounded-full" />
              <div
                className="absolute h-1.5 bg-[#90dce1] rounded-full shadow-[0_0_10px_rgba(144,220,225,0.4)]"
                style={{ left: `${range.start}%`, width: `${range.end - range.start}%` }}
              />
              <input
                type="range" value={range.start}
                onChange={e => setRange(r => ({ ...r, start: Math.min(r.end - 5, parseInt(e.target.value)) }))}
                className="absolute w-full appearance-none bg-transparent pointer-events-none custom-range-thumb z-10"
              />
              <input
                type="range" value={range.end}
                onChange={e => setRange(r => ({ ...r, end: Math.max(r.start + 5, parseInt(e.target.value)) }))}
                className="absolute w-full appearance-none bg-transparent pointer-events-none custom-range-thumb z-10"
              />
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleApplyTrim}
              disabled={isProcessing}
              className="flex-1 py-3 bg-[#90dce1] text-slate-950 text-[10px] font-black uppercase rounded-xl flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-95 disabled:opacity-50"
            >
              {isProcessing ? <RefreshCw size={14} className="animate-spin" /> : <Check size={14} />}
              {isProcessing ? 'Processing...' : 'Apply Selection'}
            </button>
            <button
              onClick={() => { setIsTrimMode(false); setRange(audioData?.range || {start:0, end:100}); }}
              disabled={isProcessing}
              className="px-4 py-3 bg-slate-800 text-slate-300 hover:text-white rounded-xl transition-all"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      <audio
        ref={audioRef}
        src={audioData?.original}
        onEnded={() => setIsPlaying(false)}
        onTimeUpdate={() => {
          if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
            if (!readOnly && audioRef.current.currentTime > (range.end / 100) * duration) {
              audioRef.current.pause();
              setIsPlaying(false);
            }
          }
        }}
        className="hidden"
      />
    </div>
  );
};
