import React, { useState, useRef } from 'react';
import { X } from 'lucide-react';
import { lightX2VVoiceClone } from '../../../services/geminiService';

interface CloneVoiceModalProps {
  isOpen: boolean;
  lightX2VConfig: { url: string; token: string };
  onClose: (newSpeakerId?: string) => void;
}

export const CloneVoiceModal: React.FC<CloneVoiceModalProps> = ({
  isOpen,
  lightX2VConfig,
  onClose
}) => {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioText, setAudioText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [voiceName, setVoiceName] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAudioFile(file);
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
    }
  };

  const handleClone = async () => {
    if (!audioFile) {
      alert('Please select an audio file first');
      return;
    }
    if (!lightX2VConfig.url || !lightX2VConfig.token) {
      alert('LightX2V configuration is missing');
      return;
    }

    setIsProcessing(true);
    try {
      // Convert file to base64
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(audioFile);
      });

      // Clone voice
      const cloneResult = await lightX2VVoiceClone(
        lightX2VConfig.url,
        lightX2VConfig.token,
        base64,
        audioText || undefined
      );
      const cloneData = JSON.parse(cloneResult);

      // Save voice with name
      if (voiceName.trim()) {
        const response = await fetch(`${lightX2VConfig.url.replace(/\/$/, '')}/api/v1/voice/clone/save`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${lightX2VConfig.token}`
          },
          body: JSON.stringify({
            speaker_id: cloneData.speaker_id,
            name: voiceName.trim()
          })
        });

        if (!response.ok) {
          throw new Error('Failed to save voice name');
        }
      }

      const newSpeakerId = cloneData.speaker_id;
      alert('Voice cloned successfully!');
      onClose(newSpeakerId);
    } catch (error: any) {
      console.error('Clone error:', error);
      alert(`Clone failed: ${error.message || 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[150] flex items-center justify-center p-4"
      onClick={() => onClose()}
    >
      <div
        className="relative w-full max-w-2xl max-h-[90vh] bg-slate-900/95 backdrop-blur-[40px] border border-slate-800/60 rounded-3xl shadow-2xl overflow-hidden flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60">
          <h3 className="text-xl font-semibold text-slate-200">Clone Voice</h3>
          <button
            onClick={() => onClose()}
            className="w-9 h-9 flex items-center justify-center bg-slate-800/80 border border-slate-700 text-slate-400 hover:text-slate-200 rounded-full transition-all"
          >
            <X size={16} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Audio File</label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*,video/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full p-6 border-2 border-dashed border-slate-700 rounded-xl hover:border-[#90dce1] transition-all text-slate-400 hover:text-slate-200"
                >
                  {audioFile ? audioFile.name : 'Click to select audio or video file'}
                </button>
                {audioUrl && <audio src={audioUrl} controls className="w-full mt-2 rounded-lg" />}
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Audio Text (Optional)</label>
                <textarea
                  value={audioText}
                  onChange={e => setAudioText(e.target.value)}
                  placeholder="Enter the text content of the audio for better accuracy..."
                  className="w-full h-24 bg-slate-800 border border-slate-700 rounded-xl p-3 text-sm text-slate-200 placeholder-slate-500 focus:border-[#90dce1] focus:outline-none resize-none"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Voice Name (Optional)</label>
                <input
                  type="text"
                  value={voiceName}
                  onChange={e => setVoiceName(e.target.value)}
                  placeholder="Give this voice a name..."
                  className="w-full bg-slate-800 border border-slate-700 rounded-xl p-3 text-sm text-slate-200 placeholder-slate-500 focus:border-[#90dce1] focus:outline-none"
                />
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => onClose()}
                className="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 text-slate-300 rounded-xl hover:bg-slate-700 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleClone}
                disabled={!audioFile || isProcessing}
                className="flex-1 px-4 py-3 bg-[#90dce1] text-white rounded-xl hover:bg-[#7dd3da] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isProcessing ? 'Processing...' : 'Clone Voice'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
