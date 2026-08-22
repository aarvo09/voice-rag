import { useState, useRef } from 'react';
import axios from 'axios';
import { Mic, Loader2, StopCircle } from 'lucide-react';
import type { VoiceQueryResponse } from '../types';

interface MicrophoneProps {
  onResult: (result: VoiceQueryResponse) => void;
  onError: (error: string) => void;
  onClear: () => void;
}

export const Microphone: React.FC<MicrophoneProps> = ({ onResult, onError, onClear }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const startRecording = async () => {
    try {
      onClear();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        await processAudio(audioBlob);
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error(err);
      onError("Microphone permission denied or unavailable.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');

      const response = await axios.post<VoiceQueryResponse>('http://localhost:8000/api/voice-query', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      onResult(response.data);
    } catch (err: any) {
      console.error(err);
      if (err.response && err.response.data && err.response.data.detail) {
        onError(err.response.data.detail);
      } else {
        onError("Backend offline or request failed.");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="my-12">
      <div className="flex flex-col items-start gap-4">
        {!isRecording && !isProcessing ? (
          <button
            onClick={startRecording}
            className="group relative flex items-center justify-center gap-4 bg-hhgoa-border text-hhgoa-bg hover:bg-hhgoa-brand transition-colors duration-300 px-8 py-6 rounded-none font-bold text-xl uppercase tracking-wider shadow-[4px_4px_0px_0px_rgba(234,88,12,1)]"
          >
            <Mic size={32} />
            <span>Talk to the Corpus</span>
          </button>
        ) : isRecording ? (
          <button
            onClick={stopRecording}
            className="group relative flex items-center justify-center gap-4 bg-red-600 text-white hover:bg-red-700 transition-colors duration-300 px-8 py-6 rounded-none font-bold text-xl uppercase tracking-wider shadow-[4px_4px_0px_0px_rgba(23,23,23,1)] animate-pulse"
          >
            <StopCircle size={32} />
            <span>Listening... (Click to Stop)</span>
          </button>
        ) : (
          <button
            disabled
            className="group relative flex items-center justify-center gap-4 bg-gray-200 text-gray-500 px-8 py-6 rounded-none font-bold text-xl uppercase tracking-wider cursor-not-allowed border-2 border-gray-300"
          >
            <Loader2 size={32} className="animate-spin" />
            <span>Processing</span>
          </button>
        )}
        
        <div className="text-xs font-bold uppercase tracking-widest text-gray-500 ml-1">
          HINT: English / हिन्दी
        </div>
      </div>

      {isProcessing && (
        <div className="mt-8 border-2 border-hhgoa-border p-4 bg-white">
          <div className="text-xs uppercase font-bold tracking-widest mb-4">Pipeline Status</div>
          <div className="flex items-center gap-2 text-sm font-medium uppercase overflow-x-auto whitespace-nowrap">
            <span className="text-hhgoa-brand">Voice</span>
            <span>→</span>
            <span className="text-hhgoa-brand animate-pulse">STT</span>
            <span>→</span>
            <span>Retrieve</span>
            <span>→</span>
            <span>Generate</span>
            <span>→</span>
            <span>Ground</span>
          </div>
        </div>
      )}
    </div>
  );
};
