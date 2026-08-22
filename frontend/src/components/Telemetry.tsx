import type { Timings } from '../types';

interface TelemetryProps {
  timings: Timings;
}

export const Telemetry: React.FC<TelemetryProps> = ({ timings }) => {
  const localMs = timings.retrieval_ms + timings.grounding_ms;
  const remoteMs = timings.stt_ms + timings.generation_ms;

  return (
    <div className="mt-16 mb-8">
      <h3 className="text-3xl font-black uppercase tracking-tighter mb-6 border-b-2 border-hhgoa-border pb-2 inline-block">
        Pipeline Telemetry
      </h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="border-2 border-hhgoa-border p-4 bg-white relative">
          <div className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-1">STT</div>
          <div className="text-sm font-medium mb-2">Sarvam</div>
          <div className="text-2xl font-black text-hhgoa-brand">{timings.stt_ms.toFixed(0)} ms</div>
        </div>
        
        <div className="border-2 border-hhgoa-border p-4 bg-white relative">
          <div className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-1">Retrieval</div>
          <div className="text-sm font-medium mb-2">E5 + FAISS / BM25</div>
          <div className="text-2xl font-black">{timings.retrieval_ms.toFixed(1)} ms</div>
        </div>
        
        <div className="border-2 border-hhgoa-border p-4 bg-white relative">
          <div className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-1">Generation</div>
          <div className="text-sm font-medium mb-2">Groq</div>
          <div className="text-2xl font-black text-hhgoa-brand">{timings.generation_ms.toFixed(0)} ms</div>
        </div>
        
        <div className="border-2 border-hhgoa-border p-4 bg-white relative">
          <div className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-1">Grounding</div>
          <div className="text-sm font-medium mb-2">Validator</div>
          <div className="text-2xl font-black">{timings.grounding_ms.toFixed(1)} ms</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="border-t-2 border-hhgoa-border pt-4">
          <div className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-2">Local Pipeline</div>
          <div className="text-lg font-medium">Retrieval + Guardrails</div>
          <div className="text-3xl font-black mt-2">{localMs.toFixed(1)} ms</div>
        </div>
        
        <div className="border-t-2 border-hhgoa-border pt-4">
          <div className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-2">API / Network</div>
          <div className="text-lg font-medium">Sarvam + Groq</div>
          <div className="text-3xl font-black mt-2 text-hhgoa-brand">{remoteMs.toFixed(0)} ms</div>
        </div>
        
        <div className="border-t-4 border-hhgoa-border pt-4 bg-gray-100 px-4 pb-4 -mx-4 md:mx-0">
          <div className="text-xs font-bold uppercase tracking-widest text-gray-800 mb-2">Total End-to-End</div>
          <div className="text-lg font-medium">Total Latency</div>
          <div className="text-4xl font-black mt-2">{timings.total_e2e_ms.toFixed(0)} ms</div>
        </div>
      </div>
    </div>
  );
};
