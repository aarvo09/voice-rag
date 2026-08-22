import type { Timings, LatencySummary } from '../types';

interface TelemetryProps {
  timings: Timings;
  latencySummary?: LatencySummary | null;
}

export const Telemetry: React.FC<TelemetryProps> = ({ timings, latencySummary }) => {
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
          {timings.generation_ms === 0 && (
            <div className="absolute top-4 right-4 text-[10px] font-black uppercase text-red-600 border border-red-600 px-1 py-0.5">
              LLM Called: No
            </div>
          )}
        </div>
        
        <div className="border-2 border-hhgoa-border p-4 bg-white relative">
          <div className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-1">Grounding</div>
          <div className="text-sm font-medium mb-2">Validator</div>
          <div className="text-2xl font-black">{timings.grounding_ms.toFixed(1)} ms</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
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

      {latencySummary && latencySummary.sample_count > 0 && (
        <div className="border-t-4 border-hhgoa-border pt-8">
          <h3 className="text-2xl font-black uppercase tracking-tighter mb-2 inline-block">
            Latency Analytics
          </h3>
          <div className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-8 flex flex-col md:flex-row justify-between md:items-end border-b border-gray-200 pb-2 gap-4">
            <span>OBSERVED SAMPLES: {latencySummary.sample_count}</span>
            <span className="flex gap-4 md:gap-12 md:pr-4">
              <span className="w-16 text-right">P50</span>
              <span className="w-16 text-right">P70</span>
              <span className="w-16 text-right">P100</span>
            </span>
          </div>

          <div className="space-y-6">
            <div>
              <div className="text-xs font-bold uppercase tracking-widest text-hhgoa-brand mb-2">Remote API</div>
              <div className="flex flex-col md:flex-row justify-between md:items-center py-2 border-b border-gray-100 gap-2 md:gap-0">
                <span className="font-bold text-gray-800 uppercase text-sm">Sarvam STT</span>
                <span className="flex gap-4 md:gap-12 md:pr-4 font-medium">
                  <span className="w-16 text-right">{latencySummary.stt_ms.p50.toFixed(0)} ms</span>
                  <span className="w-16 text-right">{latencySummary.stt_ms.p70.toFixed(0)} ms</span>
                  <span className="w-16 text-right text-hhgoa-brand font-black">{latencySummary.stt_ms.p100.toFixed(0)} ms</span>
                </span>
              </div>
              <div className="flex flex-col md:flex-row justify-between md:items-center py-2 border-b border-gray-100 gap-2 md:gap-0">
                <span className="font-bold text-gray-800 uppercase text-sm">Groq Generation</span>
                <span className="flex gap-4 md:gap-12 md:pr-4 font-medium">
                  <span className="w-16 text-right">{latencySummary.generation_ms.p50.toFixed(0)} ms</span>
                  <span className="w-16 text-right">{latencySummary.generation_ms.p70.toFixed(0)} ms</span>
                  <span className="w-16 text-right text-hhgoa-brand font-black">{latencySummary.generation_ms.p100.toFixed(0)} ms</span>
                </span>
              </div>
            </div>

            <div>
              <div className="text-xs font-bold uppercase tracking-widest text-hhgoa-brand mb-2">Local Pipeline</div>
              <div className="flex flex-col md:flex-row justify-between md:items-center py-2 border-b border-gray-100 gap-2 md:gap-0">
                <span className="font-bold text-gray-800 uppercase text-sm">Retrieval</span>
                <span className="flex gap-4 md:gap-12 md:pr-4 font-medium">
                  <span className="w-16 text-right">{latencySummary.retrieval_ms.p50.toFixed(1)} ms</span>
                  <span className="w-16 text-right">{latencySummary.retrieval_ms.p70.toFixed(1)} ms</span>
                  <span className="w-16 text-right font-black">{latencySummary.retrieval_ms.p100.toFixed(1)} ms</span>
                </span>
              </div>
              <div className="flex flex-col md:flex-row justify-between md:items-center py-2 border-b border-gray-100 gap-2 md:gap-0">
                <span className="font-bold text-gray-800 uppercase text-sm">Grounding</span>
                <span className="flex gap-4 md:gap-12 md:pr-4 font-medium">
                  <span className="w-16 text-right">{latencySummary.grounding_ms.p50.toFixed(1)} ms</span>
                  <span className="w-16 text-right">{latencySummary.grounding_ms.p70.toFixed(1)} ms</span>
                  <span className="w-16 text-right font-black">{latencySummary.grounding_ms.p100.toFixed(1)} ms</span>
                </span>
              </div>
            </div>

            <div className="pt-2">
              <div className="text-xs font-bold uppercase tracking-widest text-hhgoa-brand mb-2">Overall</div>
              <div className="flex flex-col md:flex-row justify-between md:items-center py-2 border-b border-gray-100 bg-gray-50 px-2 -mx-2 gap-2 md:gap-0">
                <span className="font-bold text-gray-800 uppercase text-sm">Total RAG</span>
                <span className="flex gap-4 md:gap-12 md:pr-2 font-medium">
                  <span className="w-16 text-right">{latencySummary.total_rag_ms.p50.toFixed(0)} ms</span>
                  <span className="w-16 text-right">{latencySummary.total_rag_ms.p70.toFixed(0)} ms</span>
                  <span className="w-16 text-right font-black">{latencySummary.total_rag_ms.p100.toFixed(0)} ms</span>
                </span>
              </div>
              <div className="flex flex-col md:flex-row justify-between md:items-center py-3 border-b-2 border-hhgoa-border bg-gray-100 px-2 -mx-2 mt-2 gap-2 md:gap-0">
                <span className="font-bold text-gray-900 uppercase">Total End-to-End</span>
                <span className="flex gap-4 md:gap-12 md:pr-2 font-black text-lg">
                  <span className="w-16 text-right">{latencySummary.total_e2e_ms.p50.toFixed(0)} ms</span>
                  <span className="w-16 text-right">{latencySummary.total_e2e_ms.p70.toFixed(0)} ms</span>
                  <span className="w-16 text-right text-hhgoa-brand">{latencySummary.total_e2e_ms.p100.toFixed(0)} ms</span>
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
