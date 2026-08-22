import type { VoiceQueryResponse } from '../types';
import { CheckCircle, XCircle } from 'lucide-react';

interface ResultsProps {
  result: VoiceQueryResponse;
}

export const Results: React.FC<ResultsProps> = ({ result }) => {
  const isRefused = result.status === 'refused_insufficient';

  return (
    <div className="flex flex-col gap-12 mt-12">
      {/* Transcript */}
      <section>
        <div className="text-xs uppercase font-bold tracking-widest text-hhgoa-brand mb-4 flex items-center gap-4">
          YOU SAID
          <span className="bg-hhgoa-border text-hhgoa-bg px-2 py-1 text-[10px]">
            {result.language === 'en-IN' ? 'ENGLISH' : result.language === 'hi-IN' ? 'हिन्दी' : result.language}
          </span>
        </div>
        <h2 className="text-3xl md:text-5xl font-black tracking-tight leading-tight">
          "{result.transcript}"
        </h2>
      </section>

      {/* Answer */}
      <section className="border-4 border-hhgoa-border bg-white p-8 md:p-12 shadow-[8px_8px_0px_0px_rgba(23,23,23,1)]">
        <div className="text-xs uppercase font-bold tracking-widest mb-8 border-b-2 border-hhgoa-border pb-4 flex justify-between items-end">
          <span className="text-2xl">ANSWER</span>
          {isRefused ? (
            <div className="flex flex-col items-end">
              <span className="text-red-600 flex items-center gap-2"><XCircle size={16} /> NO RELEVANT CONTEXT</span>
              <span className="text-gray-500 mt-1">LLM SKIPPED</span>
            </div>
          ) : (
            <div className="flex flex-col items-end">
              <span className="text-green-600 flex items-center gap-2"><CheckCircle size={16} /> GROUNDED ✓</span>
            </div>
          )}
        </div>
        
        <p className={`text-2xl md:text-4xl font-medium leading-relaxed ${isRefused ? 'text-gray-500 italic' : 'text-gray-900'}`}>
          {isRefused 
            ? "The provided knowledge base does not contain enough evidence to answer this question."
            : result.answer}
        </p>
        
        {!isRefused && result.citations && result.citations.length > 0 && (
          <div className="mt-12 pt-8 border-t-2 border-dashed border-gray-300">
            <div className="text-xs uppercase font-bold tracking-widest text-gray-500 mb-4">CITATIONS</div>
            <div className="flex flex-wrap gap-2">
              {result.citations.map((cite, idx) => (
                <span key={idx} className="bg-gray-100 border border-gray-300 px-3 py-1 text-sm font-mono">
                  {cite}
                </span>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Evidence */}
      {result.retrieved_documents && result.retrieved_documents.length > 0 && (
        <section className="mt-8">
          <h3 className="text-2xl font-black uppercase tracking-tighter mb-6">THE EVIDENCE</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {result.retrieved_documents.slice(0, 5).map((doc, idx) => (
              <div key={idx} className="border-2 border-hhgoa-border p-6 bg-white flex flex-col h-full">
                <div className="flex justify-between items-start mb-4">
                  <span className="text-xl font-black text-hhgoa-brand">{(idx + 1).toString().padStart(2, '0')}</span>
                  <div className="text-right">
                    <div className="text-xs font-mono bg-gray-100 px-2 py-1">{doc.document_id}</div>
                    <div className="text-[10px] uppercase font-bold text-gray-500 mt-1">Score: {doc.score?.toFixed(4)}</div>
                  </div>
                </div>
                <p className="text-sm text-gray-700 font-medium leading-relaxed line-clamp-4 flex-grow mb-4">
                  "{doc.text}"
                </p>
                <div className="text-[10px] uppercase font-bold tracking-widest border-t border-gray-200 pt-3 text-gray-400">
                  {doc.source || 'Knowledge Base'} • {doc.language || 'Unknown'}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
};
