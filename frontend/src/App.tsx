import { useState } from 'react';
import { Hero } from './components/Hero';
import { Microphone } from './components/Microphone';
import { Telemetry } from './components/Telemetry';
import { Results } from './components/Results';
import type { VoiceQueryResponse } from './types';

function App() {
  const [result, setResult] = useState<VoiceQueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleResult = (newResult: VoiceQueryResponse) => {
    setResult(newResult);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setResult(null);
  };

  const handleClear = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen p-6 md:p-12 lg:p-24 selection:bg-hhgoa-brand selection:text-white relative pb-32">
      {/* Decorative Marks */}
      <div className="absolute top-12 right-12 w-32 h-32 border-t-4 border-r-4 border-hhgoa-border opacity-20 pointer-events-none hidden md:block"></div>
      <div className="absolute bottom-12 left-12 w-24 h-24 border-b-4 border-l-4 border-hhgoa-border opacity-20 pointer-events-none hidden md:block"></div>
      
      <div className="max-w-6xl mx-auto relative z-10">
        <Hero />
        
        <Microphone 
          onResult={handleResult} 
          onError={handleError}
          onClear={handleClear}
        />

        {error && (
          <div className="mt-8 border-l-4 border-red-600 bg-red-50 p-6 shadow-sm">
            <h3 className="text-red-800 font-bold uppercase tracking-widest text-xs mb-2">Error</h3>
            <p className="text-red-900 font-medium text-lg">{error}</p>
          </div>
        )}

        {result && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
            <Results result={result} />
            <Telemetry timings={result.timings} />
          </div>
        )}
      </div>

      <footer className="absolute bottom-8 right-12 text-xs font-bold uppercase tracking-widest text-gray-400">
        BUILT FOR HH GOA 2026 • LESS NOISE. MORE SIGNAL.
      </footer>
    </div>
  );
}

export default App;
