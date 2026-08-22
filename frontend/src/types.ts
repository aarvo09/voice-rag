export interface Timings {
  stt_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  grounding_ms: number;
  total_rag_ms: number;
  total_e2e_ms: number;
}

export interface LatencyMetric {
  p50: number;
  p70: number;
  p100: number;
}

export interface LatencySummary {
  sample_count: number;
  updated_at: number;
  source: string;
  stt_ms: LatencyMetric;
  retrieval_ms: LatencyMetric;
  generation_ms: LatencyMetric;
  grounding_ms: LatencyMetric;
  total_rag_ms: LatencyMetric;
  total_e2e_ms: LatencyMetric;
}

export interface Document {
  document_id: string;
  score?: number;
  text?: string;
  source?: string;
  language?: string;
}

export interface VoiceQueryResponse {
  transcript: string;
  language: string;
  answer: string | null;
  grounded: boolean;
  citations: string[];
  status: string;
  timings: Timings;
  retrieved_documents: Document[];
}
