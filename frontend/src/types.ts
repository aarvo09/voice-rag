export interface Timings {
  stt_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  grounding_ms: number;
  total_rag_ms: number;
  total_e2e_ms: number;
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
