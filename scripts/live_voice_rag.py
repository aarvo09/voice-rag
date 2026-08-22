import os
import sys
import argparse
import time
import logging
import tempfile
import wave
import struct
import subprocess
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

from app.stt.sarvam import SarvamSTTProvider
from app.retrieval.metadata import CorpusMetadataLoader
from app.embeddings.model import MultilingualE5Embedder
from app.retrieval.faiss_index import FaissVectorIndex
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.retriever import ProductionRetriever, VectorRetriever
from app.generation.config import LLMConfig
from app.generation.llm import get_llm_provider
from app.guardrails.policy import GuardrailPolicy, GuardrailPolicyConfig
from app.pipeline.text_rag import TextRAGService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- AUDIO CAPTURE CONFIG ---
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 500  # Amplitude threshold for silence
SILENCE_DURATION = 1.5   # Seconds of silence to trigger stop
MIN_RECORD_DURATION = 1.0 # Minimum seconds to record

def record_until_silence():
    """Records audio from the microphone until silence is detected using arecord."""
    print("\nListening... (Speak now, and pause to finish)")
    
    # Start arecord as a subprocess
    process = subprocess.Popen(
        ['arecord', '-f', 'S16_LE', '-c', '1', '-r', '16000', '-t', 'raw', '-q'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    audio_data = []
    silent_chunks = 0
    chunk_size = 2048 # bytes (1024 samples of 16-bit)
    
    start_time = time.time()
    
    assert process.stdout is not None, "Failed to open process stdout"
    
    try:
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
                
            audio_data.append(chunk)
            
            # Calculate amplitude using struct
            # Each sample is 2 bytes (int16), so chunk_size // 2 samples
            samples = struct.unpack(f'<{len(chunk)//2}h', chunk)
            if samples:
                amplitude = max(abs(s) for s in samples)
            else:
                amplitude = 0
                
            if amplitude < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0
                
            # Check if we've been silent long enough
            total_duration = time.time() - start_time
            if total_duration > MIN_RECORD_DURATION:
                silence_seconds = silent_chunks * (chunk_size / 2 / SAMPLE_RATE)
                if silence_seconds >= SILENCE_DURATION:
                    break
    finally:
        process.terminate()
        process.wait()
                        
    print("Processing audio...")
    final_audio = b''.join(audio_data)
    
    # Save to temp wav file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(final_audio)
    
    return temp_file.name

def main():
    parser = argparse.ArgumentParser(description="Live Voice RAG with Sarvam STT and Groq")
    args = parser.parse_args()

    # 1. Initialize STT Provider
    logger.info("Initializing Sarvam STT Provider...")
    stt_provider = SarvamSTTProvider()
    
    # 2. Initialize Text RAG Pipeline
    logger.info("Initializing Text RAG components...")
    parquet_path = os.path.join(project_root, "data", "processed", "dev_corpus.parquet")
    faiss_path = os.path.join(project_root, "data", "indexes", "dev.faiss")
    bm25_path = os.path.join(project_root, "data", "indexes", "dev_bm25.pkl")
    
    metadata_loader = CorpusMetadataLoader(parquet_path)
    embedder = MultilingualE5Embedder()
    faiss_idx = FaissVectorIndex()
    faiss_idx.load(faiss_path)
    bm25_idx = BM25Retriever()
    bm25_idx.load(bm25_path)
    
    vector_retriever = VectorRetriever(
        embedder=embedder,
        faiss_index=faiss_idx,
        metadata_loader=metadata_loader
    )
    
    retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_idx,
        metadata_loader=metadata_loader
    )
    
    # Force Groq as requested
    os.environ["LLM_PROVIDER"] = "groq"
    llm_config = LLMConfig()
    llm_config.provider = "groq"
    provider = get_llm_provider(config=llm_config)
    
    # Keep max_grounding_retries at 0 (or 1) based on existing policy, but we can ensure it's 0 to be fast
    policy = GuardrailPolicy(config=GuardrailPolicyConfig(max_grounding_retries=0))
    
    rag_service = TextRAGService(retriever=retriever, provider=provider, policy=policy)
    
    logger.info("Initialization complete. Entering live Voice RAG loop.")
    
    try:
        while True:
            # Step 1: Capture Audio
            audio_path = record_until_silence()
            start_e2e = time.perf_counter()
            
            # Step 2: STT
            stt_result = stt_provider.transcribe(audio_path)
            
            # Clean up temp file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            if not stt_result.transcript.strip():
                print("No speech detected. Exiting.")
                break
                
            # Step 3: RAG
            lang = stt_result.detected_language
            lang_name = "Hindi" if "hi" in lang.lower() else "English"
            res = rag_service.run(stt_result.transcript, metadata={"language": lang_name})
            end_e2e = time.perf_counter()
            
            # Step 4: Display Output
            total_latency_ms = round((end_e2e - start_e2e) * 1000.0, 2)
            
            print("\n==================================================")
            print("LIVE VOICE RAG RESULT")
            print("==================================================")
            print(f"Transcript:         {stt_result.transcript}")
            print(f"Detected Language:  {stt_result.detected_language}")
            print(f"Status:             {res.get('status')}")
            print(f"Final Answer:       {res.get('answer')}")
            print(f"Citations:          {res.get('citations')}")
            print(f"Grounded:           {res.get('grounded')}")
            
            telemetry = res.get("telemetry", {})
            llm_called = telemetry.get('llm_ms', 0) > 0
            print(f"LLM Called:         {llm_called}")
            print("-" * 50)
            print("LATENCY BREAKDOWN:")
            print(f"  STT Latency:         {stt_result.latency_ms} ms")
            print(f"  Retrieval Latency:   {telemetry.get('retrieval_ms', 0)} ms")
            print(f"  Relevance Guard:     {telemetry.get('retrieval_guardrail_ms', 0)} ms")
            print(f"  LLM Called:          {llm_called}")
            print(f"  Groq LLM Latency:    {telemetry.get('llm_ms', 0)} ms")
            print(f"  Total RAG Latency:   {telemetry.get('total_ms', 0)} ms")
            print(f"  Total End-to-End:    {total_latency_ms} ms")
            print("==================================================\n")
            
            # Exit the loop after one successful query
            break
            
    except KeyboardInterrupt:
        print("\nExiting Live Voice RAG. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
