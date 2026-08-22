import os
import sys
import logging
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

from app.retrieval.metadata import CorpusMetadataLoader
from app.embeddings.model import MultilingualE5Embedder
from app.retrieval.faiss_index import FaissVectorIndex
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.retriever import ProductionRetriever, VectorRetriever

logging.basicConfig(level=logging.INFO)

def main():
    parquet_path = os.path.join(project_root, "data", "processed", "dev_corpus.parquet")
    faiss_path = os.path.join(project_root, "data", "indexes", "dev.faiss")
    bm25_path = os.path.join(project_root, "data", "indexes", "dev_bm25.pkl")
    
    metadata_loader = CorpusMetadataLoader(parquet_path)
    embedder = MultilingualE5Embedder()
    faiss_idx = FaissVectorIndex()
    faiss_idx.load(faiss_path)
    bm25_idx = BM25Retriever()
    bm25_idx.load(bm25_path)
    
    vector_retriever = VectorRetriever(embedder, faiss_idx, metadata_loader)
    retriever = ProductionRetriever(vector_retriever, bm25_idx, metadata_loader)
    
    # Test A: In domain
    res_a = retriever.retrieve("What were the consequences of the Manhattan Project?")
    
    # Test B: Out of domain
    res_b = retriever.retrieve("Who won the 2022 FIFA World Cup?")
    
    print("\n--- TEST A ---")
    conf = res_a.get("confidence", {})
    results = res_a.get("results", [])
    print(f"Confidence score (Dense Top-1): {conf.get('confidence_score')}")
    if len(results) > 1:
        print(f"Top 2: {results[1].get('score')}")
    print(f"Results len: {len(results)}")
    
    print("\n--- TEST B ---")
    conf = res_b.get("confidence", {})
    results = res_b.get("results", [])
    print(f"Confidence score (Dense Top-1): {conf.get('confidence_score')}")
    if len(results) > 1:
        print(f"Top 2: {results[1].get('score')}")
    print(f"Results len: {len(results)}")

if __name__ == '__main__':
    main()
