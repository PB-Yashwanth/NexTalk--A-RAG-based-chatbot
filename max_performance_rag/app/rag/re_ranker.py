"""
Re-ranker
Optional re-ranking of retrieved chunks for better relevance.
"""

from sentence_transformers import CrossEncoder
from typing import Optional

# Singleton
_reranker: Optional[CrossEncoder] = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker() -> CrossEncoder:
    """Get or initialize the re-ranker model."""
    global _reranker
    if _reranker is None:
        print(f"[ReRanker] Loading model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        print("[ReRanker] Model loaded")
    return _reranker


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int | None = None
) -> list[dict]:
    """
    Re-rank chunks based on relevance to query.
    
    Args:
        query: The user's question
        chunks: Retrieved chunks to re-rank
        top_k: Number of top results to return (None = all)
        
    Returns:
        Re-ranked list of chunks
    """
    if not chunks:
        return []
    
    reranker = get_reranker()
    
    # Create query-chunk pairs
    pairs = [(query, chunk["content"]) for chunk in chunks]
    
    # Get relevance scores
    scores = reranker.predict(pairs)
    
    # Attach scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    
    # Sort by rerank score (descending)
    sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    # Return top_k if specified
    if top_k:
        return sorted_chunks[:top_k]
    
    return sorted_chunks