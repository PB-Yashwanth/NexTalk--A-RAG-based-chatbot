"""
Embedding Service
Converts text → vector embeddings using sentence-transformers.
"""

from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME

# Singleton pattern - load model once
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global _model
    if _model is None:
        print(f"[EmbeddingService] Loading model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[EmbeddingService] Model loaded successfully")
    return _model


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats (embedding vector)
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts at once (batch processing).
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist() 
