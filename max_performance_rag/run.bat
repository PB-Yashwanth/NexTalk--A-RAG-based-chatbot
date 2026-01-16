import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle
import threading
import hashlib

class VectorStore:
    """Working vector store for your RAG system"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = []
        self.lock = threading.RLock()
        print(f"Vector Store ready (dim={embedding_dim})")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the knowledge base"""
        if not documents:
            return
            
        with self.lock:
            # Create simple embeddings
            for doc in documents:
                # Deterministic embedding from hash
                seed = int(hashlib.md5(doc.encode()).hexdigest()[:8], 16)
                np.random.seed(seed)
                embedding = np.random.randn(self.embedding_dim).astype('float32')
                embedding = embedding / np.linalg.norm(embedding)
                
                self.documents.append(doc)
                self.embeddings.append(embedding)
            
            print(f"✓ Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for relevant documents"""
        if not query or not self.documents:
            return []
            
        with self.lock:
            # Create query embedding
            seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            query_embedding = np.random.randn(self.embedding_dim).astype('float32')
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Calculate similarities
            results = []
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = float(np.dot(query_embedding, doc_embedding))
                if similarity > 0.3:  # Threshold
                    results.append((
                        self.documents[i],
                        similarity,
                        {"id": i, "length": len(self.documents[i])}
                    ))
            
            # Sort and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

# Create global instance
vector_store = VectorStore()