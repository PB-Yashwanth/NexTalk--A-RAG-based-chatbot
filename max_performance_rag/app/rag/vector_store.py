"""
Vector Store (ChromaDB)
Persistent vector database for document chunk storage + retrieval.

We keep a single Chroma collection, but store metadata (user_id, doc_id, source, page)
so retrieval can be filtered per user session (ChatGPT-like uploads).

Telemetry disabled.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import uuid
import logging

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_DIR, CHROMA_COLLECTION
from app.rag.embedding_service import embed_text, embed_texts

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        documents = [c["content"] for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = embed_texts(documents)

        metadatas: List[Dict[str, Any]] = []
        for c in chunks:
            meta: Dict[str, Any] = {"source": c.get("source", "Unknown")}
            if c.get("user_id") is not None:
                meta["user_id"] = c["user_id"]
            if c.get("doc_id") is not None:
                meta["doc_id"] = c["doc_id"]
            if c.get("page") is not None:
                meta["page"] = c["page"]
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.count() == 0:
            logger.warning("[VectorStore/Chroma] Collection is empty")
            return []

        q_emb = embed_text(query)

        # Build Chroma 'where' filter
        # ChromaDB requires $and for multiple conditions
        where = None
        if user_id and doc_ids:
            if len(doc_ids) == 1:
                where = {
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"doc_id": {"$eq": doc_ids[0]}}
                    ]
                }
            else:
                where = {
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"doc_id": {"$in": doc_ids}}
                    ]
                }
        elif user_id:
            where = {"user_id": {"$eq": user_id}}
        elif doc_ids:
            if len(doc_ids) == 1:
                where = {"doc_id": {"$eq": doc_ids[0]}}
            else:
                where = {"doc_id": {"$in": doc_ids}}

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        results: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            if dist is None:
                score = 0.0
            else:
                try:
                    score = float(1.0 / (1.0 + float(dist)))
                except Exception:
                    score = 0.0

            meta = meta or {}
            results.append(
                {
                    "content": doc,
                    "source": meta.get("source", "Unknown"),
                    "score": score,
                    "page": meta.get("page"),
                    "doc_id": meta.get("doc_id"),
                }
            )

        return results

    def clear(self) -> None:
        try:
            self.client.delete_collection(name=CHROMA_COLLECTION)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION)
        logger.info("[VectorStore/Chroma] Cleared collection")

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0


_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store