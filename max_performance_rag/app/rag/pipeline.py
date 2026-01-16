from __future__ import annotations
from typing import Optional, Dict, Any, List, Literal

from app.config import RAG_SCORE_THRESHOLD
from app.rag.vector_store import get_vector_store
from app.rag.retrieval_chain import build_rag_prompt, build_chat_prompt, get_system_prompt
from app.rag.llm_engine import generate_response
from app.rag.memory_manager import get_chat_history, add_exchange

ChatMode = Literal["auto", "rag", "chat"]

_DOC_KEYWORDS = (
    "knowledge base", "kb", "document", "documents", "docs", "pdf", "docx", "txt",
    "file", "files", "source", "sources", "context", "uploaded", "upload",
    "summarize", "summary", "from the document", "from this file", "in the file"
)


def _is_doc_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in _DOC_KEYWORDS)


def get_answer(
    question: str,
    top_k: int = 5,
    user_id: Optional[str] = None,
    use_reranker: bool = False,
    mode: ChatMode = "auto",
    active_doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    store = get_vector_store()

    # Normalize
    user_id = user_id or "anonymous"
    if mode not in ("auto", "rag", "chat"):
        mode = "auto"

    # defensive clamp
    try:
        top_k = int(top_k or 5)
    except Exception:
        top_k = 5
    top_k = max(1, min(top_k, 20))

    if not active_doc_ids:
        active_doc_ids = None

    chat_history = get_chat_history(user_id)

    # --- CHAT MODE ---
    if mode == "chat":
        prompt = build_chat_prompt(question=question, chat_history=chat_history)
        answer = generate_response(prompt=prompt, system_prompt=get_system_prompt())
        add_exchange(user_id, question, answer)
        return {"answer": answer, "sources": []}

    # --- Retrieve only if active docs exist ---
    retrieved: List[dict] = []
    if active_doc_ids:
        retrieved = store.search(
            query=question,
            top_k=top_k,
            user_id=user_id,
            doc_ids=active_doc_ids,
        )

    # --- Forced RAG requires uploaded docs ---
    if mode == "rag" and not active_doc_ids:
        answer = "No documents uploaded for this session. Please upload a document and ask again."
        add_exchange(user_id, question, answer)
        return {"answer": answer, "sources": []}

    if mode == "rag" and not retrieved:
        answer = "I searched your uploaded documents but couldn't find anything relevant."
        add_exchange(user_id, question, answer)
        return {"answer": answer, "sources": []}

    # Auto mode + no retrieval => normal chat
    if mode == "auto" and not retrieved:
        prompt = build_chat_prompt(question=question, chat_history=chat_history)
        answer = generate_response(prompt=prompt, system_prompt=get_system_prompt())
        add_exchange(user_id, question, answer)
        return {"answer": answer, "sources": []}

    # --- Compute top score ---
    top_score = 0.0
    if retrieved:
        try:
            top_score = float(retrieved[0].get("score", 0.0))
        except Exception:
            top_score = 0.0

    doc_question = _is_doc_question(question)

    # Decide whether to use RAG
    if mode == "rag":
        use_rag = True
    else:
        # auto: use RAG if strong similarity OR clearly doc-oriented
        use_rag = (bool(retrieved) and top_score >= RAG_SCORE_THRESHOLD) or (bool(retrieved) and doc_question)

    # If user is asking doc-style but similarity is weak, ask for clarification rather than hallucinate.
    if mode == "auto" and active_doc_ids and doc_question and (not retrieved or top_score < RAG_SCORE_THRESHOLD):
        answer = (
            "I can answer from your uploaded documents, but I couldn't confidently find the relevant section.\n\n"
            "Try one of these:\n"
            "- Ask using exact keywords from the document\n"
            "- Mention which page/section it’s in\n"
            "- Or paste the paragraph you want me to use"
        )
        add_exchange(user_id, question, answer)
        return {"answer": answer, "sources": []}

    if use_rag:
        prompt = build_rag_prompt(question=question, context_chunks=retrieved, chat_history=chat_history)
        sources = [
            {
                "content": ch.get("content", ""),
                "source": ch.get("source", "Unknown"),
                "score": float(ch.get("score", 0.0)),
                "page": ch.get("page"),
                "doc_id": ch.get("doc_id"),
            }
            for ch in retrieved[:top_k]
        ]
    else:
        prompt = build_chat_prompt(question=question, chat_history=chat_history)
        sources = []

    answer = generate_response(prompt=prompt, system_prompt=get_system_prompt())
    add_exchange(user_id, question, answer)

    return {"answer": answer, "sources": sources}