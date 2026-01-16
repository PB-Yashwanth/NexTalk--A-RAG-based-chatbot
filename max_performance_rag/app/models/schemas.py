"""
Pydantic models for API requests and responses.
(Pydantic v2 compatible)
"""

from typing import Optional, List

from pydantic import BaseModel, Field, constr

# Literal fallback for older Python versions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


ChatMode = Literal["auto", "rag", "chat"]

UserId = constr(strip_whitespace=True, min_length=1, max_length=64)
DocId = constr(strip_whitespace=True, min_length=1, max_length=128)


class ChatRequest(BaseModel):
    """Incoming chat request."""
    message: str = Field(..., min_length=1, max_length=4000, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")

    user_id: Optional[UserId] = Field(default=None, description="User/session identifier")

    # Pydantic v2: enforce list size with Field(min_length/max_length)
    active_doc_ids: Optional[List[DocId]] = Field(
        default=None,
        min_length=1,
        max_length=25,
        description="List of uploaded doc_ids active in this chat session (per-session RAG)",
    )

    mode: ChatMode = Field(
        default="auto",
        description='Chat mode: "auto" (rag if relevant), "rag" (force rag), "chat" (normal chat only)',
    )


class SourceChunk(BaseModel):
    """A retrieved document chunk."""
    content: str
    source: str
    score: float
    page: Optional[int] = None
    doc_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response with answer and sources."""
    answer: str
    sources: List[SourceChunk]
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


class IngestResponse(BaseModel):
    """Response after uploading + indexing a document."""
    doc_id: str
    filename: str
    chunks_added: int