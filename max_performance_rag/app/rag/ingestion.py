from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
import uuid
import logging

from pypdf import PdfReader
import docx  # python-docx

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks


def _decode_text(data: bytes) -> str:
    # utf-8 first; ignore broken bytes
    return data.decode("utf-8", errors="ignore")


def extract_text_from_upload(filename: str, data: bytes) -> str:
    name = (filename or "").lower().strip()

    if name.endswith(".txt") or name.endswith(".md"):
        return _decode_text(data)

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()

    if name.endswith(".docx"):
        d = docx.Document(BytesIO(data))
        return "\n".join([p.text for p in d.paragraphs if (p.text or "").strip()]).strip()

    raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx")


def extract_pages_from_upload(filename: str, data: bytes) -> List[Tuple[int, str]]:
    name = (filename or "").lower().strip()

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        pages: List[Tuple[int, str]] = []
        total_pages = len(reader.pages)
        logger.info(f"[Ingestion] PDF has {total_pages} pages")

        for i, page in enumerate(reader.pages, start=1):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append((i, txt))
            else:
                logger.warning(f"[Ingestion] Page {i} has no extractable text (possibly scanned/image)")

        logger.info(f"[Ingestion] Extracted text from {len(pages)} pages")
        return pages

    full = extract_text_from_upload(filename, data)
    return [(1, full)] if full else []


def build_chunks_for_store(
    *,
    filename: str,
    data: bytes,
    user_id: str,
    doc_id: Optional[str] = None,
    chunk_size: int = 900,
    overlap: int = 120,
) -> Tuple[str, List[Dict[str, Any]]]:
    doc_id = doc_id or str(uuid.uuid4())

    chunks: List[Dict[str, Any]] = []
    pages = extract_pages_from_upload(filename, data)

    for page_num, page_text in pages:
        for c in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                {
                    "content": c,
                    "source": filename,
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "page": page_num,
                }
            )

    logger.info(f"[Ingestion] Created {len(chunks)} chunks from {len(pages)} pages for {filename}")
    return doc_id, chunks