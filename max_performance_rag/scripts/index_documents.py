import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.rag.vector_store import get_vector_store

SUPPORTED_EXT = {".txt", ".pdf", ".docx"}


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        out = []
        for page in reader.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                out.append("")
        return "\n".join(out)

    if ext == ".docx":
        from docx import Document
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)

    return ""


def main():
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Recursive scan + file-only filter
    files = [
        f for f in KNOWLEDGE_BASE_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ]

    if not files:
        print(f"No documents found in: {KNOWLEDGE_BASE_DIR}")
        print("Add .txt/.pdf/.docx files then re-run.")
        return

    store = get_vector_store()

    # For shared KB, clearing avoids duplicate chunks on re-index
    store.clear()

    all_chunks = []
    for f in files:
        text = extract_text(f)
        if not text.strip():
            print(f"Skipping (no text): {f.name}")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for c in chunks:
            all_chunks.append({"content": c, "source": f.name})

        print(f"{f.name}: {len(chunks)} chunks")

    store.add_chunks(all_chunks)
    print(f"Indexed total chunks: {store.count()}")


if __name__ == "__main__":
    main()