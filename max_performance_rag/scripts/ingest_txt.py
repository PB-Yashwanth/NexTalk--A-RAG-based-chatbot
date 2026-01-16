from pathlib import Path
import argparse

from app.rag.vector_store import get_vector_store

DOCS_DIR = Path(__file__).resolve().parents[1] / "knowledge_base" / "documents"

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def main(reset: bool):
    store = get_vector_store()
    if reset:
        store.clear()

    if not DOCS_DIR.exists():
        raise SystemExit(f"Documents folder not found: {DOCS_DIR}")

    added = 0
    for fp in DOCS_DIR.glob("*.txt"):
        content = fp.read_text(encoding="utf-8", errors="ignore")
        for ch in chunk_text(content):
            store.add_chunks([{"content": ch, "source": fp.name}])
            added += 1

    print(f"Done. Added chunks: {added}. Total chunks in DB: {store.count()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Clear existing Chroma collection first")
    args = ap.parse_args()
    main(reset=args.reset)