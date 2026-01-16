"""
Test script for RAG pipeline.
Run this to verify everything works.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.pipeline import get_answer
from app.rag.vector_store import get_vector_store
from app.rag.llm_engine import check_ollama_status


def test_system():
    """Run system tests."""
    print("=" * 50)
    print("NexTalk System Test")
    print("=" * 50)
    
    # Test 1: Ollama status
    print("\n[1] Checking Ollama...")
    status = check_ollama_status()
    print(f"    Status: {status['status']}")
    print(f"    Model Ready: {status.get('model_ready', False)}")
    
    if not status.get("model_ready"):
        print(f"\n    ⚠️  Model '{status['configured_model']}' not found!")
        print(f"    Run: ollama pull {status['configured_model']}")
    
    # Test 2: Vector store
    print("\n[2] Checking Vector Store...")
    store = get_vector_store()
    print(f"    Chunks in store: {store.count()}")
    
    if store.count() == 0:
        print("    ⚠️  No documents indexed! Add documents first.")
    
    # Test 3: Query (only if we have chunks)
    if store.count() > 0:
        print("\n[3] Testing Query...")
        result = get_answer("What is this document about?", top_k=3)
        print(f"    Answer preview: {result['answer'][:100]}...")
        print(f"    Sources found: {len(result['sources'])}")
    
    print("\n" + "=" * 50)
    print("Test Complete")
    print("=" * 50)


if __name__ == "__main__":
    test_system()