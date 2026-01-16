"""
LLM Engine
Local LLM inference using Ollama.
"""

from __future__ import annotations

import ollama
from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL

_client = ollama.Client(host=OLLAMA_BASE_URL)


def generate_response(prompt: str, system_prompt: str | None = None) -> str:
    """
    Generate an assistant response using Ollama chat().

    Updates:
    - increase num_predict to reduce truncated answers (e.g., ending with "Porsche, .")
    - increase context window slightly
    - stabilize outputs (lower temperature)
    - add stop tokens to reduce role-leak artifacts
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = _client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "num_ctx": 2048,
                "num_predict": 512,
                "temperature": 0.4,
                "top_p": 0.9,
                "stop": ["\nUser:", "\nAssistant:", "\nSystem:"],
            },
            keep_alive="0s",
        )
        return response["message"]["content"]

    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower():
            return f"[ERROR] Cannot connect to Ollama. Make sure Ollama is running on {OLLAMA_BASE_URL}"
        return f"[ERROR] LLM generation failed: {error_msg}"


def check_ollama_status() -> dict:
    """Check if Ollama is running and whether configured model exists."""
    try:
        data = _client.list()

        models = data.get("models", data if isinstance(data, list) else [])
        available_models = []
        for m in models:
            name = None
            if isinstance(m, dict):
                name = m.get("name") or m.get("model")
            else:
                name = getattr(m, "name", None) or getattr(m, "model", None)

            if name:
                available_models.append(name)

        model_ready = OLLAMA_MODEL in available_models

        return {
            "status": "ok",
            "available_models": available_models,
            "configured_model": OLLAMA_MODEL,
            "model_ready": model_ready,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "configured_model": OLLAMA_MODEL,
            "model_ready": False,
        }