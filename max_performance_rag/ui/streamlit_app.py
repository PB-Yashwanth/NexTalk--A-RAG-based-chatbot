# ui/streamlit_app.py

import os
import time
import uuid
from typing import Any, Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv

# -------------------------------------------------
# ENV + CONFIG
# -------------------------------------------------
load_dotenv()

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").strip()
API_KEY = os.getenv("API_KEY", "").strip()

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

ENABLE_FAKE_STREAMING = True
STREAM_DELAY_SEC = 0.008

# -------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# -------------------------------------------------
st.set_page_config(page_title="NexTalk", layout="centered")

# -------------------------------------------------
# GLOBAL CSS (IMPORTANT – ONLY ONCE)
# -------------------------------------------------
st.markdown(
    """
<style>
/* -------- Base -------- */
html, body, [class*="css"] {
    background-color: #0E0E11;
    color: #FFFFFF;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}

/* -------- Upload Card -------- */
.nt-upload-card {
    background:#16161D;
    border:1px solid #2A2A35;
    border-radius:14px;
    padding:14px;
    margin: 12px 0 14px 0;
}
.nt-upload-title {
    color:#FFFFFF;
    font-size:14px;
    margin-bottom:6px;
}
.nt-upload-sub {
    color:#B3B3C6;
    font-size:12px;
}

/* -------- Sources -------- */
.nt-source-card {
    background:#16161D;
    border:1px solid #2A2A35;
    border-radius:12px;
    padding:12px;
    margin-bottom:10px;
}
.nt-source-top {
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:10px;
}
.nt-source-file {
    color:#FFFFFF;
    font-size:13px;
}
.nt-source-meta {
    color:#B3B3C6;
    font-size:12px;
    white-space:nowrap;
}
.nt-source-body {
    color:#B3B3C6;
    font-size:12.5px;
    margin-top:8px;
    line-height:1.45;
}
.nt-source-chip {
    display:inline-block;
    padding:2px 8px;
    border-radius:999px;
    background:#1E1E27;
    border:1px solid #2A2A35;
    color:#B3B3C6;
    font-size:12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# SPLASH (ONCE PER SESSION)
# -------------------------------------------------
def show_splash_loader(duration_sec: float = 1.0) -> None:
    holder = st.empty()
    holder.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:center;height:60vh;">
            <div style="color:#B3B3C6;font-size:14px;">Warming up NexTalk…</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(duration_sec)
    holder.empty()


if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = True
    show_splash_loader()

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
st.title("NexTalk")

if "user_id" not in st.session_state:
    st.session_state.user_id = f"user-{uuid.uuid4().hex[:8]}"
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc_ids" not in st.session_state:
    st.session_state.active_doc_ids = []

# -------------------------------------------------
# HTTP HELPERS
# -------------------------------------------------
def _headers_json() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _headers_plain() -> Dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def api_post_json(url: str, payload: dict, timeout: int = 300) -> requests.Response:
    return requests.post(url, json=payload, headers=_headers_json(), timeout=timeout)


def api_post_ingest(api_url: str, user_id: str, uploaded_file) -> requests.Response:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    data = {"user_id": user_id}
    return requests.post(f"{api_url}/ingest", headers=_headers_plain(), files=files, data=data, timeout=300)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.text_input("API URL", key="api_url")
    st.text_input("User ID", key="user_id")
    top_k = st.slider("top_k", 1, 10, 3)

    if st.button("Clear memory"):
        st.session_state.messages = []
        st.session_state.active_doc_ids = []
        st.rerun()

# -------------------------------------------------
# OPTIONAL UPLOAD (NOT REQUIRED)
# -------------------------------------------------
with st.expander("📎 Upload document (optional)", expanded=False):
    st.markdown(
        f"""
        <div class="nt-upload-card">
          <div class="nt-upload-title">
            Upload a document (TXT / PDF / DOCX • max {MAX_UPLOAD_MB}MB)
          </div>
          <div class="nt-upload-sub">
            Only needed if you want document-based answers.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload document",
        type=["txt", "pdf", "docx"],
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("Indexing document..."):
            resp = api_post_ingest(st.session_state.api_url, st.session_state.user_id, uploaded)

        if resp.status_code == 200:
            data = resp.json()
            doc_id = data.get("doc_id")
            if doc_id and doc_id not in st.session_state.active_doc_ids:
                st.session_state.active_doc_ids.append(doc_id)
            st.success(f"Indexed: {uploaded.name}")
        else:
            st.error("Failed to index document")

# -------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------------------------------
# CHAT INPUT (ALWAYS AVAILABLE)
# -------------------------------------------------
prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        payload = {
            "message": prompt,
            "user_id": st.session_state.user_id,
            "top_k": top_k,
            "active_doc_ids": st.session_state.active_doc_ids or None,
            "mode": "auto",
        }

        with st.spinner("Thinking..."):
            r = api_post_json(f"{st.session_state.api_url}/chat", payload)

        if r.status_code != 200:
            st.error("Chat failed")
        else:
            data = r.json()
            answer = data.get("answer", "")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
