# 🤖 NexTalk — A RAG-based Chatbot

NexTalk is a **Retrieval-Augmented Generation (RAG) based chatbot** capable of both **document-grounded question answering** and **normal conversational chat**.  
It combines **vector search** with **local LLM inference** to deliver accurate, context-aware responses.

---

## ✨ Features

- 🔍 Retrieval-Augmented Generation (RAG) for factual, document-based answers
- 💬 Can also act as a **normal conversational chatbot**
- 📚 Vector-based semantic search using **ChromaDB**
- 🧠 Local LLM inference using **Ollama**
- ⚡ Backend API built with **FastAPI**
- 🖥️ Interactive and simple **Streamlit UI**
- 🔒 Runs locally (no dependency on paid APIs)

---

## 🧠 Tech Stack

- **Language**: Python  
- **LLM**: Ollama  
- **Vector Database**: ChromaDB  
- **RAG Framework**: LlamaIndex  
- **Backend**: FastAPI  
- **Frontend / UI**: Streamlit  

---

## 🏗️ Architecture Overview

1. User inputs a query through the Streamlit UI  
2. Relevant documents are retrieved using **ChromaDB vector search**  
3. Retrieved context is passed to the LLM via **LlamaIndex**  
4. **Ollama** generates a grounded response  
5. FastAPI handles request–response flow  
6. The final answer is displayed in the UI  

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

- Python 3.9+
- Ollama installed and running  
- Basic understanding of Python & APIs  

---

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/PB-Yashwanth/NexTalk--A-RAG-based-chatbot.git
cd NexTalk--A-RAG-based-chatbot

pip install -r requirements.txt

uvicorn main:app --reload

streamlit run app.py

📌 Usage

Upload or index documents for knowledge-based Q&A

Ask factual questions based on indexed data

Use the chatbot for normal conversational interaction

Designed to minimize hallucinations using retrieval

📈 Future Improvements

Add authentication & session memory

Support multiple document formats (PDF, DOCX, URLs)

Improve UI/UX

Add deployment support (Docker / Cloud)

🎯 Learning Outcomes

Hands-on experience with RAG pipelines

Understanding of vector databases

Building AI systems end-to-end (data → model → UI)

Backend–frontend integration for AI applications

