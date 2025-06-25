# 📘 EduMedBot – RAG-Based AI Chatbot for Understanding PDFs

**EduMedBot** is a real-time, AI-powered chatbot built using Retrieval-Augmented Generation (RAG) that helps users understand content from research papers, class notes, and medical reports in natural language. It mimics ChatGPT-like behavior while answering only based on the uploaded PDF content.

Built using **LangChain**, **FAISS**, **Streamlit**, and **LLMs (Groq/Cohere)**, this project showcases a complete end-to-end implementation of a smart, scalable, and interactive GenAI application.

---

## Live Demo

🔗 [Try the App on Streamlit Cloud](https://pdf-chatbot-ic4lfmqkkb6ndc5hwyg3jk.streamlit.app/)

---

## 💡 Features

- Upload any PDF file (notes, papers, reports)
- RAG-based response generation using vector similarity search
- Two response modes:
  - AI-Explained (LLM generates answer using PDF context)
  - Exact Match (Top chunks from your PDF)
- Powered by Cohere for embeddings and Groq LLMs for answers
- Chat-style interface with contextual memory (like ChatGPT)
- Source chunk viewing with expand/collapse toggles
- Streamlit UI – lightweight, clean, responsive

---

## 🧱 Tech Stack

| Category           | Tools/Frameworks Used                         |
|--------------------|-----------------------------------------------|
| Embedding Model    | [Cohere API](https://cohere.com)              |
| LLM (RAG Answer)   | [Groq API – Mixtral/LLaMA3](https://groq.com) |
| Vector Search      | FAISS (Facebook AI Similarity Search)         |
| RAG Framework      | LangChain                                     |
| PDF Parsing        | pdfplumber                                    |
| Frontend           | Streamlit                                     |
| Secrets Handling   | Streamlit Secrets / .env                      |

---

## 📁 Project Structure

pdf-chatbot/
├── app.py # Streamlit UI
├── rag_pipeline.py # Embedding, retrieval, RAG query logic
├── embedding_api.py # Embedding via Cohere
├── groq_llm.py # LLM wrapper (Groq API)
├── preprocess.py # Chunking and PDF text extraction
├── vector_store.py # FAISS-based vector store
├── requirements.txt # Python dependencies
└── .env or secrets.toml # API keys for local/Streamlit Cloud

🙌 Acknowledgements
   LangChain
   FAISS
   Cohere
   Groq
   Streamlit
