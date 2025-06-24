import streamlit as st
import os
import tempfile

from rag_pipeline import build_vector_store_from_pdf, query_rag_system
from embedding_api import get_embedding

st.set_page_config(page_title="EduMedBot - PDF AI Assistant", page_icon="📘", layout="wide")
st.title("📘 EduMedBot – Understand Your PDFs with AI")
st.markdown("Upload your class notes, research papers, or reports and chat with an AI to explore their content.")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar PDF Upload
st.sidebar.header("📄 Upload Your PDF File")
pdf_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    st.sidebar.success("PDF uploaded!")
    with st.spinner("Processing the document..."):
        st.session_state.vector_store = build_vector_store_from_pdf(tmp_path)
    st.sidebar.success("📚 Document indexed successfully!")

# Chat Option Toggle
if st.session_state.vector_store:
    st.sidebar.markdown("---")
    response_style = st.sidebar.radio("🧠 Response Type", ["🤖 AI-Explained", "📄 Exact PDF Match"], horizontal=False)

    st.subheader("🧠 EduMedBot Chat History")
    if not st.session_state.chat_history:
        st.chat_message("assistant").markdown("👋 Hi! Ask me anything about your uploaded PDF.")

    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(item["question"])

        with st.chat_message("assistant"):
            if item["style"] == "ai":
                st.markdown(item["answer"])
                with st.expander("🔎 Source Chunks"):
                    for i, chunk in enumerate(item["context"]):
                        st.markdown(f"**Chunk {i+1}:**\n```text\n{chunk.strip()[:500]}\n```")
            else:
                st.markdown("📄 **Top Matching PDF Chunks:**")
                for i, chunk in enumerate(item["context"]):
                    st.code(chunk.strip(), language="text")

    # Chat Input at Bottom
    st.markdown("---")
    user_query = st.chat_input("Ask something about the PDF...")

    if user_query:
        with st.spinner("🤖 Generating answer..."):
            if response_style == "🤖 AI-Explained":
                answer, context = query_rag_system(user_query, st.session_state.vector_store)
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": answer,
                    "context": context,
                    "style": "ai"
                })
            else:
                query_embedding = get_embedding(user_query)
                if query_embedding:
                    chunks = st.session_state.vector_store.search(query_embedding[0], top_k=3)
                    st.session_state.chat_history.append({
                        "question": user_query,
                        "answer": None,
                        "context": chunks,
                        "style": "exact"
                    })
        st.rerun()

else:
    st.info("Please upload a PDF from the sidebar to get started.")

# Footer
st.markdown("---")
st.markdown("💡 Built using LangChain + FAISS + Groq + Cohere + Streamlit")
