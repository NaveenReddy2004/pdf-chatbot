import streamlit as st
import tempfile
import os
from preinstall_model import DocumentProcessor, RAGSystem

st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")
st.title("📚 PDF Chatbot with RAG")

doc_processor = DocumentProcessor()
rag_system = RAGSystem()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    result = doc_processor.process_document(file_path, uploaded_file.name)
    os.remove(file_path)

    if result["success"]:
        st.success(f"✅ Processed {uploaded_file.name} with {result['chunk_count']} chunks.")
        st.session_state["pdf_ready"] = True
    else:
        st.error(f"❌ Error: {result['error']}")

if st.session_state.get("pdf_ready", False):
    question = st.text_input("Ask a question about the PDF")
    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            chunks = rag_system.retrieve_relevant_chunks(question)
            answer = rag_system.generate_answer(question, chunks)

        st.markdown("### ✅ Answer")
        st.write(answer["answer"])
        st.markdown("#### 📁 Sources")
        st.write(answer["sources"])
        st.markdown("#### 📊 Confidence")
        st.progress(answer["confidence"])
