import streamlit as st
import os
import tempfile

from rag_pipeline import build_vector_store_from_pdf, query_rag_system

st.set_page_config(page_title="EduMed AI Chatbot", page_icon="📘", layout="wide")
st.title("EduMed AI Chatbot")
st.markdown("Helping students understand PDFs, research papers, and reports using Groq-powered AI.")

# Initialize session
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload
st.sidebar.header("Upload Your PDF")
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if pdf_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    st.sidebar.success("PDF uploaded successfully!")
    with st.spinner("Indexing PDF... Please wait..."):
        st.session_state.vector_store = build_vector_store_from_pdf(tmp_path)
    st.sidebar.success("Document indexed!")

# Chat Interface 
if st.session_state.vector_store:
    st.subheader("💬 Ask Questions About Your PDF")
    response_style = st.radio("Choose response type:", ["🤖 AI-Explained", "📄 Exact PDF Text"], horizontal=True)

    user_query = st.text_input("Enter your question here:", key="input")

    if user_query:
        with st.spinner("🧠 Processing..."):
            if response_style == "🤖 AI-Explained":
                answer, context = query_rag_system(user_query, st.session_state.vector_store)
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": answer,
                    "context": context,
                    "style": "ai"
                })
            else:
                query_embedding = get_embedding(user_query)[0]
                chunks = st.session_state.vector_store.search(query_embedding, top_k=3)
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": None,
                    "context": chunks,
                    "style": "exact"
                })


# Display Chat History
for item in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑‍🎓 You:** {item['question']}")

    if item['style'] == "ai":
        st.markdown(f"**EduMedBot (AI Explained):** {item['answer']}")
        with st.expander("Source Chunks"):
            for i, chunk in enumerate(item["context"]):
                st.markdown(f"**Chunk {i+1}:**\n```text\n{chunk.strip()[:500]}\n```")
    else:
        st.markdown(f"**Exact PDF Chunks (Top Matches):**")
        for i, chunk in enumerate(item["context"]):
            st.code(chunk.strip(), language="text")

# Footer
st.markdown("---")
st.markdown("💡 Built using LangChain + Groq API + FAISS + Streamlit")


