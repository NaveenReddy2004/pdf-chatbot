from preprocess import prepare_chunks
from embedding_api import get_embedding
from vector_store import VectorIndex
from groq_llm import GroqLLM

def build_vector_store_from_pdf(pdf_path):
    chunks = prepare_chunks(pdf_path)
    vector_store = VectorIndex(dim=384)

    for chunk in chunks:
        try:
            embedding = get_embedding(chunk)
            vector_store.add(chunk, embedding[0])
        except Exception as e:
            print("Skipping a chunk:", e)
    return vector_store

def query_rag_system(user_query, vector_store):
    query_embedding = get_embedding(user_query)[0]
    relevant_chunks = vector_store.search(query_embedding, top_k=3)

    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are a helpful assistant. Based on the following document content, answer the user's question:\n\n{context}\n\nQuestion: {user_query}\nAnswer:"""

    llm = GroqLLM()
    response = llm(prompt)
    return response, relevant_chunks
