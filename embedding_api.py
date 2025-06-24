import os
import cohere
from dotenv import load_dotenv
load_dotenv()

# Load from Streamlit secrets (if available)
try:
    import streamlit as st
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
except:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

cohere_client = cohere.Client(COHERE_API_KEY)

def get_embedding(text: str):
    print(f" Embedding input: {text[:50]}")
    
    if not text or len(text.strip()) < 5:
        print(" Text too short to embed.")
        return []
    
    try:
        response = cohere_client.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        print(" Cohere API returned embedding.")
        return [response.embeddings[0]]
    except Exception as e:
        print(" Cohere Embedding Error:", e)
        return []
