import os
import cohere
from dotenv import load_dotenv
load_dotenv()

try:
    import streamlit as st
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
except:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

cohere_client = cohere.Client(COHERE_API_KEY)

def get_embedding(text: str):
    if not text or len(text.strip()) < 5:
        print("Skipping embedding: input text too short")
        return []
    try:
        response = cohere_client.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return [response.embeddings[0]]
    except Exception as e:
        print("Cohere Embedding Error:", e)
        return []
