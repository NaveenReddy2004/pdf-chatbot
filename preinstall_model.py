import os
import uuid
from typing import List, Dict, Any
import PyPDF2
import numpy as np
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import streamlit as st
from datetime import datetime

HF_API_KEY = st.secrets["HF_API_KEY"]
HF_API_URL = "https://api-inference.huggingface.co/models/bert-large-uncased-whole-word-masking-finetuned-squad"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_CHUNKS_FOR_CONTEXT = 3

documents_store = {}
embeddings_store = {}

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        return text

    def create_text_chunks(self, text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embedding_model.encode(texts, convert_to_tensor=False))

    def process_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        text = self.extract_text_from_pdf(file_path)
        chunks = self.create_text_chunks(text)
        embeddings = self.generate_embeddings(chunks)

        doc_id = str(uuid.uuid4())
        documents_store[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "chunks": chunks,
            "processed_at": datetime.now().isoformat(),
            "chunk_count": len(chunks)
        }
        embeddings_store[doc_id] = embeddings

        return {
            "success": True,
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "filename": filename
        }

class RAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad') 
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


    def retrieve_relevant_chunks(self, query: str, top_k=MAX_CHUNKS_FOR_CONTEXT) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding)

        all_chunks, all_embeddings = [], []
        for doc_id, doc_data in documents_store.items():
            for i, chunk in enumerate(doc_data['chunks']):
                all_chunks.append({
                    "text": chunk,
                    "source": doc_data["filename"],
                    "doc_id": doc_id,
                    "chunk_id": i
                })
                all_embeddings.append(embeddings_store[doc_id][i])

        if not all_embeddings:
            return []

        similarities = cosine_similarity(query_embedding, np.array(all_embeddings))[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            {**all_chunks[i], "similarity": float(similarities[i])}
            for i in top_indices if similarities[i] > 0.3
        ]

    def generate_answer_with_llm(self, question: str, context: str, chunks: List[Dict]) -> str:
    inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
    return answer.strip()
