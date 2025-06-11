import os
import json
import uuid
from typing import List, Dict, Any
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from dotenv import load_dotenv
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    logger.error("Hugging Face API key not found in .env file")
    raise ValueError("HF_API_KEY is required in .env file")

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'Uploads'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_CHUNKS_FOR_CONTEXT = 5
HF_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store documents and embeddings
documents_store = {}
embeddings_store = {}
embedding_model = None


class DocumentProcessor:
    def __init__(self):
        self.load_embedding_model()

    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        global embedding_model
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def create_text_chunks(self, text: str, chunk_size: int = CHUNK_SIZE,
                           overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())

            if i + chunk_size >= len(words):
                break

        return chunks

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        try:
            embeddings = embedding_model.encode(texts, convert_to_tensor=False)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def process_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process a single document"""
        try:
            text = self.extract_text_from_pdf(file_path)
            chunks = self.create_text_chunks(text)
            embeddings = self.generate_embeddings(chunks)

            doc_id = str(uuid.uuid4())
            document_data = {
                'id': doc_id,
                'filename': filename,
                'chunks': chunks,
                'processed_at': datetime.now().isoformat(),
                'chunk_count': len(chunks)
            }

            documents_store[doc_id] = document_data
            embeddings_store[doc_id] = embeddings

            logger.info(f"Processed document {filename}: {len(chunks)} chunks created")

            return {
                'success': True,
                'doc_id': doc_id,
                'chunk_count': len(chunks),
                'filename': filename
            }

        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            return {'success': False, 'error': str(e)}


class RAGSystem:
    def __init__(self):
        self.setup_llm()

    def setup_llm(self):
        """Setup the Hugging Face API client"""
        try:
            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json",
                "Cache-Control": "no-cache"
            }
            payload = {
                "inputs": {"question": "Test connectivity", "context": "Test"},
                "parameters": {"max_length": 10}
            }
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to connect to Hugging Face API: {response.status_code} - {response.text}")
                raise ValueError(f"Failed to connect to Hugging Face API: {response.status_code} - {response.text}")
            logger.info("Hugging Face API setup completed")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to setup Hugging Face API: {e}")
            raise

    def retrieve_relevant_chunks(self, query: str, top_k: int = MAX_CHUNKS_FOR_CONTEXT) -> List[Dict]:
        """Retrieve most relevant chunks for the query"""
        if not documents_store or not embeddings_store:
            logger.warning("No documents or embeddings available for retrieval")
            return []

        try:
            query_embedding = embedding_model.encode([query], convert_to_tensor=False)
            query_embedding = np.array(query_embedding)

            all_chunks = []
            all_embeddings = []

            for doc_id, doc_data in documents_store.items():
                doc_embeddings = embeddings_store[doc_id]
                for i, chunk in enumerate(doc_data['chunks']):
                    all_chunks.append({
                        'text': chunk,
                        'source': doc_data['filename'],
                        'doc_id': doc_id,
                        'chunk_id': i
                    })
                    all_embeddings.append(doc_embeddings[i])

            if not all_embeddings:
                logger.warning("No chunks available for embedding comparison")
                return []

            all_embeddings = np.array(all_embeddings)
            similarities = cosine_similarity(query_embedding, all_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]

            relevant_chunks = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Increased threshold
                    chunk_data = all_chunks[idx].copy()
                    chunk_data['similarity'] = float(similarities[idx])
                    relevant_chunks.append(chunk_data)

            logger.info(f"Query: {query}")
            logger.info(f"Retrieved {len(relevant_chunks)} chunks: {[c['text'][:50] + '...' for c in relevant_chunks]}")
            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            return []

    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer using retrieved chunks"""
        if not relevant_chunks:
            logger.warning(f"No relevant chunks found for query: {query}")
            return {
                'answer': "I couldn't find relevant information in the uploaded documents. Please try rephrasing your question or upload additional documents.",
                'sources': [],
                'confidence': 0.0,
                'chunks_used': 0
            }

        context_parts = []
        sources = set()

        for chunk in relevant_chunks:
            context_parts.append(f"From {chunk['source']}: {chunk['text']}")
            sources.add(chunk['source'])

        context = "\n\n".join(context_parts)

        answer = self.generate_answer_with_llm(query, context, relevant_chunks)

        return {
            'answer': answer,
            'sources': list(sources),
            'confidence': relevant_chunks[0]['similarity'] if relevant_chunks else 0.0,
            'chunks_used': len(relevant_chunks)
        }

    def generate_answer_with_llm(self, query: str, context: str, chunks: List[Dict]) -> str:
        """Generate answer using Hugging Face API"""
        try:
            prompt = f"""Based on the following medical report excerpts, provide a concise and accurate answer to the question. Use clear, empathetic language suitable for a patient, avoiding complex medical jargon. Ensure the answer directly addresses the question and includes relevant details from the context.

Context:
{context}

Question: {query}

Answer:"""

            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json",
                "Cache-Control": "no-cache"
            }
            payload = {
                "inputs": {"question": query, "context": context},
                "parameters": {
                    "max_length": 200,  # Increased for detailed answers
                    "top_k": 50,
                    "top_p": 0.95,
                    "early_stopping": True
                }
            }

            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Hugging Face API error: {response.status_code} - {response.text}")
                return self.generate_extractive_answer(query, context, chunks)

            result = response.json()
            answer = result.get("answer", "").strip() if isinstance(result, dict) else ""

            if not answer:
                return self.generate_extractive_answer(query, context, chunks)

            return answer

        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating Hugging Face API answer: {e}")
            return self.generate_extractive_answer(query, context, chunks)

    def generate_extractive_answer(self, query: str, context: str, chunks: List[Dict]) -> str:
        """Generate answer using simple extraction method"""
        try:
            sentences = []
            for chunk in chunks[:3]:
                chunk_sentences = chunk['text'].split('. ')
                for sentence in chunk_sentences:
                    if len(sentence.strip()) > 10:
                        sentences.append(sentence.strip())

            query_words = set(query.lower().split())
            scored_sentences = []

            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                if overlap > 0:
                    scored_sentences.append((sentence, overlap))

            scored_sentences.sort(key=lambda x: x[1], reverse=True)

            if scored_sentences:
                answer_sentences = [s[0] for s in scored_sentences[:2]]  # Up to 2 sentences
                answer = '. '.join(answer_sentences)
                if not answer.endswith('.'):
                    answer += '.'
            else:
                answer = "I couldn't find relevant information in the documents."

            return answer

        except Exception as e:
            logger.error(f"Error in extractive answer generation: {e}")
            return "I couldn't find relevant information in the documents."


# Initialize processors
doc_processor = DocumentProcessor()
rag_system = RAGSystem()


@app.route('/', methods=['GET'])
def serve_index():
    """Serve the index.html frontend"""
    try:
        return send_file('index.html')
    except FileNotFoundError:
        logger.error("index.html not found in project directory")
        return jsonify({'success': False, 'error': 'index.html not found'}), 404


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'documents_count': len(documents_store),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process PDF file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Only PDF files are allowed'}), 400

        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        result = doc_processor.process_document(file_path, filename)
        os.remove(file_path)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question about uploaded documents"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'success': False, 'error': 'No question provided'}), 400

        question = data['question'].strip()
        if not question:
            return jsonify({'success': False, 'error': 'Empty question'}), 400

        if not documents_store:
            return jsonify({
                'success': False,
                'error': 'No documents uploaded. Please upload PDF files first.'
            }), 400

        relevant_chunks = rag_system.retrieve_relevant_chunks(question)
        result = rag_system.generate_answer(question, relevant_chunks)

        return jsonify({
            'success': True,
            'answer': result['answer'],
            'sources': result['sources'],
            'confidence': result['confidence'],
            'chunks_used': len(relevant_chunks)
        })

    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear_documents():
    """Clear all uploaded documents"""
    try:
        global documents_store, embeddings_store
        documents_store.clear()
        embeddings_store.clear()
        if os.path.exists(UPLOAD_FOLDER):
            import shutil
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER)

        return jsonify({
            'success': True,
            'message': 'All documents and uploads cleared'
        })

    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        doc_list = []
        for doc_id, doc_data in documents_store.items():
            doc_list.append({
                'id': doc_id,
                'filename': doc_data['filename'],
                'chunk_count': doc_data['chunk_count'],
                'processed_at': doc_data['processed_at']
            })

        return jsonify({
            'success': True,
            'documents': doc_list,
            'total_count': len(doc_list)
        })

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Starting PDF RAG Chatbot Server...")
    print("📚 Features:")
    print("  - PDF text extraction")
    print("  - Semantic chunking with embeddings")
    print("  - RAG-based question answering with Hugging Face API (deepset/roberta-base-squad2)")
    print("  - Multi-document support")
    print("\n🔧 Configuration:")
    print(f"  - Chunk size: {CHUNK_SIZE}")
    print(f"  - Chunk overlap: {CHUNK_OVERLAP}")
    print(f"  - Max chunks for context: {MAX_CHUNKS_FOR_CONTEXT}")
    print(f"  - Upload folder: {UPLOAD_FOLDER}")
    print("\n🌐 Server starting on http://localhost:5000")
    print("📄 Access the chatbot at http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)