from pdf_loader import extract_text_from_pdf
from chunker import chunk_text

def prepare_chunks(pdf_path: str):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    return chunks
