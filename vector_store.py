import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add(self, text_chunk: str, embedding: list):
        self.index.add(np.array([embedding]).astype("float32"))
        self.text_chunks.append(text_chunk)

    def search(self, query_embedding: list, top_k=3):
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), top_k)
        return [self.text_chunks[i] for i in I[0]]
