import faiss
import numpy as np

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_top_k(index, query_vector, chunks, k=3):
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]
