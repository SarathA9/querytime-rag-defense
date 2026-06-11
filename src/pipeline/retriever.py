import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class Retriever:
    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1", device: str = "cpu"):
        # Always run the embedder on CPU — preserves full GPU VRAM for the LLM.
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_embedding_dimension()
        self.index = None
        self.documents: List[str] = []
        self.embeddings: np.ndarray = None

    def build_index(self, documents: List[str], batch_size: int = 64) -> None:
        self.documents = list(documents)
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.embeddings = embeddings.astype(np.float32)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_emb, top_k)
        docs = [self.documents[i] for i in indices[0]]
        embs = self.embeddings[indices[0]]
        return docs, scores[0], embs, query_emb[0]

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def add_documents(self, new_docs: List[str]) -> None:
        new_embs = self.encode(new_docs).astype(np.float32)
        self.documents.extend(new_docs)
        self.embeddings = np.vstack([self.embeddings, new_embs])
        self.index.add(new_embs)
