from typing import Dict, List, Optional
from .retriever import Retriever
from .generator import Generator


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        defense=None,
    ) -> Dict:
        k = top_k or self.top_k
        docs, scores, doc_embs, query_emb = self.retriever.retrieve(question, top_k=k)

        defense_result = None
        context_docs = docs

        if defense is not None:
            defense_result = defense.detect(query_emb, doc_embs, scores, docs)
            # Fall back to all docs if defense filters everything
            context_docs = defense_result["clean_docs"] or docs

        answer = self.generator.generate(question, context_docs)

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": docs,
            "retrieval_scores": scores.tolist(),
            "doc_embeddings": doc_embs,
            "query_embedding": query_emb,
            "defense": defense_result,
        }
