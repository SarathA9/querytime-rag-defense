"""
Rank-Faithfulness Consistency (RFC) — query-time backdoor detection.

For each retrieved document d_i:
  - retrieval_score(d_i)  : cosine_sim(query, d_i)          [from FAISS]
  - faithfulness_score(d_i): cosine_sim(d_i, centroid(d_{j≠i}))
  - rfc_score(d_i)        : retrieval_score - faithfulness_score

A poisoned document ranks high for the query but is semantically isolated
from the rest of the retrieved context, giving a large positive RFC score.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List


class RFCDetector:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def compute_rfc_scores(
        self,
        query_embedding: np.ndarray,   # (dim,)
        doc_embeddings: np.ndarray,    # (k, dim)
        retrieval_scores: np.ndarray,  # (k,)  cosine sims from FAISS
    ) -> np.ndarray:
        k = len(doc_embeddings)
        rfc_scores = np.zeros(k, dtype=np.float32)

        for i in range(k):
            other = [j for j in range(k) if j != i]
            if not other:
                continue
            centroid = doc_embeddings[other].mean(axis=0, keepdims=True)
            faithfulness = cosine_similarity(doc_embeddings[i : i + 1], centroid)[0, 0]
            rfc_scores[i] = retrieval_scores[i] - faithfulness

        return rfc_scores

    def detect(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        retrieval_scores: np.ndarray,
        documents: List[str],
    ) -> Dict:
        rfc_scores = self.compute_rfc_scores(query_embedding, doc_embeddings, retrieval_scores)
        flags = rfc_scores > self.threshold

        return {
            "rfc_scores": rfc_scores.tolist(),
            "flags": flags.tolist(),
            "clean_docs": [d for d, f in zip(documents, flags) if not f],
            "flagged_docs": [d for d, f in zip(documents, flags) if f],
            "n_flagged": int(flags.sum()),
        }
