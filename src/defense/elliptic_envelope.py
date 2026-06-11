"""
Ingestion-time baseline: EllipticEnvelope anomaly detection.
Fits on clean corpus embeddings; flags outlier documents before indexing.

Embeddings are PCA-reduced before the EllipticEnvelope fit. The Minimum Covariance
Determinant estimator behind EllipticEnvelope needs samples >> dimensions; at the
raw 768-d embedding size it is both statistically degenerate and pathologically
slow (minutes per fit). PCA to ~50 components makes it well-posed and fast (~2s)
while preserving the dominant variance used for outlier scoring.
"""

import numpy as np
import joblib
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from typing import List, Tuple


class EllipticEnvelopeDetector:
    def __init__(
        self,
        contamination: float = 0.1,
        n_components: int = 50,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_components = n_components
        self.random_state = random_state
        self.pca: PCA = None
        self.detector: EllipticEnvelope = None
        self.fitted = False

    def fit(self, embeddings: np.ndarray) -> None:
        n_comp = min(self.n_components, embeddings.shape[1], embeddings.shape[0] - 1)
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        reduced = self.pca.fit_transform(embeddings)
        self.detector = EllipticEnvelope(
            contamination=self.contamination, random_state=self.random_state
        )
        self.detector.fit(reduced)
        self.fitted = True

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        return self.detector.predict(self.pca.transform(embeddings))  # 1 inlier, -1 outlier

    def filter_corpus(
        self, documents: List[str], embeddings: np.ndarray
    ) -> Tuple[List[str], np.ndarray, List[int]]:
        preds = self.predict(embeddings)
        clean_mask = preds == 1
        clean_docs = [d for d, m in zip(documents, clean_mask) if m]
        clean_embs = embeddings[clean_mask]
        flagged_indices = [i for i, m in enumerate(clean_mask) if not m]
        return clean_docs, clean_embs, flagged_indices

    def save(self, path: str) -> None:
        joblib.dump({"pca": self.pca, "detector": self.detector}, path)

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        self.pca = obj["pca"]
        self.detector = obj["detector"]
        self.fitted = True
