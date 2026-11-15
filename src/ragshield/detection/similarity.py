"""Similarity and outlier-based poison detection.

Poisoned or anomalous documents frequently deviate from legitimate domain clusters
or exhibit artificial clustering patterns. This detector identifies embedding-space
outliers and abnormal similarity distributions relative to benign reference representations.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.detection.base import PoisonDetector, DetectionResult, ThreatLevel, ScanResult


class SimilarityDetector(PoisonDetector):
    """Detect poisoned document chunks based on embedding similarity and outlier patterns.

    Identifies:
    1. Outlier embeddings with low similarity to known benign corpus clusters.
    2. Dense suspicious clusters formed by injected adversarial variants.
    3. Structural anomalies in embedding norm, entropy, and coordinate sparsity.

    Args:
        cluster_threshold: Cosine similarity threshold for detecting adversarial clusters.
        outlier_threshold: Maximum distance threshold beyond which a chunk is deemed an outlier.
        min_cluster_size: Minimum number of documents constituting a suspicious cluster.
        embedder: Optional embedding generator function or model.
    """

    def __init__(
        self,
        cluster_threshold: float = 0.95,
        outlier_threshold: float = 0.45,
        min_cluster_size: int = 3,
        embedder: Optional[Any] = None,
    ):
        self.cluster_threshold = cluster_threshold
        self.outlier_threshold = outlier_threshold
        self.min_cluster_size = min_cluster_size
        self.embedder = embedder
        self._similarity_matrix = None
        self._documents = None
        self._reference_centroids: Optional[np.ndarray] = None
        self._reference_embeddings: Optional[np.ndarray] = None

    def _ensure_embedding(self, document: Document) -> Optional[np.ndarray]:
        """Ensure document has an embedding, using embedder if available."""
        if document.embedding is not None:
            arr = np.array(document.embedding, dtype=np.float32)
            return arr.flatten()
        if self.embedder is not None and document.content:
            try:
                emb = self.embedder.embed(document.content)
                if hasattr(emb, "ndim") and emb.ndim > 1:
                    vec = emb[0].flatten()
                else:
                    vec = np.array(emb, dtype=np.float32).flatten()
                document.embedding = vec.tolist()
                return vec
            except Exception:
                pass
        return None

    def fit_reference_corpus(self, embeddings: List[List[float]], num_clusters: int = 5) -> None:
        """Fit reference distribution centroids from a trusted benign corpus."""
        if not embeddings:
            return
        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        normalized = arr / norms
        self._reference_embeddings = normalized

        if len(arr) <= num_clusters:
            self._reference_centroids = normalized
        else:
            try:
                from sklearn.cluster import MiniBatchKMeans
                kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init=3)
                kmeans.fit(normalized)
                centers = kmeans.cluster_centers_
                self._reference_centroids = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
            except Exception:
                mean_center = normalized.mean(axis=0, keepdims=True)
                self._reference_centroids = mean_center / (np.linalg.norm(mean_center) + 1e-10)

    def _compute_similarity_matrix(self, documents: List[Document]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        embeddings = []
        for doc in documents:
            emb = self._ensure_embedding(doc)
            if emb is None:
                raise ValueError(f"Document {doc.doc_id} has no embedding")
            embeddings.append(emb)

        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        normalized = arr / norms
        return np.dot(normalized, normalized.T)

    def _detect_clusters(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """Detect suspicious document clusters."""
        n = similarity_matrix.shape[0]
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            cluster = [i]
            for j in range(i + 1, n):
                if j not in visited and similarity_matrix[i, j] > self.cluster_threshold:
                    cluster.append(j)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
                visited.update(cluster)

        return clusters

    def _detect_outliers(self, similarity_matrix: np.ndarray) -> List[int]:
        """Detect outlier documents with low average similarity."""
        n = similarity_matrix.shape[0]
        if n <= 1:
            return []
        outliers = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            avg_similarity = float(similarity_matrix[i, mask].mean())
            if avg_similarity < (1.0 - self.outlier_threshold):
                outliers.append(i)
        return outliers

    def fit(self, knowledge_base: KnowledgeBase) -> None:
        """Fit detector on existing knowledge base documents."""
        self._documents = list(knowledge_base.documents)
        if self._documents:
            self._similarity_matrix = self._compute_similarity_matrix(self._documents)
            embs = [d.embedding for d in self._documents if d.embedding is not None]
            if embs:
                self.fit_reference_corpus(embs)

    def detect(self, document: Document) -> DetectionResult:
        """Detect outlier or anomaly characteristics in document embedding."""
        emb = self._ensure_embedding(document)
        if emb is None:
            return DetectionResult(
                is_poisoned=False,
                confidence=0.0,
                threat_level=ThreatLevel.NONE,
                reason="No embedding available for similarity analysis",
                score=0.0,
                metadata={"outlier_score": 0.0},
            )

        norm = float(np.linalg.norm(emb))
        norm_suspicious = (norm < 0.5 or norm > 2.0)

        emb_abs = np.abs(emb)
        emb_norm = emb_abs / (emb_abs.sum() + 1e-10)
        entropy = float(-np.sum(emb_norm * np.log(emb_norm + 1e-10)))
        entropy_suspicious = (entropy < 2.0)

        sparsity = float(np.sum(np.abs(emb) < 0.001) / len(emb))
        sparsity_suspicious = (sparsity > 0.90)

        # Check against reference centroids if available
        reference_outlier_distance = 0.0
        if self._reference_centroids is not None and len(self._reference_centroids) > 0:
            normed_emb = (emb / (norm + 1e-10)).flatten()
            sims = np.dot(self._reference_centroids, normed_emb)
            max_sim = float(np.max(sims))
            reference_outlier_distance = float(max(0.0, 1.0 - max_sim))

        stat_flags = sum([norm_suspicious, entropy_suspicious, sparsity_suspicious])
        base_score = stat_flags / 3.0

        if self._reference_centroids is not None:
            combined_risk = 0.7 * min(1.0, reference_outlier_distance / max(self.outlier_threshold, 0.01)) + 0.3 * base_score
        else:
            combined_risk = base_score

        is_poisoned = (stat_flags >= 2) or (reference_outlier_distance > self.outlier_threshold > 0.0)

        reasons = []
        if norm_suspicious:
            reasons.append(f"unusual embedding norm ({norm:.2f})")
        if entropy_suspicious:
            reasons.append(f"low embedding entropy ({entropy:.2f})")
        if sparsity_suspicious:
            reasons.append(f"unusual sparsity ({sparsity:.2f})")
        if reference_outlier_distance > self.outlier_threshold > 0.0:
            reasons.append(f"reference cluster outlier (dist={reference_outlier_distance:.2f} > {self.outlier_threshold:.2f})")

        reason = ", ".join(reasons) if reasons else "Normal embedding distribution"

        threat_level = ThreatLevel.NONE
        if is_poisoned:
            threat_level = ThreatLevel.HIGH if combined_risk > 0.75 else ThreatLevel.MEDIUM

        return DetectionResult(
            is_poisoned=is_poisoned,
            confidence=float(min(0.95, max(0.2, combined_risk))),
            threat_level=threat_level,
            reason=reason,
            score=float(np.clip(combined_risk, 0.0, 1.0)),
            metadata={
                "norm": norm,
                "entropy": entropy,
                "sparsity": sparsity,
                "reference_outlier_distance": reference_outlier_distance,
                "stat_flags": stat_flags,
                "outlier_score": float(np.clip(combined_risk, 0.0, 1.0)),
            },
        )

    def scan_knowledge_base(
        self, knowledge_base: KnowledgeBase, threshold: Optional[float] = None
    ) -> ScanResult:
        """Scan knowledge base for clusters and outliers."""
        if threshold is not None:
            self.cluster_threshold = threshold

        self.fit(knowledge_base)
        clusters = self._detect_clusters(self._similarity_matrix)
        outliers = self._detect_outliers(self._similarity_matrix)

        suspicious_indices = set()
        for cluster in clusters:
            suspicious_indices.update(cluster)
        suspicious_indices.update(outliers)

        poisoned_docs = []
        for idx in suspicious_indices:
            doc = self._documents[idx]
            in_cluster = any(idx in cluster for cluster in clusters)
            is_outlier = idx in outliers

            if in_cluster:
                reason = f"Part of suspicious cluster (similarity > {self.cluster_threshold})"
                threat_level = ThreatLevel.HIGH
            else:
                reason = f"Outlier document (avg similarity < {1.0 - self.outlier_threshold:.2f})"
                threat_level = ThreatLevel.MEDIUM

            result = DetectionResult(
                is_poisoned=True,
                confidence=0.85 if in_cluster else 0.65,
                threat_level=threat_level,
                reason=reason,
                score=1.0 if in_cluster else 0.75,
                metadata={"in_cluster": in_cluster, "is_outlier": is_outlier},
            )
            poisoned_docs.append((doc, result))

        total = knowledge_base.size()
        clean_docs = total - len(poisoned_docs)
        detection_rate = (len(poisoned_docs) / total * 100) if total > 0 else 0.0

        return ScanResult(
            total_documents=total,
            poisoned_docs=poisoned_docs,
            clean_docs=clean_docs,
            detection_rate=detection_rate,
        )
