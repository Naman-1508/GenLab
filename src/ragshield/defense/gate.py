"""Pre-Ingestion Security Gate and Hybrid Ensemble Controller.

Orchestrates:
1. Multi-format document ingestion and chunking.
2. The four security detectors:
   - Enhanced Perplexity Analysis
   - Similarity / Outlier Analysis
   - Semantic Analysis
   - Direct Prompt-Injection Analysis
3. Hybrid Ensemble Risk Fusion and decision routing.
4. Physical quarantine isolation (preventing poisoned chunks from reaching vector storage).
5. Explainable AI (XAI) attribution reporting.
6. Cryptographic integrity layer (Merkle Tree, Vector Commitment, Audit Log).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np

from ragshield.core.document import Document
from ragshield.core.knowledge_base import KnowledgeBase
from ragshield.core.embedder import Embedder, MockEmbedder
from ragshield.ingestion.processor import DocumentIngestionPipeline, IngestionChunkConfig
from ragshield.detection.perplexity import PerplexityDetector
from ragshield.detection.similarity import SimilarityDetector
from ragshield.detection.semantic import SemanticDetector
from ragshield.detection.prompt_injection import PromptInjectionDetector
from ragshield.detection.fusion import HybridRiskFusion, FusionWeights, UnifiedSecurityResult
from ragshield.defense.quarantine import QuarantineManager, QuarantineEntry
from ragshield.xai.explainer import SecurityExplainer, ExplanationReport
from ragshield.integrity.merkle_tree import MerkleTree, HashAlgorithm
from ragshield.integrity.vector_commit import HashBasedCommitment, Commitment
from ragshield.integrity.audit_log import AuditLog, OperationType


@dataclass
class GateIngestionResult:
    """Detailed result of processing a document chunk through the Pre-Ingestion Security Gate."""

    chunk_id: str
    success: bool
    action_taken: str  # "ingested" or "quarantined"
    unified_risk_score: float
    decision: str  # "safe" or "suspicious"
    dominant_detector: str
    detector_scores: Dict[str, float]
    detector_contributions: Dict[str, float]
    reasons: List[str]
    flagged_evidence: List[str]
    xai_report: ExplanationReport
    document: Document
    commitment_hash: Optional[str] = None
    merkle_root: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "success": self.success,
            "action_taken": self.action_taken,
            "unified_risk_score": self.unified_risk_score,
            "decision": self.decision,
            "dominant_detector": self.dominant_detector,
            "detector_scores": self.detector_scores,
            "detector_contributions": self.detector_contributions,
            "reasons": self.reasons,
            "flagged_evidence": self.flagged_evidence,
            "xai_summary": self.xai_report.summary_narrative,
            "commitment_hash": self.commitment_hash,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
        }


@dataclass
class BatchDocumentIngestionReport:
    """Summary of batch document ingestion through the security gate."""

    document_name: str
    total_chunks: int
    safe_chunks: int
    quarantined_chunks: int
    quarantine_rate: float
    chunk_results: List[GateIngestionResult]
    overall_status: str  # "clean", "partially_quarantined", "fully_quarantined"
    merkle_root: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_name": self.document_name,
            "total_chunks": self.total_chunks,
            "safe_chunks": self.safe_chunks,
            "quarantined_chunks": self.quarantined_chunks,
            "quarantine_rate": self.quarantine_rate,
            "overall_status": self.overall_status,
            "merkle_root": self.merkle_root,
            "chunk_results": [r.to_dict() for r in self.chunk_results],
        }


class PreIngestionSecurityGate:
    """Pre-Ingestion Security Gate protecting the RAG knowledge base.

    All incoming content must pass this gate before embedding and indexing.
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        embedder: Optional[Embedder] = None,
        fusion_weights: Optional[FusionWeights] = None,
        decision_threshold: float = 0.50,
        chunk_config: Optional[IngestionChunkConfig] = None,
        use_transformer_perplexity: bool = False,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.embedder = embedder or MockEmbedder()

        # Ingestion pipeline
        self.ingestion_pipeline = DocumentIngestionPipeline(config=chunk_config)

        # 4 Detectors
        self.perplexity_detector = PerplexityDetector(use_transformer=use_transformer_perplexity)
        self.similarity_detector = SimilarityDetector(embedder=self.embedder)
        self.semantic_detector = SemanticDetector()
        self.prompt_injection_detector = PromptInjectionDetector()

        # Hybrid Ensemble Fusion
        self.fusion = HybridRiskFusion(
            perplexity_detector=self.perplexity_detector,
            similarity_detector=self.similarity_detector,
            semantic_detector=self.semantic_detector,
            prompt_injection_detector=self.prompt_injection_detector,
            weights=fusion_weights or FusionWeights(),
            decision_threshold=decision_threshold,
        )

        # Explainable AI
        self.explainer = SecurityExplainer()

        # Quarantine Subsystem (physical isolation)
        self.quarantine_manager = QuarantineManager()

        # Cryptographic Integrity Layer
        self.merkle_tree = MerkleTree(algorithm=HashAlgorithm.SHA256)
        self.vector_committer = HashBasedCommitment()
        self.audit_log = AuditLog()

        self._vector_commitments: Dict[str, Commitment] = {}
        self._accepted_docs_content: List[str] = []

    def ingest_chunk(self, chunk: Document, source_name: str = "stream") -> GateIngestionResult:
        """Process an individual chunk through the 4-detector security gate.
        
        Implements the Conservative Fail-Closed Principle:
        Any internal verification error or unhandled detector exception routes the chunk
        directly to quarantine, strictly preventing unverified content from entering the trusted KB.
        """
        try:
            # 1. Execute Hybrid Ensemble Risk Fusion
            fusion_result = self.fusion.evaluate_chunk(chunk)

            # 2. Generate Explainable AI (XAI) Report
            xai_report = self.explainer.explain(fusion_result, chunk)
        except Exception as e:
            # Conservative Fail-Closed routing
            self.quarantine_manager.quarantine(
                document=chunk,
                reason=f"Conservative Fail-Closed: Security evaluation error ({type(e).__name__}: {str(e)})",
                detection_result=None,
                metadata={"source": source_name, "risk_score": 1.0, "fail_closed": True, "error": str(e)},
            )
            self.audit_log.log_operation(
                OperationType.POISON_DETECTED,
                {"chunk_id": chunk.doc_id, "source": source_name, "error": str(e), "fail_closed": True},
            )
            return GateIngestionResult(
                chunk_id=chunk.doc_id,
                success=False,
                action_taken="quarantined",
                unified_risk_score=1.0,
                decision="suspicious",
                dominant_detector="system_integrity",
                detector_scores={"perplexity": 0.0, "similarity": 0.0, "semantic": 0.0, "prompt_injection": 0.0},
                detector_contributions={"perplexity": 25.0, "similarity": 25.0, "semantic": 25.0, "prompt_injection": 25.0},
                reasons=[f"Conservative Fail-Closed: Security analysis failed ({type(e).__name__})"],
                flagged_evidence=[],
                xai_report=None,
                document=chunk,
                commitment_hash=None,
                merkle_root=self.merkle_tree.get_root_hash(),
            )

        if fusion_result.is_suspicious:
            # Route to Quarantine
            self.quarantine_manager.quarantine(
                document=chunk,
                reason=xai_report.summary_narrative,
                detection_result=None,
                metadata={
                    "source": source_name,
                    "risk_score": fusion_result.unified_risk_score,
                    "dominant_detector": fusion_result.dominant_detector,
                    "flagged_evidence": fusion_result.flagged_evidence,
                },
            )

            # Log security event
            self.audit_log.log_operation(
                OperationType.POISON_DETECTED,
                {
                    "chunk_id": chunk.doc_id,
                    "source": source_name,
                    "risk_score": fusion_result.unified_risk_score,
                    "dominant_detector": fusion_result.dominant_detector,
                    "reasons": fusion_result.reasons,
                    "evidence": fusion_result.flagged_evidence,
                },
            )

            # Strictly ensure document is NOT added to knowledge base
            return GateIngestionResult(
                chunk_id=chunk.doc_id,
                success=False,
                action_taken="quarantined",
                unified_risk_score=fusion_result.unified_risk_score,
                decision=fusion_result.decision,
                dominant_detector=fusion_result.dominant_detector,
                detector_scores=fusion_result.detector_scores,
                detector_contributions=fusion_result.detector_contributions,
                reasons=fusion_result.reasons,
                flagged_evidence=fusion_result.flagged_evidence,
                xai_report=xai_report,
                document=chunk,
                commitment_hash=None,
                merkle_root=self.merkle_tree.get_root_hash(),
            )

        else:
            # Safe chunk: proceed to embedding generation and trusted vector storage
            if chunk.embedding is None:
                emb = self.embedder.embed(chunk.content)
                if hasattr(emb, "ndim") and emb.ndim > 1:
                    chunk.embedding = emb[0].tolist()
                elif hasattr(emb, "tolist"):
                    chunk.embedding = emb.tolist()
                else:
                    chunk.embedding = list(emb)

            # Generate vector commitment for embedding integrity
            commitment = self.vector_committer.commit(np.array(chunk.embedding, dtype=np.float32))
            self._vector_commitments[chunk.doc_id] = commitment

            # Store in trusted vector DB
            self.knowledge_base.add_document(chunk)

            # Update Merkle Tree
            self._accepted_docs_content.append(chunk.content)
            current_root = self.merkle_tree.build(self._accepted_docs_content)

            # Log trusted addition to audit log
            self.audit_log.log_operation(
                OperationType.DOCUMENT_ADD,
                {
                    "chunk_id": chunk.doc_id,
                    "source": source_name,
                    "risk_score": fusion_result.unified_risk_score,
                    "commitment": commitment.value,
                    "merkle_root": current_root,
                },
            )

            return GateIngestionResult(
                chunk_id=chunk.doc_id,
                success=True,
                action_taken="ingested",
                unified_risk_score=fusion_result.unified_risk_score,
                decision=fusion_result.decision,
                dominant_detector=fusion_result.dominant_detector,
                detector_scores=fusion_result.detector_scores,
                detector_contributions=fusion_result.detector_contributions,
                reasons=fusion_result.reasons,
                flagged_evidence=fusion_result.flagged_evidence,
                xai_report=xai_report,
                document=chunk,
                commitment_hash=commitment.value,
                merkle_root=current_root,
            )

    def ingest_document(
        self,
        source: Union[str, bytes],
        file_type: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> BatchDocumentIngestionReport:
        """Process an entire document (PDF, TXT, MD, or raw string) through the security gate."""
        chunks = self.ingestion_pipeline.process(
            source=source,
            file_type=file_type,
            source_name=source_name,
        )

        name = source_name or (os.path.basename(source) if isinstance(source, str) and os.path.exists(source) else "document")

        results: List[GateIngestionResult] = []
        for c in chunks:
            res = self.ingest_chunk(c, source_name=name)
            results.append(res)

        total = len(results)
        safe_count = sum(1 for r in results if r.success)
        quarantine_count = total - safe_count
        q_rate = (quarantine_count / total * 100.0) if total > 0 else 0.0

        if quarantine_count == 0:
            status = "clean"
        elif safe_count == 0:
            status = "fully_quarantined"
        else:
            status = "partially_quarantined"

        return BatchDocumentIngestionReport(
            document_name=name,
            total_chunks=total,
            safe_chunks=safe_count,
            quarantined_chunks=quarantine_count,
            quarantine_rate=round(q_rate, 2),
            chunk_results=results,
            overall_status=status,
            merkle_root=self.merkle_tree.get_root_hash(),
        )

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify cryptographic integrity of the trusted knowledge base and audit chain."""
        # 1. Audit log verification
        audit_valid = self.audit_log.verify_chain()

        # 2. Merkle Tree verification
        merkle_root = self.merkle_tree.get_root_hash()
        kb_docs = [d.content for d in self.knowledge_base.documents]

        merkle_valid = True
        if kb_docs:
            temp_tree = MerkleTree(algorithm=HashAlgorithm.SHA256)
            recomputed_root = temp_tree.build(kb_docs)
            merkle_valid = (recomputed_root == merkle_root)

        # 3. Vector commitment verification
        commitment_errors = []
        for doc in self.knowledge_base.documents:
            if doc.doc_id in self._vector_commitments:
                comm = self._vector_commitments[doc.doc_id]
                vec = np.array(doc.embedding, dtype=np.float32)
                if not self.vector_committer.verify(vec, comm):
                    commitment_errors.append(doc.doc_id)

        all_ok = audit_valid and merkle_valid and (len(commitment_errors) == 0)

        # Log integrity verification event
        self.audit_log.log_operation(
            OperationType.INTEGRITY_CHECK,
            {
                "audit_chain_valid": audit_valid,
                "merkle_valid": merkle_valid,
                "commitment_violations": len(commitment_errors),
                "overall_integrity": all_ok,
            },
        )

        return {
            "overall_integrity_verified": all_ok,
            "merkle_root": merkle_root,
            "merkle_tree_valid": merkle_valid,
            "audit_chain_valid": audit_valid,
            "audit_log_entries": len(self.audit_log.entries),
            "vector_commitments_audited": len(self._vector_commitments),
            "commitment_violations": commitment_errors,
            "timestamp": datetime.now().isoformat(),
        }
