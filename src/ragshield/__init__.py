"""
RAG-Shield: Security Framework for Retrieval-Augmented Generation Systems
"""

__version__ = "0.2.0"

from ragshield.core import (
    RAGSystem,
    KnowledgeBase,
    Document,
    DocumentMetadata,
    Embedder,
    SentenceTransformerEmbedder,
    MockEmbedder,
    SecureRAGPipeline,
    SecureRAGResponse,
)
from ragshield.detection import (
    PoisonDetector,
    PerplexityDetector,
    SimilarityDetector,
    SemanticDetector,
    PromptInjectionDetector,
    HybridRiskFusion,
    FusionWeights,
    UnifiedSecurityResult,
    create_poison_detector,
)
from ragshield.defense import (
    RAGShield,
    PreIngestionSecurityGate,
    GateIngestionResult,
    BatchDocumentIngestionReport,
    QuarantineManager,
)
from ragshield.xai import SecurityExplainer, ExplanationReport
from ragshield.ingestion import DocumentIngestionPipeline, IngestionChunkConfig
from ragshield.integrity import MerkleTree, HashBasedCommitment, AuditLog

__all__ = [
    "RAGSystem",
    "KnowledgeBase",
    "Document",
    "DocumentMetadata",
    "Embedder",
    "SentenceTransformerEmbedder",
    "MockEmbedder",
    "SecureRAGPipeline",
    "SecureRAGResponse",
    "PoisonDetector",
    "PerplexityDetector",
    "SimilarityDetector",
    "SemanticDetector",
    "PromptInjectionDetector",
    "HybridRiskFusion",
    "FusionWeights",
    "UnifiedSecurityResult",
    "create_poison_detector",
    "RAGShield",
    "PreIngestionSecurityGate",
    "GateIngestionResult",
    "BatchDocumentIngestionReport",
    "QuarantineManager",
    "SecurityExplainer",
    "ExplanationReport",
    "DocumentIngestionPipeline",
    "IngestionChunkConfig",
    "MerkleTree",
    "HashBasedCommitment",
    "AuditLog",
]
