"""Document ingestion, normalization, and chunking module for RAG-Shield."""

from ragshield.ingestion.processor import (
    DocumentIngestionPipeline,
    IngestionChunkConfig,
    ProcessedChunk,
)

__all__ = [
    "DocumentIngestionPipeline",
    "IngestionChunkConfig",
    "ProcessedChunk",
]

