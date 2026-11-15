"""Document processing, extraction, normalization, and chunking pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional, Union
import uuid

from ragshield.core.document import Document, DocumentMetadata


@dataclass
class IngestionChunkConfig:
    """Configuration for document chunking.

    Attributes:
        chunk_size: Target size of each chunk (in characters).
        chunk_overlap: Overlap between consecutive chunks (in characters).
        min_chunk_size: Minimum characters to form a valid chunk.
        preserve_sentences: Whether to split at sentence boundaries when possible.
    """

    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    preserve_sentences: bool = True


@dataclass
class ProcessedChunk:
    """Represents a chunk extracted from an ingested document."""

    chunk_id: str
    parent_doc_id: str
    chunk_index: int
    total_chunks: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> Document:
        """Convert chunk into a RAG-Shield Document."""
        doc_meta = DocumentMetadata(
            source=self.metadata.get("source", ""),
            timestamp=datetime.now(),
            author=self.metadata.get("author", "ingestion_pipeline"),
            doc_type=self.metadata.get("doc_type", "chunk"),
            tags=self.metadata.get("tags", ["chunk"]),
            custom={
                "parent_doc_id": self.parent_doc_id,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                **self.metadata,
            },
        )
        return Document(
            content=self.content,
            doc_id=self.chunk_id,
            metadata=doc_meta,
        )


class DocumentIngestionPipeline:
    """Ingestion pipeline supporting multi-format document loading, normalization, and chunking."""

    def __init__(self, config: Optional[IngestionChunkConfig] = None):
        self.config = config or IngestionChunkConfig()

    def normalize_text(self, text: str) -> str:
        """Clean and normalize extracted text.

        Removes non-printable control characters (except common whitespace),
        normalizes unicode, and collapses excessive whitespace.
        """
        if not text:
            return ""

        # Remove zero-width spaces and invisible control characters
        text = re.sub(r"[\u200b-\u200f\u2028-\u202f\ufeff]", "", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace excessive consecutive spaces or tabs with single space
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3 or more newlines into double newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def extract_from_txt(self, file_path_or_bytes: Union[str, bytes]) -> str:
        """Extract text from TXT or Markdown content."""
        if isinstance(file_path_or_bytes, bytes):
            return file_path_or_bytes.decode("utf-8", errors="ignore")
        elif os.path.exists(file_path_or_bytes):
            with open(file_path_or_bytes, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            # Direct string content
            return str(file_path_or_bytes)

    def extract_from_pdf(self, file_path_or_bytes: Union[str, bytes]) -> str:
        """Extract text from PDF file using pypdf."""
        try:
            import pypdf
            import io

            if isinstance(file_path_or_bytes, bytes):
                stream = io.BytesIO(file_path_or_bytes)
                reader = pypdf.PdfReader(stream)
            elif os.path.exists(file_path_or_bytes):
                reader = pypdf.PdfReader(file_path_or_bytes)
            else:
                raise ValueError(f"PDF file not found: {file_path_or_bytes}")

            extracted_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                extracted_pages.append(page_text)

            return "\n\n".join(extracted_pages)
        except Exception as e:
            # Fallback for plain-text or if pypdf parsing fails
            if isinstance(file_path_or_bytes, bytes):
                return file_path_or_bytes.decode("utf-8", errors="ignore")
            elif os.path.exists(file_path_or_bytes):
                with open(file_path_or_bytes, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

    def load_document(
        self,
        source: Union[str, bytes],
        file_type: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """Load document text based on file format.

        Returns:
            Tuple of (raw_text, source_identifier, inferred_format)
        """
        if isinstance(source, str) and os.path.exists(source):
            source_name = source_name or os.path.basename(source)
            ext = os.path.splitext(source)[1].lower().lstrip(".")
            inferred_type = file_type or ext or "txt"

            if inferred_type == "pdf":
                raw_text = self.extract_from_pdf(source)
            else:
                raw_text = self.extract_from_txt(source)
            return raw_text, source_name, inferred_type

        elif isinstance(source, bytes):
            source_name = source_name or "stream_document"
            inferred_type = file_type or "txt"
            if inferred_type == "pdf":
                raw_text = self.extract_from_pdf(source)
            else:
                raw_text = self.extract_from_txt(source)
            return raw_text, source_name, inferred_type

        else:
            # Raw string
            source_name = source_name or "raw_text_input"
            inferred_type = file_type or "txt"
            return str(source), source_name, inferred_type

    def chunk_text(
        self, text: str, parent_doc_id: str, source_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedChunk]:
        """Divide normalized text into overlapping chunks.

        Handles sentence boundaries and preserves context continuity.
        """
        normalized = self.normalize_text(text)
        if not normalized:
            return []

        metadata = source_metadata or {}
        chunk_size = max(100, self.config.chunk_size)
        overlap = min(self.config.chunk_overlap, chunk_size // 2)

        raw_chunks = []
        if len(normalized) <= chunk_size:
            raw_chunks.append(normalized)
        else:
            start = 0
            text_len = len(normalized)
            while start < text_len:
                end = min(start + chunk_size, text_len)
                # Try breaking at a sentence boundary if configured
                if self.config.preserve_sentences and end < text_len:
                    # Look backwards for a sentence end
                    search_zone = normalized[max(start, end - 100):end]
                    match = re.search(r"[.!?]\s+", search_zone)
                    if match:
                        end = max(start, end - 100) + match.end()

                chunk_str = normalized[start:end].strip()
                if len(chunk_str) >= self.config.min_chunk_size or not raw_chunks:
                    raw_chunks.append(chunk_str)

                if end >= text_len:
                    break
                start = max(start + 1, end - overlap)

        total = len(raw_chunks)
        processed: List[ProcessedChunk] = []
        for idx, c in enumerate(raw_chunks):
            chunk_id = f"{parent_doc_id}_chunk_{idx}"
            chunk_meta = dict(metadata)
            chunk_meta.update({
                "chunk_index": idx,
                "total_chunks": total,
                "char_length": len(c),
            })
            processed.append(
                ProcessedChunk(
                    chunk_id=chunk_id,
                    parent_doc_id=parent_doc_id,
                    chunk_index=idx,
                    total_chunks=total,
                    content=c,
                    metadata=chunk_meta,
                )
            )

        return processed

    def process(
        self,
        source: Union[str, bytes],
        file_type: Optional[str] = None,
        source_name: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Document]:
        """Ingest, normalize, chunk, and return Document chunk instances ready for security analysis."""
        raw_text, identifier, fmt = self.load_document(source, file_type, source_name)
        parent_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"

        chunks = self.chunk_text(
            text=raw_text,
            parent_doc_id=parent_id,
            source_metadata={"source": identifier, "file_type": fmt, "ingested_at": datetime.now().isoformat()},
        )

        return [c.to_document() for c in chunks]

