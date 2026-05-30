class RAGError(Exception):
    """Base exception for RAG-related failures."""


class RAGEmbeddingError(RAGError):
    """Raised when embedding generation fails."""


class RAGIngestionError(RAGError):
    """Raised when explicit job-description ingestion fails."""
