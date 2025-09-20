"""Pydantic models for the RAG system"""

# Document models
# Chapter models
from .chapter import (ChapterContent, ChapterContentResponse, ChunkComparison,
                      ChunkingComparisonResponse, ChunkingResult,
                      ChunkListResponse, ChunkMetadata, ChunkQuery,
                      ChunkSearchResult, TextChunk)
from .document import (ChapterSummary, ConfirmedTOC, ConfirmTOCResponse,
                       DocumentChaptersResponse, DocumentListResponse,
                       DocumentMetadata, DocumentStats, DocumentUpload,
                       ErrorResponse, MappedChapter, PageRange, ParsedTOC,
                       ParseTOCResponse, Subsection, TOCChapter,
                       TOCConfirmation, ValidationError)

__all__ = [
    # Core data structures
    "PageRange",
    "Subsection",
    "TOCChapter",
    "MappedChapter",
    
    # TOC processing
    "ParsedTOC",
    "ConfirmedTOC",
    "DocumentUpload",
    "TOCConfirmation",
    
    # Document metadata
    "ChapterSummary",
    "DocumentMetadata", 
    "DocumentStats",
    
    # Chunk structures
    "ChunkMetadata",
    "TextChunk",
    "ChapterContent",
    "ChunkingResult",
    
    # Query and search
    "ChunkQuery",
    "ChunkSearchResult",
    "ChunkComparison",
    
    # API responses
    "ParseTOCResponse",
    "ConfirmTOCResponse",
    "DocumentListResponse",
    "DocumentChaptersResponse",
    "ChapterContentResponse",
    "ChunkListResponse",
    "ChunkingComparisonResponse",
    
    # Error handling
    "ErrorResponse",
    "ValidationError"
]