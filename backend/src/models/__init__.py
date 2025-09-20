"""Pydantic models for the RAG system"""

# Simple chunk models (NEW)
# Document models (EXISTING - keeping for TOC parsing)
from .document import (ChapterSummary, ConfirmedTOC, ConfirmTOCResponse,
                       DocumentChaptersResponse, DocumentListResponse,
                       DocumentMetadata, DocumentStats, DocumentUpload,
                       ErrorResponse, MappedChapter, PageRange, ParsedTOC,
                       ParseTOCResponse, Subsection, TOCChapter,
                       TOCConfirmation, ValidationError)
from .simple_chunk import (ContextExpandedResult, SimpleChunk, SimpleChunkList,
                           SimpleChunkListResponse, SimpleChunkResponse)

__all__ = [
    # Simple chunk models (NEW)
    "SimpleChunk",
    "SimpleChunkList", 
    "SimpleChunkResponse",
    "SimpleChunkListResponse",
    "ContextExpandedResult",
    
    # Core data structures (EXISTING)
    "PageRange",
    "Subsection", 
    "TOCChapter",
    "MappedChapter",
    
    # TOC processing (EXISTING)
    "ParsedTOC",
    "ConfirmedTOC", 
    "DocumentUpload",
    "TOCConfirmation",
    
    # Document metadata (EXISTING)
    "ChapterSummary",
    "DocumentMetadata",
    "DocumentStats",
    
    # API responses (EXISTING + NEW)
    "ParseTOCResponse",
    "ConfirmTOCResponse", 
    "DocumentListResponse",
    "DocumentChaptersResponse",
    "SimpleChunkResponse",
    "SimpleChunkListResponse",
    
    # Error handling (EXISTING)
    "ErrorResponse",
    "ValidationError"
]