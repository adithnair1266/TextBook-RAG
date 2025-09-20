from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SimpleChunk(BaseModel):
    """Simple chunk model - just the essentials"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., min_length=1, description="Chunk content")
    chapter_name: str = Field(..., description="Chapter this chunk belongs to")
    paragraph_index: int = Field(..., ge=0, description="Position within chapter (0-based)")
    page_number: int = Field(..., ge=1, description="Page number in original document")
    word_count: int = Field(..., ge=0, description="Word count in chunk")
    
    # Simple navigation
    prev_chunk_id: Optional[str] = Field(None, description="Previous chunk in sequence")
    next_chunk_id: Optional[str] = Field(None, description="Next chunk in sequence")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @property
    def display_title(self) -> str:
        """Human-readable title for this chunk"""
        return f"{self.chapter_name} - Paragraph {self.paragraph_index + 1}"
    
    @property
    def context_info(self) -> str:
        """Brief context information"""
        return f"Page {self.page_number}, {self.chapter_name}"


class SimpleChunkList(BaseModel):
    """Collection of simple chunks with metadata"""
    chunks: list[SimpleChunk] = Field(..., description="List of chunks")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks")
    chapter_name: str = Field(..., description="Chapter name")
    total_word_count: int = Field(..., ge=0, description="Total words across all chunks")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[SimpleChunk]:
        """Get chunk by ID"""
        return next((chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None)
    
    def get_chunks_by_page(self, page_number: int) -> list[SimpleChunk]:
        """Get all chunks from a specific page"""
        return [chunk for chunk in self.chunks if chunk.page_number == page_number]
    
    def get_paragraph_range(self, start_idx: int, end_idx: int) -> list[SimpleChunk]:
        """Get chunks within paragraph index range"""
        return [
            chunk for chunk in self.chunks 
            if start_idx <= chunk.paragraph_index <= end_idx
        ]


# Response models for API
class SimpleChunkResponse(BaseModel):
    """API response for single chunk"""
    doc_id: str = Field(..., description="Document ID")
    chunk: SimpleChunk = Field(..., description="The chunk data")
    context_available: bool = Field(True, description="Whether context expansion is available")


class SimpleChunkListResponse(BaseModel):
    """API response for chunk list"""
    doc_id: str = Field(..., description="Document ID")
    chunks: list[SimpleChunk] = Field(..., description="List of chunks")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks")
    chapter_filter: Optional[str] = Field(None, description="Chapter filter applied")
    page_filter: Optional[int] = Field(None, description="Page filter applied")


class ContextExpandedResult(BaseModel):
    """Search result with context expansion"""
    target_chunk: SimpleChunk = Field(..., description="The matching chunk")
    context_chunks: list[SimpleChunk] = Field(..., description="Surrounding context chunks")
    context_text: str = Field(..., description="Merged context as readable text")
    similarity_score: float = Field(..., ge=0, le=1, description="Search similarity score")
    expansion_type: str = Field(..., description="Type of context expansion used")
    total_words: int = Field(..., ge=0, description="Total words in expanded context")