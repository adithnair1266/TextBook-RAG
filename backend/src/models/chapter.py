from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from models.document import PageRange, Subsection


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk"""
    chunk_id: int = Field(..., ge=0, description="Chunk ID within chapter")
    word_count: int = Field(..., ge=0, description="Word count in chunk")
    chapter_title: str = Field(..., description="Parent chapter title")
    chapter_page_start: int = Field(..., ge=1, description="Chapter start page")
    chapter_page_end: int = Field(..., ge=1, description="Chapter end page")
    subsections: List[Subsection] = Field(default_factory=list, description="Relevant subsections")
    created_at: datetime = Field(..., description="Chunk creation timestamp")
    chunking_strategy: str = Field(..., description="Strategy used for chunking")
    
    # Optional fields for different chunking strategies
    overlap_words: Optional[int] = Field(None, ge=0, description="Words overlapping with previous chunk")
    semantic_boundary: Optional[bool] = Field(None, description="Whether chunk ends at semantic boundary")
    sentence_count: Optional[int] = Field(None, ge=0, description="Number of sentences in chunk")


class TextChunk(BaseModel):
    """Individual text chunk with content and metadata"""
    chunk_id: int = Field(..., ge=0, description="Chunk ID within chapter")
    text: str = Field(..., min_length=1, description="Chunk content")
    word_count: int = Field(..., ge=0, description="Word count in chunk")
    chapter_title: str = Field(..., description="Parent chapter title")
    chapter_page_start: int = Field(..., ge=1, description="Chapter start page")
    chapter_page_end: int = Field(..., ge=1, description="Chapter end page")
    subsections: List[Subsection] = Field(default_factory=list, description="Relevant subsections")
    created_at: datetime = Field(..., description="Chunk creation timestamp")
    chunking_strategy: str = Field(..., description="Strategy used for chunking")
    
    # Optional fields for different chunking strategies
    overlap_words: Optional[int] = Field(None, ge=0, description="Words overlapping with previous chunk")
    semantic_boundary: Optional[bool] = Field(None, description="Whether chunk ends at semantic boundary")
    sentence_count: Optional[int] = Field(None, ge=0, description="Number of sentences in chunk")
    
    # Additional context when chunk is retrieved
    doc_id: Optional[str] = Field(None, description="Document ID")
    chapter_filename: Optional[str] = Field(None, description="Chapter filename")
    
    @validator('word_count')
    def word_count_matches_text(cls, v, values):
        if 'text' in values:
            actual_count = len(values['text'].split())
            # Allow small discrepancies due to cleaning/processing
            if abs(v - actual_count) > 5:
                raise ValueError(f'Word count mismatch: stated {v}, actual ~{actual_count}')
        return v


class ChapterContent(BaseModel):
    """Complete chapter content with chunks"""
    title: str = Field(..., description="Chapter title")
    toc_page: int = Field(..., ge=1, description="Original TOC page number")
    page_range: PageRange = Field(..., description="Chapter page range")
    text: str = Field(..., description="Full chapter text")
    chunks: List[TextChunk] = Field(..., description="Chapter chunks")
    subsections: List[Subsection] = Field(default_factory=list, description="Chapter subsections")
    extracted_at: datetime = Field(..., description="Extraction timestamp")
    word_count: int = Field(..., ge=0, description="Total word count")
    chunk_count: int = Field(..., ge=0, description="Number of chunks")
    page_count: int = Field(..., ge=1, description="Number of pages")
    chunking_strategy: str = Field(..., description="Chunking strategy used")
    
    @validator('chunk_count')
    def chunk_count_matches_list(cls, v, values):
        if 'chunks' in values and v != len(values['chunks']):
            raise ValueError('chunk_count must match length of chunks list')
        return v
    
    @validator('word_count')
    def word_count_reasonable(cls, v, values):
        if 'text' in values:
            actual_count = len(values['text'].split())
            # Allow reasonable discrepancy for cleaning
            if abs(v - actual_count) > actual_count * 0.1:  # 10% tolerance
                raise ValueError(f'Word count seems incorrect: stated {v}, actual ~{actual_count}')
        return v


class ChunkingResult(BaseModel):
    """Result of chunking operation"""
    strategy: str = Field(..., description="Chunking strategy used")
    config: Dict[str, Any] = Field(..., description="Chunking configuration")
    chunks: List[TextChunk] = Field(..., description="Generated chunks")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks")
    total_words: int = Field(..., ge=0, description="Total word count")
    avg_chunk_size: float = Field(..., ge=0, description="Average chunk size in words")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    @validator('total_chunks')
    def total_chunks_matches_list(cls, v, values):
        if 'chunks' in values and v != len(values['chunks']):
            raise ValueError('total_chunks must match length of chunks list')
        return v
    
    @validator('avg_chunk_size')
    def avg_chunk_size_calculation(cls, v, values):
        if 'chunks' in values and 'total_words' in values and len(values['chunks']) > 0:
            expected_avg = values['total_words'] / len(values['chunks'])
            if abs(v - expected_avg) > 1:  # Allow 1 word difference for rounding
                raise ValueError(f'Average chunk size incorrect: stated {v}, expected ~{expected_avg}')
        return v


class ChunkQuery(BaseModel):
    """Query for finding chunks"""
    doc_id: Optional[str] = Field(None, description="Filter by document ID")
    chapter_title: Optional[str] = Field(None, description="Filter by chapter title")
    min_words: Optional[int] = Field(None, ge=0, description="Minimum word count")
    max_words: Optional[int] = Field(None, ge=0, description="Maximum word count")
    chunking_strategy: Optional[str] = Field(None, description="Filter by chunking strategy")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of chunks to return")
    
    @validator('max_words')
    def max_words_greater_than_min(cls, v, values):
        if v is not None and 'min_words' in values and values['min_words'] is not None:
            if v < values['min_words']:
                raise ValueError('max_words must be >= min_words')
        return v


class ChunkSearchResult(BaseModel):
    """Result of chunk search"""
    chunks: List[TextChunk] = Field(..., description="Found chunks")
    total_found: int = Field(..., ge=0, description="Total chunks found")
    query: ChunkQuery = Field(..., description="Original query")
    
    @validator('total_found')
    def total_found_matches_list(cls, v, values):
        if 'chunks' in values and v != len(values['chunks']):
            raise ValueError('total_found must match length of chunks list')
        return v


class ChunkComparison(BaseModel):
    """Comparison between different chunking strategies"""
    strategy_a: str = Field(..., description="First chunking strategy")
    strategy_b: str = Field(..., description="Second chunking strategy")
    chapter_title: str = Field(..., description="Chapter being compared")
    
    chunks_a: int = Field(..., ge=0, description="Number of chunks in strategy A")
    chunks_b: int = Field(..., ge=0, description="Number of chunks in strategy B")
    
    avg_size_a: float = Field(..., ge=0, description="Average chunk size in strategy A")
    avg_size_b: float = Field(..., ge=0, description="Average chunk size in strategy B")
    
    size_variance_a: float = Field(..., ge=0, description="Size variance in strategy A")
    size_variance_b: float = Field(..., ge=0, description="Size variance in strategy B")
    
    recommendation: Optional[str] = Field(None, description="Recommended strategy")
    notes: Optional[str] = Field(None, description="Additional notes")


# Response Models
class ChapterContentResponse(BaseModel):
    """Response for chapter content request"""
    doc_id: str = Field(..., description="Document ID")
    chapter: ChapterContent = Field(..., description="Chapter content and chunks")


class ChunkListResponse(BaseModel):
    """Response for chunk list request"""
    doc_id: str = Field(..., description="Document ID")
    chunks: List[TextChunk] = Field(..., description="List of chunks")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks")
    
    @validator('total_chunks')
    def total_chunks_matches_list(cls, v, values):
        if 'chunks' in values and v != len(values['chunks']):
            raise ValueError('total_chunks must match length of chunks list')
        return v


class ChunkingComparisonResponse(BaseModel):
    """Response for chunking strategy comparison"""
    comparisons: List[ChunkComparison] = Field(..., description="Strategy comparisons")
    overall_recommendation: Optional[str] = Field(None, description="Overall recommendation")
    summary: Dict[str, Any] = Field(..., description="Comparison summary")