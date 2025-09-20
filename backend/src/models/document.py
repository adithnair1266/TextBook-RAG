from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PageRange(BaseModel):
    """Page range for chapters"""
    start: int = Field(..., ge=1, description="Starting page number")
    end: int = Field(..., ge=1, description="Ending page number")
    
    @validator('end')
    def end_must_be_after_start(cls, v, values):
        if 'start' in values and v < values['start']:
            raise ValueError('End page must be >= start page')
        return v


class Subsection(BaseModel):
    """Subsection within a chapter"""
    title: str = Field(..., min_length=1, description="Subsection title")
    start_page: int = Field(..., ge=1, description="Starting page number")
    toc_start_page: Optional[int] = Field(None, description="Original TOC page number")


class TOCChapter(BaseModel):
    """Chapter as parsed from TOC"""
    title: str = Field(..., min_length=1, description="Chapter title")
    start_page: int = Field(..., ge=1, description="Starting page number")
    subsections: List[Subsection] = Field(default_factory=list, description="Chapter subsections")


class MappedChapter(TOCChapter):
    """Chapter with page mapping applied"""
    toc_start_page: int = Field(..., ge=1, description="Original TOC page number")
    end_page: Optional[int] = Field(None, ge=1, description="Ending page number")


class ParsedTOC(BaseModel):
    """Parsed table of contents"""
    chapters: List[TOCChapter] = Field(..., min_items=1, description="List of chapters")
    page_chunks_processed: Optional[int] = Field(None, description="Number of page chunks processed")
    total_chapters: int = Field(..., ge=0, description="Total number of chapters found")
    
    @validator('total_chapters')
    def total_chapters_matches_list(cls, v, values):
        if 'chapters' in values and v != len(values['chapters']):
            raise ValueError('total_chapters must match length of chapters list')
        return v


class ConfirmedTOC(BaseModel):
    """User-confirmed and edited TOC"""
    chapters: List[TOCChapter] = Field(..., min_items=1, description="Confirmed chapters")


class DocumentUpload(BaseModel):
    """Document upload request"""
    toc_pages: str = Field(..., description="TOC pages (e.g., '9-22' or '9,10,11')")
    
    @validator('toc_pages')
    def validate_toc_pages_format(cls, v):
        # Basic validation - detailed parsing happens in utils
        if not v.strip():
            raise ValueError('TOC pages cannot be empty')
        
        # Check for valid formats
        if not any(char in v for char in [',', '-']) and not v.isdigit():
            raise ValueError('TOC pages must be in format "9-22", "9,10,11", or "9"')
        
        return v.strip()


class TOCConfirmation(BaseModel):
    """TOC confirmation request"""
    doc_id: str = Field(..., min_length=1, description="Document ID")
    content_starts_at: int = Field(..., ge=1, description="Page where content begins")
    toc_data: ConfirmedTOC = Field(..., description="Confirmed TOC data")


class ChapterSummary(BaseModel):
    """Summary of a processed chapter"""
    title: str = Field(..., description="Chapter title")
    filename: str = Field(..., description="Chapter filename on disk")
    page_range: PageRange = Field(..., description="Page range")
    word_count: int = Field(..., ge=0, description="Word count")
    chunk_count: int = Field(..., ge=0, description="Number of chunks")
    chunking_strategy: str = Field(..., description="Chunking strategy used")


class DocumentMetadata(BaseModel):
    """Complete document metadata"""
    doc_id: str = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Original filename")
    page_count: int = Field(..., ge=1, description="Total pages in PDF")
    status: str = Field(default="processed", description="Processing status")
    chapters_count: int = Field(..., ge=0, description="Number of chapters")
    total_word_count: int = Field(..., ge=0, description="Total word count")
    total_chunk_count: int = Field(..., ge=0, description="Total chunk count")
    content_starts_at: int = Field(..., ge=1, description="Page where content starts")
    processed_at: datetime = Field(..., description="Processing timestamp")
    chunking_strategy: str = Field(..., description="Chunking strategy used")
    chapters: List[ChapterSummary] = Field(..., description="Chapter summaries")
    
    @validator('chapters_count')
    def chapters_count_matches_list(cls, v, values):
        if 'chapters' in values and v != len(values['chapters']):
            raise ValueError('chapters_count must match length of chapters list')
        return v


class DocumentStats(BaseModel):
    """Statistics about all processed documents"""
    total_documents: int = Field(..., ge=0, description="Total number of documents")
    total_chapters: int = Field(..., ge=0, description="Total number of chapters")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks")
    total_words: int = Field(..., ge=0, description="Total word count")
    chunking_strategies: Dict[str, int] = Field(..., description="Chunking strategies used")
    latest_document: Optional[DocumentMetadata] = Field(None, description="Most recent document")


# Request/Response Models
class ParseTOCResponse(BaseModel):
    """Response from TOC parsing"""
    doc_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    page_count: int = Field(..., ge=1, description="PDF page count")
    toc_pages: List[int] = Field(..., description="TOC page numbers")
    parsed_toc: ParsedTOC = Field(..., description="Parsed TOC data")
    message: str = Field(..., description="Success message")


class ConfirmTOCResponse(BaseModel):
    """Response from TOC confirmation"""
    doc_id: str = Field(..., description="Document ID")
    status: str = Field(default="processed", description="Processing status")
    chapters_count: int = Field(..., ge=0, description="Number of chapters")
    total_word_count: int = Field(..., ge=0, description="Total word count")
    total_chunk_count: int = Field(..., ge=0, description="Total chunk count")
    chapters: List[ChapterSummary] = Field(..., description="Chapter summaries")
    message: str = Field(..., description="Success message")


class DocumentListResponse(BaseModel):
    """Response for document list"""
    documents: List[DocumentMetadata] = Field(..., description="List of documents")


class DocumentChaptersResponse(BaseModel):
    """Response for document chapters"""
    doc_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    chapters_count: int = Field(..., ge=0, description="Number of chapters")
    total_chunk_count: int = Field(..., ge=0, description="Total chunk count")
    chapters: List[ChapterSummary] = Field(..., description="Chapter summaries")


class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    doc_id: Optional[str] = Field(None, description="Document ID if applicable")


class ValidationError(BaseModel):
    """Validation error details"""
    valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    failed_chapters: Optional[List[str]] = Field(None, description="Chapters that failed validation")