from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request for asking a question"""
    query: str = Field(..., min_length=1, description="The question to ask")
    max_chunks: Optional[int] = Field(2, ge=1, le=5, description="Maximum target chunks to use")
    search_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Additional search parameters (chapter_filter, context_strategy, etc.)"
    )


class CitationInfo(BaseModel):
    """Citation information for response"""
    chapter_name: str = Field(..., description="Chapter name")
    page_number: int = Field(..., description="Page number")
    chunk_id: str = Field(..., description="Source chunk ID")
    citation_text: str = Field(..., description="Formatted citation")


class AnswerResponse(BaseModel):
    """Response for question answering"""
    doc_id: str = Field(..., description="Document ID")
    query: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    citations: List[CitationInfo] = Field(..., description="Source citations")
    
    # Metadata
    chunks_used: int = Field(..., description="Number of target chunks used")
    total_context_words: int = Field(..., description="Total words in context")
    generation_time: Optional[float] = Field(None, description="Time to generate answer")
    
    # Search info
    search_strategy: str = Field(..., description="Search strategy used")
    context_expansion: bool = Field(..., description="Whether context was expanded")
    
    # Success/error info
    success: bool = Field(..., description="Whether generation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Additional info
    has_embeddings: bool = Field(True, description="Whether document has embeddings")
    architecture: str = Field("paragraph_based", description="System architecture")


class QuickAnswerResponse(BaseModel):
    """Simplified response for quick queries"""
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(..., description="Simple source list")
    success: bool = Field(..., description="Whether generation succeeded")


class ConversationRequest(BaseModel):
    """Request for conversation-based question answering"""
    query: str = Field(..., min_length=1, description="The question to ask")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Previous Q&A pairs"
    )
    max_chunks: Optional[int] = Field(2, ge=1, le=5, description="Maximum target chunks to use")


class ConversationResponse(BaseModel):
    """Response for conversation-based question answering"""
    doc_id: str = Field(..., description="Document ID")
    conversation_id: str = Field(..., description="Conversation ID")
    query: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    citations: List[CitationInfo] = Field(..., description="Source citations")
    
    # Conversation context
    conversation_length: int = Field(..., description="Number of turns in conversation")
    context_used_previous: bool = Field(..., description="Whether previous context was used")
    
    # Standard metadata
    chunks_used: int = Field(..., description="Number of target chunks used") 
    success: bool = Field(..., description="Whether generation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")