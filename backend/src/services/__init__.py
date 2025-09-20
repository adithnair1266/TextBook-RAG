"""Services for the RAG system"""

# TOC Services
# Chunking Services (NEW - simplified)
from .chunking_service import (ChunkingConfig, ChunkingService,
                               ParagraphChunker, chunk_chapter_content,
                               default_chunking_service,
                               get_chunking_strategy_info)
# Context Services (NEW)
from .context_service import (ContextService, default_context_service,
                              expand_with_context, get_page_context,
                              get_paragraph_context)
# Document Services (UPDATED)
from .document_service import (DocumentProcessor, DocumentReader,
                               default_document_processor)
# Embedding Services (UPDATED) 
from .embedding_service import (ChunkFAISSIndex, EmbeddingModel,
                                EmbeddingService, default_embedding_service,
                                embed_document_chunks, search_document_chunks)
from .toc_service import (ChunkedTOCParser, SinglePassTOCParser,
                          TOCParsingStrategy, TOCService, default_toc_service,
                          parse_toc_single_pass, parse_toc_with_page_chunks)
# Validation Services (UNCHANGED)
from .validation_service import (apply_page_offset_mapping,
                                 calculate_chapter_boundaries,
                                 validate_chapter_data,
                                 validate_chapter_mapping,
                                 validate_chapters_exist,
                                 validate_content_starts_at,
                                 validate_toc_structure)

__all__ = [
    # TOC Services
    "TOCParsingStrategy",
    "ChunkedTOCParser", 
    "SinglePassTOCParser",
    "TOCService",
    "default_toc_service",
    "parse_toc_with_page_chunks",
    "parse_toc_single_pass",
    
    # Chunking Services (NEW)
    "ChunkingConfig",
    "ParagraphChunker",
    "ChunkingService", 
    "default_chunking_service",
    "chunk_chapter_content",
    "get_chunking_strategy_info",
    
    # Context Services (NEW)
    "ContextService",
    "default_context_service", 
    "get_paragraph_context",
    "get_page_context", 
    "expand_with_context",
    
    # Document Services (UPDATED)
    "DocumentProcessor",
    "DocumentReader",
    "default_document_processor",
    
    # Embedding Services (UPDATED)
    "EmbeddingModel", 
    "ChunkFAISSIndex",
    "EmbeddingService",
    "default_embedding_service",
    "embed_document_chunks",
    "search_document_chunks",
    
    # Validation Services (UNCHANGED)
    "validate_toc_structure",
    "apply_page_offset_mapping",
    "validate_chapter_mapping", 
    "calculate_chapter_boundaries",
    "validate_content_starts_at",
    "validate_chapters_exist",
    "validate_chapter_data"
]