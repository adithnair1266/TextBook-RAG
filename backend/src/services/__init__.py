"""Services for the RAG system - True Hierarchical Implementation"""

# TOC Services
# Hierarchical Chunking Services
from .chunking_service import (ChunkingConfig, ChunkingService,
                               HierarchicalChunk, HierarchicalChunkTree,
                               TrueHierarchicalChunker, chunk_chapter_content,
                               default_chunking_service)
# Hierarchical Document Services
from .document_service import (DocumentReader, HierarchicalDocumentProcessor,
                               HierarchicalDocumentReader,
                               default_document_processor)
# Hierarchical Embedding Services
from .embedding_service import (EmbeddingModel, EmbeddingService,
                                HierarchicalEmbeddingService,
                                HierarchicalFAISSIndex,
                                HierarchicalSearchResult,
                                default_embedding_service,
                                embed_document_chunks, search_document_chunks)
from .toc_service import (ChunkedTOCParser, SinglePassTOCParser,
                          TOCParsingStrategy, TOCService, default_toc_service,
                          parse_toc_single_pass, parse_toc_with_page_chunks)
# Validation Services
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
    
    # Hierarchical Chunking Services
    "ChunkingConfig",
    "HierarchicalChunk",
    "HierarchicalChunkTree", 
    "TrueHierarchicalChunker",
    "ChunkingService",
    "default_chunking_service",
    "chunk_chapter_content",
    
    # Hierarchical Document Services
    "HierarchicalDocumentProcessor",
    "HierarchicalDocumentReader",
    "DocumentReader",  # Factory function for backward compatibility
    "default_document_processor",
    
    # Hierarchical Embedding Services
    "EmbeddingModel",
    "HierarchicalFAISSIndex",
    "HierarchicalEmbeddingService",
    "HierarchicalSearchResult",
    "EmbeddingService",
    "default_embedding_service",
    "embed_document_chunks",
    "search_document_chunks",
    
    # Validation Services
    "validate_toc_structure",
    "apply_page_offset_mapping",
    "validate_chapter_mapping", 
    "calculate_chapter_boundaries",
    "validate_content_starts_at",
    "validate_chapters_exist",
    "validate_chapter_data"
]