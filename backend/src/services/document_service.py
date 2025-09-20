import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pymupdf4llm
from models.simple_chunk import SimpleChunk, SimpleChunkList
from utils.file_utils import sanitize_chapter_filename
from utils.text_utils import clean_chapter_content, extract_word_count

from .chunking_service import ChunkingService, default_chunking_service
from .embedding_service import EmbeddingService, default_embedding_service


class DocumentProcessor:
    """Document processor for paragraph-based chunking"""
    
    def __init__(self, chunking_service: ChunkingService = None, embedding_service: EmbeddingService = None):
        self.chunking_service = chunking_service or default_chunking_service
        self.embedding_service = embedding_service or default_embedding_service
    
    def set_chunking_service(self, chunking_service: ChunkingService):
        """Switch to a different chunking service"""
        self.chunking_service = chunking_service
    
    def set_embedding_service(self, embedding_service: EmbeddingService):
        """Switch to a different embedding service"""
        self.embedding_service = embedding_service
    
    async def extract_chapter_chunks(self, pdf_path: str, chapter: Dict[str, Any]) -> SimpleChunkList:
        """Extract and chunk content for a single chapter"""
        start_page = chapter["start_page"]
        end_page = chapter["end_page"]
        
        # Extract pages (zero-indexed for pymupdf4llm)
        chapter_pages = list(range(start_page - 1, end_page))
        
        print(f"Extracting '{chapter['title']}': pages {start_page}-{end_page}")
        
        # Extract markdown content
        chapter_markdown = pymupdf4llm.to_markdown(pdf_path, pages=chapter_pages)
        
        if not chapter_markdown or len(chapter_markdown.strip()) < 100:
            raise ValueError(f"Insufficient content extracted from pages {start_page}-{end_page}")
        
        # Clean the content before chunking
        cleaned_content = clean_chapter_content(chapter_markdown)
        
        # Chunk into paragraphs
        chunk_list = self.chunking_service.chunk_chapter_content(cleaned_content, chapter)
        
        print(f"✅ Chunked: {chapter['title']} ({len(chunk_list.chunks)} paragraphs)")
        
        return chunk_list
    
    async def process_all_chapters(self, pdf_path: str, chapters_with_boundaries: List[Dict[str, Any]]) -> List[SimpleChunkList]:
        """Extract and chunk all chapters"""
        all_chapter_chunks = []
        total_chunks = 0
        
        print(f"Processing {len(chapters_with_boundaries)} chapters into paragraphs...")
        
        for chapter in chapters_with_boundaries:
            try:
                chunk_list = await self.extract_chapter_chunks(pdf_path, chapter)
                all_chapter_chunks.append(chunk_list)
                
                chunk_count = len(chunk_list.chunks)
                total_chunks += chunk_count
                
                print(f"✅ Processed: {chapter['title']} ({chunk_count} paragraphs, {chunk_list.total_word_count} words)")
                
            except Exception as e:
                print(f"❌ Failed to process '{chapter['title']}': {str(e)}")
                raise ValueError(f"Chapter processing failed for '{chapter['title']}': {str(e)}")
        
        print(f"Total chapters processed: {len(all_chapter_chunks)}")
        print(f"Total paragraphs created: {total_chunks}")
        
        return all_chapter_chunks
    
    def save_chapter_chunks_to_disk(self, doc_id: str, chapter_chunk_lists: List[SimpleChunkList], documents_dir: Path) -> List[Dict[str, Any]]:
        """Save chapter chunks to disk and return summaries"""
        chapters_dir = documents_dir / f"{doc_id}_chapters"
        chapters_dir.mkdir(exist_ok=True)
        
        saved_chapters = []
        
        for chunk_list in chapter_chunk_lists:
            chapter_title = chunk_list.chapter_name
            
            # Create sanitized filename
            filename = sanitize_chapter_filename(chapter_title)
            chapter_path = chapters_dir / f"{filename}.json"
            
            # Get page range from chunks
            if chunk_list.chunks:
                min_page = min(chunk.page_number for chunk in chunk_list.chunks)
                max_page = max(chunk.page_number for chunk in chunk_list.chunks)
                page_range = {"start": min_page, "end": max_page}
            else:
                page_range = {"start": 1, "end": 1}
            
            # Prepare chapter data for saving
            chapter_data = {
                "chapter_info": {
                    "title": chapter_title,
                    "page_range": page_range,
                    "total_word_count": chunk_list.total_word_count,
                    "chunk_count": chunk_list.total_chunks,
                    "processed_at": datetime.utcnow().isoformat(),
                    "chunking_strategy": "ParagraphChunker"
                },
                "chunks": [chunk.dict() for chunk in chunk_list.chunks]
            }
            
            # Save chapter chunks
            with open(chapter_path, "w", encoding="utf-8") as f:
                json.dump(chapter_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Create summary for metadata
            chapter_summary = {
                "title": chapter_title,
                "filename": f"{filename}.json",
                "page_range": page_range,
                "word_count": chunk_list.total_word_count,
                "chunk_count": chunk_list.total_chunks,
                "chunking_strategy": "ParagraphChunker"
            }
            
            saved_chapters.append(chapter_summary)
        
        print(f"Saved {len(saved_chapters)} chapter chunk files to {chapters_dir}")
        return saved_chapters
    
    def create_document_metadata(self, doc_id: str, filename: str, page_count: int, 
                               content_starts_at: int, saved_chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create document metadata"""
        total_word_count = sum(ch["word_count"] for ch in saved_chapters)
        total_chunk_count = sum(ch["chunk_count"] for ch in saved_chapters)
        
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "page_count": page_count,
            "status": "processed",
            "chapters_count": len(saved_chapters),
            "total_word_count": total_word_count,
            "total_chunk_count": total_chunk_count,
            "content_starts_at": content_starts_at,
            "processed_at": datetime.utcnow().isoformat(),
            "chunking_strategy": "ParagraphChunker",
            "architecture": "paragraph_based",
            "chapters": saved_chapters
        }
        
        return metadata
    
    def save_document_metadata(self, metadata: Dict[str, Any], documents_dir: Path):
        """Save document metadata to disk"""
        doc_id = metadata["doc_id"]
        metadata_path = documents_dir / f"{doc_id}_metadata.json"
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Saved metadata to {metadata_path}")
    
    def save_toc_data(self, doc_id: str, confirmed_toc: Dict[str, Any], 
                     content_starts_at: int, chapters_with_mapping: List[Dict[str, Any]], 
                     documents_dir: Path):
        """Save TOC data for reference"""
        toc_data_record = {
            "confirmed_toc": confirmed_toc,
            "content_starts_at": content_starts_at,
            "chapters_with_mapping": chapters_with_mapping,
            "processing_type": "paragraph_based",
            "saved_at": datetime.utcnow().isoformat()
        }
        
        toc_path = documents_dir / f"{doc_id}_toc.json"
        with open(toc_path, "w", encoding="utf-8") as f:
            json.dump(toc_data_record, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Saved TOC data to {toc_path}")
    
    async def process_document_with_embeddings(self, pdf_path: str, chapters_with_boundaries: List[Dict[str, Any]], 
                                             doc_id: str, documents_dir: Path, 
                                             confirmed_toc: Dict[str, Any], content_starts_at: int, 
                                             chapters_with_mapping: List[Dict[str, Any]], page_count: int) -> Dict[str, Any]:
        """Complete document processing pipeline with paragraph chunks and embeddings"""
        
        print(f"Starting paragraph-based document processing for {doc_id}...")
        print("📄 Processing with simple paragraph chunking")
        
        # Extract and chunk all chapters
        all_chapter_chunks = await self.process_all_chapters(pdf_path, chapters_with_boundaries)
        
        print(f"📊 Processing mode: paragraph_based")
        
        # Generate embeddings from paragraph chunks
        embedding_success = False
        embedding_error = None
        
        try:
            print("🔗 Generating embeddings for paragraph chunks...")
            embedding_info = self.embedding_service.embed_document_chunks(doc_id, all_chapter_chunks)
            
            print(f"✅ Generated embeddings:")
            print(f"   📈 Total chunks: {embedding_info.get('total_chunks', 0)}")
            print(f"   📚 Chapters: {embedding_info.get('total_chapters', 0)}")
            
            embedding_success = True
            
        except Exception as e:
            embedding_error = str(e)
            print(f"⚠️ Warning: Failed to generate embeddings: {embedding_error}")
            print("Continuing without embeddings - search functionality will not be available")
        
        # Save chapter chunks to disk
        saved_chapters = self.save_chapter_chunks_to_disk(doc_id, all_chapter_chunks, documents_dir)
        
        # Save TOC data
        self.save_toc_data(doc_id, confirmed_toc, content_starts_at, chapters_with_mapping, documents_dir)
        
        # Create and save document metadata
        metadata = self.create_document_metadata(
            doc_id, "uploaded_document.pdf", 
            page_count, content_starts_at, saved_chapters
        )
        
        # Add embedding information to metadata
        if embedding_success:
            metadata["has_embeddings"] = True
            metadata["embedding_model"] = self.embedding_service.model_name
            metadata["embedding_dimension"] = self.embedding_service.embedding_model.embedding_dim
            metadata["embedding_type"] = "paragraph_based"
            metadata["embedding_info"] = {
                "total_embedded_chunks": embedding_info.get("total_chunks", 0),
                "chapters_embedded": embedding_info.get("total_chapters", 0)
            }
        else:
            metadata["has_embeddings"] = False
            metadata["embedding_error"] = embedding_error
        
        self.save_document_metadata(metadata, documents_dir)
        
        # Create processing summary
        total_chunks = sum(len(chunk_list.chunks) for chunk_list in all_chapter_chunks)
        
        processing_summary = {
            "doc_id": doc_id,
            "processing_mode": "paragraph_based",
            "chapters_processed": len(saved_chapters),
            "total_chunks": total_chunks,
            "chunking_strategy": "ParagraphChunker",
            "embeddings_generated": embedding_success,
            "embedding_type": "paragraph_based" if embedding_success else None
        }
        
        return processing_summary


class DocumentReader:
    """Reader service for paragraph-based documents"""
    
    def __init__(self, documents_dir: Path):
        self.documents_dir = documents_dir
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get metadata for all processed documents"""
        documents = []
        
        for metadata_path in self.documents_dir.glob("*_metadata.json"):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                documents.append(metadata)
            except Exception as e:
                print(f"Error reading metadata {metadata_path}: {str(e)}")
                continue
        
        # Sort by processed date (newest first)
        documents.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
        return documents
    
    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a specific document"""
        metadata_path = self.documents_dir / f"{doc_id}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_id}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_chapter_chunks(self, doc_id: str, chapter_filename: str) -> Dict[str, Any]:
        """Get chunks for a specific chapter"""
        chapter_path = self.documents_dir / f"{doc_id}_chapters" / chapter_filename
        
        if not chapter_path.exists():
            raise FileNotFoundError(f"Chapter chunks not found: {chapter_filename}")
        
        with open(chapter_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_all_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks from all chapters as a flat list"""
        metadata = self.get_document_metadata(doc_id)
        all_chunks = []
        
        for chapter_info in metadata["chapters"]:
            try:
                chapter_data = self.get_chapter_chunks(doc_id, chapter_info["filename"])
                chunks = chapter_data.get("chunks", [])
                
                # Add document context to each chunk
                for chunk in chunks:
                    chunk["doc_id"] = doc_id
                    chunk["chapter_filename"] = chapter_info["filename"]
                    all_chunks.append(chunk)
                
            except Exception as e:
                print(f"Error reading chapter chunks {chapter_info['filename']}: {str(e)}")
                continue
        
        return all_chunks
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists"""
        metadata_path = self.documents_dir / f"{doc_id}_metadata.json"
        return metadata_path.exists()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about all processed documents"""
        documents = self.get_all_documents()
        
        if not documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_words": 0
            }
        
        total_chunks = sum(doc.get("total_chunk_count", 0) for doc in documents)
        total_words = sum(doc.get("total_word_count", 0) for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_chapters": sum(doc.get("chapters_count", 0) for doc in documents),
            "total_chunks": total_chunks,
            "total_words": total_words,
            "architecture": "paragraph_based",
            "chunking_strategy": "ParagraphChunker",
            "latest_document": documents[0] if documents else None
        }


# Default instances
default_document_processor = DocumentProcessor()

