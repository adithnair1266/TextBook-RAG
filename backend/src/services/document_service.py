import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pymupdf4llm
from utils.file_utils import sanitize_chapter_filename
from utils.text_utils import clean_chapter_content, extract_word_count

from .chunking_service import (ChunkingService, HierarchicalChunkTree,
                               default_chunking_service)
from .embedding_service import EmbeddingService, default_embedding_service


class HierarchicalDocumentProcessor:
    """Document processor for true hierarchical chunking with tree structures"""
    
    def __init__(self, chunking_service: ChunkingService = None, embedding_service: EmbeddingService = None):
        self.chunking_service = chunking_service or default_chunking_service
        self.embedding_service = embedding_service or default_embedding_service
    
    def set_chunking_service(self, chunking_service: ChunkingService):
        """Switch to a different chunking service"""
        self.chunking_service = chunking_service
    
    def set_embedding_service(self, embedding_service: EmbeddingService):
        """Switch to a different embedding service"""
        self.embedding_service = embedding_service
    
    async def extract_chapter_tree(self, pdf_path: str, chapter: Dict[str, Any]) -> HierarchicalChunkTree:
        """Extract and chunk content for a single chapter into hierarchical tree"""
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
        
        # Build hierarchical chunk tree
        chunk_tree = self.chunking_service.chunk_chapter_content(cleaned_content, chapter)
        
        # Add metadata to tree
        tree_metadata = {
            "chapter_title": chapter["title"],
            "page_range": {"start": start_page, "end": end_page},
            "total_word_count": extract_word_count(cleaned_content),
            "extracted_at": datetime.utcnow().isoformat()
        }
        
        # Store metadata in tree (add to first chunk as chapter metadata)
        root_chunks = chunk_tree.get_chunks_by_level(1)
        if root_chunks:
            root_chunks[0].metadata.update(tree_metadata)
        
        return chunk_tree
    
    async def process_all_chapters(self, pdf_path: str, chapters_with_boundaries: List[Dict[str, Any]]) -> List[HierarchicalChunkTree]:
        """Extract and chunk all chapters into hierarchical trees"""
        all_chapter_trees = []
        total_chunks = 0
        
        print(f"Extracting and building hierarchical trees for {len(chapters_with_boundaries)} chapters...")
        
        for chapter in chapters_with_boundaries:
            try:
                chapter_tree = await self.extract_chapter_tree(pdf_path, chapter)
                all_chapter_trees.append(chapter_tree)
                
                # Count chunks across all levels
                tree_chunk_count = len(chapter_tree.get_all_chunks())
                level_breakdown = {
                    f"level_{i}": len(chapter_tree.get_chunks_by_level(i)) 
                    for i in range(1, 5)
                }
                
                level_info = ", ".join([
                    f"L{level[-1]}:{count}" for level, count in level_breakdown.items() if count > 0
                ])
                
                print(f"✅ Built tree: {chapter['title']} ({tree_chunk_count} chunks: {level_info})")
                total_chunks += tree_chunk_count
                
            except Exception as e:
                print(f"❌ Failed to build tree for '{chapter['title']}': {str(e)}")
                raise ValueError(f"Chapter tree building failed for '{chapter['title']}': {str(e)}")
        
        print(f"Total chapters processed: {len(all_chapter_trees)}")
        print(f"Total chunks created: {total_chunks}")
        
        return all_chapter_trees
    
    def save_chapter_trees_to_disk(self, doc_id: str, chapter_trees: List[HierarchicalChunkTree], documents_dir: Path) -> List[Dict[str, Any]]:
        """Save hierarchical chapter trees to disk and return summaries"""
        chapters_dir = documents_dir / f"{doc_id}_chapters"
        chapters_dir.mkdir(exist_ok=True)
        
        saved_chapters = []
        
        for i, chapter_tree in enumerate(chapter_trees):
            # Get chapter info from root chunk
            root_chunks = chapter_tree.get_chunks_by_level(1)
            if not root_chunks:
                print(f"Warning: No root chunk found for chapter tree {i}")
                continue
            
            root_chunk = root_chunks[0]
            chapter_title = root_chunk.metadata.get("chapter_title", f"Chapter {i+1}")
            
            # Create sanitized filename
            filename = sanitize_chapter_filename(chapter_title)
            chapter_path = chapters_dir / f"{filename}.json"
            
            # Prepare tree data for saving
            tree_data = {
                "chapter_info": {
                    "title": chapter_title,
                    "page_range": root_chunk.metadata.get("page_range", {}),
                    "total_word_count": root_chunk.metadata.get("total_word_count", 0),
                    "extracted_at": root_chunk.metadata.get("extracted_at"),
                    "is_hierarchical": True
                },
                "chunk_tree": chapter_tree.to_dict(),
                "statistics": self._calculate_tree_statistics(chapter_tree)
            }
            
            # Save chapter tree
            with open(chapter_path, "w", encoding="utf-8") as f:
                json.dump(tree_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Create summary for metadata
            tree_stats = tree_data["statistics"]
            chapter_summary = {
                "title": chapter_title,
                "filename": f"{filename}.json",
                "page_range": root_chunk.metadata.get("page_range", {}),
                "word_count": root_chunk.metadata.get("total_word_count", 0),
                "chunk_count": tree_stats["total_chunks"],
                "chunking_strategy": "TrueHierarchicalChunker",
                "is_hierarchical": True,
                "tree_depth": tree_stats["max_depth"],
                "level_breakdown": tree_stats["chunks_by_level"]
            }
            
            saved_chapters.append(chapter_summary)
        
        print(f"Saved {len(saved_chapters)} hierarchical chapter trees to {chapters_dir}")
        return saved_chapters
    
    def _calculate_tree_statistics(self, tree: HierarchicalChunkTree) -> Dict[str, Any]:
        """Calculate statistics for a hierarchical chunk tree"""
        all_chunks = tree.get_all_chunks()
        
        if not all_chunks:
            return {
                "total_chunks": 0,
                "max_depth": 0,
                "chunks_by_level": {},
                "avg_word_count": 0,
                "total_word_count": 0
            }
        
        chunks_by_level = {}
        total_words = 0
        max_depth = 0
        
        for chunk in all_chunks:
            level = chunk.level
            chunks_by_level[f"level_{level}"] = chunks_by_level.get(f"level_{level}", 0) + 1
            total_words += chunk.word_count
            max_depth = max(max_depth, level)
        
        return {
            "total_chunks": len(all_chunks),
            "max_depth": max_depth,
            "chunks_by_level": chunks_by_level,
            "avg_word_count": total_words / len(all_chunks) if all_chunks else 0,
            "total_word_count": total_words
        }
    
    def create_document_metadata(self, doc_id: str, filename: str, page_count: int, 
                               content_starts_at: int, saved_chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create document metadata for hierarchical structure"""
        total_word_count = sum(ch["word_count"] for ch in saved_chapters)
        total_chunk_count = sum(ch["chunk_count"] for ch in saved_chapters)
        
        # Aggregate hierarchical statistics
        total_level_breakdown = {}
        max_tree_depth = 0
        
        for chapter in saved_chapters:
            chapter_levels = chapter.get("level_breakdown", {})
            max_tree_depth = max(max_tree_depth, chapter.get("tree_depth", 0))
            
            for level, count in chapter_levels.items():
                total_level_breakdown[level] = total_level_breakdown.get(level, 0) + count
        
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
            "chunking_strategy": "TrueHierarchicalChunker",
            "is_hierarchical": True,
            "tree_structure": {
                "max_depth": max_tree_depth,
                "total_level_breakdown": total_level_breakdown,
                "chapters_with_trees": len(saved_chapters)
            },
            "chapters": saved_chapters
        }
        
        return metadata
    
    def save_document_metadata(self, metadata: Dict[str, Any], documents_dir: Path):
        """Save document metadata to disk"""
        doc_id = metadata["doc_id"]
        metadata_path = documents_dir / f"{doc_id}_metadata.json"
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Saved hierarchical metadata to {metadata_path}")
    
    def save_toc_data(self, doc_id: str, confirmed_toc: Dict[str, Any], 
                     content_starts_at: int, chapters_with_mapping: List[Dict[str, Any]], 
                     documents_dir: Path):
        """Save TOC data for reference"""
        toc_data_record = {
            "confirmed_toc": confirmed_toc,
            "content_starts_at": content_starts_at,
            "chapters_with_mapping": chapters_with_mapping,
            "processing_type": "hierarchical_trees",
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
        """Complete document processing pipeline with hierarchical trees and embeddings"""
        
        print(f"Starting hierarchical document processing for {doc_id}...")
        print("🌳 Building hierarchical chunk trees with true parent-child relationships")
        
        # Extract and build hierarchical trees for all chapters
        all_chapter_trees = await self.process_all_chapters(pdf_path, chapters_with_boundaries)
        
        print(f"📊 Processing mode: hierarchical_trees")
        
        # Generate embeddings from hierarchical trees
        embedding_success = False
        embedding_error = None
        
        try:
            print("🔗 Generating hierarchical embeddings with tree navigation...")
            embedding_info = self.embedding_service.embed_document_chunks(doc_id, all_chapter_trees)
            
            print(f"✅ Generated tree-based embeddings:")
            print(f"   📈 Total chunks: {embedding_info.get('total_chunks', 0)}")
            print(f"   🌲 Tree depth: {embedding_info.get('tree_depth', 0)} levels")
            
            level_stats = embedding_info.get('level_statistics', {})
            for level_name, stats in level_stats.items():
                print(f"   📊 {level_name}: {stats['chunk_count']} chunks")
                
            embedding_success = True
            
        except Exception as e:
            embedding_error = str(e)
            print(f"⚠️ Warning: Failed to generate embeddings: {embedding_error}")
            print("Continuing without embeddings - search functionality will not be available")
        
        # Save chapter trees to disk
        saved_chapters = self.save_chapter_trees_to_disk(doc_id, all_chapter_trees, documents_dir)
        
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
            metadata["embedding_type"] = "hierarchical_tree"
            metadata["embedding_info"] = {
                "total_embedded_chunks": embedding_info.get("total_chunks", 0),
                "tree_depth": embedding_info.get("tree_depth", 0),
                "level_statistics": embedding_info.get("level_statistics", {})
            }
        else:
            metadata["has_embeddings"] = False
            metadata["embedding_error"] = embedding_error
        
        self.save_document_metadata(metadata, documents_dir)
        
        # Create processing summary
        total_chunks = sum(len(tree.get_all_chunks()) for tree in all_chapter_trees)
        
        processing_summary = {
            "doc_id": doc_id,
            "processing_mode": "hierarchical_trees",
            "chapters_processed": len(saved_chapters),
            "total_chunks": total_chunks,
            "chunking_strategy": "TrueHierarchicalChunker",
            "embeddings_generated": embedding_success,
            "embedding_type": "hierarchical_tree" if embedding_success else None,
            "tree_statistics": {
                "max_depth": metadata["tree_structure"]["max_depth"],
                "level_breakdown": metadata["tree_structure"]["total_level_breakdown"]
            }
        }
        
        return processing_summary


class HierarchicalDocumentReader:
    """Reader service for hierarchical documents with tree navigation"""
    
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
    
    def get_chapter_tree(self, doc_id: str, chapter_filename: str) -> Dict[str, Any]:
        """Get full hierarchical tree for a specific chapter"""
        chapter_path = self.documents_dir / f"{doc_id}_chapters" / chapter_filename
        
        if not chapter_path.exists():
            raise FileNotFoundError(f"Chapter tree not found: {chapter_filename}")
        
        with open(chapter_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_all_chunks_from_trees(self, doc_id: str, level_filter: int = None) -> List[Dict[str, Any]]:
        """Get all chunks from all chapter trees, optionally filtered by level"""
        metadata = self.get_document_metadata(doc_id)
        all_chunks = []
        
        for chapter_info in metadata["chapters"]:
            try:
                chapter_data = self.get_chapter_tree(doc_id, chapter_info["filename"])
                tree_data = chapter_data.get("chunk_tree", {})
                
                # Reconstruct tree and get chunks
                from .chunking_service import HierarchicalChunkTree
                tree = HierarchicalChunkTree.from_dict(tree_data)
                
                if level_filter is not None:
                    chunks = tree.get_chunks_by_level(level_filter)
                else:
                    chunks = tree.get_all_chunks()
                
                # Convert chunks to dictionaries and add context
                for chunk in chunks:
                    chunk_dict = chunk.to_dict()
                    chunk_dict["doc_id"] = doc_id
                    chunk_dict["chapter_filename"] = chapter_info["filename"]
                    all_chunks.append(chunk_dict)
                
            except Exception as e:
                print(f"Error reading chapter tree {chapter_info['filename']}: {str(e)}")
                continue
        
        return all_chunks
    
    def get_tree_navigation(self, doc_id: str, chapter_filename: str, chunk_id: str) -> Dict[str, Any]:
        """Get hierarchical navigation for a specific chunk"""
        try:
            chapter_data = self.get_chapter_tree(doc_id, chapter_filename)
            tree_data = chapter_data.get("chunk_tree", {})
            
            from .chunking_service import HierarchicalChunkTree
            tree = HierarchicalChunkTree.from_dict(tree_data)
            
            chunk = tree.get_chunk(chunk_id)
            if not chunk:
                return {"error": "Chunk not found"}
            
            context_window = tree.get_context_window(chunk_id, levels_up=2, levels_down=1)
            
            return {
                "target_chunk": chunk.to_dict(),
                "ancestors": [ancestor.to_dict() for ancestor in context_window.get("ancestors", [])],
                "descendants": [desc.to_dict() for desc in context_window.get("descendants", [])],
                "navigation_path": " > ".join(chunk.path),
                "context_text": context_window.get("context_text", "")
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_hierarchical_statistics(self, doc_id: str) -> Dict[str, Any]:
        """Get detailed hierarchical statistics for a document"""
        try:
            metadata = self.get_document_metadata(doc_id)
            
            if not metadata.get("is_hierarchical"):
                return {"error": "Document does not use hierarchical chunking"}
            
            tree_structure = metadata.get("tree_structure", {})
            
            # Get detailed chunk information
            all_chunks = self.get_all_chunks_from_trees(doc_id)
            
            # Calculate advanced statistics
            level_stats = {}
            total_relationships = 0
            
            for level in range(1, tree_structure.get("max_depth", 0) + 1):
                level_chunks = [c for c in all_chunks if c["level"] == level]
                if level_chunks:
                    avg_children = sum(len(c.get("children_ids", [])) for c in level_chunks) / len(level_chunks)
                    total_relationships += sum(len(c.get("children_ids", [])) for c in level_chunks)
                    
                    level_stats[f"level_{level}"] = {
                        "chunk_count": len(level_chunks),
                        "avg_word_count": sum(c["word_count"] for c in level_chunks) / len(level_chunks),
                        "avg_children": avg_children,
                        "sample_paths": [c.get("path", []) for c in level_chunks[:3]]
                    }
            
            return {
                "doc_id": doc_id,
                "tree_structure": tree_structure,
                "level_statistics": level_stats,
                "total_chunks": len(all_chunks),
                "total_relationships": total_relationships,
                "navigation_depth": tree_structure.get("max_depth", 0)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
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
                "hierarchical_documents": 0,
                "total_chunks": 0,
                "total_relationships": 0
            }
        
        hierarchical_docs = [doc for doc in documents if doc.get("is_hierarchical")]
        
        total_chunks = sum(doc.get("total_chunk_count", 0) for doc in documents)
        total_words = sum(doc.get("total_word_count", 0) for doc in documents)
        
        # Aggregate hierarchical statistics
        total_level_breakdown = {}
        total_relationships = 0
        
        for doc in hierarchical_docs:
            tree_structure = doc.get("tree_structure", {})
            level_breakdown = tree_structure.get("total_level_breakdown", {})
            
            for level, count in level_breakdown.items():
                total_level_breakdown[level] = total_level_breakdown.get(level, 0) + count
            
            # Estimate relationships (each non-leaf chunk has children)
            non_leaf_levels = ["level_1", "level_2", "level_3"]
            for level in non_leaf_levels:
                if level in level_breakdown:
                    total_relationships += level_breakdown[level]
        
        return {
            "total_documents": len(documents),
            "hierarchical_documents": len(hierarchical_docs),
            "flat_documents": len(documents) - len(hierarchical_docs),
            "total_chapters": sum(doc.get("chapters_count", 0) for doc in documents),
            "total_chunks": total_chunks,
            "total_words": total_words,
            "hierarchical_statistics": {
                "total_level_breakdown": total_level_breakdown,
                "total_relationships": total_relationships,
                "avg_tree_depth": sum(
                    doc.get("tree_structure", {}).get("max_depth", 0) 
                    for doc in hierarchical_docs
                ) / len(hierarchical_docs) if hierarchical_docs else 0
            },
            "latest_document": documents[0] if documents else None
        }


# Default instances
default_document_processor = HierarchicalDocumentProcessor()
default_document_reader = HierarchicalDocumentReader


# Convenience function for backward compatibility
def DocumentReader(documents_dir: Path):
    """Factory function for document reader"""
    return HierarchicalDocumentReader(documents_dir)