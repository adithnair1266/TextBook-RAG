import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")
    SentenceTransformer = None

from models.simple_chunk import (ContextExpandedResult, SimpleChunk,
                                 SimpleChunkList)

from .context_service import ContextService, default_context_service


class EmbeddingModel:
    """Wrapper for sentence transformer models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers package required. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        print(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            return np.array([])
        
        print(f"Generating embeddings for {len(texts)} texts...")
        try:
            # Convert to numpy array and normalize
            embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
            # Normalize for cosine similarity (FAISS IndexFlatIP expects normalized vectors)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.embed_texts([text])[0]


class ChunkFAISSIndex:
    """FAISS index for paragraph chunk storage and retrieval"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Cosine similarity
        self.chunk_metadata: List[Dict[str, Any]] = []  # Stores chunk metadata
        self.chunk_id_to_index: Dict[str, int] = {}  # Maps chunk_id to FAISS index position
    
    def add_chunks(self, chunks: List[SimpleChunk], embeddings: np.ndarray):
        """Add chunks with their embeddings"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata for each chunk
        for i, chunk in enumerate(chunks):
            faiss_index = self.index.ntotal - len(chunks) + i
            
            # Store chunk metadata
            chunk_metadata = {
                # Core chunk info
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "chapter_name": chunk.chapter_name,
                "paragraph_index": chunk.paragraph_index,
                "page_number": chunk.page_number,
                "word_count": chunk.word_count,
                
                # Navigation
                "prev_chunk_id": chunk.prev_chunk_id,
                "next_chunk_id": chunk.next_chunk_id,
                
                # Context info
                "context_info": chunk.context_info,
                "display_title": chunk.display_title,
                
                # Index tracking
                "faiss_index": faiss_index,
                "embedded_at": datetime.utcnow().isoformat()
            }
            
            self.chunk_metadata.append(chunk_metadata)
            self.chunk_id_to_index[chunk.chunk_id] = faiss_index
        
        print(f"Added {len(chunks)} chunks to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, 
               chapter_filter: Optional[str] = None,
               page_filter: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Search for similar chunks with optional filtering"""
        if self.index.ntotal == 0:
            return np.array([]), []
        
        # Ensure query is normalized and correct shape
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search more results than needed for filtering
        search_k = min(k * 5, self.index.ntotal) if (chapter_filter or page_filter) else k
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Get metadata and apply filters
        results_metadata = []
        final_scores = []
        
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunk_metadata):
                metadata = self.chunk_metadata[idx].copy()
                
                # Apply chapter filter
                if chapter_filter and metadata["chapter_name"] != chapter_filter:
                    continue
                
                # Apply page filter
                if page_filter and metadata["page_number"] != page_filter:
                    continue
                
                metadata["similarity_score"] = float(scores[0][i])
                results_metadata.append(metadata)
                final_scores.append(scores[0][i])
                
                # Stop when we have enough results
                if len(results_metadata) >= k:
                    break
        
        return np.array(final_scores), results_metadata
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by ID"""
        faiss_index = self.chunk_id_to_index.get(chunk_id)
        if faiss_index is not None and faiss_index < len(self.chunk_metadata):
            return self.chunk_metadata[faiss_index].copy()
        return None
    
    def save(self, index_path: Path, metadata_path: Path):
        """Save index and metadata to disk"""
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        save_data = {
            "chunk_metadata": self.chunk_metadata,
            "chunk_id_to_index": self.chunk_id_to_index
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    def load(self, index_path: Path, metadata_path: Path):
        """Load index and metadata from disk"""
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Index or metadata file not found")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        self.chunk_metadata = save_data.get("chunk_metadata", [])
        self.chunk_id_to_index = save_data.get("chunk_id_to_index", {})
        
        print(f"Loaded index from {index_path} with {self.index.ntotal} chunks")


class EmbeddingService:
    """Embedding service for paragraph-based chunks with context expansion"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_dir: Path = None):
        self.model_name = model_name
        self.data_dir = data_dir or Path("../data")
        self.embeddings_dir = self.data_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = EmbeddingModel(model_name)
        
        # Context service for result expansion
        self.context_service = default_context_service
        
        # Cache for loaded indexes
        self.loaded_indexes: Dict[str, ChunkFAISSIndex] = {}
        self.loaded_chunks: Dict[str, List[SimpleChunk]] = {}
    
    def embed_document_chunks(self, doc_id: str, chapter_chunk_lists: List[SimpleChunkList]) -> Dict[str, Any]:
        """Generate embeddings for paragraph chunks"""
        print(f"Generating embeddings for document: {doc_id}")
        
        # Create document embedding directory
        doc_embedding_dir = self.embeddings_dir / doc_id
        doc_embedding_dir.mkdir(exist_ok=True)
        
        # Collect all chunks from all chapters
        all_chunks = []
        for chunk_list in chapter_chunk_lists:
            all_chunks.extend(chunk_list.chunks)
        
        if not all_chunks:
            raise ValueError("No chunks found to embed")
        
        # Prepare texts for embedding (with chapter context)
        texts_to_embed = []
        for chunk in all_chunks:
            # Add chapter context to embedding text
            enhanced_text = f"Chapter: {chunk.chapter_name}\n\n{chunk.text}"
            texts_to_embed.append(enhanced_text)
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_texts(texts_to_embed)
        
        # Create and populate FAISS index
        index = ChunkFAISSIndex(self.embedding_model.embedding_dim)
        index.add_chunks(all_chunks, embeddings)
        
        # Save index and metadata
        index_path = doc_embedding_dir / "chunks.faiss"
        metadata_path = doc_embedding_dir / "chunks_metadata.json"
        index.save(index_path, metadata_path)
        
        # Save embedding info
        embedding_info = {
            "doc_id": doc_id,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_model.embedding_dim,
            "embedding_type": "paragraph_based",
            "total_chunks": len(all_chunks),
            "total_chapters": len(chapter_chunk_lists),
            "chapter_breakdown": {
                chunk_list.chapter_name: len(chunk_list.chunks) 
                for chunk_list in chapter_chunk_lists
            },
            "created_at": datetime.utcnow().isoformat(),
            "index_path": str(index_path),
            "metadata_path": str(metadata_path)
        }
        
        info_path = doc_embedding_dir / "embedding_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Generated embeddings: {len(all_chunks)} chunks across {len(chapter_chunk_lists)} chapters")
        
        return embedding_info
    
    def load_document_assets(self, doc_id: str) -> Tuple[ChunkFAISSIndex, List[SimpleChunk]]:
        """Load index and chunks for a document"""
        cache_key = doc_id
        
        if cache_key in self.loaded_indexes and cache_key in self.loaded_chunks:
            return self.loaded_indexes[cache_key], self.loaded_chunks[cache_key]
        
        doc_embedding_dir = self.embeddings_dir / doc_id
        
        # Check if files exist
        index_path = doc_embedding_dir / "chunks.faiss"
        metadata_path = doc_embedding_dir / "chunks_metadata.json"
        
        if not all(path.exists() for path in [index_path, metadata_path]):
            raise FileNotFoundError(f"Embeddings not found for document: {doc_id}")
        
        # Load index
        index = ChunkFAISSIndex(self.embedding_model.embedding_dim)
        index.load(index_path, metadata_path)
        
        # Convert metadata back to SimpleChunk objects for context service
        chunks = []
        for chunk_meta in index.chunk_metadata:
            chunk = SimpleChunk(
                chunk_id=chunk_meta["chunk_id"],
                text=chunk_meta["text"],
                chapter_name=chunk_meta["chapter_name"],
                paragraph_index=chunk_meta["paragraph_index"],
                page_number=chunk_meta["page_number"],
                word_count=chunk_meta["word_count"]
            )
            chunk.prev_chunk_id = chunk_meta.get("prev_chunk_id")
            chunk.next_chunk_id = chunk_meta.get("next_chunk_id")
            chunks.append(chunk)
        
        # Cache the loaded assets
        self.loaded_indexes[cache_key] = index
        self.loaded_chunks[cache_key] = chunks
        
        print(f"Loaded assets for document: {doc_id} ({index.index.ntotal} chunks)")
        return index, chunks
    
    def search_document(self, doc_id: str, query: str, k: int = 5,
                       expand_context: bool = True,
                       context_strategy: str = "paragraph",
                       context_window: int = 2,
                       chapter_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search document with context expansion"""
        try:
            # Load document assets
            index, all_chunks = self.load_document_assets(doc_id)
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_single(query)
            
            # Search with optional filtering
            scores, results_metadata = index.search(
                query_embedding, k, chapter_filter=chapter_filter
            )
            
            if not results_metadata:
                return []
            
            # Convert results to expanded context if requested
            search_results = []
            for i, metadata in enumerate(results_metadata):
                # Create SimpleChunk from metadata
                target_chunk = SimpleChunk(
                    chunk_id=metadata["chunk_id"],
                    text=metadata["text"],
                    chapter_name=metadata["chapter_name"],
                    paragraph_index=metadata["paragraph_index"],
                    page_number=metadata["page_number"],
                    word_count=metadata["word_count"]
                )
                
                if expand_context:
                    # Use context service to expand
                    expanded_result = self.context_service.expand_search_result(
                        target_chunk=target_chunk,
                        all_chunks=all_chunks,
                        similarity_score=float(scores[i]),
                        strategy=context_strategy,
                        context_window=context_window
                    )
                    search_results.append(expanded_result.dict())
                else:
                    # Return basic result
                    basic_result = {
                        "chunk_id": metadata["chunk_id"],
                        "text": metadata["text"],
                        "chapter_name": metadata["chapter_name"],
                        "paragraph_index": metadata["paragraph_index"],
                        "page_number": metadata["page_number"],
                        "similarity_score": float(scores[i]),
                        "context_info": metadata["context_info"]
                    }
                    search_results.append(basic_result)
            
            return search_results
            
        except FileNotFoundError:
            print(f"No embeddings found for document: {doc_id}")
            return []
        except Exception as e:
            print(f"Search error for document {doc_id}: {str(e)}")
            return []
    
    def get_chunk_context(self, doc_id: str, chunk_id: str, 
                         strategy: str = "paragraph", 
                         window: int = 2) -> Dict[str, Any]:
        """Get context for a specific chunk"""
        try:
            index, all_chunks = self.load_document_assets(doc_id)
            
            # Find target chunk
            target_chunk = None
            for chunk in all_chunks:
                if chunk.chunk_id == chunk_id:
                    target_chunk = chunk
                    break
            
            if not target_chunk:
                return {"error": "Chunk not found"}
            
            # Get context using context service
            expanded_result = self.context_service.expand_search_result(
                target_chunk=target_chunk,
                all_chunks=all_chunks,
                similarity_score=1.0,  # Not a search result
                strategy=strategy,
                context_window=window
            )
            
            return expanded_result.dict()
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings"""
        stats = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_model.embedding_dim,
            "embedding_type": "paragraph_based",
            "documents_with_embeddings": [],
            "total_chunks": 0,
            "total_documents": 0
        }
        
        for doc_dir in self.embeddings_dir.iterdir():
            if doc_dir.is_dir():
                info_path = doc_dir / "embedding_info.json"
                if info_path.exists():
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            doc_info = json.load(f)
                        
                        stats["documents_with_embeddings"].append(doc_info)
                        stats["total_documents"] += 1
                        stats["total_chunks"] += doc_info.get("total_chunks", 0)
                        
                    except Exception as e:
                        print(f"Error reading embedding info for {doc_dir.name}: {str(e)}")
        
        return stats


# Default service instance
default_embedding_service = EmbeddingService()


# Convenience functions
def embed_document_chunks(doc_id: str, chapter_chunk_lists: List[SimpleChunkList]) -> Dict[str, Any]:
    """Generate embeddings for document chunks (convenience function)"""
    return default_embedding_service.embed_document_chunks(doc_id, chapter_chunk_lists)


def search_document_chunks(doc_id: str, query: str, k: int = 5, 
                          expand_context: bool = True, 
                          context_strategy: str = "paragraph") -> List[Dict[str, Any]]:
    """Search document chunks with context expansion (convenience function)"""
    return default_embedding_service.search_document(
        doc_id, query, k, expand_context, context_strategy
    )