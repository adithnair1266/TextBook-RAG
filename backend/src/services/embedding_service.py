import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")
    SentenceTransformer = None

from .chunking_service import HierarchicalChunk, HierarchicalChunkTree


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


class HierarchicalFAISSIndex:
    """FAISS index for hierarchical chunk storage and retrieval"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Cosine similarity
        self.chunk_metadata: List[Dict[str, Any]] = []  # Stores chunk metadata + hierarchy info
        self.chunk_id_to_index: Dict[str, int] = {}  # Maps chunk_id to FAISS index position
    
    def add_chunks(self, chunks: List[HierarchicalChunk], embeddings: np.ndarray, tree: HierarchicalChunkTree):
        """Add hierarchical chunks with their embeddings"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store clean metadata for each chunk (no context pollution)
        for i, chunk in enumerate(chunks):
            faiss_index = self.index.ntotal - len(chunks) + i
            
            # Build clean metadata - hierarchy stored for post-retrieval use
            chunk_metadata = {
                # Core chunk info
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,  # Clean chunk text only
                "level": chunk.level,
                "title": chunk.title,
                "word_count": chunk.word_count,
                "path": chunk.path,
                
                # Hierarchy navigation (for post-retrieval context)
                "parent_id": chunk.parent_id,
                "children_ids": chunk.children_ids,
                
                # Original metadata
                **chunk.metadata,
                
                # Index tracking
                "faiss_index": faiss_index,
                "embedded_at": datetime.utcnow().isoformat()
            }
            
            self.chunk_metadata.append(chunk_metadata)
            self.chunk_id_to_index[chunk.chunk_id] = faiss_index
        
        print(f"Added {len(chunks)} hierarchical chunks to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, level_filter: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Search for similar chunks with optional level filtering"""
        if self.index.ntotal == 0:
            return np.array([]), []
        
        # Ensure query is normalized and correct shape
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search more results than needed for filtering
        search_k = min(k * 3, self.index.ntotal) if level_filter else k
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Get metadata and apply level filtering
        results_metadata = []
        final_scores = []
        
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunk_metadata):
                metadata = self.chunk_metadata[idx].copy()
                
                # Apply level filter if specified
                if level_filter is None or metadata["level"] == level_filter:
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
        
        print(f"Saved hierarchical index to {index_path} and metadata to {metadata_path}")
    
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
        
        print(f"Loaded hierarchical index from {index_path} with {self.index.ntotal} chunks")


class HierarchicalSearchResult:
    """Enhanced search result with hierarchical navigation"""
    
    def __init__(self, chunk_metadata: Dict[str, Any], tree: HierarchicalChunkTree, similarity_score: float):
        self.chunk_metadata = chunk_metadata
        self.tree = tree
        self.similarity_score = similarity_score
        self.chunk_id = chunk_metadata["chunk_id"]
    
    def get_context_expansion(self, levels_up: int = 1) -> Dict[str, Any]:
        """Get expanded context by walking up the hierarchy"""
        context_window = self.tree.get_context_window(self.chunk_id, levels_up=levels_up)
        
        return {
            "target_chunk": self.chunk_metadata,
            "similarity_score": self.similarity_score,
            "ancestors": [chunk.to_dict() for chunk in context_window.get("ancestors", [])],
            "enriched_context": context_window.get("context_text", ""),
            "navigation_path": " > ".join(self.chunk_metadata.get("path", []))
        }
    
    def get_drill_down(self, levels_down: int = 1) -> Dict[str, Any]:
        """Get drill-down details by exploring children"""
        context_window = self.tree.get_context_window(self.chunk_id, levels_down=levels_down)
        
        return {
            "target_chunk": self.chunk_metadata,
            "similarity_score": self.similarity_score,
            "descendants": [chunk.to_dict() for chunk in context_window.get("descendants", [])],
            "navigation_path": " > ".join(self.chunk_metadata.get("path", []))
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            **self.chunk_metadata,
            "similarity_score": self.similarity_score,
            "navigation_path": " > ".join(self.chunk_metadata.get("path", [])),
            "has_parent": self.chunk_metadata.get("parent_id") is not None,
            "has_children": len(self.chunk_metadata.get("children_ids", [])) > 0
        }


class HierarchicalEmbeddingService:
    """True hierarchical embedding service with tree-based search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_dir: Path = None):
        self.model_name = model_name
        self.data_dir = data_dir or Path("../data")
        self.embeddings_dir = self.data_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = EmbeddingModel(model_name)
        
        # Cache for loaded indexes and trees
        self.loaded_indexes: Dict[str, HierarchicalFAISSIndex] = {}
        self.loaded_trees: Dict[str, HierarchicalChunkTree] = {}
    
    def embed_document_tree(self, doc_id: str, chapter_trees: List[HierarchicalChunkTree]) -> Dict[str, Any]:
        """Generate embeddings for hierarchical chunk trees"""
        print(f"Generating clean embeddings for document: {doc_id}")
        
        # Create document embedding directory
        doc_embedding_dir = self.embeddings_dir / doc_id
        doc_embedding_dir.mkdir(exist_ok=True)
        
        # Combine all trees into a single tree structure
        combined_tree = HierarchicalChunkTree()
        all_chunks = []
        
        for chapter_tree in chapter_trees:
            # Add all chunks from this chapter to combined tree
            for chunk in chapter_tree.get_all_chunks():
                combined_tree.add_chunk(chunk)
                all_chunks.append(chunk)
        
        if not all_chunks:
            raise ValueError("No chunks found to embed")
        
        # Prepare CLEAN texts for embedding (no metadata pollution)
        texts_to_embed = []
        for chunk in all_chunks:
            clean_text = self._create_clean_text(chunk)
            texts_to_embed.append(clean_text)
        
        print(f"Embedding {len(texts_to_embed)} chunks with clean text (no metadata pollution)")
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_texts(texts_to_embed)
        
        # Create and populate hierarchical FAISS index
        index = HierarchicalFAISSIndex(self.embedding_model.embedding_dim)
        index.add_chunks(all_chunks, embeddings, combined_tree)
        
        # Save tree structure
        tree_path = doc_embedding_dir / "chunk_tree.json"
        with open(tree_path, 'w', encoding='utf-8') as f:
            json.dump(combined_tree.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save index and metadata
        index_path = doc_embedding_dir / "hierarchical.faiss"
        metadata_path = doc_embedding_dir / "hierarchical_metadata.json"
        index.save(index_path, metadata_path)
        
        # Save embedding info
        level_stats = {}
        for level in range(1, 5):
            level_chunks = [c for c in all_chunks if c.level == level]
            if level_chunks:
                level_stats[f"level_{level}"] = {
                    "chunk_count": len(level_chunks),
                    "avg_word_count": sum(c.word_count for c in level_chunks) / len(level_chunks)
                }
        
        embedding_info = {
            "doc_id": doc_id,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_model.embedding_dim,
            "embedding_type": "hierarchical_tree_clean",
            "total_chunks": len(all_chunks),
            "total_chapters": len(chapter_trees),
            "level_statistics": level_stats,
            "tree_depth": max(chunk.level for chunk in all_chunks),
            "created_at": datetime.utcnow().isoformat(),
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "tree_path": str(tree_path),
            "clean_embeddings": True  # Flag to indicate no metadata pollution
        }
        
        info_path = doc_embedding_dir / "embedding_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Generated clean hierarchical embeddings: {len(all_chunks)} chunks")
        print(f"   Tree depth: {embedding_info['tree_depth']} levels")
        print("   Embeddings: Clean chunk text only (no metadata pollution)")
        for level_name, stats in level_stats.items():
            print(f"   {level_name}: {stats['chunk_count']} chunks")
        
        return embedding_info
    
    def _create_clean_text(self, chunk: HierarchicalChunk) -> str:
        """Create CLEAN text for embedding - no metadata pollution"""
        # Return only the actual chunk content - no path, no parent context, no title
        return chunk.text
    
    def load_document_assets(self, doc_id: str) -> Tuple[HierarchicalFAISSIndex, HierarchicalChunkTree]:
        """Load index and tree for a document"""
        cache_key = doc_id
        
        if cache_key in self.loaded_indexes and cache_key in self.loaded_trees:
            return self.loaded_indexes[cache_key], self.loaded_trees[cache_key]
        
        doc_embedding_dir = self.embeddings_dir / doc_id
        
        # Check if files exist
        index_path = doc_embedding_dir / "hierarchical.faiss"
        metadata_path = doc_embedding_dir / "hierarchical_metadata.json"
        tree_path = doc_embedding_dir / "chunk_tree.json"
        
        if not all(path.exists() for path in [index_path, metadata_path, tree_path]):
            raise FileNotFoundError(f"Hierarchical embeddings not found for document: {doc_id}")
        
        # Load index
        index = HierarchicalFAISSIndex(self.embedding_model.embedding_dim)
        index.load(index_path, metadata_path)
        
        # Load tree
        with open(tree_path, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        tree = HierarchicalChunkTree.from_dict(tree_data)
        
        # Cache the loaded assets
        self.loaded_indexes[cache_key] = index
        self.loaded_trees[cache_key] = tree
        
        print(f"Loaded hierarchical assets for document: {doc_id} ({index.index.ntotal} chunks)")
        return index, tree
    
    def search_document(self, doc_id: str, query: str, k: int = 5, 
                       level_filter: Optional[int] = None,
                       expand_context: bool = True,
                       context_levels: int = 1) -> List[Dict[str, Any]]:
        """Search document with POST-RETRIEVAL context expansion"""
        try:
            # Load document assets
            index, tree = self.load_document_assets(doc_id)
            
            # Generate query embedding from CLEAN query text
            query_embedding = self.embedding_model.embed_single(query)
            
            # Search with optional level filtering (clean embeddings)
            scores, results_metadata = index.search(query_embedding, k, level_filter)
            
            if not results_metadata:
                return []
            
            # ADD CONTEXT AFTER RETRIEVAL (not during embedding)
            search_results = []
            for i, metadata in enumerate(results_metadata):
                if expand_context:
                    # Get the chunk from tree
                    chunk = tree.get_chunk(metadata["chunk_id"])
                    if chunk:
                        # Build rich context AFTER finding relevant chunks
                        enriched_result = {
                            # Core chunk data
                            "chunk_id": metadata["chunk_id"],
                            "text": metadata["text"],  # Clean chunk text
                            "level": metadata["level"],
                            "similarity_score": float(scores[i]),
                            
                            # POST-RETRIEVAL context expansion
                            "parent_context": self._get_parent_context_post_retrieval(chunk, tree),
                            "navigation_path": " > ".join(metadata.get("path", [])),
                            "children_summary": self._get_children_summary_post_retrieval(chunk, tree),
                            
                            # Ancestor context if requested
                            "ancestors": self._get_ancestors_context(chunk, tree, context_levels),
                            
                            # Original metadata
                            **{k: v for k, v in metadata.items() if k not in ["text", "chunk_id", "level"]}
                        }
                        search_results.append(enriched_result)
                    else:
                        # Fallback to basic result
                        search_results.append({
                            **metadata,
                            "similarity_score": float(scores[i])
                        })
                else:
                    # Return basic result without context expansion
                    search_results.append({
                        **metadata,
                        "similarity_score": float(scores[i])
                    })
            
            return search_results
            
        except FileNotFoundError:
            print(f"No hierarchical embeddings found for document: {doc_id}")
            return []
        except Exception as e:
            print(f"Search error for document {doc_id}: {str(e)}")
            return []
    
    def _get_parent_context_post_retrieval(self, chunk: HierarchicalChunk, tree: HierarchicalChunkTree) -> Optional[str]:
        """Get parent context AFTER retrieval - not during embedding"""
        parent = tree.get_parent(chunk.chunk_id)
        if parent and parent.level < chunk.level:
            return f"From {parent.title}: {parent.text[:200]}{'...' if len(parent.text) > 200 else ''}"
        return None

    def _get_children_summary_post_retrieval(self, chunk: HierarchicalChunk, tree: HierarchicalChunkTree) -> Optional[str]:
        """Get children summary AFTER retrieval"""
        if chunk.level >= 3:  # Only for chapters and sections
            return None
        
        children = tree.get_children(chunk.chunk_id)
        if children:
            titles = [child.title for child in children[:3]]  # First 3 children
            more_count = len(children) - 3
            summary = ", ".join(titles)
            if more_count > 0:
                summary += f", and {more_count} more"
            return f"Contains: {summary}"
        return None

    def _get_ancestors_context(self, chunk: HierarchicalChunk, tree: HierarchicalChunkTree, levels: int) -> List[Dict[str, Any]]:
        """Get ancestor context for navigation"""
        ancestors = []
        current = tree.get_parent(chunk.chunk_id)
        count = 0
        
        while current and count < levels:
            ancestors.append({
                "chunk_id": current.chunk_id,
                "title": current.title,
                "level": current.level,
                "text_preview": current.text[:150] + "..." if len(current.text) > 150 else current.text
            })
            current = tree.get_parent(current.chunk_id)
            count += 1
        
        return ancestors
    
    def adaptive_search(self, doc_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Adaptive search that automatically selects best strategy"""
        query_lower = query.lower()
        
        # Classify query and determine search strategy
        if any(term in query_lower for term in ["overview", "summary", "what is", "about"]):
            # High-level queries: prefer chapters and sections
            level_preference = [1, 2]
            expand_down = True
        elif any(term in query_lower for term in ["how to", "steps", "process", "procedure"]):
            # Process queries: prefer paragraphs with context
            level_preference = [3]
            expand_down = False
        elif any(term in query_lower for term in ["definition", "means", "define"]):
            # Definition queries: prefer sentences and paragraphs
            level_preference = [4, 3]
            expand_down = False
        else:
            # General queries: balanced approach
            level_preference = None
            expand_down = False
        
        if level_preference:
            # Search preferred levels and combine results
            all_results = []
            remaining_k = k
            
            for level in level_preference:
                if remaining_k <= 0:
                    break
                
                level_results = self.search_document(
                    doc_id, query, k=remaining_k, 
                    level_filter=level, expand_context=True
                )
                
                all_results.extend(level_results)
                remaining_k -= len(level_results)
            
            # Sort by similarity and return top k
            all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            return all_results[:k]
        else:
            # Standard search across all levels
            return self.search_document(doc_id, query, k=k, expand_context=True)
    
    def get_chunk_navigation(self, doc_id: str, chunk_id: str) -> Dict[str, Any]:
        """Get navigation options for a specific chunk"""
        try:
            index, tree = self.load_document_assets(doc_id)
            
            chunk_metadata = index.get_chunk_by_id(chunk_id)
            if not chunk_metadata:
                return {"error": "Chunk not found"}
            
            result = HierarchicalSearchResult(chunk_metadata, tree, 1.0)
            
            return {
                "chunk": chunk_metadata,
                "context_expansion": result.get_context_expansion(levels_up=2),
                "drill_down": result.get_drill_down(levels_down=2),
                "siblings": self._get_sibling_chunks(chunk_id, tree, index)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_sibling_chunks(self, chunk_id: str, tree: HierarchicalChunkTree, 
                           index: HierarchicalFAISSIndex) -> List[Dict[str, Any]]:
        """Get sibling chunks at the same level"""
        chunk = tree.get_chunk(chunk_id)
        if not chunk or not chunk.parent_id:
            return []
        
        parent = tree.get_chunk(chunk.parent_id)
        if not parent:
            return []
        
        siblings = []
        for sibling_id in parent.children_ids:
            if sibling_id != chunk_id:
                sibling_metadata = index.get_chunk_by_id(sibling_id)
                if sibling_metadata:
                    siblings.append(sibling_metadata)
        
        return siblings
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about hierarchical embeddings"""
        stats = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_model.embedding_dim,
            "embedding_type": "hierarchical_tree_clean",
            "documents_with_embeddings": [],
            "total_embeddings_by_level": {f"level_{i}": 0 for i in range(1, 5)},
            "total_documents": 0
        }
        
        for doc_dir in self.embeddings_dir.iterdir():
            if doc_dir.is_dir():
                info_path = doc_dir / "embedding_info.json"
                if info_path.exists():
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            doc_info = json.load(f)
                        
                        # Accept both old and new embedding types
                        if doc_info.get("embedding_type") in ["hierarchical_tree", "hierarchical_tree_clean"]:
                            stats["documents_with_embeddings"].append(doc_info)
                            stats["total_documents"] += 1
                            
                            # Aggregate level statistics
                            level_stats = doc_info.get("level_statistics", {})
                            for level_name, level_info in level_stats.items():
                                if level_name in stats["total_embeddings_by_level"]:
                                    stats["total_embeddings_by_level"][level_name] += level_info.get("chunk_count", 0)
                        
                    except Exception as e:
                        print(f"Error reading embedding info for {doc_dir.name}: {str(e)}")
        
        return stats


# Main EmbeddingService class (backward compatible)
class EmbeddingService:
    """Main embedding service using clean hierarchical approach"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_dir: Path = None):
        self.model_name = model_name
        self.embedding_model = EmbeddingModel(model_name)
        
        # Use hierarchical service as the main implementation
        self.hierarchical_service = HierarchicalEmbeddingService(model_name, data_dir)
    
    def embed_document_chunks(self, doc_id: str, chapters_data: List[Any]) -> Dict[str, Any]:
        """Generate embeddings for document chunks (supports hierarchical trees)"""
        
        # Check if this is hierarchical tree data
        if (chapters_data and 
            hasattr(chapters_data[0], 'get_all_chunks') and 
            callable(getattr(chapters_data[0], 'get_all_chunks'))):
            # This is a list of HierarchicalChunkTree objects
            print("Detected hierarchical chunk trees, using clean embedding approach")
            return self.hierarchical_service.embed_document_tree(doc_id, chapters_data)
        else:
            # For backward compatibility, handle flat chunk data
            raise ValueError("This service now only supports hierarchical chunk trees. Please update your chunking service.")
    
    def search_document(self, doc_id: str, query: str, k: int = 5, search_mode: str = "adaptive") -> List[Dict[str, Any]]:
        """Search document using clean embeddings with post-retrieval context"""
        
        if search_mode == "adaptive":
            return self.hierarchical_service.adaptive_search(doc_id, query, k)
        elif search_mode.startswith("level_"):
            # Extract level number
            try:
                level = int(search_mode.split("_")[1])
                return self.hierarchical_service.search_document(doc_id, query, k, level_filter=level)
            except (IndexError, ValueError):
                return self.hierarchical_service.search_document(doc_id, query, k)
        else:
            return self.hierarchical_service.search_document(doc_id, query, k)
    
    def get_chunk_navigation(self, doc_id: str, chunk_id: str) -> Dict[str, Any]:
        """Get hierarchical navigation for a chunk"""
        return self.hierarchical_service.get_chunk_navigation(doc_id, chunk_id)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        return self.hierarchical_service.get_embedding_stats()


# Default service instance
default_embedding_service = EmbeddingService()


# Convenience functions
def embed_document_chunks(doc_id: str, chapter_trees: List[Any]) -> Dict[str, Any]:
    """Generate embeddings for document chunk trees (convenience function)"""
    return default_embedding_service.embed_document_chunks(doc_id, chapter_trees)


def search_document_chunks(doc_id: str, query: str, k: int = 5, search_mode: str = "adaptive") -> List[Dict[str, Any]]:
    """Search document chunks with hierarchical navigation (convenience function)"""
    return default_embedding_service.search_document(doc_id, query, k, search_mode)