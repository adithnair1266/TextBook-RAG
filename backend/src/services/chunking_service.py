import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.text_utils import extract_word_count, split_paragraph_by_sentences


@dataclass
class ChunkingConfig:
    """Configuration for hierarchical chunking"""
    min_section_words: int = 200      # Minimum words to create a section
    min_paragraph_words: int = 50     # Minimum words for paragraph chunks
    max_paragraph_words: int = 400    # Split large paragraphs
    sentence_threshold: int = 20      # Minimum words for sentence chunks
    context_overlap: int = 25         # Words to overlap for context


class HierarchicalChunk:
    """Single chunk in a hierarchical tree structure"""
    
    def __init__(
        self,
        chunk_id: str,
        text: str,
        level: int,
        title: str = "",
        parent_id: Optional[str] = None,
        **metadata
    ):
        self.chunk_id = chunk_id
        self.text = text.strip()
        self.level = level  # 1=chapter, 2=section, 3=paragraph, 4=sentence
        self.title = title
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        self.path: List[str] = []  # Full path from root
        self.word_count = extract_word_count(self.text)
        self.created_at = datetime.utcnow().isoformat()
        
        # Store additional metadata
        self.metadata = {
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "chapter_title": metadata.get("chapter_title", ""),
            "subsection_title": metadata.get("subsection_title", ""),
            **metadata
        }
    
    def add_child(self, child_id: str):
        """Add a child chunk ID"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def set_path(self, path: List[str]):
        """Set the full hierarchical path"""
        self.path = path.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "level": self.level,
            "title": self.title,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "path": self.path,
            "word_count": self.word_count,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalChunk":
        """Create from dictionary"""
        chunk = cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            level=data["level"],
            title=data.get("title", ""),
            parent_id=data.get("parent_id"),
            **data.get("metadata", {})
        )
        chunk.children_ids = data.get("children_ids", [])
        chunk.path = data.get("path", [])
        chunk.word_count = data.get("word_count", extract_word_count(chunk.text))
        chunk.created_at = data.get("created_at", datetime.utcnow().isoformat())
        return chunk


class HierarchicalChunkTree:
    """Manages the hierarchical chunk tree structure"""
    
    def __init__(self):
        self.chunks: Dict[str, HierarchicalChunk] = {}
        self.root_ids: List[str] = []  # Top-level chapter IDs
    
    def add_chunk(self, chunk: HierarchicalChunk):
        """Add chunk to tree and maintain relationships"""
        self.chunks[chunk.chunk_id] = chunk
        
        # Add to parent's children
        if chunk.parent_id and chunk.parent_id in self.chunks:
            self.chunks[chunk.parent_id].add_child(chunk.chunk_id)
        
        # Track root chunks (chapters)
        if chunk.level == 1:
            self.root_ids.append(chunk.chunk_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[HierarchicalChunk]:
        """Get chunk by ID"""
        return self.chunks.get(chunk_id)
    
    def get_parent(self, chunk_id: str) -> Optional[HierarchicalChunk]:
        """Get parent chunk"""
        chunk = self.chunks.get(chunk_id)
        if chunk and chunk.parent_id:
            return self.chunks.get(chunk.parent_id)
        return None
    
    def get_children(self, chunk_id: str) -> List[HierarchicalChunk]:
        """Get direct children chunks"""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return []
        
        return [self.chunks[child_id] for child_id in chunk.children_ids 
                if child_id in self.chunks]
    
    def get_ancestors(self, chunk_id: str) -> List[HierarchicalChunk]:
        """Get all ancestor chunks (parent, grandparent, etc.)"""
        ancestors = []
        current = self.get_parent(chunk_id)
        
        while current:
            ancestors.append(current)
            current = self.get_parent(current.chunk_id)
        
        return ancestors
    
    def get_descendants(self, chunk_id: str) -> List[HierarchicalChunk]:
        """Get all descendant chunks (recursive)"""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return []
        
        descendants = []
        for child_id in chunk.children_ids:
            if child_id in self.chunks:
                child = self.chunks[child_id]
                descendants.append(child)
                descendants.extend(self.get_descendants(child_id))
        
        return descendants
    
    def get_context_window(self, chunk_id: str, levels_up: int = 1, levels_down: int = 0) -> Dict[str, Any]:
        """Get hierarchical context around a chunk"""
        target_chunk = self.chunks.get(chunk_id)
        if not target_chunk:
            return {}
        
        # Get ancestors up to specified levels
        ancestors = self.get_ancestors(chunk_id)[:levels_up]
        
        # Get descendants down to specified levels
        descendants = []
        if levels_down > 0:
            descendants = self._get_descendants_by_level(chunk_id, levels_down)
        
        return {
            "target": target_chunk,
            "ancestors": ancestors,
            "descendants": descendants,
            "context_text": self._build_context_text(target_chunk, ancestors)
        }
    
    def _get_descendants_by_level(self, chunk_id: str, max_levels: int) -> List[HierarchicalChunk]:
        """Get descendants up to max_levels deep"""
        if max_levels <= 0:
            return []
        
        descendants = []
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return descendants
        
        # Get direct children
        for child_id in chunk.children_ids:
            if child_id in self.chunks:
                child = self.chunks[child_id]
                descendants.append(child)
                
                # Recursively get their descendants
                if max_levels > 1:
                    descendants.extend(self._get_descendants_by_level(child_id, max_levels - 1))
        
        return descendants
    
    def _build_context_text(self, target: HierarchicalChunk, ancestors: List[HierarchicalChunk]) -> str:
        """Build enriched context text for embedding"""
        context_parts = []
        
        # Add hierarchical path
        if ancestors:
            path_titles = [ancestor.title for ancestor in reversed(ancestors) if ancestor.title]
            if path_titles:
                context_parts.append("Context: " + " > ".join(path_titles))
        
        # Add target content
        if target.title:
            context_parts.append(f"Section: {target.title}")
        
        context_parts.append(target.text)
        
        return "\n".join(context_parts)
    
    def get_all_chunks(self) -> List[HierarchicalChunk]:
        """Get all chunks as a flat list"""
        return list(self.chunks.values())
    
    def get_chunks_by_level(self, level: int) -> List[HierarchicalChunk]:
        """Get all chunks at a specific level"""
        return [chunk for chunk in self.chunks.values() if chunk.level == level]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire tree to dictionary"""
        return {
            "chunks": {chunk_id: chunk.to_dict() for chunk_id, chunk in self.chunks.items()},
            "root_ids": self.root_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalChunkTree":
        """Create tree from dictionary"""
        tree = cls()
        
        # First pass: create all chunks
        for chunk_id, chunk_data in data.get("chunks", {}).items():
            chunk = HierarchicalChunk.from_dict(chunk_data)
            tree.chunks[chunk_id] = chunk
        
        # Second pass: rebuild relationships
        tree.root_ids = data.get("root_ids", [])
        
        return tree


class TrueHierarchicalChunker:
    """True hierarchical chunking that builds nested tree structures"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
    
    def chunk_content(self, content: str, chapter_info: Dict[str, Any]) -> HierarchicalChunkTree:
        """Build hierarchical chunk tree from chapter content"""
        
        chapter_title = chapter_info.get("title", "Unknown Chapter")
        print(f"Building hierarchical chunk tree for '{chapter_title}'...")
        
        tree = HierarchicalChunkTree()
        
        # Level 1: Create root chapter chunk
        chapter_id = self._generate_id("ch")
        chapter_chunk = HierarchicalChunk(
            chunk_id=chapter_id,
            text=content,
            level=1,
            title=chapter_title,
            chapter_title=chapter_title,
            page_start=chapter_info.get("start_page"),
            page_end=chapter_info.get("end_page")
        )
        chapter_chunk.set_path([chapter_title])
        tree.add_chunk(chapter_chunk)
        
        # Level 2: Extract sections from content
        sections = self._extract_sections(content, chapter_info)
        for section_info in sections:
            section_id = self._generate_id("sec")
            section_chunk = HierarchicalChunk(
                chunk_id=section_id,
                text=section_info["text"],
                level=2,
                title=section_info["title"],
                parent_id=chapter_id,
                chapter_title=chapter_title,
                subsection_title=section_info["title"],
                page_start=chapter_info.get("start_page"),
                page_end=chapter_info.get("end_page")
            )
            section_chunk.set_path([chapter_title, section_info["title"]])
            tree.add_chunk(section_chunk)
            
            # Level 3: Extract paragraphs from section
            paragraphs = self._extract_paragraphs(section_info["text"])
            for para_idx, para_text in enumerate(paragraphs):
                para_id = self._generate_id("para")
                para_chunk = HierarchicalChunk(
                    chunk_id=para_id,
                    text=para_text,
                    level=3,
                    title=f"Paragraph {para_idx + 1}",
                    parent_id=section_id,
                    chapter_title=chapter_title,
                    subsection_title=section_info["title"],
                    page_start=chapter_info.get("start_page"),
                    page_end=chapter_info.get("end_page")
                )
                para_chunk.set_path([chapter_title, section_info["title"], f"Paragraph {para_idx + 1}"])
                tree.add_chunk(para_chunk)
                
                # Level 4: Extract key sentences from paragraph (if large enough)
                if extract_word_count(para_text) > self.config.max_paragraph_words:
                    sentences = self._extract_key_sentences(para_text)
                    for sent_idx, sent_text in enumerate(sentences):
                        sent_id = self._generate_id("sent")
                        sent_chunk = HierarchicalChunk(
                            chunk_id=sent_id,
                            text=sent_text,
                            level=4,
                            title=f"Sentence {sent_idx + 1}",
                            parent_id=para_id,
                            chapter_title=chapter_title,
                            subsection_title=section_info["title"],
                            page_start=chapter_info.get("start_page"),
                            page_end=chapter_info.get("end_page")
                        )
                        sent_chunk.set_path([chapter_title, section_info["title"], f"Paragraph {para_idx + 1}", f"Sentence {sent_idx + 1}"])
                        tree.add_chunk(sent_chunk)
        
        total_chunks = len(tree.get_all_chunks())
        level_counts = {i: len(tree.get_chunks_by_level(i)) for i in range(1, 5)}
        
        print(f"  Created hierarchical tree: {total_chunks} total chunks")
        print(f"    Level 1 (chapters): {level_counts[1]}")
        print(f"    Level 2 (sections): {level_counts[2]}")
        print(f"    Level 3 (paragraphs): {level_counts[3]}")
        print(f"    Level 4 (sentences): {level_counts[4]}")
        
        return tree
    
    def _extract_sections(self, content: str, chapter_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract sections from chapter content using TOC subsections and content analysis"""
        subsections = chapter_info.get("subsections", [])
        
        if subsections:
            # Use TOC subsections as guide
            return self._extract_sections_from_toc(content, subsections)
        else:
            # Fall back to content-based section detection
            return self._extract_sections_from_content(content)
    
    def _extract_sections_from_toc(self, content: str, subsections: List[Dict]) -> List[Dict[str, str]]:
        """Extract sections based on TOC subsection information"""
        sections = []
        
        # Simple approach: divide content roughly by subsection count
        content_length = len(content)
        section_size = content_length // max(len(subsections), 1)
        
        for i, subsection in enumerate(subsections):
            start_pos = i * section_size
            end_pos = (i + 1) * section_size if i < len(subsections) - 1 else content_length
            
            section_text = content[start_pos:end_pos].strip()
            if section_text and extract_word_count(section_text) >= self.config.min_section_words:
                sections.append({
                    "title": subsection.get("title", f"Section {i + 1}"),
                    "text": section_text
                })
        
        return sections if sections else [{"title": "Main Section", "text": content}]
    
    def _extract_sections_from_content(self, content: str) -> List[Dict[str, str]]:
        """Extract sections based on markdown headers or content patterns"""
        # Look for markdown headers (##, ###)
        header_pattern = r'^(#{2,4})\s+(.+)$'
        lines = content.split('\n')
        
        sections = []
        current_section = []
        current_title = "Introduction"
        
        for line in lines:
            match = re.match(header_pattern, line, re.MULTILINE)
            if match:
                # Save previous section
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text and extract_word_count(section_text) >= self.config.min_section_words:
                        sections.append({
                            "title": current_title,
                            "text": section_text
                        })
                
                # Start new section
                current_title = match.group(2)
                current_section = [line]
            else:
                current_section.append(line)
        
        # Save final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append({
                    "title": current_title,
                    "text": section_text
                })
        
        # If no sections found, treat entire content as one section
        return sections if sections else [{"title": "Main Section", "text": content}]
    
    def _extract_paragraphs(self, section_text: str) -> List[str]:
        """Extract meaningful paragraphs from section text"""
        # Split by double newlines
        raw_paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
        
        paragraphs = []
        for para in raw_paragraphs:
            word_count = extract_word_count(para)
            
            # Skip very short paragraphs
            if word_count < self.config.min_paragraph_words:
                continue
            
            # Split very long paragraphs
            if word_count > self.config.max_paragraph_words:
                sentences = split_paragraph_by_sentences(para)
                # Group sentences into reasonable paragraphs
                current_para = []
                current_words = 0
                
                for sentence in sentences:
                    sentence_words = extract_word_count(sentence)
                    if current_words + sentence_words > self.config.max_paragraph_words and current_para:
                        paragraphs.append(' '.join(current_para))
                        current_para = [sentence]
                        current_words = sentence_words
                    else:
                        current_para.append(sentence)
                        current_words += sentence_words
                
                if current_para:
                    paragraphs.append(' '.join(current_para))
            else:
                paragraphs.append(para)
        
        return paragraphs if paragraphs else [section_text]
    
    def _extract_key_sentences(self, paragraph_text: str) -> List[str]:
        """Extract key sentences from large paragraphs"""
        sentences = split_paragraph_by_sentences(paragraph_text)
        
        # Filter out very short sentences
        key_sentences = [
            sent for sent in sentences 
            if extract_word_count(sent) >= self.config.sentence_threshold
        ]
        
        return key_sentences if key_sentences else sentences
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID for chunks"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"


class ChunkingService:
    """Main chunking service using true hierarchical chunking"""
    
    def __init__(self, chunker: TrueHierarchicalChunker = None):
        self.chunker = chunker or TrueHierarchicalChunker()
    
    def chunk_chapter_content(self, content: str, chapter_info: Dict[str, Any]) -> HierarchicalChunkTree:
        """Chunk content into hierarchical tree structure"""
        return self.chunker.chunk_content(content, chapter_info)
    
    def set_chunker(self, chunker: TrueHierarchicalChunker):
        """Switch to a different chunker"""
        self.chunker = chunker


# Default service instance with true hierarchical chunking
default_chunking_service = ChunkingService()


# Convenience function for backward compatibility
def chunk_chapter_content(chapter_text: str, chapter_info: Dict[str, Any]) -> HierarchicalChunkTree:
    """Chunk chapter content using hierarchical strategy"""
    return default_chunking_service.chunk_chapter_content(chapter_text, chapter_info)