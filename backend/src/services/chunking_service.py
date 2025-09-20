import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from models.simple_chunk import SimpleChunk, SimpleChunkList
from utils.text_utils import extract_word_count


class ChunkingConfig:
    """Configuration for paragraph chunking"""
    def __init__(self):
        self.min_paragraph_words: int = 30      # Skip very short paragraphs
        self.max_paragraph_words: int = 500     # Split very long paragraphs
        self.overlap_sentences: int = 1         # Sentences to overlap between chunks
        self.preserve_sentences: bool = True    # Don't split mid-sentence


class ParagraphChunker:
    """Paragraph-based chunker - no artificial hierarchy"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
    
    def chunk_chapter(self, content: str, chapter_info: Dict[str, Any]) -> SimpleChunkList:
        """Chunk chapter content into paragraphs"""
        
        chapter_title = chapter_info.get("title", "Unknown Chapter")
        start_page = chapter_info.get("start_page", 1)
        end_page = chapter_info.get("end_page", start_page)
        
        print(f"Chunking '{chapter_title}' into paragraphs (pages {start_page}-{end_page})")
        
        # Extract paragraphs from content
        raw_paragraphs = self._extract_paragraphs(content)
        
        # Convert to SimpleChunk objects with proper indexing
        chunks = []
        total_pages = end_page - start_page + 1
        
        for i, paragraph_text in enumerate(raw_paragraphs):
            # Estimate page number for this paragraph
            page_progress = i / max(len(raw_paragraphs) - 1, 1)  # 0.0 to 1.0
            estimated_page = start_page + int(page_progress * total_pages)
            estimated_page = min(estimated_page, end_page)  # Don't exceed end page
            
            chunk_id = self._generate_chunk_id(chapter_title, i)
            
            chunk = SimpleChunk(
                chunk_id=chunk_id,
                text=paragraph_text,
                chapter_name=chapter_title,
                paragraph_index=i,
                page_number=estimated_page,
                word_count=extract_word_count(paragraph_text)
            )
            
            chunks.append(chunk)
        
        # Link neighboring chunks
        self._link_chunks(chunks)
        
        # Calculate totals
        total_words = sum(chunk.word_count for chunk in chunks)
        
        print(f"  Created {len(chunks)} paragraph chunks ({total_words} words)")
        
        return SimpleChunkList(
            chunks=chunks,
            total_chunks=len(chunks),
            chapter_name=chapter_title,
            total_word_count=total_words
        )
    
    def _extract_paragraphs(self, content: str) -> List[str]:
        """Extract meaningful paragraphs from content"""
        
        # Split by double newlines (standard paragraph separator)
        raw_paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        final_paragraphs = []
        
        for para in raw_paragraphs:
            word_count = extract_word_count(para)
            
            # Skip very short paragraphs (likely noise)
            if word_count < self.config.min_paragraph_words:
                continue
            
            # Handle very long paragraphs
            if word_count > self.config.max_paragraph_words:
                # Split long paragraphs by sentences
                split_paras = self._split_long_paragraph(para)
                final_paragraphs.extend(split_paras)
            else:
                final_paragraphs.append(para)
        
        # If no good paragraphs found, split by single newlines
        if not final_paragraphs:
            print("  Warning: No good paragraphs found, falling back to line splitting")
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            final_paragraphs = [line for line in lines if extract_word_count(line) >= 10]
        
        return final_paragraphs
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split a long paragraph into smaller chunks at sentence boundaries"""
        
        # Simple sentence splitting (can be improved with proper NLP)
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        chunks = []
        current_chunk = []
        current_words = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = extract_word_count(sentence)
            
            # If adding this sentence would exceed max, save current chunk
            if (current_words + sentence_words > self.config.max_paragraph_words 
                and current_chunk):
                
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_words = sentence_words
            else:
                current_chunk.append(sentence)
                current_words += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [paragraph]
    
    def _link_chunks(self, chunks: List[SimpleChunk]) -> None:
        """Link neighboring chunks for navigation"""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.prev_chunk_id = chunks[i-1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i+1].chunk_id
    
    def _generate_chunk_id(self, chapter_title: str, paragraph_index: int) -> str:
        """Generate unique chunk ID"""
        # Create short chapter identifier
        chapter_id = re.sub(r'[^a-zA-Z0-9]', '', chapter_title.lower())[:10]
        return f"{chapter_id}_p{paragraph_index:03d}_{uuid.uuid4().hex[:6]}"


class ChunkingService:
    """Main chunking service using paragraph chunking"""
    
    def __init__(self, chunker: ParagraphChunker = None):
        self.chunker = chunker or ParagraphChunker()
        self.strategy_name = "ParagraphChunker"
    
    def chunk_chapter_content(self, content: str, chapter_info: Dict[str, Any]) -> SimpleChunkList:
        """Chunk content into paragraph structure"""
        return self.chunker.chunk_chapter(content, chapter_info)
    
    def set_chunker(self, chunker: ParagraphChunker):
        """Switch to a different chunker"""
        self.chunker = chunker
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about current chunking strategy"""
        return {
            "strategy_name": self.strategy_name,
            "chunker_class": self.chunker.__class__.__name__,
            "config": {
                "min_paragraph_words": self.chunker.config.min_paragraph_words,
                "max_paragraph_words": self.chunker.config.max_paragraph_words,
                "overlap_sentences": self.chunker.config.overlap_sentences,
                "preserve_sentences": self.chunker.config.preserve_sentences
            },
            "levels": ["paragraphs_only"],
            "features": [
                "paragraph_splitting",
                "sentence_boundary_preservation", 
                "neighboring_chunk_linking",
                "page_estimation"
            ]
        }


# Default service instance
default_chunking_service = ChunkingService()


# Convenience functions
def chunk_chapter_content(chapter_text: str, chapter_info: Dict[str, Any]) -> SimpleChunkList:
    """Chunk chapter content using paragraph strategy"""
    return default_chunking_service.chunk_chapter_content(chapter_text, chapter_info)


def get_chunking_strategy_info() -> Dict[str, Any]:
    """Get current chunking strategy info"""
    return default_chunking_service.get_strategy_info()