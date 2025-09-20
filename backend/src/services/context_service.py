from typing import List, Optional

from models.simple_chunk import ContextExpandedResult, SimpleChunk


class ContextService:
    """Handle context expansion for search results"""
    
    def get_neighboring_paragraphs(self, target_chunk: SimpleChunk, 
                                 all_chunks: List[SimpleChunk], 
                                 context_window: int = 2) -> List[SimpleChunk]:
        """Get neighboring paragraphs within same chapter"""
        
        # Filter to same chapter
        chapter_chunks = [
            chunk for chunk in all_chunks 
            if chunk.chapter_name == target_chunk.chapter_name
        ]
        
        # Sort by paragraph index
        chapter_chunks.sort(key=lambda x: x.paragraph_index)
        
        # Find target position
        target_idx = next(
            (i for i, chunk in enumerate(chapter_chunks) 
             if chunk.chunk_id == target_chunk.chunk_id), 
            None
        )
        
        if target_idx is None:
            return [target_chunk]
        
        # Get context window around target
        start_idx = max(0, target_idx - context_window)
        end_idx = min(len(chapter_chunks), target_idx + context_window + 1)
        
        return chapter_chunks[start_idx:end_idx]
    
    def get_page_context(self, target_chunk: SimpleChunk, 
                        all_chunks: List[SimpleChunk], 
                        page_window: int = 1) -> List[SimpleChunk]:
        """Get all chunks from target page + adjacent pages"""
        
        target_page = target_chunk.page_number
        context_pages = range(
            target_page - page_window, 
            target_page + page_window + 1
        )
        
        # Filter to relevant pages in same chapter
        page_chunks = [
            chunk for chunk in all_chunks 
            if (chunk.page_number in context_pages and 
                chunk.chapter_name == target_chunk.chapter_name)
        ]
        
        # Sort by page number, then paragraph index
        page_chunks.sort(key=lambda x: (x.page_number, x.paragraph_index))
        return page_chunks
    
    def get_mixed_context(self, target_chunk: SimpleChunk, 
                         all_chunks: List[SimpleChunk],
                         paragraph_window: int = 1,
                         page_window: int = 0) -> List[SimpleChunk]:
        """Get context using both paragraph and page strategies"""
        
        # Start with paragraph neighbors
        para_chunks = self.get_neighboring_paragraphs(
            target_chunk, all_chunks, paragraph_window
        )
        
        # Add page context if requested
        if page_window > 0:
            page_chunks = self.get_page_context(
                target_chunk, all_chunks, page_window
            )
            
            # Merge and deduplicate
            all_context_chunks = para_chunks + page_chunks
            seen_ids = set()
            unique_chunks = []
            
            for chunk in all_context_chunks:
                if chunk.chunk_id not in seen_ids:
                    unique_chunks.append(chunk)
                    seen_ids.add(chunk.chunk_id)
            
            # Re-sort by position
            unique_chunks.sort(key=lambda x: (x.page_number, x.paragraph_index))
            return unique_chunks
        
        return para_chunks
    
    def merge_context_chunks(self, chunks: List[SimpleChunk], 
                           include_page_markers: bool = True) -> str:
        """Merge chunks into coherent context text"""
        if not chunks:
            return ""
        
        if not include_page_markers:
            # Simple concatenation
            return "\n\n".join(chunk.text for chunk in chunks)
        
        # Group by page for better formatting
        pages = {}
        for chunk in chunks:
            if chunk.page_number not in pages:
                pages[chunk.page_number] = []
            pages[chunk.page_number].append(chunk)
        
        context_parts = []
        for page_num in sorted(pages.keys()):
            page_chunks = sorted(pages[page_num], key=lambda x: x.paragraph_index)
            
            # Only add page marker if multiple pages
            if len(pages) > 1:
                page_text = "\n\n".join(chunk.text for chunk in page_chunks)
                context_parts.append(f"[Page {page_num}]\n{page_text}")
            else:
                page_text = "\n\n".join(chunk.text for chunk in page_chunks)
                context_parts.append(page_text)
        
        return "\n\n---\n\n".join(context_parts) if len(pages) > 1 else context_parts[0]
    
    def expand_search_result(self, target_chunk: SimpleChunk,
                           all_chunks: List[SimpleChunk],
                           similarity_score: float,
                           strategy: str = "paragraph",
                           context_window: int = 2) -> ContextExpandedResult:
        """Expand a search result with context"""
        
        # Choose expansion strategy
        if strategy == "paragraph":
            context_chunks = self.get_neighboring_paragraphs(
                target_chunk, all_chunks, context_window
            )
            expansion_type = f"neighboring_paragraphs_window_{context_window}"
            
        elif strategy == "page":
            context_chunks = self.get_page_context(
                target_chunk, all_chunks, context_window
            )
            expansion_type = f"page_window_{context_window}"
            
        elif strategy == "mixed":
            context_chunks = self.get_mixed_context(
                target_chunk, all_chunks, 
                paragraph_window=context_window, 
                page_window=1
            )
            expansion_type = f"mixed_para_{context_window}_page_1"
            
        else:
            # No expansion
            context_chunks = [target_chunk]
            expansion_type = "no_expansion"
        
        # Create merged context text
        context_text = self.merge_context_chunks(
            context_chunks, 
            include_page_markers=(strategy in ["page", "mixed"])
        )
        
        # Calculate total words
        total_words = sum(chunk.word_count for chunk in context_chunks)
        
        return ContextExpandedResult(
            target_chunk=target_chunk,
            context_chunks=context_chunks,
            context_text=context_text,
            similarity_score=similarity_score,
            expansion_type=expansion_type,
            total_words=total_words
        )
    
    def get_sequential_chunks(self, start_chunk: SimpleChunk,
                            all_chunks: List[SimpleChunk],
                            count: int = 5,
                            direction: str = "forward") -> List[SimpleChunk]:
        """Get sequential chunks in reading order"""
        
        # Filter to same chapter and sort
        chapter_chunks = [
            chunk for chunk in all_chunks 
            if chunk.chapter_name == start_chunk.chapter_name
        ]
        chapter_chunks.sort(key=lambda x: x.paragraph_index)
        
        # Find start position
        start_idx = next(
            (i for i, chunk in enumerate(chapter_chunks) 
             if chunk.chunk_id == start_chunk.chunk_id), 
            None
        )
        
        if start_idx is None:
            return [start_chunk]
        
        if direction == "forward":
            end_idx = min(len(chapter_chunks), start_idx + count)
            return chapter_chunks[start_idx:end_idx]
        elif direction == "backward":
            start_idx = max(0, start_idx - count + 1)
            return chapter_chunks[start_idx:start_idx + count]
        else:
            # Both directions
            window = count // 2
            start_idx = max(0, start_idx - window)
            end_idx = min(len(chapter_chunks), start_idx + count)
            return chapter_chunks[start_idx:end_idx]
    
    def find_related_chunks_by_page(self, target_chunk: SimpleChunk,
                                  all_chunks: List[SimpleChunk]) -> dict:
        """Find all chunks related to target by page proximity"""
        
        target_page = target_chunk.page_number
        
        # Group chunks by page distance
        page_groups = {
            "same_page": [],
            "adjacent_pages": [],
            "nearby_pages": []
        }
        
        for chunk in all_chunks:
            if chunk.chapter_name != target_chunk.chapter_name:
                continue
                
            page_distance = abs(chunk.page_number - target_page)
            
            if page_distance == 0:
                page_groups["same_page"].append(chunk)
            elif page_distance == 1:
                page_groups["adjacent_pages"].append(chunk)
            elif page_distance <= 3:
                page_groups["nearby_pages"].append(chunk)
        
        # Sort each group
        for group in page_groups.values():
            group.sort(key=lambda x: (x.page_number, x.paragraph_index))
        
        return page_groups


# Default service instance
default_context_service = ContextService()


# Convenience functions
def get_paragraph_context(target_chunk: SimpleChunk, all_chunks: List[SimpleChunk], 
                         window: int = 2) -> List[SimpleChunk]:
    """Get neighboring paragraphs (convenience function)"""
    return default_context_service.get_neighboring_paragraphs(target_chunk, all_chunks, window)


def get_page_context(target_chunk: SimpleChunk, all_chunks: List[SimpleChunk], 
                    window: int = 1) -> List[SimpleChunk]:
    """Get page context (convenience function)"""
    return default_context_service.get_page_context(target_chunk, all_chunks, window)


def expand_with_context(target_chunk: SimpleChunk, all_chunks: List[SimpleChunk],
                       similarity_score: float, strategy: str = "paragraph") -> ContextExpandedResult:
    """Expand search result with context (convenience function)"""
    return default_context_service.expand_search_result(
        target_chunk, all_chunks, similarity_score, strategy
    )