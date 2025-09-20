import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pymupdf4llm
from services.agents import Agent
from utils.text_utils import clean_toc_content

# TOC parsing prompt - can be easily swapped for different strategies
TOC_PARSING_PROMPT = """
You are a precise TOC parser. Extract individual chapters from this section of a table of contents.

IMPORTANT RULES:
- Look for entries that start with "Chapter X:" - these are the main chapters
- Ignore "PHASE", "PART", or section headers - they are just organizational groupings
- Each "Chapter X:" entry should be a separate top-level chapter
- Extract subsections within each chapter if they exist (like "1.1", "1.2", etc.)
- This might be a partial TOC section - only extract chapters that are COMPLETE in this section

Return ONLY a valid JSON object with this exact structure:
{
  "chapters": [
    {
      "title": "Chapter 1: Introduction to Systems Analysis and Design",
      "start_page": 2,
      "subsections": [
        {"title": "1.1 Information Technology", "start_page": 3},
        {"title": "1.2 Information Systems", "start_page": 4}
      ]
    }
  ]
}

Rules:
- Extract ALL complete "Chapter X:" entries in this section
- Include subsections (numbered like 1.1, 2.1, etc.) when present
- Page numbers must be integers
- If a chapter seems incomplete (title without page number), skip it
- Return ONLY the JSON object, no explanation
"""


class TOCParsingStrategy:
    """Base class for different TOC parsing strategies"""
    
    def __init__(self, model: str = "llama-3.1-8b-instant", server: str = "groq"):
        self.model = model
        self.server = server
        self.agent = Agent(
            model=self.model,
            systemPrompt=TOC_PARSING_PROMPT,
            server=self.server
        )
    
    async def parse_toc_content(self, content: str, context: str = "") -> Dict[str, Any]:
        """Parse TOC content using LLM"""
        prompt = f"Parse this section of a table of contents{context}:\n\n{content}"
        
        response = self.agent.runAgent(prompt)
        
        if not response:
            return {"chapters": []}
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"chapters": []}


class ChunkedTOCParser(TOCParsingStrategy):
    """Parse TOC by splitting pages into chunks"""
    
    def __init__(self, max_pages_per_chunk: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_pages_per_chunk = max_pages_per_chunk
    
    async def parse_toc_pages(self, pdf_path: str, toc_pages: List[int]) -> Dict[str, Any]:
        """Parse TOC by splitting pages evenly across multiple LLM calls"""
        
        # Split pages into chunks
        page_chunks = []
        for i in range(0, len(toc_pages), self.max_pages_per_chunk):
            chunk_pages = toc_pages[i:i + self.max_pages_per_chunk]
            page_chunks.append(chunk_pages)
        
        print(f"Split {len(toc_pages)} TOC pages into {len(page_chunks)} chunks:")
        for i, chunk in enumerate(page_chunks):
            print(f"  Chunk {i+1}: pages {chunk}")
        
        all_chapters = []
        
        for i, chunk_pages in enumerate(page_chunks):
            try:
                print(f"Processing chunk {i+1}/{len(page_chunks)} (pages {chunk_pages})...")
                
                # Extract markdown for this chunk of pages
                zero_indexed_pages = [p - 1 for p in chunk_pages]
                chunk_markdown = pymupdf4llm.to_markdown(pdf_path, pages=zero_indexed_pages)
                
                if not chunk_markdown or len(chunk_markdown.strip()) < 20:
                    print(f"Warning: No meaningful content in chunk {i+1}")
                    continue
                
                # Clean the content
                cleaned_chunk = clean_toc_content(chunk_markdown)
                
                print(f"  Chunk {i+1} content: {len(cleaned_chunk)} chars")
                
                # Add context about this being a partial TOC
                context = f" (pages {chunk_pages}, chunk {i+1} of {len(page_chunks)})"
                
                chunk_data = await self.parse_toc_content(cleaned_chunk, context)
                
                # Extract chapters from this chunk
                chunk_chapters = chunk_data.get("chapters", [])
                if chunk_chapters:
                    all_chapters.extend(chunk_chapters)
                    print(f"✓ Extracted {len(chunk_chapters)} chapters from chunk {i+1}")
                else:
                    print(f"Warning: No chapters found in chunk {i+1}")
                    
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        # Sort chapters by start_page to ensure correct order
        if all_chapters:
            try:
                all_chapters.sort(key=lambda x: int(x.get("start_page", 0)))
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not sort chapters by page number: {str(e)}")
        
        print(f"Total chapters extracted: {len(all_chapters)}")
        
        return {
            "chapters": all_chapters,
            "page_chunks_processed": len(page_chunks),
            "total_chapters": len(all_chapters)
        }


class SinglePassTOCParser(TOCParsingStrategy):
    """Parse entire TOC in a single LLM call"""
    
    async def parse_toc_pages(self, pdf_path: str, toc_pages: List[int]) -> Dict[str, Any]:
        """Parse entire TOC in one go"""
        print(f"Processing all TOC pages in single pass: {toc_pages}")
        
        # Extract markdown for all pages
        zero_indexed_pages = [p - 1 for p in toc_pages]
        toc_markdown = pymupdf4llm.to_markdown(pdf_path, pages=zero_indexed_pages)
        
        if not toc_markdown or len(toc_markdown.strip()) < 20:
            return {"chapters": [], "error": "No meaningful content extracted"}
        
        # Clean the content
        cleaned_content = clean_toc_content(toc_markdown)
        
        print(f"Extracted content: {len(cleaned_content)} chars")
        
        # Parse with LLM
        context = f" (pages {toc_pages}, single pass)"
        toc_data = await self.parse_toc_content(cleaned_content, context)
        
        chapters = toc_data.get("chapters", [])
        print(f"Extracted {len(chapters)} chapters")
        
        return {
            "chapters": chapters,
            "total_chapters": len(chapters)
        }


class TOCService:
    """Main TOC service that can use different parsing strategies"""
    
    def __init__(self, strategy: TOCParsingStrategy = None):
        self.strategy = strategy or ChunkedTOCParser()
    
    def set_strategy(self, strategy: TOCParsingStrategy):
        """Switch to a different parsing strategy"""
        self.strategy = strategy
    
    async def parse_toc(self, pdf_path: str, toc_pages: List[int]) -> Dict[str, Any]:
        """Parse TOC using the current strategy"""
        return await self.strategy.parse_toc_pages(pdf_path, toc_pages)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available parsing strategies"""
        return [
            "chunked",      # ChunkedTOCParser - split pages into chunks
            "single_pass",  # SinglePassTOCParser - process all pages at once
        ]
    
    def create_strategy(self, strategy_name: str, **kwargs) -> TOCParsingStrategy:
        """Factory method to create parsing strategies"""
        strategies = {
            "chunked": ChunkedTOCParser,
            "single_pass": SinglePassTOCParser,
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
        
        return strategies[strategy_name](**kwargs)


# Default service instance
default_toc_service = TOCService()


# Convenience functions for backward compatibility
async def parse_toc_with_page_chunks(pdf_path: str, toc_pages: List[int], max_pages_per_chunk: int = 3) -> Dict[str, Any]:
    """Parse TOC using chunked strategy (backward compatibility)"""
    strategy = ChunkedTOCParser(max_pages_per_chunk=max_pages_per_chunk)
    service = TOCService(strategy)
    return await service.parse_toc(pdf_path, toc_pages)


async def parse_toc_single_pass(pdf_path: str, toc_pages: List[int]) -> Dict[str, Any]:
    """Parse TOC using single pass strategy"""
    strategy = SinglePassTOCParser()
    service = TOCService(strategy)
    return await service.parse_toc(pdf_path, toc_pages)