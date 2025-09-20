import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from models.simple_chunk import SimpleChunk
from pydantic import BaseModel, Field
from services.agents import Agent, ContextWindowExceededException


class CitationSource(BaseModel):
    """Citation source information"""
    chapter_name: str = Field(..., description="Chapter name")
    page_number: int = Field(..., description="Page number")
    chunk_id: str = Field(..., description="Source chunk ID")
    
    def format_citation(self) -> str:
        """Format as [Chapter, Page]"""
        return f"[{self.chapter_name}, Page {self.page_number}]"


class AnswerResult(BaseModel):
    """Answer generation result"""
    answer: str = Field(..., description="Generated answer")
    citations: List[CitationSource] = Field(..., description="Source citations")
    context_used: str = Field(..., description="Context text used for generation")
    chunks_used: int = Field(..., description="Number of target chunks used")
    total_context_words: int = Field(..., description="Total words in context")
    generation_time: Optional[float] = Field(None, description="Time to generate answer")
    success: bool = Field(True, description="Whether generation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class AnswerService:
    """Single-shot answer generation service"""
    
    def __init__(self, agent: Agent = None):
        # Use default Groq agent if none provided
        self.agent = agent or Agent(
            model="llama-3.1-8b-instant",
            systemPrompt=self._get_system_prompt(),
            server="groq"
        )
    
    def _get_system_prompt(self) -> str:
        """RAG-specific system prompt"""
        return """You are a helpful assistant that answers questions based on provided document context.

INSTRUCTIONS:
- Use ONLY information from the provided context
- Only take relevant information from context to answer the question asked, ignore non relevant information from context
- Provide comprehensive, well-structured answers 
- If the context doesn't contain enough information, say so clearly
- Do not make up information not in the context
- Be concise but thorough
- Return with proper markdown format
"""
    
    def generate_answer(self, query: str, search_results: List[Dict], 
                       max_target_chunks: int = 2) -> AnswerResult:
        """Generate answer from search results with expanded context"""
        
        start_time = datetime.utcnow()
        
        try:
            # Take top N target chunks with their expanded context
            selected_results = search_results[:max_target_chunks]
            
            if not selected_results:
                return AnswerResult(
                    answer="No relevant information found in the document.",
                    citations=[],
                    context_used="",
                    chunks_used=0,
                    total_context_words=0,
                    success=False,
                    error_message="No search results provided"
                )
            
            # Build context and collect citations
            context_parts = []
            citations = []
            seen_chunks = set()  # Deduplicate overlapping chunks
            
            for i, result in enumerate(selected_results):
                if 'target_chunk' in result:
                    # Context-expanded result
                    target_chunk = result['target_chunk']
                    context_chunks = result.get('context_chunks', [target_chunk])
                    
                    # Add citation for target chunk
                    citations.append(CitationSource(
                        chapter_name=target_chunk['chapter_name'],
                        page_number=target_chunk['page_number'],
                        chunk_id=target_chunk['chunk_id']
                    ))
                    
                    # Use the merged context text if available, otherwise build it
                    if 'context_text' in result and result['context_text']:
                        context_text = result['context_text']
                    else:
                        # Build context from chunks
                        context_text = self._merge_context_chunks(context_chunks)
                    
                    # Add section header
                    section_header = f"=== Source {i+1}: {target_chunk['chapter_name']} (Page {target_chunk['page_number']}) ==="
                    context_parts.append(f"{section_header}\n{context_text}")
                    
                else:
                    # Basic result (no context expansion)
                    citations.append(CitationSource(
                        chapter_name=result['chapter_name'],
                        page_number=result['page_number'],
                        chunk_id=result['chunk_id']
                    ))
                    
                    section_header = f"=== Source {i+1}: {result['chapter_name']} (Page {result['page_number']}) ==="
                    context_parts.append(f"{section_header}\n{result['text']}")
            
            # Combine all context
            full_context = "\n\n".join(context_parts)
            
            # Build the prompt
            prompt = self._build_prompt(query, full_context)
            
            # Generate answer
            try:
                raw_answer = self.agent.runAgent(prompt)
                
                if not raw_answer:
                    return AnswerResult(
                        answer="I apologize, but I was unable to generate an answer at this time.",
                        citations=citations,
                        context_used=full_context,
                        chunks_used=len(selected_results),
                        total_context_words=len(full_context.split()),
                        success=False,
                        error_message="LLM returned empty response"
                    )
                
                # Clean up the answer
                cleaned_answer = self._clean_answer(raw_answer)
                
                # Calculate metrics
                generation_time = (datetime.utcnow() - start_time).total_seconds()
                
                return AnswerResult(
                    answer=cleaned_answer,
                    citations=citations,
                    context_used=full_context,
                    chunks_used=len(selected_results),
                    total_context_words=len(full_context.split()),
                    generation_time=generation_time,
                    success=True
                )
                
            except ContextWindowExceededException:
                return AnswerResult(
                    answer="The context for this question is too large. Try asking a more specific question.",
                    citations=citations,
                    context_used=full_context,
                    chunks_used=len(selected_results),
                    total_context_words=len(full_context.split()),
                    success=False,
                    error_message="Context window exceeded"
                )
                
        except Exception as e:
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            return AnswerResult(
                answer="I apologize, but an error occurred while generating the answer.",
                citations=[],
                context_used="",
                chunks_used=0,
                total_context_words=0,
                generation_time=generation_time,
                success=False,
                error_message=str(e)
            )
    
    def _merge_context_chunks(self, chunks: List[Dict]) -> str:
        """Merge context chunks into readable text"""
        if not chunks:
            return ""
        
        # Sort by paragraph index to maintain reading order
        sorted_chunks = sorted(chunks, key=lambda x: x.get('paragraph_index', 0))
        
        # Check if chunks span multiple pages
        pages = set(chunk.get('page_number', 1) for chunk in sorted_chunks)
        
        if len(pages) > 1:
            # Group by page for better formatting
            page_groups = {}
            for chunk in sorted_chunks:
                page = chunk.get('page_number', 1)
                if page not in page_groups:
                    page_groups[page] = []
                page_groups[page].append(chunk['text'])
            
            # Format with page markers
            page_texts = []
            for page in sorted(page_groups.keys()):
                page_text = "\n\n".join(page_groups[page])
                page_texts.append(f"[Page {page}]\n{page_text}")
            
            return "\n\n---\n\n".join(page_texts)
        else:
            # Simple concatenation for single page
            return "\n\n".join(chunk['text'] for chunk in sorted_chunks)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the RAG prompt"""
        return f"""Based on the following context from the document, provide a comprehensive answer to the user's question.

CONTEXT:
{context}

QUESTION: {query}

Provide a detailed answer using only the information from the context above."""
    
    def _clean_answer(self, raw_answer: str) -> str:
        """Clean up the generated answer"""
        # Remove any unwanted prefixes
        answer = raw_answer.strip()
        
        # Remove common LLM prefixes
        prefixes_to_remove = [
            "Based on the provided context,",
            "According to the document,",
            "The context indicates that",
            "From the given information,"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        return answer
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the answer service"""
        return {
            "service_type": "single_shot_answer_generation",
            "model": self.agent.model,
            "server": self.agent.server,
            "max_target_chunks": 2,
            "citation_format": "[Chapter Name, Page X]",
            "features": [
                "context_expansion_aware",
                "automatic_citation_extraction", 
                "context_window_handling",
                "multi_page_context_formatting"
            ]
        }


# Default service instance
default_answer_service = AnswerService()


# Convenience functions
def generate_answer(query: str, search_results: List[Dict], max_chunks: int = 2) -> AnswerResult:
    """Generate answer from search results (convenience function)"""
    return default_answer_service.generate_answer(query, search_results, max_chunks)


def ask_question(query: str, search_results: List[Dict]) -> str:
    """Simple interface - just return the answer text"""
    result = default_answer_service.generate_answer(query, search_results)
    return result.answer if result.success else "Error generating answer."