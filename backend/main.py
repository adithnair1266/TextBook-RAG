import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pymupdf4llm
from backend.src.services.agents import Agent
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG System API", version="5.2.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "file://"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directories
DATA_DIR = Path("../data")
DOCUMENTS_DIR = DATA_DIR / "documents"

# Create directories
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

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

@app.get("/")
async def root():
    return {"message": "RAG System API v5.2 - Clean Chunking (No Overlap)", "status": "healthy"}

@app.post("/parse-toc")
async def parse_toc(
    file: UploadFile = File(...),
    toc_pages: str = Form(...),
):
    """Parse TOC and return results for user confirmation/editing"""
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    doc_id = str(uuid.uuid4())
    
    try:
        # Save PDF temporarily
        file_path = DOCUMENTS_DIR / f"{doc_id}_temp.pdf"
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Get PDF info
        import fitz
        doc = fitz.open(str(file_path))
        page_count = doc.page_count
        doc.close()
        
        print(f"Uploaded: {file.filename} ({page_count} pages)")
        
        # Parse TOC pages
        page_list = parse_toc_pages(toc_pages, page_count)
        
        # Extract and parse TOC with LLM
        print(f"Parsing TOC from pages {page_list}...")
        parsed_toc = await parse_toc_with_page_chunks(str(file_path), page_list)
        
        # Validate basic structure
        if not parsed_toc.get("chapters"):
            file_path.unlink()  # Cleanup
            raise HTTPException(
                status_code=422,
                detail="No chapters found in TOC. Please check the page range."
            )
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "page_count": page_count,
            "toc_pages": page_list,
            "parsed_toc": parsed_toc,
            "message": f"Found {len(parsed_toc['chapters'])} chapters. Please review and confirm."
        }
        
    except HTTPException:
        # Cleanup and re-raise
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        raise
    except Exception as e:
        # Cleanup and raise error
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        print(f"TOC parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TOC parsing failed: {str(e)}")

@app.post("/confirm-toc")
async def confirm_toc(
    doc_id: str = Form(...),
    content_starts_at: int = Form(...),
    toc_data: str = Form(...)  # JSON string of the confirmed/edited TOC
):
    """Process document with confirmed/edited TOC data"""
    
    try:
        # Parse the confirmed TOC data
        confirmed_toc = json.loads(toc_data)
        
        # Check if temp PDF exists
        temp_file_path = DOCUMENTS_DIR / f"{doc_id}_temp.pdf"
        if not temp_file_path.exists():
            raise HTTPException(status_code=404, detail="Temporary file not found. Please re-upload.")
        
        # Move temp file to permanent location
        file_path = DOCUMENTS_DIR / f"{doc_id}.pdf"
        temp_file_path.rename(file_path)
        
        # Get PDF info
        import fitz
        doc = fitz.open(str(file_path))
        page_count = doc.page_count
        doc.close()
        
        print(f"Processing document with {len(confirmed_toc.get('chapters', []))} chapters")
        print(f"Content starts at page: {content_starts_at}")
        
        # Validate content_starts_at
        if content_starts_at < 1 or content_starts_at > page_count:
            cleanup_document_files(doc_id)
            raise HTTPException(
                status_code=400,
                detail=f"content_starts_at ({content_starts_at}) must be between 1 and {page_count}"
            )
        
        # Validate confirmed TOC structure
        validation_result = validate_toc_structure(confirmed_toc, page_count)
        if not validation_result["valid"]:
            cleanup_document_files(doc_id)
            raise HTTPException(
                status_code=422,
                detail=f"TOC validation failed: {'; '.join(validation_result['errors'])}"
            )
        
        # Apply page offset mapping
        chapters_with_mapping = apply_page_offset_mapping(
            confirmed_toc.get("chapters", []), 
            content_starts_at
        )
        
        # Validate chapter mapping (first 3 chapters)
        print("Validating chapter mappings...")
        mapping_validation = await validate_chapter_mapping(
            str(file_path), 
            chapters_with_mapping[:3],
            page_count
        )
        
        if not mapping_validation["valid"]:
            cleanup_document_files(doc_id)
            raise HTTPException(
                status_code=422,
                detail=f"Chapter mapping failed: {'; '.join(mapping_validation['errors'])}"
            )
        
        # Calculate chapter boundaries
        chapters_with_boundaries = calculate_chapter_boundaries(
            chapters_with_mapping, 
            page_count
        )
        
        print(f"Extracting and chunking {len(chapters_with_boundaries)} chapters...")
        
        # Extract ALL chapters with clean chunking
        all_chapters_data = []
        total_chunks = 0
        
        for chapter in chapters_with_boundaries:
            try:
                chapter_data = await extract_chapter_content(str(file_path), chapter)
                all_chapters_data.append(chapter_data)
                total_chunks += chapter_data["chunk_count"]
                print(f"✓ Extracted: {chapter['title']} ({chapter_data['chunk_count']} clean chunks)")
                
            except Exception as e:
                print(f"✗ Failed to extract chapter '{chapter['title']}': {str(e)}")
                cleanup_document_files(doc_id)
                raise HTTPException(
                    status_code=500,
                    detail=f"Chapter extraction failed for '{chapter['title']}': {str(e)}"
                )
        
        # Save everything
        chapters_dir = DOCUMENTS_DIR / f"{doc_id}_chapters"
        chapters_dir.mkdir(exist_ok=True)
        
        saved_chapters = []
        for chapter_data in all_chapters_data:
            # Create sanitized filename
            filename = sanitize_chapter_filename(chapter_data["title"])
            chapter_path = chapters_dir / f"{filename}.json"
            
            # Save chapter
            with open(chapter_path, "w", encoding="utf-8") as f:
                json.dump(chapter_data, f, indent=2, ensure_ascii=False)
            
            saved_chapters.append({
                "title": chapter_data["title"],
                "filename": f"{filename}.json",
                "page_range": chapter_data["page_range"],
                "word_count": chapter_data["word_count"],
                "chunk_count": chapter_data["chunk_count"]
            })
        
        # Save TOC data
        toc_data_record = {
            "confirmed_toc": confirmed_toc,
            "content_starts_at": content_starts_at,
            "chapters_with_mapping": chapters_with_mapping
        }
        
        toc_path = DOCUMENTS_DIR / f"{doc_id}_toc.json"
        with open(toc_path, "w", encoding="utf-8") as f:
            json.dump(toc_data_record, f, indent=2, ensure_ascii=False)
        
        # Save document metadata
        metadata = {
            "doc_id": doc_id,
            "filename": "uploaded_document.pdf",  # We don't have original filename here
            "page_count": page_count,
            "status": "processed",
            "chapters_count": len(saved_chapters),
            "total_word_count": sum(ch["word_count"] for ch in saved_chapters),
            "total_chunk_count": total_chunks,
            "content_starts_at": content_starts_at,
            "processed_at": datetime.utcnow().isoformat(),
            "chapters": saved_chapters
        }
        
        metadata_path = DOCUMENTS_DIR / f"{doc_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return {
            "doc_id": doc_id,
            "status": "processed",
            "chapters_count": len(saved_chapters),
            "total_word_count": metadata["total_word_count"],
            "total_chunk_count": total_chunks,
            "chapters": saved_chapters,
            "message": f"Successfully extracted {len(saved_chapters)} chapters with {total_chunks} clean chunks"
        }
        
    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid TOC data format")
    except Exception as e:
        print(f"Document processing error: {str(e)}")
        cleanup_document_files(doc_id)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.delete("/cancel-processing/{doc_id}")
async def cancel_processing(doc_id: str):
    """Cancel processing and cleanup temporary files"""
    temp_file_path = DOCUMENTS_DIR / f"{doc_id}_temp.pdf"
    
    if temp_file_path.exists():
        temp_file_path.unlink()
        return {"message": "Processing cancelled and temporary files cleaned up"}
    else:
        raise HTTPException(status_code=404, detail="No processing found to cancel")

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    documents = []
    for metadata_path in DOCUMENTS_DIR.glob("*_metadata.json"):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            documents.append(metadata)
        except Exception as e:
            print(f"Error reading metadata {metadata_path}: {str(e)}")
            continue
    
    return {"documents": documents}

@app.get("/documents/{doc_id}/chapters")
async def get_document_chapters(doc_id: str):
    """Get chapters for a specific document"""
    metadata_path = DOCUMENTS_DIR / f"{doc_id}_metadata.json"
    
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return {
            "doc_id": doc_id,
            "filename": metadata["filename"],
            "chapters_count": metadata["chapters_count"],
            "total_chunk_count": metadata.get("total_chunk_count", 0),
            "chapters": metadata["chapters"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading document data: {str(e)}")

@app.get("/documents/{doc_id}/chapters/{chapter_filename}")
async def get_chapter_content(doc_id: str, chapter_filename: str):
    """Get content of a specific chapter including chunks"""
    chapter_path = DOCUMENTS_DIR / f"{doc_id}_chapters" / chapter_filename
    
    if not chapter_path.exists():
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    try:
        with open(chapter_path, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)
        
        return chapter_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading chapter: {str(e)}")

# Helper functions

def parse_toc_pages(toc_pages: str, page_count: int) -> List[int]:
    """Parse TOC pages input and validate"""
    try:
        if "," in toc_pages:
            page_list = [int(p.strip()) for p in toc_pages.split(",")]
        elif "-" in toc_pages:
            start_str, end_str = toc_pages.split("-", 1)
            start_page = int(start_str.strip())
            end_page = int(end_str.strip())
            if start_page < 1 or end_page < start_page:
                raise ValueError("Invalid page range")
            page_list = list(range(start_page, end_page + 1))
        else:
            single_page = int(toc_pages.strip())
            if single_page < 1:
                raise ValueError("Page number must be positive")
            page_list = [single_page]
        
        # Validate pages against PDF
        max_page = max(page_list)
        if max_page > page_count:
            raise ValueError(f"Page {max_page} exceeds document length ({page_count} pages)")
        
        return page_list
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid toc_pages format: {str(e)}")

def sanitize_chapter_filename(title: str) -> str:
    """Sanitize chapter title for filesystem"""
    # Remove common prefixes
    sanitized = re.sub(r'^Chapter\s+\d+\s*:?\s*', '', title, flags=re.IGNORECASE)
    sanitized = re.sub(r'^\d+\.?\d*\s*', '', sanitized)  # Remove leading numbers like "2.1"
    
    # Replace special characters and spaces
    sanitized = re.sub(r'[^\w\s-]', '', sanitized)  # Keep only alphanumeric, spaces, hyphens
    sanitized = re.sub(r'\s+', '_', sanitized.strip())  # Replace spaces with underscores
    sanitized = sanitized.lower()
    
    # Truncate if too long
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed_chapter"
    
    return sanitized

def clean_chapter_content(content: str) -> str:
    """Clean extracted chapter content before chunking"""
    
    # Generic copyright boilerplate removal (works for multiple publishers)
    copyright_patterns = [
        r'Copyright \d{4}.*?All Rights Reserved.*?',
        r'Editorial review has deemed.*?overall learning experience.*?',
        r'May not be copied, scanned, or duplicated.*?',
        r'No part of this publication may be reproduced.*?',
        r'All rights reserved\. No part of this work.*?',
        r'[A-Z][a-z]+ Learning reserves.*?require it\.',
        r'[A-Z][a-z]+ reserves the right.*?time\.',
        r'Printed in.*?United States.*?',
        r'ISBN.*?\d{4}\s*'
    ]
    
    for pattern in copyright_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up broken table formatting
    content = re.sub(r'</?(?:table|tr|td|th|br)\b[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\|[\s\-\|]*\|', '', content)  # Remove table separators
    content = re.sub(r'\|.*?\|', '', content)  # Remove table content lines
    
    # Fix broken formatting patterns
    content = re.sub(r'(\w+)<br>([^<\n]*)<br>', r'\1 \2', content)  # Fix <br> breaks
    
    # Clean up OCR artifacts and formatting issues
    content = re.sub(r'­\s*', '', content)  # Remove soft hyphens
    content = re.sub(r'\u0007', '', content)  # Remove bell characters
    content = re.sub(r'[^\x00-\x7F]+', ' ', content)  # Remove non-ASCII chars
    
    # Fix missing apostrophes and spaces
    content = re.sub(r'\b(\w+)\s+s\b', r"\1's", content)  # Fix "company s" -> "company's"
    content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Fix "asacorporate" -> "as a corporate"
    content = re.sub(r'i Phone', 'iPhone', content, flags=re.IGNORECASE)
    content = re.sub(r'onadaily', 'on a daily', content)
    content = re.sub(r'witharealistic', 'with a realistic', content)
    
    # Fix page number artifacts
    content = re.sub(r'\n\s*\d+­?\s*\n', '\n', content)  # Remove standalone page numbers
    content = re.sub(r'Phase \d+ Systems Planning\s*\d+­?\s*', '', content)
    
    # Clean up excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 consecutive newlines
    content = re.sub(r' {3,}', ' ', content)  # Max 2 consecutive spaces
    content = re.sub(r'\t+', ' ', content)  # Replace tabs with spaces
    
    # Remove figure/table references that are orphaned
    content = re.sub(r'\*\*FIGURE \d+-\d+\*\*\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'Source:\s*[^\n]*$', '', content, flags=re.MULTILINE)
    
    # Clean up repeated asterisks and formatting
    content = re.sub(r'\*{3,}', '**', content)  # Reduce multiple asterisks
    content = re.sub(r'(\*\*\w+\*\*)\s*(\*\*\w+\*\*)', r'\1 \2', content)  # Fix bold formatting
    
    # Final cleanup
    content = content.strip()
    
    return content

def clean_toc_content(content: str) -> str:
    """Remove noise from TOC content before LLM processing"""
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip obvious noise
        if re.match(r'^\s*\d+\s*$', stripped):  # Page numbers alone
            continue
        if re.match(r'^\s*(Contents|Table of Contents)\s*$', stripped, re.I):
            continue
        if re.match(r'^\s*(Page|Pg\.?)\s*\d*\s*$', stripped, re.I):
            continue
        if len(stripped) < 3:  # Very short lines
            continue
        if re.match(r'^\s*\.{3,}\s*$', stripped):  # Dot leaders only
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

async def parse_toc_with_page_chunks(pdf_path: str, toc_pages: List[int], max_pages_per_chunk: int = 3) -> Dict[str, Any]:
    """Parse TOC by splitting pages evenly across multiple LLM calls"""
    
    # Split pages into chunks
    page_chunks = []
    for i in range(0, len(toc_pages), max_pages_per_chunk):
        chunk_pages = toc_pages[i:i + max_pages_per_chunk]
        page_chunks.append(chunk_pages)
    
    print(f"Split {len(toc_pages)} TOC pages into {len(page_chunks)} chunks:")
    for i, chunk in enumerate(page_chunks):
        print(f"  Chunk {i+1}: pages {chunk}")
    
    all_chapters = []
    agent = Agent(
        model="llama-3.1-8b-instant",
        systemPrompt=TOC_PARSING_PROMPT,
        server="groq"
    )
    
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
            prompt = f"Parse this section of a table of contents (pages {chunk_pages}, chunk {i+1} of {len(page_chunks)}):\n\n{cleaned_chunk}"
            
            response = agent.runAgent(prompt)
            
            if not response:
                print(f"Warning: Empty response for chunk {i+1}")
                continue
            
            # Parse JSON response
            try:
                chunk_data = json.loads(response.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    chunk_data = json.loads(json_match.group())
                else:
                    print(f"Warning: Could not parse JSON from chunk {i+1}")
                    continue
            
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

def chunk_chapter_content(chapter_text: str, chapter_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chunk chapter content into optimal sizes for retrieval - NO OVERLAP"""
    
    TARGET_WORDS = 260  # ~350 tokens
    MIN_WORDS = 150     # ~200 tokens  
    MAX_WORDS = 375     # ~500 tokens
    
    # Split into paragraphs and filter out very short ones
    raw_paragraphs = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]
    paragraphs = []
    
    # Filter and clean paragraphs
    for para in raw_paragraphs:
        # Skip very short paragraphs (likely artifacts)
        if len(para.split()) < 5:
            continue
            
        # Skip paragraphs that are mostly formatting artifacts
        if re.match(r'^[\*\-\|\s]+$', para):
            continue
            
        # Skip standalone page numbers or headers
        if re.match(r'^\d+[\-­]?\s*$', para):
            continue
            
        # Skip figure references without content
        if re.match(r'^(FIGURE|TABLE|Source:)', para) and len(para.split()) < 10:
            continue
            
        paragraphs.append(para)
    
    if not paragraphs:
        # Fallback: if no good paragraphs found, use sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', chapter_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return create_sentence_based_chunks(sentences, chapter_info, TARGET_WORDS, MAX_WORDS)
    
    chunks = []
    current_chunk_paras = []
    current_word_count = 0
    
    chapter_title = chapter_info.get("title", "Unknown Chapter")
    
    for paragraph in paragraphs:
        para_word_count = len(paragraph.split())
        
        # Handle very short paragraphs (always merge)
        if para_word_count < 30:
            current_chunk_paras.append(paragraph)
            current_word_count += para_word_count
            
        # Handle very long paragraphs
        elif para_word_count > MAX_WORDS:
            # Save current chunk if exists
            if current_chunk_paras:
                chunks.append(create_chunk(current_chunk_paras, chapter_info, len(chunks)))
                current_chunk_paras = []
                current_word_count = 0
            
            # Split oversized paragraph by sentences
            if para_word_count > MAX_WORDS * 1.5:
                sentences = split_paragraph_by_sentences(paragraph)
                sentence_chunks = group_sentences_into_chunks(sentences, TARGET_WORDS, MAX_WORDS)
                for sentence_chunk in sentence_chunks:
                    chunks.append(create_chunk([sentence_chunk], chapter_info, len(chunks)))
            else:
                # Keep as standalone chunk (slightly over limit is OK)
                chunks.append(create_chunk([paragraph], chapter_info, len(chunks)))
                
        # Handle normal paragraphs
        else:
            # Check if adding this paragraph exceeds MAX_WORDS (strict limit)
            if current_word_count + para_word_count > MAX_WORDS and current_chunk_paras:
                # Save current chunk
                chunks.append(create_chunk(current_chunk_paras, chapter_info, len(chunks)))
                
                # Start new chunk (NO OVERLAP)
                current_chunk_paras = [paragraph]
                current_word_count = para_word_count
            else:
                # Add to current chunk
                current_chunk_paras.append(paragraph)
                current_word_count += para_word_count
    
    # Save final chunk
    if current_chunk_paras:
        chunks.append(create_chunk(current_chunk_paras, chapter_info, len(chunks)))
    
    # Merge tiny chunks with neighbors
    chunks = merge_tiny_chunks(chunks, MIN_WORDS)
    
    # Final validation and cleanup
    chunks = validate_and_clean_chunks(chunks, MIN_WORDS, MAX_WORDS)
    
    print(f"  Chunked '{chapter_title}': {len(chunks)} clean chunks (no overlap)")
    
    return chunks

def create_sentence_based_chunks(sentences: List[str], chapter_info: Dict[str, Any], target_words: int, max_words: int) -> List[Dict[str, Any]]:
    """Fallback chunking method using sentences when paragraph detection fails"""
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        if current_word_count + sentence_words > max_words and current_chunk:
            chunks.append(create_chunk([' '.join(current_chunk)], chapter_info, len(chunks)))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    if current_chunk:
        chunks.append(create_chunk([' '.join(current_chunk)], chapter_info, len(chunks)))
    
    return chunks

def validate_and_clean_chunks(chunks: List[Dict[str, Any]], min_words: int, max_words: int) -> List[Dict[str, Any]]:
    """Final validation and cleanup of chunks"""
    valid_chunks = []
    
    for chunk in chunks:
        chunk_text = chunk["text"].strip()
        word_count = len(chunk_text.split())
        
        # Skip chunks that are too small or mostly garbage
        if word_count < 50:  # Increased minimum
            continue
            
        # Skip chunks that are mostly formatting artifacts
        if len(re.sub(r'[^\w\s]', '', chunk_text).strip()) < 100:
            continue
            
        # Enforce hard maximum (split if too large)
        if word_count > max_words * 1.2:
            # Split oversized chunk into smaller pieces
            sentences = split_paragraph_by_sentences(chunk_text)
            sentence_chunks = group_sentences_into_chunks(sentences, max_words // 2, max_words)
            for i, sentence_chunk in enumerate(sentence_chunks):
                new_chunk = chunk.copy()
                new_chunk["text"] = sentence_chunk
                new_chunk["word_count"] = len(sentence_chunk.split())
                new_chunk["chunk_id"] = len(valid_chunks)
                valid_chunks.append(new_chunk)
        else:
            # Update word count to be accurate
            chunk["word_count"] = word_count
            valid_chunks.append(chunk)
    
    # Re-index chunks
    for idx, chunk in enumerate(valid_chunks):
        chunk["chunk_id"] = idx
    
    return valid_chunks

def create_chunk(paragraphs: List[str], chapter_info: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
    """Create a chunk object with metadata"""
    chunk_text = '\n\n'.join(paragraphs)
    word_count = len(chunk_text.split())
    
    return {
        "chunk_id": chunk_index,
        "text": chunk_text,
        "word_count": word_count,
        "chapter_title": chapter_info.get("title", ""),
        "chapter_page_start": chapter_info.get("start_page", 0),
        "chapter_page_end": chapter_info.get("end_page", 0),
        "subsections": chapter_info.get("subsections", []),
        "created_at": datetime.utcnow().isoformat()
    }

def split_paragraph_by_sentences(paragraph: str) -> List[str]:
    """Split a very long paragraph into sentences"""
    # Simple sentence splitting (can be improved with proper NLP)
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    return [s.strip() for s in sentences if s.strip()]

def group_sentences_into_chunks(sentences: List[str], target_words: int, max_words: int) -> List[str]:
    """Group sentences into appropriately sized chunks"""
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        if current_word_count + sentence_words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def merge_tiny_chunks(chunks: List[Dict[str, Any]], min_words: int) -> List[Dict[str, Any]]:
    """Merge chunks that are too small with their neighbors"""
    if len(chunks) <= 1:
        return chunks
    
    merged_chunks = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        
        # If chunk is too small, try to merge with next chunk
        if current_chunk["word_count"] < min_words and i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            combined_words = current_chunk["word_count"] + next_chunk["word_count"]
            
            # Only merge if combined size is reasonable
            if combined_words <= 450:  # Slightly above max to allow merging
                merged_text = current_chunk["text"] + "\n\n" + next_chunk["text"]
                
                merged_chunk = current_chunk.copy()
                merged_chunk["text"] = merged_text
                merged_chunk["word_count"] = combined_words
                
                merged_chunks.append(merged_chunk)
                i += 2  # Skip next chunk since we merged it
            else:
                # Keep current chunk even if small
                merged_chunks.append(current_chunk)
                i += 1
        else:
            merged_chunks.append(current_chunk)
            i += 1
    
    # Re-index chunks
    for idx, chunk in enumerate(merged_chunks):
        chunk["chunk_id"] = idx
    
    return merged_chunks

async def extract_chapter_content(pdf_path: str, chapter: Dict[str, Any]) -> Dict[str, Any]:
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
    
    # Chunk the cleaned chapter content (NO OVERLAP)
    chunks = chunk_chapter_content(cleaned_content, chapter)
    
    # Count words (rough estimate)
    word_count = len(cleaned_content.split())
    
    return {
        "title": chapter["title"],
        "toc_page": chapter.get("toc_start_page", chapter["start_page"]),
        "page_range": {
            "start": start_page,
            "end": end_page
        },
        "text": cleaned_content.strip(),
        "chunks": chunks,
        "subsections": chapter.get("subsections", []),
        "extracted_at": datetime.utcnow().isoformat(),
        "word_count": word_count,
        "chunk_count": len(chunks),
        "page_count": len(chapter_pages)
    }

def cleanup_document_files(doc_id: str):
    """Clean up all files for a document"""
    patterns = [
        f"{doc_id}.pdf",
        f"{doc_id}_temp.pdf",
        f"{doc_id}_metadata.json",
        f"{doc_id}_toc.json"
    ]
    
    for pattern in patterns:
        file_path = DOCUMENTS_DIR / pattern
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"Cleaned up: {pattern}")
            except Exception as e:
                print(f"Error cleaning up {pattern}: {str(e)}")
    
    # Clean up chapters directory
    chapters_dir = DOCUMENTS_DIR / f"{doc_id}_chapters"
    if chapters_dir.exists():
        try:
            import shutil
            shutil.rmtree(chapters_dir)
            print(f"Cleaned up chapters directory: {doc_id}_chapters")
        except Exception as e:
            print(f"Error cleaning up chapters directory: {str(e)}")

def validate_toc_structure(toc_data: Dict[str, Any], max_pages: int) -> Dict[str, Any]:
    """Validate parsed TOC structure"""
    errors = []
    
    if "chapters" not in toc_data:
        errors.append("Missing 'chapters' key")
        return {"valid": False, "errors": errors}
    
    chapters = toc_data["chapters"]
    if not isinstance(chapters, list) or len(chapters) == 0:
        errors.append("Chapters must be a non-empty list")
        return {"valid": False, "errors": errors}
    
    prev_page = 0
    for i, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            errors.append(f"Chapter {i+1} is not a dictionary")
            continue
            
        if "title" not in chapter or "start_page" not in chapter:
            errors.append(f"Chapter {i+1} missing title or start_page")
            continue
        
        try:
            start_page = int(chapter["start_page"])
            if start_page <= prev_page:
                errors.append(f"Chapter {i+1} page {start_page} not sequential")
            prev_page = start_page
        except (ValueError, TypeError):
            errors.append(f"Chapter {i+1} start_page is not a valid integer")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "chapters_count": len(chapters)
    }

def apply_page_offset_mapping(chapters: List[Dict[str, Any]], content_starts_at: int) -> List[Dict[str, Any]]:
    """Apply page offset mapping to convert TOC page numbers to physical PDF pages"""
    if not chapters:
        return chapters
    
    # Calculate offset
    first_toc_page = min(chapter["start_page"] for chapter in chapters)
    offset = content_starts_at - first_toc_page
    
    print(f"Page offset calculation: first TOC page = {first_toc_page}, offset = {offset}")
    
    chapters_with_mapping = []
    for chapter in chapters:
        chapter_copy = chapter.copy()
        chapter_copy["toc_start_page"] = chapter["start_page"]
        chapter_copy["start_page"] = chapter["start_page"] + offset
        
        # Handle subsections
        if "subsections" in chapter and isinstance(chapter["subsections"], list):
            mapped_subsections = []
            for subsection in chapter["subsections"]:
                if isinstance(subsection, dict) and "start_page" in subsection:
                    subsection_copy = subsection.copy()
                    subsection_copy["toc_start_page"] = subsection["start_page"]
                    subsection_copy["start_page"] = subsection["start_page"] + offset
                    mapped_subsections.append(subsection_copy)
                else:
                    mapped_subsections.append(subsection)
            chapter_copy["subsections"] = mapped_subsections
        
        chapters_with_mapping.append(chapter_copy)
    
    return chapters_with_mapping

async def validate_chapter_mapping(pdf_path: str, chapters: List[Dict[str, Any]], page_count: int) -> Dict[str, Any]:
    """Validate that chapter titles appear on their mapped physical pages"""
    validation_results = {
        "valid": True,
        "errors": [],
        "failed_chapters": []
    }
    
    for chapter in chapters:
        chapter_title = chapter.get("title", "")
        mapped_page = chapter.get("start_page")
        
        if not chapter_title or not mapped_page:
            validation_results["errors"].append(f"Chapter missing title or mapped page")
            continue
        
        if mapped_page < 1 or mapped_page > page_count:
            validation_results["errors"].append(
                f"Chapter '{chapter_title}': Mapped page {mapped_page} outside PDF bounds"
            )
            validation_results["failed_chapters"].append(chapter_title)
            continue
        
        # Simple validation - check if chapter appears on or near the mapped page
        clean_title = clean_chapter_title(chapter_title)
        found_title = False
        
        for offset in [0, 1, 2, -1, -2]:
            check_page = mapped_page + offset
            if 1 <= check_page <= page_count:
                try:
                    page_text = pymupdf4llm.to_markdown(pdf_path, pages=[check_page - 1])
                    if page_text and fuzzy_title_match(clean_title, page_text):
                        found_title = True
                        break
                except Exception:
                    continue
        
        if not found_title:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Chapter '{chapter_title}': Title not found near page {mapped_page}"
            )
            validation_results["failed_chapters"].append(chapter_title)
    
    return validation_results

def clean_chapter_title(title: str) -> str:
    """Clean chapter title for comparison"""
    cleaned = re.sub(r'^Chapter\s+\d+\s*:?\s*', '', title, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\d+\.?\s*', '', cleaned)
    return cleaned.strip()

def fuzzy_title_match(clean_title: str, page_text: str) -> bool:
    """Check if title appears in page text with fuzzy matching"""
    if not clean_title or not page_text:
        return False
    
    clean_title_lower = clean_title.lower()
    page_text_lower = page_text.lower()
    
    if clean_title_lower in page_text_lower:
        return True
    
    title_words = [word for word in re.findall(r'\w+', clean_title_lower) if len(word) > 2]
    if not title_words:
        return False
    
    found_words = sum(1 for word in title_words if word in page_text_lower)
    return (found_words / len(title_words)) >= 0.7

def calculate_chapter_boundaries(chapters: List[Dict[str, Any]], pdf_page_count: int) -> List[Dict[str, Any]]:
    """Calculate start and end pages for each chapter"""
    chapters_with_boundaries = []
    
    for i, chapter in enumerate(chapters):
        chapter_copy = chapter.copy()
        start_page = chapter["start_page"]
        
        if i < len(chapters) - 1:
            end_page = chapters[i + 1]["start_page"] - 1
        else:
            end_page = pdf_page_count
        
        chapter_copy["end_page"] = end_page
        chapters_with_boundaries.append(chapter_copy)
    
    return chapters_with_boundaries

if __name__ == "__main__":
    import uvicorn
    print("Starting RAG System API v5.2 - Clean Chunking (No Overlap)...")
    print("Data directory:", DATA_DIR.absolute())
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)