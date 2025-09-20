import io
import json
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# Import our models
from models import (ConfirmedTOC, ConfirmTOCResponse, ContextExpandedResult,
                    DocumentChaptersResponse, DocumentListResponse,
                    ErrorResponse, ParseTOCResponse, SimpleChunkListResponse)
from models.answer import AnswerResponse, CitationInfo, QuestionRequest
from PIL import Image
# Import our services - FIXED DocumentReader import
from services import (apply_page_offset_mapping, calculate_chapter_boundaries,
                      default_chunking_service, default_context_service,
                      default_document_processor, default_embedding_service,
                      default_toc_service, validate_chapter_data,
                      validate_chapter_mapping, validate_chapters_exist,
                      validate_content_starts_at, validate_toc_structure)
from services.answer_service import default_answer_service
# Import DocumentReader directly to avoid recursion
from services.document_service import DocumentReader
# Import utilities
from utils import (cleanup_document_files, ensure_directory_exists,
                   get_pdf_info, parse_toc_pages)

app = FastAPI(
    title="RAG System API", 
    version="9.0.0",
    description="Simple paragraph-based RAG system with context expansion"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "file://"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = Path("../data")
DOCUMENTS_DIR = DATA_DIR / "documents"

# Initialize services - FIXED: Use DocumentReader class directly
ensure_directory_exists(DOCUMENTS_DIR)
document_reader = DocumentReader(DOCUMENTS_DIR)

@app.get("/")
async def root():
    return {
        "message": "RAG System API v9.0 - Simple & Effective", 
        "status": "healthy",
        "architecture": "paragraph_based",
        "features": [
            "📄 Simple paragraph chunking",
            "🔍 Semantic search with FAISS", 
            "📖 Context expansion (neighboring paragraphs)",
            "🎯 Chapter and page filtering",
            "📊 Clean data storage",
            "⚡ Fast processing",
            "🎨 Multiple TOC parsing strategies"
        ]
    }

@app.get("/info")
async def get_system_info():
    """Get system information and capabilities"""
    return {
        "version": "9.0.0",
        "architecture": "paragraph_based",
        "chunking_strategy": default_chunking_service.get_strategy_info(),
        "toc_strategies": default_toc_service.get_available_strategies(),
        "current_toc_strategy": default_toc_service.strategy.__class__.__name__,
        "embedding_model": default_embedding_service.model_name,
        "embedding_dimension": default_embedding_service.embedding_model.embedding_dim,
        "context_strategies": ["paragraph", "page", "mixed"],
        "search_features": [
            "similarity_search",
            "chapter_filtering", 
            "context_expansion",
            "neighboring_paragraphs"
        ]
    }

@app.post("/parse-toc", response_model=ParseTOCResponse)
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
        pdf_info = get_pdf_info(file_path)
        page_count = pdf_info["page_count"]
        
        print(f"Uploaded: {file.filename} ({page_count} pages)")
        
        # Parse TOC pages
        page_list = parse_toc_pages(toc_pages, page_count)
        
        # Extract and parse TOC
        print(f"Parsing TOC from pages {page_list} using {default_toc_service.strategy.__class__.__name__}...")
        parsed_toc_data = await default_toc_service.parse_toc(str(file_path), page_list)
        
        # Validate basic structure
        if not parsed_toc_data.get("chapters"):
            file_path.unlink()  # Cleanup
            raise HTTPException(
                status_code=422,
                detail="No chapters found in TOC. Please check the page range."
            )
        
        return ParseTOCResponse(
            doc_id=doc_id,
            filename=file.filename,
            page_count=page_count,
            toc_pages=page_list,
            parsed_toc=parsed_toc_data,
            message=f"Found {len(parsed_toc_data['chapters'])} chapters. Please review and confirm."
        )
        
    except HTTPException:
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        raise
    except Exception as e:
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        print(f"TOC parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TOC parsing failed: {str(e)}")

@app.post("/confirm-toc", response_model=ConfirmTOCResponse)
async def confirm_toc(
    doc_id: str = Form(...),
    content_starts_at: int = Form(...),
    toc_data: str = Form(...)  # JSON string of the confirmed/edited TOC
):
    """Process document with confirmed/edited TOC data using paragraph chunking"""
    
    try:
        # Parse the confirmed TOC data
        confirmed_toc_dict = json.loads(toc_data)
        confirmed_toc = ConfirmedTOC(**confirmed_toc_dict)
        
        # Check if temp PDF exists
        temp_file_path = DOCUMENTS_DIR / f"{doc_id}_temp.pdf"
        if not temp_file_path.exists():
            raise HTTPException(status_code=404, detail="Temporary file not found. Please re-upload.")
        
        # Move temp file to permanent location
        file_path = DOCUMENTS_DIR / f"{doc_id}.pdf"
        temp_file_path.rename(file_path)
        
        # Get PDF info
        pdf_info = get_pdf_info(file_path)
        page_count = pdf_info["page_count"]
        
        print(f"Processing document with {len(confirmed_toc.chapters)} chapters")
        print(f"Content starts at page: {content_starts_at}")
        print(f"Using chunking strategy: {default_chunking_service.strategy_name}")
        print(f"Using embedding model: {default_embedding_service.model_name}")
        print("📄 Processing with simple paragraph chunking...")
        print("   Split chapters into meaningful paragraphs")
        print("   Link neighboring paragraphs for context")
        print("   Generate embeddings with chapter context")
        
        # Validate content_starts_at
        content_validation = validate_content_starts_at(content_starts_at, page_count)
        if not content_validation["valid"]:
            cleanup_document_files(doc_id, DOCUMENTS_DIR)
            raise HTTPException(status_code=400, detail=content_validation["error"])
        
        # Validate confirmed TOC structure
        toc_validation = validate_toc_structure(confirmed_toc.dict(), page_count)
        if not toc_validation["valid"]:
            cleanup_document_files(doc_id, DOCUMENTS_DIR)
            raise HTTPException(
                status_code=422,
                detail=f"TOC validation failed: {'; '.join(toc_validation['errors'])}"
            )
        
        # Validate chapter data
        chapter_validation = validate_chapter_data([ch.dict() for ch in confirmed_toc.chapters])
        if not chapter_validation["valid"]:
            cleanup_document_files(doc_id, DOCUMENTS_DIR)
            raise HTTPException(
                status_code=422,
                detail=f"Chapter validation failed: {'; '.join(chapter_validation['errors'])}"
            )
        
        # Apply page offset mapping
        chapters_with_mapping = apply_page_offset_mapping(
            [ch.dict() for ch in confirmed_toc.chapters], 
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
            cleanup_document_files(doc_id, DOCUMENTS_DIR)
            raise HTTPException(
                status_code=422,
                detail=f"Chapter mapping failed: {'; '.join(mapping_validation['errors'])}"
            )
        
        # Calculate chapter boundaries
        chapters_with_boundaries = calculate_chapter_boundaries(
            chapters_with_mapping, 
            page_count
        )
        
        print(f"Processing {len(chapters_with_boundaries)} chapters into paragraphs...")
        
        # Use the paragraph processing pipeline
        try:
            processing_summary = await default_document_processor.process_document_with_embeddings(
                str(file_path),
                chapters_with_boundaries,
                doc_id,
                DOCUMENTS_DIR,
                confirmed_toc.dict(),
                content_starts_at,
                chapters_with_mapping,
                page_count
            )
            
            print("✅ Document processing completed successfully!")
            print(f"   📄 Architecture: {processing_summary['processing_mode']}")
            print(f"   📚 Chapters: {processing_summary['chapters_processed']}")
            print(f"   📝 Paragraphs: {processing_summary['total_chunks']}")
            print(f"   🔧 Strategy: {processing_summary['chunking_strategy']}")
            
            if processing_summary['embeddings_generated']:
                print(f"   ✅ Embeddings: {processing_summary.get('embedding_type', 'generated')}")
            else:
                print("   ⚠️ Embeddings: Failed (search not available)")
            
            # Get the final metadata for response
            final_metadata = document_reader.get_document_metadata(doc_id)
            
            # Create response message
            message = f"Successfully processed {processing_summary['chapters_processed']} chapters into {processing_summary['total_chunks']} paragraphs"
                
            if processing_summary['embeddings_generated']:
                message += f" and generated embeddings for semantic search"
            else:
                message += " (embeddings failed - search not available)"
        
        except Exception as e:
            print(f"Document processing failed: {str(e)}")
            cleanup_document_files(doc_id, DOCUMENTS_DIR)
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        
        return ConfirmTOCResponse(
            doc_id=doc_id,
            status="processed",
            chapters_count=final_metadata["chapters_count"],
            total_word_count=final_metadata["total_word_count"],
            total_chunk_count=final_metadata["total_chunk_count"],
            chapters=final_metadata["chapters"],
            message=message
        )
        
    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid TOC data format")
    except Exception as e:
        print(f"Document processing error: {str(e)}")
        cleanup_document_files(doc_id, DOCUMENTS_DIR)
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

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all processed documents"""
    documents = document_reader.get_all_documents()
    return DocumentListResponse(documents=documents)

@app.get("/documents/{doc_id}/chapters", response_model=DocumentChaptersResponse)
async def get_document_chapters(doc_id: str):
    """Get chapters for a specific document"""
    try:
        metadata = document_reader.get_document_metadata(doc_id)
        
        return DocumentChaptersResponse(
            doc_id=doc_id,
            filename=metadata["filename"],
            chapters_count=metadata["chapters_count"],
            total_chunk_count=metadata.get("total_chunk_count", 0),
            chapters=metadata["chapters"]
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading document data: {str(e)}")

@app.get("/documents/{doc_id}/chapters/{chapter_filename}")
async def get_chapter_chunks(doc_id: str, chapter_filename: str):
    """Get chunks for a specific chapter"""
    try:
        chapter_data = document_reader.get_chapter_chunks(doc_id, chapter_filename)
        
        return {
            "doc_id": doc_id,
            "chapter_filename": chapter_filename,
            "chapter_info": chapter_data.get("chapter_info", {}),
            "chunks": chapter_data.get("chunks", []),
            "total_chunks": len(chapter_data.get("chunks", [])),
            "architecture": "paragraph_based"
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chapter not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading chapter: {str(e)}")

@app.get("/documents/{doc_id}/chunks", response_model=SimpleChunkListResponse)
async def get_document_chunks(
    doc_id: str,
    chapter_filter: Optional[str] = Query(None, description="Filter by chapter name"),
    page_filter: Optional[int] = Query(None, description="Filter by page number")
):
    """Get all chunks from a document with optional filtering"""
    try:
        all_chunks = document_reader.get_all_chunks(doc_id)
        
        # Apply filters
        filtered_chunks = all_chunks
        
        if chapter_filter:
            filtered_chunks = [
                chunk for chunk in filtered_chunks 
                if chunk.get("chapter_name") == chapter_filter
            ]
        
        if page_filter:
            filtered_chunks = [
                chunk for chunk in filtered_chunks 
                if chunk.get("page_number") == page_filter
            ]
        
        return SimpleChunkListResponse(
            doc_id=doc_id,
            chunks=filtered_chunks,
            total_chunks=len(filtered_chunks),
            chapter_filter=chapter_filter,
            page_filter=page_filter
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading chunks: {str(e)}")

@app.get("/search/{doc_id}")
async def search_document(
    doc_id: str, 
    query: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    expand_context: bool = Query(True, description="Include neighboring paragraphs"),
    context_strategy: str = Query("paragraph", description="Context strategy: paragraph, page, mixed"),
    context_window: int = Query(2, ge=0, le=5, description="Context window size"),
    chapter_filter: Optional[str] = Query(None, description="Filter by chapter name")
):
    """Search document using paragraph-based semantic search with context expansion"""
    try:
        if not document_reader.document_exists(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if document has embeddings
        try:
            metadata = document_reader.get_document_metadata(doc_id)
            if not metadata.get("has_embeddings", False):
                raise HTTPException(
                    status_code=422, 
                    detail="Document does not have embeddings. Search not available."
                )
        except:
            pass  # Continue with search attempt
        
        # Perform search
        results = default_embedding_service.search_document(
            doc_id, query, k, expand_context, context_strategy, context_window, chapter_filter
        )
        
        if not results:
            return {
                "doc_id": doc_id,
                "query": query,
                "results": [],
                "total_results": 0,
                "message": "No results found. Document may not have embeddings generated."
            }
        
        # Add search context information
        search_info = {
            "doc_id": doc_id,
            "query": query,
            "expand_context": expand_context,
            "context_strategy": context_strategy,
            "context_window": context_window,
            "chapter_filter": chapter_filter,
            "results": results,
            "total_results": len(results),
            "embedding_model": default_embedding_service.model_name,
            "architecture": "paragraph_based"
        }
        
        return search_info
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search/{doc_id}/context/{chunk_id}")
async def get_chunk_context(
    doc_id: str, 
    chunk_id: str,
    strategy: str = Query("paragraph", description="Context strategy: paragraph, page, mixed"),
    window: int = Query(2, ge=0, le=5, description="Context window size")
):
    """Get context for a specific chunk"""
    try:
        context_result = default_embedding_service.get_chunk_context(
            doc_id, chunk_id, strategy, window
        )
        
        if "error" in context_result:
            raise HTTPException(status_code=404, detail=context_result["error"])
        
        return {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "strategy": strategy,
            "window": window,
            "context": context_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting context: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        doc_stats = document_reader.get_document_stats()
        embedding_stats = default_embedding_service.get_embedding_stats()
        
        combined_stats = {
            **doc_stats,
            "embedding_stats": embedding_stats,
            "system_info": {
                "architecture": "paragraph_based",
                "chunking_strategy": default_chunking_service.strategy_name,
                "toc_strategy": default_toc_service.strategy.__class__.__name__,
                "embedding_model": default_embedding_service.model_name,
                "embedding_type": "paragraph_based",
                "context_strategies": ["paragraph", "page", "mixed"],
                "version": "9.0.0"
            }
        }
        
        return combined_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/embeddings/stats")
async def get_embedding_stats():
    """Get embedding statistics"""
    try:
        stats = default_embedding_service.get_embedding_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embedding stats: {str(e)}")

@app.post("/embeddings/{doc_id}/regenerate")
async def regenerate_embeddings(doc_id: str):
    """Regenerate embeddings for a document"""
    try:
        if not document_reader.document_exists(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document metadata
        metadata = document_reader.get_document_metadata(doc_id)
        
        # Load all chapter chunks
        chapter_chunk_lists = []
        for chapter_info in metadata["chapters"]:
            chapter_data = document_reader.get_chapter_chunks(doc_id, chapter_info["filename"])
            chunks = chapter_data.get("chunks", [])
            
            # Convert to SimpleChunk objects
            from models.simple_chunk import SimpleChunk, SimpleChunkList
            simple_chunks = []
            for chunk_data in chunks:
                chunk = SimpleChunk(**chunk_data)
                simple_chunks.append(chunk)
            
            chunk_list = SimpleChunkList(
                chunks=simple_chunks,
                total_chunks=len(simple_chunks),
                chapter_name=chapter_info["title"],
                total_word_count=sum(chunk.word_count for chunk in simple_chunks)
            )
            chapter_chunk_lists.append(chunk_list)
        
        if not chapter_chunk_lists:
            raise HTTPException(status_code=404, detail="No chunk data found for document")
        
        # Generate new embeddings
        embedding_info = default_embedding_service.embed_document_chunks(doc_id, chapter_chunk_lists)
        
        return {
            "doc_id": doc_id,
            "message": "Embeddings regenerated successfully",
            "embedding_info": embedding_info,
            "architecture": "paragraph_based"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Embedding regeneration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to regenerate embeddings: {str(e)}")

@app.post("/switch-toc-strategy")
async def switch_toc_strategy(strategy_name: str, **config):
    """Switch the default TOC parsing strategy"""
    try:
        new_strategy = default_toc_service.create_strategy(strategy_name, **config)
        default_toc_service.set_strategy(new_strategy)
        
        return {
            "message": f"Switched to {strategy_name} TOC parsing strategy",
            "strategy": strategy_name,
            "config": config,
            "chunking_strategy": default_chunking_service.strategy_name
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/switch-embedding-model")
async def switch_embedding_model(model_name: str):
    """Switch the embedding model"""
    try:
        from services.embedding_service import EmbeddingService

        # Create new embedding service with different model
        new_embedding_service = EmbeddingService(model_name=model_name)
        
        # Update the document processor to use new embedding service
        default_document_processor.set_embedding_service(new_embedding_service)
        
        # Update global embedding service
        global default_embedding_service
        default_embedding_service = new_embedding_service
        
        return {
            "message": f"Switched to {model_name} embedding model",
            "model_name": model_name,
            "embedding_dimension": new_embedding_service.embedding_model.embedding_dim,
            "embedding_type": "paragraph_based",
            "note": "New documents will use this model. Existing embeddings unchanged."
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to switch embedding model: {str(e)}")

@app.post("/ask/{doc_id}", response_model=AnswerResponse)
async def ask_question(
    doc_id: str,
    request: QuestionRequest
):
    """Ask a question and get an AI-generated answer with citations"""
    try:
        if not document_reader.document_exists(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if document has embeddings
        try:
            metadata = document_reader.get_document_metadata(doc_id)
            if not metadata.get("has_embeddings", False):
                raise HTTPException(
                    status_code=422, 
                    detail="Document does not have embeddings. Question answering not available."
                )
        except:
            pass  # Continue with attempt
        
        # Extract search parameters
        search_params = request.search_params or {}
        k = search_params.get('k', 5)  # Get more results than needed for answer generation
        expand_context = search_params.get('expand_context', True)
        context_strategy = search_params.get('context_strategy', 'paragraph')
        context_window = search_params.get('context_window', 2)
        chapter_filter = search_params.get('chapter_filter')
        
        # First, perform semantic search
        search_results = default_embedding_service.search_document(
            doc_id, request.query, k, expand_context, context_strategy, context_window, chapter_filter
        )
        
        if not search_results:
            return AnswerResponse(
                doc_id=doc_id,
                query=request.query,
                answer="No relevant information found in the document for your question.",
                citations=[],
                chunks_used=0,
                total_context_words=0,
                search_strategy=context_strategy,
                context_expansion=expand_context,
                success=False,
                error_message="No search results found",
                has_embeddings=metadata.get("has_embeddings", True),
                architecture="paragraph_based"
            )
        
        # Generate answer using search results
        answer_result = default_answer_service.generate_answer(
            request.query, 
            search_results, 
            request.max_chunks
        )
        
        # Convert citations to response format
        citation_infos = [
            CitationInfo(
                chapter_name=citation.chapter_name,
                page_number=citation.page_number,
                chunk_id=citation.chunk_id,
                citation_text=citation.format_citation()
            )
            for citation in answer_result.citations
        ]
        
        return AnswerResponse(
            doc_id=doc_id,
            query=request.query,
            answer=answer_result.answer,
            citations=citation_infos,
            chunks_used=answer_result.chunks_used,
            total_context_words=answer_result.total_context_words,
            generation_time=answer_result.generation_time,
            search_strategy=context_strategy,
            context_expansion=expand_context,
            success=answer_result.success,
            error_message=answer_result.error_message,
            has_embeddings=metadata.get("has_embeddings", True),
            architecture="paragraph_based"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Question answering error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.get("/ask/{doc_id}/simple")
async def ask_simple_question(
    doc_id: str, 
    query: str = Query(..., description="Question to ask"),
    max_chunks: int = Query(2, ge=1, le=5, description="Maximum chunks to use")
):
    """Simple question interface - just returns the answer text"""
    try:
        # Use the main ask endpoint logic but return simplified response
        request = QuestionRequest(query=query, max_chunks=max_chunks)
        response = await ask_question(doc_id, request)
        
        if response.success:
            return {
                "answer": response.answer,
                "sources": [citation.citation_text for citation in response.citations],
                "chunks_used": response.chunks_used
            }
        else:
            return {
                "answer": response.answer,
                "sources": [],
                "error": response.error_message
            }
            
    except HTTPException as e:
        return {
            "answer": "Error: " + e.detail,
            "sources": [],
            "error": e.detail
        }
    except Exception as e:
        return {
            "answer": "An error occurred while processing your question.",
            "sources": [],
            "error": str(e)
        }

@app.post("/extract-question")
async def extract_question(file: UploadFile = File(...)):
    """Extract text from uploaded image using OCR"""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are supported")
    
    try:
        # Read the uploaded image
        image_content = await file.read()
        
        # Save to temporary file (PyMuPDF4LLM needs file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_content)
            temp_file_path = temp_file.name
        
        try:
            # Use PyMuPDF4LLM to extract text from image
            import pymupdf4llm
            extracted_text = pymupdf4llm.to_markdown(temp_file_path)
            print(extracted_text)
            
            if not extracted_text or len(extracted_text.strip()) < 5:
                return {
                    "success": False,
                    "extracted_text": "",
                    "message": "No text found in image. Try a clearer photo or type the question manually."
                }
            
            # Basic text cleaning
            cleaned_text = clean_ocr_text(extracted_text)
            
            return {
                "success": True,
                "extracted_text": cleaned_text,
                "raw_text": extracted_text,
                "message": f"Extracted {len(cleaned_text)} characters. Review and search if correct."
            }
            
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        print(f"OCR extraction error: {str(e)}")
        return {
            "success": False,
            "extracted_text": "",
            "message": f"OCR failed: {str(e)}. Please try typing the question manually."
        }


def clean_ocr_text(text: str) -> str:
    """Basic OCR text cleaning"""
    import re

    # Remove markdown formatting that PyMuPDF4LLM adds
    text = re.sub(r'#{1,6}\s*', '', text)  # Remove headers
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Remove bold/italic
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code formatting
    
    # Clean up common OCR artifacts
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' {3,}', ' ', text)  # Max 2 spaces
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    
    # Fix common OCR errors
    text = re.sub(r'\b[Il1]\b', 'I', text)  # Fix I/l/1 confusion
    text = re.sub(r'\b[O0]\b', '0', text)  # Fix O/0 confusion
    
    # Remove obvious page artifacts
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Chapter \d+', '', text, flags=re.IGNORECASE)
    print(text)
    
    return text.strip()

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG System API v9.0 - Simple & Effective!")
    print("📁 Data directory:", DATA_DIR.absolute())
    print(f"📖 TOC Strategy: {default_toc_service.strategy.__class__.__name__}")
    print(f"📝 Chunking Strategy: {default_chunking_service.strategy_name}")
    print("   📄 Simple paragraph-based chunking")
    print("   🔗 Neighboring paragraph context")
    print("   📊 Clean storage with minimal complexity")
    
    # Check if embedding model loads successfully
    try:
        print(f"🤖 Embedding Model: {default_embedding_service.model_name}")
        print(f"🔢 Embedding Dimension: {default_embedding_service.embedding_model.embedding_dim}")
        print("🔍 Search Features: semantic similarity + context expansion")
        print("✅ Paragraph-based embeddings ready!")
    except Exception as e:
        print(f"⚠️  Warning: Embedding service not available: {str(e)}")
        print("📦 Install with: pip install sentence-transformers faiss-cpu")
    
    print("\n🎯 New in v9.0 - Simplified Architecture:")
    print("   • Removed artificial hierarchy (60% less code)")
    print("   • Simple paragraph chunking with context expansion")
    print("   • Clean data storage (JSON lists, not trees)")
    print("   • Same search quality with less complexity")
    print("   • Neighboring paragraph context strategy")
    print("   • Chapter and page filtering")
    
    print("\n📡 Key API Endpoints:")
    print("   • GET /search/{doc_id} - Semantic search with context")
    print("   • GET /search/{doc_id}/context/{chunk_id} - Get chunk context")
    print("   • GET /documents/{doc_id}/chunks - Browse all paragraphs")
    print("   • POST /confirm-toc - Process with paragraph chunking")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)