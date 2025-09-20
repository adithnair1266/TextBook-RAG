import re
import shutil
from pathlib import Path
from typing import List, Union


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


def cleanup_document_files(doc_id: str, documents_dir: Path):
    """Clean up all files for a document"""
    patterns = [
        f"{doc_id}.pdf",
        f"{doc_id}_temp.pdf",
        f"{doc_id}_metadata.json",
        f"{doc_id}_toc.json"
    ]
    
    for pattern in patterns:
        file_path = documents_dir / pattern
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"Cleaned up: {pattern}")
            except Exception as e:
                print(f"Error cleaning up {pattern}: {str(e)}")
    
    # Clean up chapters directory
    chapters_dir = documents_dir / f"{doc_id}_chapters"
    if chapters_dir.exists():
        try:
            shutil.rmtree(chapters_dir)
            print(f"Cleaned up chapters directory: {doc_id}_chapters")
        except Exception as e:
            print(f"Error cleaning up chapters directory: {str(e)}")


def parse_toc_pages(toc_pages: str, page_count: int) -> List[int]:
    """Parse TOC pages input and validate"""
    from fastapi import HTTPException
    
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


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed"""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def format_file_size(bytes_size: int) -> str:
    """Format file size in human readable format"""
    if bytes_size == 0:
        return '0 Bytes'
    
    k = 1024
    sizes = ['Bytes', 'KB', 'MB', 'GB']
    i = 0
    
    while bytes_size >= k and i < len(sizes) - 1:
        bytes_size /= k
        i += 1
    
    return f"{bytes_size:.2f} {sizes[i]}"


def get_pdf_info(pdf_path: Union[str, Path]) -> dict:
    """Get basic PDF information"""
    import fitz
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(str(pdf_path))
    try:
        return {
            "page_count": doc.page_count,
            "file_size": pdf_path.stat().st_size,
            "file_size_formatted": format_file_size(pdf_path.stat().st_size)
        }
    finally:
        doc.close()