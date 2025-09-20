from pathlib import Path
from typing import Any, Dict, List

import pymupdf4llm
from utils.text_utils import clean_chapter_title, fuzzy_title_match


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


def validate_content_starts_at(content_starts_at: int, page_count: int) -> Dict[str, Any]:
    """Validate content_starts_at parameter"""
    if content_starts_at < 1 or content_starts_at > page_count:
        return {
            "valid": False,
            "error": f"content_starts_at ({content_starts_at}) must be between 1 and {page_count}"
        }
    
    return {"valid": True}


def validate_chapters_exist(chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that at least one chapter exists"""
    if not chapters or len(chapters) == 0:
        return {
            "valid": False,
            "error": "At least one chapter must be specified"
        }
    
    return {"valid": True}


def validate_chapter_data(chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate individual chapter data structure"""
    errors = []
    
    for i, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            errors.append(f"Chapter {i+1}: Must be a dictionary")
            continue
        
        if "title" not in chapter or not chapter["title"].strip():
            errors.append(f"Chapter {i+1}: Missing or empty title")
        
        if "start_page" not in chapter:
            errors.append(f"Chapter {i+1}: Missing start_page")
        else:
            try:
                start_page = int(chapter["start_page"])
                if start_page < 1:
                    errors.append(f"Chapter {i+1}: start_page must be positive")
            except (ValueError, TypeError):
                errors.append(f"Chapter {i+1}: start_page must be a valid integer")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }