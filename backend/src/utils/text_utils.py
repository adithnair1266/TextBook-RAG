import re
from typing import List


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
    content = re.sub(r'Â­\s*', '', content)  # Remove soft hyphens
    content = re.sub(r'\u0007', '', content)  # Remove bell characters
    content = re.sub(r'[^\x00-\x7F]+', ' ', content)  # Remove non-ASCII chars
    
    # Fix missing apostrophes and spaces
    content = re.sub(r'\b(\w+)\s+s\b', r"\1's", content)  # Fix "company s" -> "company's"
    content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Fix "asacorporate" -> "as a corporate"
    content = re.sub(r'i Phone', 'iPhone', content, flags=re.IGNORECASE)
    content = re.sub(r'onadaily', 'on a daily', content)
    content = re.sub(r'witharealistic', 'with a realistic', content)
    
    # Fix page number artifacts
    content = re.sub(r'\n\s*\d+Â­?\s*\n', '\n', content)  # Remove standalone page numbers
    content = re.sub(r'Phase \d+ Systems Planning\s*\d+Â­?\s*', '', content)
    
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


def split_paragraph_by_sentences(paragraph: str) -> List[str]:
    """Split a very long paragraph into sentences"""
    # Simple sentence splitting (can be improved with proper NLP)
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    return [s.strip() for s in sentences if s.strip()]


def extract_word_count(text: str) -> int:
    """Extract word count from text"""
    if not text:
        return 0
    return len(text.split())


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."