"""OCR module for extracting text from scanned PDFs using Unstructured."""

import logging
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)


def extract_text_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using OCR (for scanned/image PDFs).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dicts with keys: 'text', 'page_number'
    """
    logger.info(f"Extracting text with OCR from {pdf_path}")
    
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="ocr_only",
            infer_table_structure=False,
        )
        
        pages = []
        current_page = 1
        current_text = []
        
        for element in elements:
            page_number = getattr(element.metadata, 'page_number', current_page)
            
            if page_number != current_page and current_text:
                pages.append({
                    'text': '\n'.join(current_text),
                    'page_number': current_page
                })
                current_text = []
                current_page = page_number
            
            if hasattr(element, 'text') and element.text:
                current_text.append(element.text.strip())
        
        if current_text:
            pages.append({
                'text': '\n'.join(current_text),
                'page_number': current_page
            })
        
        logger.info(f"Extracted {len(pages)} pages via OCR")
        return pages
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise


def detect_scanned_pdf(pages: List[Dict[str, Any]]) -> bool:
    """
    Detect if PDF is scanned/image-based based on text extraction quality.
    
    Args:
        pages: List of page dicts with 'text' and 'page_number' keys
        
    Returns:
        True if PDF appears to be scanned (low text content or many empty pages)
    """
    if not pages:
        return True
    
    total_chars = sum(len(page.get('text', '')) for page in pages)
    avg_chars_per_page = total_chars / len(pages)
    
    empty_pages = sum(1 for page in pages if len(page.get('text', '').strip()) == 0)
    empty_ratio = empty_pages / len(pages)
    
    is_scanned = avg_chars_per_page < 50 or empty_ratio > 0.5
    
    logger.info(
        f"PDF scan detection: avg_chars={avg_chars_per_page:.1f}, "
        f"empty_ratio={empty_ratio:.2f}, is_scanned={is_scanned}"
    )
    
    return is_scanned

