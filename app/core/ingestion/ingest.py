"""Core ingestion pipeline functions."""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse

import requests
from pypdf import PdfReader

from app.core.ingestion.ocr import extract_text_with_ocr, detect_scanned_pdf
from app.core.ingestion.chunking import chunk_text
from app.core.ingestion.build_bm25 import build_bm25_index
from app.core.ingestion.build_vector_index import build_vector_index

logger = logging.getLogger(__name__)

# Maximum PDF file size (100MB)
MAX_PDF_SIZE = 100 * 1024 * 1024
# Lock file timeout (5 minutes)
LOCK_TIMEOUT = 300


def extract_text_embedded(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with embedded text (non-scanned).
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of dicts with 'text' and 'page_number' keys
    """
    logger.info(f"Extracting embedded text from {pdf_path}")
    
    pages = []
    reader = PdfReader(pdf_path)
    
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        pages.append({
            'text': text,
            'page_number': i
        })
    
    logger.info(f"Extracted {len(pages)} pages with embedded text")
    return pages


def download_pdf(url: str, output_path: Path) -> None:
    """
    Download PDF from URL with validation and size limits.
    
    Args:
        url: URL of PDF to download
        output_path: Path to save PDF file
        
    Raises:
        ValueError: If URL is invalid or file is too large
        requests.RequestException: If download fails
    """
    # Validate URL
    parsed = urlparse(url)
    if parsed.scheme not in ['http', 'https']:
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")
    if not parsed.netloc:
        raise ValueError("Invalid URL: missing domain")
    
    logger.info(f"Downloading PDF from {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    # Stream download to check size
    response = requests.get(url, headers=headers, timeout=30, stream=True)
    response.raise_for_status()
    
    total_size = 0
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                total_size += len(chunk)
                if total_size > MAX_PDF_SIZE:
                    raise ValueError(
                        f"PDF too large: {total_size} bytes (max: {MAX_PDF_SIZE} bytes). "
                        "File download aborted."
                    )
                f.write(chunk)
    
    logger.info(f"PDF saved to {output_path} ({total_size} bytes)")


def _acquire_lock(lock_file: Path) -> bool:
    """
    Acquire ingestion lock file (cross-platform).
    
    Args:
        lock_file: Path to lock file
        
    Returns:
        True if lock acquired, False if another process is ingesting
    """
    if lock_file.exists():
        # Check if lock is stale (older than timeout)
        try:
            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age > LOCK_TIMEOUT:
                logger.warning(f"Removing stale lock file (age: {lock_age:.1f}s)")
                lock_file.unlink()
            else:
                return False
        except OSError:
            # Lock file might have been removed, try to acquire
            pass
    
    try:
        # Create lock file atomically
        lock_file.touch(exist_ok=False)
        return True
    except FileExistsError:
        return False
    except OSError as e:
        logger.warning(f"Could not create lock file: {e}")
        return False


def _release_lock(lock_file: Path) -> None:
    """Release ingestion lock file."""
    try:
        if lock_file.exists():
            lock_file.unlink()
    except OSError as e:
        logger.warning(f"Could not remove lock file: {e}")


def process_document(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Process PDF through complete ingestion pipeline (thread-safe).
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save artifacts
        
    Returns:
        Dictionary with processing results:
        - status: 'success'
        - pages: Number of pages processed
        - chunks: Number of chunks created
        - extraction_method: Method used ('embedded', 'OCR', or 'unstructured_auto')
        - output_dir: Path to artifacts directory
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If path is invalid or not a PDF file, or if another ingestion is in progress
    """
    # Validate input path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    if pdf_path.suffix.lower() != '.pdf':
        raise ValueError(f"File must be a PDF (got: {pdf_path.suffix})")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Acquire lock to prevent concurrent ingestion
    lock_file = output_dir / ".ingestion.lock"
    if not _acquire_lock(lock_file):
        raise ValueError(
            f"Another ingestion is in progress. Lock file exists: {lock_file}. "
            f"Please wait for the current ingestion to complete."
        )
    
    try:
        logger.info("Attempting embedded text extraction...")
        pages = extract_text_embedded(str(pdf_path))
        
        is_scanned = detect_scanned_pdf(pages)
        
        if is_scanned:
            logger.info("PDF appears to be scanned. Switching to OCR...")
            try:
                pages = extract_text_with_ocr(str(pdf_path))
                extraction_method = "OCR"
            except Exception as ocr_error:
                logger.warning(f"OCR failed: {ocr_error}. Trying unstructured auto strategy...")
                from unstructured.partition.pdf import partition_pdf
                elements = partition_pdf(
                    filename=str(pdf_path),
                    strategy="auto",
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
                extraction_method = "unstructured_auto"
        else:
            extraction_method = "embedded"
        
        logger.info(f"Text extraction completed using {extraction_method} method")
        
        chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=150)
        
        if not chunks:
            raise ValueError("No chunks created from pages. Check if pages contain text.")
        
        # Write artifacts atomically to prevent corruption on concurrent ingestion
        chunks_path = output_dir / "chunks.json"
        chunks_data = [
            {
                'chunk_id': c['chunk_id'],
                'text': c['text'],
                'page_number': c['page_number'],
                'metadata': c['metadata']
            }
            for c in chunks
        ]
        
        # Atomic write: write to temp file first, then rename
        import shutil
        temp_chunks = chunks_path.with_suffix('.json.tmp')
        try:
            with open(temp_chunks, 'w') as f:
                json.dump(chunks_data, f, indent=2)
            shutil.move(str(temp_chunks), str(chunks_path))
            logger.info(f"Chunks saved to {chunks_path}")
        except Exception as e:
            if temp_chunks.exists():
                temp_chunks.unlink()
            raise
        
        build_bm25_index(chunks, output_dir)
        build_vector_index(chunks, output_dir)
        
        logger.info("Ingestion pipeline completed successfully!")
        logger.info(f"Artifacts saved to {output_dir}")
        
        return {
            'status': 'success',
            'pages': len(pages),
            'chunks': len(chunks),
            'extraction_method': extraction_method,
            'output_dir': str(output_dir)
        }
    finally:
        # Always release lock, even on error
        _release_lock(lock_file)


def ingest_from_url(url: str, output_dir: Path) -> Dict[str, Any]:
    """
    Ingest document from URL with validation.
    
    Args:
        url: URL of PDF to ingest
        output_dir: Directory to save artifacts
        
    Returns:
        Dictionary with processing results (same format as process_document)
        
    Raises:
        ValueError: If URL is invalid
        requests.RequestException: If download fails
    """
    # Validate URL format
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")
    
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL format: {url}")
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    try:
        download_pdf(url, tmp_path)
        return process_document(tmp_path, output_dir)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
