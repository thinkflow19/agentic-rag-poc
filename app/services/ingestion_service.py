"""Service for document ingestion."""

import logging
from pathlib import Path

from app.core.ingestion.ingest import ingest_from_url as _ingest_from_url

logger = logging.getLogger(__name__)


class IngestionService:
    """Service layer for document ingestion."""
    
    @staticmethod
    def ingest_from_url(url: str, output_dir: Path) -> dict:
        """
        Ingest document from URL.
        
        Args:
            url: URL of PDF to ingest
            output_dir: Directory to save artifacts
            
        Returns:
            Dictionary with processing results:
            - status: 'success'
            - pages: Number of pages processed
            - chunks: Number of chunks created
            - extraction_method: Method used
            - output_dir: Path to artifacts directory
        """
        return _ingest_from_url(url, output_dir)

