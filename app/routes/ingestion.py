"""Ingestion API routes."""

import logging
from flask import Blueprint, request, jsonify, current_app
from pathlib import Path

from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

ingestion_bp = Blueprint('ingestion', __name__)
ingestion_service = IngestionService()


@ingestion_bp.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON response with status 'healthy'
    """
    return jsonify({'status': 'healthy'})


@ingestion_bp.route('/ingest', methods=['POST'])
def ingest():
    """
    Ingest document from URL.
    
    Request body (JSON):
        - url: URL of PDF to ingest (required)
        - output_dir: Optional directory path for artifacts
        
    Returns:
        JSON response with processing results or error message
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        output_dir = Path(data.get('output_dir', current_app.config['ARTIFACTS_DIR']))
        
        result = ingestion_service.ingest_from_url(url, output_dir)
        return jsonify(result), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

