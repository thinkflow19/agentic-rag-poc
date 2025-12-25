"""Query API routes."""

import logging
from flask import Blueprint, request, jsonify

from app.services.query_service import QueryService

logger = logging.getLogger(__name__)

query_bp = Blueprint('query', __name__)
query_service = QueryService()


@query_bp.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON response with status and agent_ready flag
    """
    try:
        is_ready = query_service.is_ready()
        return jsonify({
            'status': 'healthy' if is_ready else 'unhealthy',
            'agent_ready': is_ready
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@query_bp.route('/query', methods=['POST'])
def query():
    """
    Process a query through the agent.
    
    Request body (JSON):
        - query: User query string (required)
        - chat_history: Optional list of previous messages
        
    Returns:
        JSON response with agent response or error message
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        user_query = data.get('query', '').strip()
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        chat_history = data.get('chat_history', [])
        
        result = query_service.process_query(user_query, chat_history)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

