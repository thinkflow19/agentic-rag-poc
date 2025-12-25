"""Route blueprints."""

from app.routes.ingestion import ingestion_bp
from app.routes.query import query_bp

__all__ = ['ingestion_bp', 'query_bp']

