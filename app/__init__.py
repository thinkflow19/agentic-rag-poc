"""Flask application factory."""

import os
import logging
from flask import Flask
from dotenv import load_dotenv

from app.config import config
from app.errors import register_error_handlers

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name=None):
    """
    Create and configure Flask application.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing').
                     If None, uses FLASK_ENV environment variable or defaults to 'development'.
        
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    
    config_name = config_name or os.environ.get('FLASK_ENV', 'development')
    app.config.from_object(config[config_name])
    
    register_error_handlers(app)
    
    from app.routes import ingestion_bp, query_bp
    app.register_blueprint(ingestion_bp, url_prefix='/api/ingestion')
    app.register_blueprint(query_bp, url_prefix='/api/query')
    
    logger.info(f"Flask app created with config: {config_name}")
    return app

