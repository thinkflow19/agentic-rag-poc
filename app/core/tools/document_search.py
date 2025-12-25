"""Document search tool for LangChain agent."""

import logging
import re
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from langchain.tools import tool
from app.core.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

_retriever: Optional[HybridRetriever] = None
_retriever_lock = threading.Lock()


def _get_retriever(artifacts_dir: Path = None) -> HybridRetriever:
    """
    Lazy load retriever singleton (thread-safe).
    
    Args:
        artifacts_dir: Optional path to artifacts directory. Uses config default if None.
        
    Returns:
        HybridRetriever instance
    """
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            # Double-check pattern to prevent race condition
            if _retriever is None:
                if artifacts_dir is None:
                    from app.config import Config
                    artifacts_dir = Config.ARTIFACTS_DIR
                _retriever = HybridRetriever(Path(artifacts_dir))
    return _retriever


def clear_retriever_cache():
    """
    Clear the retriever cache to force reload of indexes (thread-safe).
    Call this after ingesting new documents to ensure fresh retrieval.
    """
    global _retriever
    with _retriever_lock:
        _retriever = None


@tool
def document_search(query: str) -> Dict[str, Any]:
    """
    Search documents using hybrid retrieval (BM25 + vector search).
    Returns evidence chunks with page numbers, not final answers.
    
    Args:
        query: The search query. Can include identifiers like "HD-7961" or natural language.
        
    Returns:
        Dictionary with:
        - results: List of evidence chunks with chunk_id, page, text, and ranking info
        - confidence: "high" | "medium" | "low" based on retrieval quality
    """
    try:
        if not query or not query.strip():
            return {
                "results": [],
                "confidence": "low"
            }
        
        retriever = _get_retriever()
        search_results = retriever.search(query, top_k=5, use_hybrid=True)
        
        if not search_results:
            return {
                "results": [],
                "confidence": "low"
            }
        
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "chunk_id": result.get("chunk_id", ""),
                "page": result.get("page_number", 0),
                "text": result.get("text", ""),
                "bm25_rank": result.get("bm25_rank"),
                "vector_rank": result.get("vector_rank"),
                "hybrid_rank": result.get("hybrid_rank", 0)
            })
        
        confidence = _determine_confidence(query, search_results)
        
        return {
            "results": formatted_results,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Document search failed: {e}", exc_info=True)
        return {
            "results": [],
            "confidence": "low"
        }


def _determine_confidence(query: str, results: list) -> str:
    """
    Determine confidence level based on query and results.
    
    Args:
        query: Search query string
        results: List of search result dictionaries
        
    Returns:
        "high", "medium", or "low" confidence level
    """
    if not results:
        return "low"
    
    id_pattern = re.compile(r'\b[A-Z]{1,5}-\d+\b')
    query_has_id = bool(id_pattern.search(query))
    
    if query_has_id:
        query_ids = set(id_pattern.findall(query))
        for result in results:
            text = result.get("text", "")
            text_ids = set(id_pattern.findall(text))
            if query_ids & text_ids:
                return "high"
    
    top_result = results[0]
    if top_result.get("bm25_rank") is not None and top_result.get("bm25_rank", 999) <= 3:
        return "high"
    
    if len(results) >= 3:
        return "medium"
    
    return "low"

