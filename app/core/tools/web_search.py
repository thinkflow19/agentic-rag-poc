"""Web search tool for LangChain agent."""

import logging
import os
import threading
import warnings
from typing import Dict, Any

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger(__name__)

# Suppress deprecation warnings for fallback import
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community.tools.tavily_search")
    try:
        from langchain_tavily import TavilySearch
        TAVILY_AVAILABLE = True
        TAVILY_CLASS = TavilySearch
    except ImportError:
        try:
            from langchain_community.tools import TavilySearchResults
            TAVILY_AVAILABLE = True
            TAVILY_CLASS = TavilySearchResults
        except ImportError:
            TAVILY_AVAILABLE = False
            TAVILY_CLASS = None

_search_tool = None
_search_tool_lock = threading.Lock()


def clear_search_cache():
    """
    Clear the search tool cache to force reload (thread-safe).
    Useful when API key changes or to reset the search tool instance.
    """
    global _search_tool
    with _search_tool_lock:
        _search_tool = None


def _get_search_tool():
    """
    Get web search tool (Tavily if available, else DuckDuckGo) - thread-safe.
    Uses lazy loading and singleton pattern with double-check locking.
    
    Returns:
        LangChain search tool instance (TavilySearch, TavilySearchResults, or DuckDuckGoSearchRun)
    """
    global _search_tool
    if _search_tool is None:
        with _search_tool_lock:
            # Double-check pattern to prevent race condition
            if _search_tool is None:
                tavily_api_key = os.getenv("TAVILY_API_KEY")
                if tavily_api_key and TAVILY_AVAILABLE and TAVILY_CLASS:
                    logger.info("Using Tavily for web search")
                    if TAVILY_CLASS.__name__ == "TavilySearch":
                        _search_tool = TAVILY_CLASS(max_results=5, api_key=tavily_api_key)
                    else:
                        _search_tool = TAVILY_CLASS(max_results=5, api_key=tavily_api_key)
                else:
                    logger.info("Using DuckDuckGo for web search (Tavily API key not found or package not available)")
                    _search_tool = DuckDuckGoSearchRun()
    return _search_tool


@tool
def web_search(query: str) -> Dict[str, Any]:
    """
    Search the web for information.
    Returns evidence snippets and URLs, not final answers.
    
    Use this tool when:
    - Document search returns low confidence or no results
    - Query is about general knowledge not in the documents
    - Query requires current/recent information
    
    Args:
        query: The search query
        
    Returns:
        Dictionary with:
        - results: List of search results with content and url
        - source: "tavily" or "duckduckgo"
    """
    try:
        if not query or not query.strip():
            return {
                "results": [],
                "source": "error"
            }
        
        search_tool = _get_search_tool()
        tavily_key = os.getenv("TAVILY_API_KEY")
        formatted_results = []
        
        raw_results = search_tool.invoke(query)
        
        if tavily_key and TAVILY_AVAILABLE:
            if isinstance(raw_results, dict) and "results" in raw_results:
                for item in raw_results.get("results", []):
                    formatted_results.append({
                        "content": item.get("content", item.get("snippet", "")),
                        "url": item.get("url", "")
                    })
            elif isinstance(raw_results, list):
                for item in raw_results:
                    formatted_results.append({
                        "content": item.get("content", item.get("snippet", "")),
                        "url": item.get("url", "")
                    })
        else:
            if hasattr(search_tool, 'api_wrapper') and hasattr(search_tool.api_wrapper, 'results'):
                structured = search_tool.api_wrapper.results(query, max_results=5)
                for item in structured:
                    formatted_results.append({
                        "content": item.get("snippet", item.get("body", "")),
                        "url": item.get("link", item.get("href", ""))
                    })
            else:
                if isinstance(raw_results, str) and raw_results:
                    formatted_results.append({
                        "content": raw_results,
                        "url": ""
                    })
        
        source = "tavily" if os.getenv("TAVILY_API_KEY") else "duckduckgo"
        
        return {
            "results": formatted_results,
            "source": source
        }
        
    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)
        return {
            "results": [],
            "source": "error"
        }

