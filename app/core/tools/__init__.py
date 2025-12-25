"""LangChain tools for agent."""

from app.core.tools.document_search import document_search
from app.core.tools.web_search import web_search

__all__ = ['document_search', 'web_search']
