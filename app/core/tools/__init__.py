"""LangChain tools for agent."""

from app.core.tools.document_search import document_search
from app.core.tools.internet_search import internet_search

__all__ = ['document_search', 'internet_search']
