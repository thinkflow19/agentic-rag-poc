"""Hybrid retrieval combining BM25 and vector search with RRF fusion."""

import logging
import pickle
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.retrieval.fusion import reciprocal_rank_fusion, boost_bm25_for_identifiers

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 and vector search with RRF fusion."""
    
    def __init__(self, artifacts_dir: Path):
        """
        Initialize retriever by loading all artifacts.
        
        Args:
            artifacts_dir: Path to directory containing ingestion artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load BM25 index, vector index, and chunk metadata."""
        chunks_path = self.artifacts_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}. Run ingestion first.")
        with open(chunks_path, 'r') as f:
            self.chunks = json.load(f)
        
        if not self.chunks:
            raise ValueError("Chunks file is empty. Run ingestion first.")
        
        self.chunk_lookup = {chunk['chunk_id']: chunk for chunk in self.chunks}
        
        bm25_path = self.artifacts_dir / "bm25_index.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}. Run ingestion first.")
        with open(bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data['bm25']
            self.bm25_tokenized_corpus = bm25_data['tokenized_corpus']
            self.bm25_chunk_ids = bm25_data['chunk_ids']
        
        vector_index_path = self.artifacts_dir / "vector.index"
        if not vector_index_path.exists():
            raise FileNotFoundError(f"Vector index not found: {vector_index_path}. Run ingestion first.")
        self.vector_index = faiss.read_index(str(vector_index_path))
        
        vector_meta_path = self.artifacts_dir / "vector_store_meta.json"
        if not vector_meta_path.exists():
            raise FileNotFoundError(f"Vector metadata not found: {vector_meta_path}. Run ingestion first.")
        with open(vector_meta_path, 'r') as f:
            vector_meta = json.load(f)
            self.vector_chunk_ids = vector_meta['chunk_ids']
            self.vector_model_name = vector_meta['model_name']
        
        self.embedding_model = SentenceTransformer(self.vector_model_name)
        
        logger.info(
            f"Loaded retriever: {len(self.chunks)} chunks, "
            f"BM25 index, FAISS index ({self.vector_index.ntotal} vectors)"
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 (same logic as ingestion).
        
        Args:
            text: Text string to tokenize
            
        Returns:
            List of tokens (identifiers preserved as single tokens)
        """
        id_pattern = re.compile(r'\b([A-Z]{1,5}-\d+)\b')
        protected = {}
        counter = 0
        
        def replace_id(match):
            nonlocal counter
            token = f"__ID_{counter}__"
            protected[token] = match.group(1)
            counter += 1
            return token
        
        text_protected = id_pattern.sub(replace_id, text)
        tokens = re.findall(r'\b\w+\b', text_protected.lower())
        tokens = [protected.get(t, t) for t in tokens]
        
        return tokens
    
    def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Run BM25 search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of result dicts with 'chunk_id', 'score', 'text', 'page_number'
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self.bm25_chunk_ids[idx]
                chunk_data = self.chunk_lookup.get(chunk_id, {})
                
                results.append({
                    'chunk_id': chunk_id,
                    'score': float(scores[idx]),
                    'text': chunk_data.get('text', ''),
                    'page_number': chunk_data.get('page_number', 0)
                })
        
        return results
    
    def _vector_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Run vector similarity search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of result dicts with 'chunk_id', 'score', 'text', 'page_number'
        """
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype('float32')
        
        k = min(top_k, self.vector_index.ntotal)
        distances, indices = self.vector_index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.vector_chunk_ids):
                chunk_id = self.vector_chunk_ids[idx]
                chunk_data = self.chunk_lookup.get(chunk_id, {})
                
                results.append({
                    'chunk_id': chunk_id,
                    'score': float(distance),
                    'text': chunk_data.get('text', ''),
                    'page_number': chunk_data.get('page_number', 0)
                })
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: If True, use RRF fusion. If False, return BM25 only.
            
        Returns:
            List of results with chunk_id, text, page_number, and ranking info
            (bm25_rank, vector_rank, hybrid_rank, rrf_score)
        """
        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []
        
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        vector_results = self._vector_search(query, top_k=top_k * 2)
        
        bm25_results = boost_bm25_for_identifiers(query, bm25_results)
        
        if not use_hybrid:
            return bm25_results[:top_k]
        
        fused_results = reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            k=60,
            top_k=top_k
        )
        
        return fused_results

