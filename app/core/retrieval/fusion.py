"""Reciprocal Rank Fusion (RRF) for combining BM25 and vector search results."""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Fuse BM25 and vector search results using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) for each result set.
    
    Args:
        bm25_results: List of results from BM25 search with 'chunk_id' and 'score'
        vector_results: List of results from vector search with 'chunk_id' and 'score'
        k: RRF constant (typically 60)
        top_k: Number of final results to return
        
    Returns:
        List of fused results sorted by RRF score, with 'chunk_id', 'rrf_score', 
        'bm25_rank', 'vector_rank', 'hybrid_rank'
    """
    # Handle empty result lists
    if not bm25_results and not vector_results:
        return []
    
    # Build score maps
    bm25_scores = {}
    vector_scores = {}
    
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result.get('chunk_id')
        if not chunk_id:  # Skip results without chunk_id
            continue
        bm25_scores[chunk_id] = {
            'rank': rank,
            'score': result.get('score', 0.0),
            'result': result
        }
    
    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result.get('chunk_id')
        if not chunk_id:  # Skip results without chunk_id
            continue
        vector_scores[chunk_id] = {
            'rank': rank,
            'score': result.get('score', 0.0),
            'result': result
        }
    
    # Collect all unique chunk IDs
    all_chunk_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
    
    if not all_chunk_ids:
        return []
    
    # Calculate RRF scores
    fused_results = []
    
    for chunk_id in all_chunk_ids:
        rrf_score = 0.0
        
        bm25_rank = None
        vector_rank = None
        
        if chunk_id in bm25_scores:
            bm25_rank = bm25_scores[chunk_id]['rank']
            rrf_score += 1.0 / (k + bm25_rank)
        
        if chunk_id in vector_scores:
            vector_rank = vector_scores[chunk_id]['rank']
            rrf_score += 1.0 / (k + vector_rank)
        
        # Get the result data (prefer BM25 if both exist, as it has more metadata)
        result_data = (
            bm25_scores[chunk_id]['result'] 
            if chunk_id in bm25_scores 
            else vector_scores[chunk_id]['result']
        )
        
        fused_results.append({
            'chunk_id': chunk_id,
            'rrf_score': rrf_score,
            'bm25_rank': bm25_rank,
            'vector_rank': vector_rank,
            **result_data  # Include original result data
        })
    
    # Sort by RRF score (descending)
    fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
    
    # Add hybrid_rank
    for rank, result in enumerate(fused_results, start=1):
        result['hybrid_rank'] = rank
    
    # Return top_k
    return fused_results[:top_k]


def boost_bm25_for_identifiers(
    query: str,
    bm25_results: List[Dict[str, Any]],
    boost_factor: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Boost BM25 results if query contains identifier patterns.
    
    Args:
        query: Search query
        bm25_results: BM25 search results
        boost_factor: Multiplier for boosting scores
        
    Returns:
        Boosted results (same structure as input)
    """
    # Check if query contains identifier pattern
    id_pattern = re.compile(r'\b[A-Z]{1,5}-\d+\b')
    has_identifier = bool(id_pattern.search(query))
    
    if not has_identifier:
        return bm25_results
    
    logger.info(f"Query contains identifier pattern. Boosting BM25 results by {boost_factor}x")
    
    # Boost scores for results that might contain the identifier
    boosted = []
    for result in bm25_results:
        result_copy = result.copy()
        result_copy['score'] = result_copy.get('score', 0.0) * boost_factor
        boosted.append(result_copy)
    
    return boosted

