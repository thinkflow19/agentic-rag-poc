"""BM25 indexing module for full-text search."""

import logging
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25, keeping identifiers like HD-7961 as single tokens.
    
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


def build_bm25_index(
    chunks: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Build and persist BM25 index from chunks.
    
    Args:
        chunks: List of chunk dicts with 'text' and 'chunk_id' keys
        output_dir: Directory to save artifacts (bm25_index.pkl and bm25_mapping.json)
    """
    logger.info(f"Building BM25 index from {len(chunks)} chunks")
    
    tokenized_corpus = []
    chunk_ids = []
    
    for chunk in chunks:
        text = chunk.get('text', '')
        chunk_id = chunk.get('chunk_id', '')
        
        tokens = tokenize(text)
        tokenized_corpus.append(tokens)
        chunk_ids.append(chunk_id)
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Atomic write: write to temp file first, then rename
    import shutil
    index_path = output_dir / "bm25_index.pkl"
    temp_index = index_path.with_suffix('.pkl.tmp')
    try:
        with open(temp_index, 'wb') as f:
            pickle.dump({
                'bm25': bm25,
                'tokenized_corpus': tokenized_corpus,
                'chunk_ids': chunk_ids
            }, f)
        shutil.move(str(temp_index), str(index_path))
        logger.info(f"BM25 index saved to {index_path}")
    except Exception as e:
        if temp_index.exists():
            temp_index.unlink()
        raise
    
    # Atomic write for mapping
    mapping = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
    mapping_path = output_dir / "bm25_mapping.json"
    temp_mapping = mapping_path.with_suffix('.json.tmp')
    try:
        with open(temp_mapping, 'w') as f:
            json.dump(mapping, f, indent=2)
        shutil.move(str(temp_mapping), str(mapping_path))
        logger.info(f"BM25 mapping saved to {mapping_path}")
    except Exception as e:
        if temp_mapping.exists():
            temp_mapping.unlink()
        raise

