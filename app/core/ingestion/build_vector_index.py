"""Vector indexing module using SentenceTransformers and FAISS."""

import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


def build_vector_index(
    chunks: List[Dict[str, Any]],
    output_dir: Path,
    model_name: str = "all-MiniLM-L6-v2"
) -> None:
    """
    Build and persist FAISS vector index from chunks.
    
    Args:
        chunks: List of chunk dicts with 'text' and 'chunk_id' keys
        output_dir: Directory to save artifacts (vector.index and vector_store_meta.json)
        model_name: SentenceTransformer model name for embeddings
    """
    logger.info(f"Building vector index from {len(chunks)} chunks using {model_name}")
    
    model = SentenceTransformer(model_name)
    
    texts = [chunk.get('text', '') for chunk in chunks]
    chunk_ids = [chunk.get('chunk_id', '') for chunk in chunks]
    
    logger.info("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    dim = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    
    # Atomic write: write to temp file first, then rename
    import shutil
    index_path = output_dir / "vector.index"
    temp_index = index_path.with_suffix('.index.tmp')
    try:
        faiss.write_index(index, str(temp_index))
        shutil.move(str(temp_index), str(index_path))
        logger.info(f"FAISS index saved to {index_path}")
    except Exception as e:
        if temp_index.exists():
            temp_index.unlink()
        raise
    
    # Atomic write for metadata
    metadata = {
        'chunk_ids': chunk_ids,
        'dimension': dim,
        'model_name': model_name,
        'num_vectors': len(chunks)
    }
    
    meta_path = output_dir / "vector_store_meta.json"
    temp_meta = meta_path.with_suffix('.json.tmp')
    try:
        with open(temp_meta, 'w') as f:
            json.dump(metadata, f, indent=2)
        shutil.move(str(temp_meta), str(meta_path))
        logger.info(f"Vector metadata saved to {meta_path}")
    except Exception as e:
        if temp_meta.exists():
            temp_meta.unlink()
        raise

