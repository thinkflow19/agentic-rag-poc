"""Chunking module for splitting text into overlapping chunks with metadata."""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def chunk_text(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks, preserving identifiers like HD-7961.
    
    Args:
        pages: List of dicts with 'text' and 'page_number' keys
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of chunk dicts with keys: 'chunk_id', 'text', 'page_number', 'metadata'
    """
    logger.info(f"Chunking {len(pages)} pages (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    chunks = []
    chunk_id = 0
    
    id_pattern = re.compile(r'\b[A-Z]{1,5}-\d+\b')
    
    for page in pages:
        text = page.get('text', '')
        page_number = page.get('page_number', 1)
        
        if not text.strip():
            continue
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        text_offset = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_start = text.find(sentence, text_offset)
            if sentence_start == -1:
                sentence_start = text_offset
            sentence_end = sentence_start + len(sentence)
            sentence_has_id = bool(id_pattern.search(text[sentence_start:sentence_end]))
            
            text_offset = sentence_end + 1
            
            if current_length + len(sentence) > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'chunk_id': f"chunk_{chunk_id}",
                    'text': chunk_text,
                    'page_number': page_number,
                    'metadata': {
                        'page_number': page_number,
                        'chunk_index': chunk_id
                    }
                })
                chunk_id += 1
                
                if chunk_overlap > 0:
                    overlap_text = chunk_text[-chunk_overlap:]
                    overlap_start = max(0, overlap_text.find(' '))
                    if overlap_start > 0:
                        overlap_text = overlap_text[overlap_start:].strip()
                    current_chunk = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0
            
            if sentence_has_id and current_length > 0:
                if current_length + len(sentence) > chunk_size * 1.5:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append({
                            'chunk_id': f"chunk_{chunk_id}",
                            'text': chunk_text,
                            'page_number': page_number,
                            'metadata': {
                                'page_number': page_number,
                                'chunk_index': chunk_id
                            }
                        })
                        chunk_id += 1
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence) + 1
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'chunk_id': f"chunk_{chunk_id}",
                'text': chunk_text,
                'page_number': page_number,
                'metadata': {
                    'page_number': page_number,
                    'chunk_index': chunk_id
                }
            })
            chunk_id += 1
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

