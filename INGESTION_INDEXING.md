# Ingestion & Indexing Pipeline

## Overview

Our ingestion pipeline is a **completely offline process** that processes PDFs and builds searchable indexes. It's separate from the runtime system, which only loads pre-built indexes.

## Complete Ingestion Flow

```
PDF URL
  ↓
[1] Download PDF
  ↓
[2] Extract Text (Embedded or OCR)
  ↓
[3] Chunk Text (with overlap & identifier protection)
  ↓
[4] Save Chunks Metadata (chunks.json)
  ↓
[5] Build BM25 Index (keyword search)
  ↓
[6] Build Vector Index (semantic search)
  ↓
Artifacts Directory (ready for runtime)
```

---

## Step-by-Step Breakdown

### Step 1: Download PDF

**File**: `app/core/ingestion/ingest.py` → `download_pdf()`

```python
def download_pdf(url: str, output_path: Path) -> None:
    headers = {'User-Agent': 'Mozilla/5.0...'}
    response = requests.get(url, headers=headers, timeout=30)
    # Save to temporary file
```

**What happens**:
- Downloads PDF from URL
- Saves to temporary file
- Uses proper User-Agent to avoid 406 errors

---

### Step 2: Extract Text

**File**: `app/core/ingestion/ingest.py` → `extract_text_embedded()` + `extract_text_with_ocr()`

**Process**:

1. **Try Embedded Text First** (fast)
   ```python
   reader = PdfReader(pdf_path)
   for page in reader.pages:
       text = page.extract_text()  # pypdf extraction
   ```

2. **Detect if Scanned**
   ```python
   is_scanned = detect_scanned_pdf(pages)
   # Checks: average chars per page < 50 OR >50% pages empty
   ```

3. **If Scanned → Use OCR**
   ```python
   if is_scanned:
       pages = extract_text_with_ocr(pdf_path)  # unstructured with OCR
   ```

**Output**: List of pages with `{'text': str, 'page_number': int}`

---

### Step 3: Chunk Text

**File**: `app/core/ingestion/chunking.py` → `chunk_text()`

**Key Features**:

1. **Sentence-Based Chunking**
   ```python
   sentences = re.split(r'(?<=[.!?])\s+', text)
   # Splits on sentence boundaries
   ```

2. **Identifier Protection**
   ```python
   id_pattern = re.compile(r'\b[A-Z]{1,5}-\d+\b')
   # Detects patterns like "HD-7961"
   # Ensures identifiers are NOT split across chunks
   ```

3. **Overlapping Chunks**
   ```python
   chunk_size = 1000      # Target: 1000 characters
   chunk_overlap = 150    # Overlap: 150 characters
   # Ensures context is preserved at chunk boundaries
   ```

4. **Metadata Preservation**
   ```python
   chunk = {
       'chunk_id': 'chunk_0',
       'text': '...',
       'page_number': 1,
       'metadata': {
           'page_number': 1,
           'chunk_index': 0
       }
   }
   ```

**Output**: List of chunks with metadata

---

### Step 4: Save Chunks Metadata

**File**: `app/core/ingestion/ingest.py` → `process_document()`

```python
chunks_path = output_dir / "chunks.json"
chunks_data = [
    {
        'chunk_id': c['chunk_id'],
        'text': c['text'],
        'page_number': c['page_number'],
        'metadata': c['metadata']
    }
    for c in chunks
]
json.dump(chunks_data, f, indent=2)
```

**Output**: `chunks.json` - All chunk text and metadata for lookup

---

### Step 5: Build BM25 Index

**File**: `app/core/ingestion/build_bm25.py` → `build_bm25_index()`

**Process**:

1. **Tokenize Each Chunk**
   ```python
   def tokenize(text: str) -> List[str]:
       # 1. Protect identifiers (HD-7961 → __ID_0__)
       id_pattern = re.compile(r'\b([A-Z]{1,5}-\d+)\b')
       text_protected = id_pattern.sub(replace_id, text)
       
       # 2. Tokenize words
       tokens = re.findall(r'\b\w+\b', text_protected.lower())
       
       # 3. Restore identifiers
       tokens = [protected.get(t, t) for t in tokens]
       return tokens
   ```

2. **Build BM25 Index**
   ```python
   tokenized_corpus = [tokenize(chunk['text']) for chunk in chunks]
   bm25 = BM25Okapi(tokenized_corpus)  # rank_bm25 library
   ```

3. **Persist Index**
   ```python
   pickle.dump({
       'bm25': bm25,                    # The BM25Okapi object
       'tokenized_corpus': tokenized_corpus,  # Original tokens
       'chunk_ids': chunk_ids           # Mapping: index → chunk_id
   }, f)
   ```

4. **Save Mapping**
   ```python
   mapping = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
   json.dump(mapping, f)  # bm25_mapping.json
   ```

**Output Files**:
- `bm25_index.pkl` - Pickled BM25Okapi object
- `bm25_mapping.json` - Chunk ID to index mapping

**What BM25 Index Contains**:
- Term frequencies for all tokens
- Document frequencies
- IDF (Inverse Document Frequency) weights
- Ready to score queries instantly

---

### Step 6: Build Vector Index

**File**: `app/core/ingestion/build_vector_index.py` → `build_vector_index()`

**Process**:

1. **Load Embedding Model**
   ```python
   model = SentenceTransformer("all-MiniLM-L6-v2")
   # 384-dimensional embeddings
   ```

2. **Generate Embeddings**
   ```python
   texts = [chunk['text'] for chunk in chunks]
   embeddings = model.encode(
       texts,
       show_progress_bar=True,
       normalize_embeddings=True  # L2 normalized for cosine similarity
   )
   # Shape: (num_chunks, 384)
   ```

3. **Build FAISS Index**
   ```python
   index = faiss.IndexFlatIP(dim)  # Inner Product (cosine with normalized)
   index.add(embeddings.astype('float32'))
   ```

4. **Persist Index**
   ```python
   faiss.write_index(index, "vector.index")
   ```

5. **Save Metadata**
   ```python
   metadata = {
       'chunk_ids': chunk_ids,      # Mapping: index → chunk_id
       'dimension': 384,
       'model_name': 'all-MiniLM-L6-v2',
       'num_vectors': len(chunks)
   }
   json.dump(metadata, f)  # vector_store_meta.json
   ```

**Output Files**:
- `vector.index` - FAISS binary index
- `vector_store_meta.json` - Metadata and chunk ID mapping

**What Vector Index Contains**:
- 384-dimensional embeddings for each chunk
- FAISS index structure for fast similarity search
- Ready for semantic search queries

---

## Artifacts Directory Structure

After ingestion, `app/core/ingestion/artifacts/` contains:

```
artifacts/
├── chunks.json              # All chunk text + metadata
├── bm25_index.pkl           # BM25 search index
├── bm25_mapping.json        # BM25 chunk ID mapping
├── vector.index             # FAISS vector index
└── vector_store_meta.json  # Vector index metadata
```

---

## Index Details

### BM25 Index (`bm25_index.pkl`)

**What it is**:
- Pickled `BM25Okapi` object from `rank_bm25` library
- Contains pre-computed term frequencies and IDF weights
- Can score queries instantly without rebuilding

**How it works**:
```python
# During ingestion
bm25 = BM25Okapi(tokenized_corpus)  # Build once

# During runtime (query time)
query_tokens = tokenize("Find HD-7961")
scores = bm25.get_scores(query_tokens)  # Fast scoring
top_indices = np.argsort(scores)[::-1][:top_k]
```

**Key Features**:
- ✅ Identifier preservation (HD-7961 stays as one token)
- ✅ Fast query scoring (pre-computed statistics)
- ✅ Persisted for fast loading

---

### Vector Index (`vector.index`)

**What it is**:
- FAISS `IndexFlatIP` (Inner Product index)
- Contains 384-dimensional embeddings for each chunk
- Optimized for cosine similarity search

**How it works**:
```python
# During ingestion
embeddings = model.encode(chunks)  # Generate once
index.add(embeddings)  # Build FAISS index

# During runtime (query time)
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding, top_k)  # Fast search
```

**Key Features**:
- ✅ Semantic understanding (finds similar concepts)
- ✅ Fast similarity search (FAISS optimized)
- ✅ Normalized embeddings (cosine similarity)

---

## Why This Architecture?

### 1. **Separation of Concerns**
- **Ingestion**: Heavy computation (OCR, embedding generation)
- **Runtime**: Fast queries (just load and search)

### 2. **Performance**
- Indexes built once, loaded many times
- No need to rebuild on every query
- Fast startup (load pickles vs. rebuild from scratch)

### 3. **Scalability**
- Ingestion can run on powerful machines
- Runtime can run on lighter servers
- Indexes can be shared across instances

### 4. **Flexibility**
- Can rebuild indexes without affecting runtime
- Can experiment with different chunking/indexing strategies
- Can version indexes independently

---

## Comparison: Our Approach vs. LangChain

### Our Approach (Current)

**Ingestion**:
```python
# Build indexes offline
build_bm25_index(chunks, output_dir)      # Custom, identifier-aware
build_vector_index(chunks, output_dir)     # FAISS directly
```

**Runtime**:
```python
# Load pre-built indexes
bm25 = pickle.load('bm25_index.pkl')      # Instant load
vector_index = faiss.read_index('vector.index')  # Instant load
```

**Advantages**:
- ✅ Fast startup (load pickles)
- ✅ Custom identifier handling
- ✅ Full control over indexing

---

### LangChain Approach (Alternative)

**Ingestion**:
```python
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

# Would need to rebuild from documents each time
bm25_retriever = BM25Retriever.from_documents(documents)
vector_store = FAISS.from_documents(documents, embeddings)
```

**Runtime**:
```python
# Would need to load documents and rebuild
# OR implement custom persistence
```

**Challenges**:
- ❌ No built-in persistence (need custom solution)
- ❌ Identifier handling requires custom preprocessing
- ❌ Less control over indexing process

---

## Index Loading (Runtime)

When the system runs a query, it loads indexes:

**File**: `app/core/retrieval/hybrid.py` → `HybridRetriever.__init__()`

```python
def _load_artifacts(self):
    # Load chunks
    with open('chunks.json', 'r') as f:
        self.chunks = json.load(f)
    
    # Load BM25 index
    with open('bm25_index.pkl', 'rb') as f:
        bm25_data = pickle.load(f)
        self.bm25 = bm25_data['bm25']  # Ready to use!
    
    # Load vector index
    self.vector_index = faiss.read_index('vector.index')  # Ready to use!
    
    # Load embedding model
    self.embedding_model = SentenceTransformer(model_name)
```

**Key Point**: Indexes are **loaded once** and reused for all queries (singleton pattern).

---

## Summary

### Ingestion Pipeline
1. ✅ Download PDF
2. ✅ Extract text (embedded or OCR)
3. ✅ Chunk with identifier protection
4. ✅ Save chunks metadata
5. ✅ Build BM25 index (keyword search)
6. ✅ Build vector index (semantic search)

### Index Types
- **BM25**: Keyword-based, identifier-aware, fast scoring
- **Vector**: Semantic-based, concept matching, FAISS-optimized

### Architecture Benefits
- ✅ Fast startup (load pre-built indexes)
- ✅ Separation of ingestion and runtime
- ✅ Full control over indexing process
- ✅ Custom identifier handling

This architecture ensures reliable identifier lookup (HD-7961) while maintaining fast semantic search capabilities.

