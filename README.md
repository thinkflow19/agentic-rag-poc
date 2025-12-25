# Agentic RAG POC

A production-quality proof-of-concept demonstrating **agentic AI orchestration** combined with **advanced retrieval engineering**. This system showcases deep understanding of hybrid search, OCR-aware ingestion, and proper separation of concerns between ingestion and runtime.

## 🎯 Problem Statement

Traditional RAG systems often suffer from:
- **Limited retrieval strategies**: Relying solely on semantic search misses exact matches (e.g., contract IDs like "HD-7961")
- **Poor handling of scanned documents**: No OCR support for image-based PDFs
- **Tight coupling**: Ingestion and runtime logic mixed together
- **Tool boundary confusion**: Tools generating answers instead of returning evidence
- **Single retrieval method**: Missing the benefits of combining keyword and semantic search

This POC addresses all these issues with a clean, production-ready architecture.

## 🏗️ Architecture

### Core Principles

1. **Separation of Ingestion and Runtime**: Ingestion is a completely separate offline process. Runtime never calls ingestion logic.
2. **Evidence-Only Tools**: Tools return structured evidence. The agent synthesizes the final answer.
3. **True Hybrid Retrieval**: BM25 (keyword) + Vector (semantic) with Reciprocal Rank Fusion (RRF).
4. **OCR-Aware**: Automatically detects and handles scanned PDFs.
5. **Deterministic Identifier Lookup**: BM25 ensures exact matches for identifiers like "HD-7961".

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION (Offline)                       │
├─────────────────────────────────────────────────────────────┤
│  1. Download PDF                                             │
│  2. Extract text (embedded or OCR)                          │
│  3. Chunk with metadata preservation                         │
│  4. Build BM25 index                                         │
│  5. Build FAISS vector index                                 │
│  6. Persist artifacts                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Artifacts   │
                    │  (Local Files)│
                    └───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    RUNTIME (Online)                          │
├─────────────────────────────────────────────────────────────┤
│  Agent (LangChain)                                           │
│    │                                                         │
│    ├──► document_search tool                                 │
│    │      └──► HybridRetriever                               │
│    │            ├──► BM25 Search                              │
│    │            ├──► Vector Search                            │
│    │            └──► RRF Fusion                               │
│    │                                                         │
│    └──► web_search tool                                       │
│          └──► Tavily / DuckDuckGo                           │
│                                                              │
│  Agent synthesizes answer from evidence + adds citations     │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
agentic-rag-poc/
├── app/                        # Flask application (unified)
│   ├── __init__.py             # Application factory
│   ├── config.py               # Configuration classes
│   ├── errors.py               # Error handlers
│   ├── routes/                 # API route blueprints
│   │   ├── __init__.py
│   │   ├── ingestion.py        # Ingestion endpoints
│   │   └── query.py            # Query endpoints
│   ├── services/               # Business logic layer
│   │   ├── __init__.py
│   │   ├── ingestion_service.py
│   │   └── query_service.py
│   └── core/                   # Core domain logic (unified)
│       ├── agent/               # Agent orchestration
│       │   ├── __init__.py
│       │   └── agent.py        # Agent creation
│       ├── ingestion/          # Document processing
│       │   ├── __init__.py
│       │   ├── ingest.py       # Core ingestion functions
│       │   ├── ocr.py          # OCR functionality
│       │   ├── chunking.py     # Text chunking
│       │   ├── build_bm25.py   # BM25 indexing
│       │   ├── build_vector_index.py
│       │   └── artifacts/      # Generated artifacts
│       ├── retrieval/          # Hybrid retrieval
│       │   ├── __init__.py
│       │   ├── hybrid.py       # Hybrid retriever
│       │   └── fusion.py       # RRF fusion
│       ├── tools/              # LangChain tools
│       │   ├── __init__.py
│       │   ├── document_search.py
│       │   └── web_search.py
│       └── prompts/            # Agent prompts
│           └── agent_prompt.txt
│
├── eval/                       # Evaluation scripts
│   └── test_queries.py
│
├── app_streamlit.py            # Streamlit chat UI
├── run.py                      # Flask application runner
├── README.md
├── STRUCTURE.md                # Detailed structure docs
├── requirements.txt
└── .env.example                # Environment template
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **OpenAI API key** (required)
- **Tavily API key** (optional, for better web search)
- **Poppler** (for OCR): 
  - macOS: `brew install poppler`
  - Linux: `apt-get install poppler-utils`
  - Windows: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)

### Installation

1. **Clone and navigate to the repository**:
   ```bash
   cd agentic-rag-poc
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   # Optionally add TAVILY_API_KEY for better web search
   ```

## 📖 Usage Guide

### Step 1: Ingest Documents

Ingest a PDF document using the Flask API:

**Option A: Using Flask API (Recommended)**

```bash
# Start Flask server
python run.py
# Server runs on http://localhost:5001

# In another terminal, ingest a document
curl -X POST http://localhost:5001/api/ingestion/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.brewnquefestival.com/wp-content/uploads/2019/11/Sample-Certificate-of-Insurance-and-Endorsement-Forms.pdf"
  }'
```

**Option B: Using Python directly**

```python
from pathlib import Path
from app.core.ingestion.ingest import ingest_from_url

# Ingest document
result = ingest_from_url(
    url="https://www.brewnquefestival.com/wp-content/uploads/2019/11/Sample-Certificate-of-Insurance-and-Endorsement-Forms.pdf",
    output_dir=Path("app/core/ingestion/artifacts")
)
print(result)
```

**What happens during ingestion**:
1. PDF is downloaded
2. Text is extracted (embedded or OCR if scanned)
3. Text is chunked with overlap (preserving page numbers)
4. BM25 index is built and saved
5. FAISS vector index is built and saved
6. All artifacts are persisted to `app/core/ingestion/artifacts/`

**Output**: You'll see logs indicating:
- Which extraction method was used (embedded vs OCR)
- Number of pages processed
- Number of chunks created
- Index build progress

### Step 2: Query the System

#### Option A: Streamlit Chat UI (Easiest)

```bash
streamlit run app_streamlit.py
```

This opens a web interface in your browser where you can:
- Ask questions about the ingested documents
- See the agent's reasoning process
- View tool usage and citations

#### Option B: Flask API

```bash
# Ensure Flask server is running
python run.py

# In another terminal, query the system
curl -X POST http://localhost:5001/api/query/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find contract id HD-7961",
    "chat_history": []
  }'
```

**Response**:
```json
{
  "response": "The contract ID HD-7961 appears on page 2...",
  "status": "success"
}
```

#### Option C: Python Programmatically

```python
from app.core.agent.agent import create_agent_executor

# Create agent
agent = create_agent_executor()

# Query
result = agent({
    "input": "Find contract id HD-7961",
    "chat_history": []
})

print(result["output"])
```

### Step 3: Evaluate

Run the test suite:

```bash
python eval/test_queries.py
```

This runs predefined test queries and shows:
- Tool usage
- Evidence retrieved
- Final synthesized answers

## 🔍 Example Queries

### Identifier Lookup (BM25 Strong)

```
Query: "Find contract id HD-7961"
Expected: Agent uses document_search, finds exact match, cites page number
```

### Semantic Search (Vector Strong)

```
Query: "Summarize the certificate of insurance form"
Expected: Agent uses document_search, retrieves relevant chunks, synthesizes summary
```

### Hybrid Query

```
Query: "What does HD-7961 say about coverage limits?"
Expected: BM25 finds HD-7961, vector finds coverage context, RRF fuses results
```

### Web Fallback

```
Query: "What is BM25?"
Expected: document_search returns low confidence, agent calls web_search
```

## 🛠️ API Reference

### Flask API Endpoints

#### Ingestion Endpoints

**Health Check**
```bash
GET /api/ingestion/health
```

**Ingest Document**
```bash
POST /api/ingestion/ingest
Content-Type: application/json

{
  "url": "https://example.com/document.pdf",
  "output_dir": "app/core/ingestion/artifacts"  # Optional, defaults to config
}
```

#### Query Endpoints

**Health Check**
```bash
GET /api/query/health
```

**Process Query**
```bash
POST /api/query/query
Content-Type: application/json

{
  "query": "Your question here",
  "chat_history": []  # Optional, for conversation context
}
```

## 🧠 How It Works

### Ingestion Flow

1. **Download**: PDF fetched via `requests`
2. **Extract**: 
   - Try `pypdf` for embedded text
   - If detected as scanned → `unstructured` with OCR
3. **Chunk**: 
   - Size: ~1000 chars, overlap: 150 chars
   - Preserves page numbers in metadata
   - Protects identifiers from being split
4. **Index BM25**:
   - Tokenize chunks (preserving identifiers)
   - Build `BM25Okapi` index
   - Persist with chunk ID mapping
5. **Index Vector**:
   - Embed chunks with SentenceTransformers (`all-MiniLM-L6-v2`)
   - Build FAISS `IndexFlatIP` (cosine similarity)
   - Normalize embeddings
   - Persist index + metadata

### Runtime Flow

1. **Agent receives query**
2. **Routing decision** (from prompt):
   - Identifier pattern or document intent → `document_search` first
   - General knowledge → `web_search`
3. **Document Search**:
   - Load artifacts (lazy singleton)
   - Run BM25 search (top K)
   - Run vector search (top K)
   - Boost BM25 if identifier detected
   - Fuse with RRF (k=60)
   - Return evidence with confidence
4. **Web Search** (if needed):
   - Use Tavily if API key available
   - Otherwise DuckDuckGo
   - Return snippets + URLs
5. **Agent synthesis**:
   - Read evidence from tools
   - Synthesize coherent answer
   - Add citations (page numbers for docs, URLs for web)
   - State which tools were used

### Hybrid Retrieval Details

**Reciprocal Rank Fusion (RRF)**:

For each result in final ranking:
```
rrf_score = (1 / (60 + bm25_rank)) + (1 / (60 + vector_rank))
```

Results are sorted by `rrf_score` descending.

**Identifier Boosting**:

If query matches pattern `[A-Z]{1,5}-\d+`:
- BM25 scores multiplied by 2.0
- Ensures identifier matches rank higher

## 📦 Technical Details

### Dependencies

- **LangChain 1.1.x**: Agent framework
- **FAISS**: Vector similarity search
- **rank-bm25**: BM25 implementation
- **SentenceTransformers**: Embeddings
- **Unstructured**: OCR for PDFs
- **pypdf**: Embedded text extraction
- **Flask**: API framework
- **Streamlit**: Chat UI

### Index Formats

- **BM25**: Pickled `BM25Okapi` object + JSON mapping
- **Vector**: FAISS binary index + JSON metadata
- **Chunks**: JSON array with text, page numbers, metadata

### Tool Interface

Both tools follow LangChain's `@tool` decorator pattern and return structured dictionaries. The agent receives evidence, not answers.

## 🧪 Testing

The evaluation script (`eval/test_queries.py`) includes:

1. **Identifier lookup**: "Find contract id HD-7961"
2. **Location query**: "Where does HD-7961 appear?"
3. **Summarization**: "Summarize the certificate of insurance form"
4. **Web fallback**: "What is BM25?"

Each test prints:
- Tool usage (from agent logs)
- Evidence retrieved
- Final synthesized answer

## ⚠️ Troubleshooting

### "OPENAI_API_KEY not found"
- Ensure `.env` file exists with `OPENAI_API_KEY=your_key`
- Or set environment variable: `export OPENAI_API_KEY=your_key`

### "Artifacts not found"
- Run ingestion first via Flask API or Python
- Check that `app/core/ingestion/artifacts/` contains required files:
  - `chunks.json`
  - `bm25_index.pkl`
  - `vector.index`
  - `vector_store_meta.json`

### OCR fails
- Ensure `unstructured[pdf]` is installed
- Check PDF is not corrupted
- Verify poppler is installed: `brew install poppler` (macOS)
- Verify sufficient disk space for temporary files

### BM25 not finding identifiers
- Check identifier format matches pattern `[A-Z]{1,5}-\d+`
- Verify identifier wasn't split during chunking (check logs)
- Ensure BM25 index was built successfully

### Streamlit UI not loading
- Ensure streamlit is installed: `pip install streamlit`
- Check that artifacts exist (run ingestion first)
- Verify OPENAI_API_KEY is set in `.env`

### Flask server errors
- Check that port 5001 is not in use
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check logs for specific error messages

## 🔮 Limitations & Future Work

### Current Limitations

- Single document ingestion (can be extended)
- Local indexes only (no managed vector DB)
- No incremental updates (full re-ingestion required)
- Fixed chunk size (could be adaptive)

### Potential Enhancements

- Multi-document support with namespace separation
- Query expansion and rewriting
- Re-ranking with cross-encoder models
- Streaming ingestion for large document sets
- Metadata filtering (date ranges, document types)
- Hybrid search tuning (learned fusion weights)

## 📄 License

This is a proof-of-concept for evaluation purposes.
