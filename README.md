# Simple RAG System

A local, open-source Retrieval-Augmented Generation (RAG) system with hybrid search capabilities. Built with LangChain, ChromaDB, and Ollama.

## Table of Contents

- [Quick Start](#quick-start)
- [Step-by-Step Overview](#step-by-step-overview)
  - [Step 1: Setup & Data](#step-1-setup--data)
  - [Step 2: Embeddings & Storage](#step-2-embeddings--storage)
  - [Step 3: Hybrid Retriever](#step-3-hybrid-retriever)
  - [Step 4: LLM Generation](#step-4-llm-generation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/gustavo30rocha/simple-rag.git
cd simple-rag
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama chromadb sentence-transformers pypdf "unstructured[md]" rank-bm25 wikipedia

# Pull Ollama model 
ollama pull llama3:8b

# Prepare documents and create database
python scrape_wikipedia.py # Optional: download test articles
python create_database.py # Add new documents incrementally
python create_database.py --clear # Clear the database

# Query with hybrid search
python query_data.py "Your question" --hybrid
```

## Step-by-Step Overview

- **Step 1: Setup & Data** — Load PDF/Markdown files
- **Step 2: Embeddings & Storage** — Generate embeddings; store in ChromaDB; build BM25 index
- **Step 3: Hybrid Retriever** — Combine vector (dense) with BM25 (sparse); return top K
- **Step 4: LLM Generation** — Generate answers grounded in retrieved context

### Step 1: Setup & Data

**Design Rationale**: Used LangChain's document loaders for ingestion to leverage proven, well-maintained parsers while keeping the pipeline straightforward and extensible.

**Document Loading**:
- **PDF**: `PyPDFDirectoryLoader` recursively loads PDFs from `data/` directory
- **Markdown**: `DirectoryLoader` with `glob="*.md"` pattern, recursive search
- **Metadata**: Preserves `source` (file path) and `page` (page number for PDFs)

**Text Chunking**:
- **Splitter**: `RecursiveCharacterTextSplitter`
- **Parameters**: `chunk_size=1000` characters, `chunk_overlap=150` characters

**Chunk ID Generation**:
- **Format**: `{source}:{page}:{chunk_index}`
- **Example**: `data/document.pdf:0:0` (first chunk of first page)
- **Purpose**: Enables incremental updates (only adds new chunks by ID)

**Output**: List of `Document` objects with content and metadata (source, page, id)

### Step 2: Embeddings & Storage

**Embedding Model**: `BAAI/bge-small-en-v1.5`
- **Dimensions**: 384
- **Why This Model**: Trained specifically for retrieval tasks; better semantic understanding than general-purpose models; efficient size
- **Alternative Models**: `sentence-transformers/all-mpnet-base-v2` (better quality, 768 dims), `BAAI/bge-base-en-v1.5` (larger version)

**Vector Storage (ChromaDB)**:
- **Distance Metric**: Cosine distance (configured via `collection_metadata={"hnsw:space": "cosine"}`)
- **Persistence**: Stores in `chroma/` directory
- **Note**: ChromaDB returns cosine **distance** (0-2, lower = more similar), not similarity

**BM25 Index Building**:
- **Algorithm**: BM25Okapi from `rank-bm25`
- **Tokenization**: Simple regex `\b\w+\b` on lowercase text (used for testing purposes)
- **Storage**: Saves index to `chroma/bm25_index.pkl` and document references to `chroma/bm25_docs.pkl`

**Incremental Updates**:
- Checks existing chunk IDs in database
- Only adds chunks with new IDs
- Rebuilds BM25 index with all chunks (old + new) to keep it synchronized

**Artifacts**:
- `chroma/` directory with ChromaDB database
- `chroma/bm25_index.pkl` (BM25 index)
- `chroma/bm25_docs.pkl` (document references)

### Step 3: Hybrid Retriever

**Design Rationale**: After testing various retrieval approaches, a hybrid method that merges semantic understanding with keyword matching was selected. This dual-strategy approach addresses the limitations of single-method retrieval systems.

**Why Not Dense-Only?**
While vector embeddings excel at understanding semantic relationships and context, they struggle with:
- Exact keyword matching (e.g., searching for "Python" may return snake-related content)
- Technical terms, symbols, and rare entity names
- Queries requiring precise lexical matches

A pure dense retriever lacks a direct path for keyword-based recall, which limits precision on technical queries without additional re-ranking layers.

**Solution: Hybrid Approach**
Combine two complementary retrieval methods:
- **Dense (Vector) Search**: Uses `BAAI/bge-small-en-v1.5` embeddings to find semantically similar content
- **Sparse (BM25) Search**: Performs keyword-based matching on raw text

The final relevance score is a weighted combination: `final_score = α × dense_score + (1−α) × sparse_score`, where `α` is controlled via `--hybrid-weight`.

**Benefits**:
- Captures both conceptual similarity and exact term matches
- More robust across diverse query types (natural language vs. technical terms)
- Tunable balance between semantic and keyword matching

**Implementation**:

1. **BM25 Search**:
   - Tokenizes query: `tokenize(query_text)` → lowercase words
   - Scores all documents: `bm25.get_scores(query_tokens)` → raw BM25 scores
   - Normalizes scores: min-max normalization to 0-1 range

2. **Vector Search**:
   - Embeds query using same embedding model
   - Retrieves top K×3 documents via cosine distance (retrieves 3× more than requested to ensure good BM25 matches aren't missed during merging)
   - Converts distance to similarity: `similarity = 1 - distance`
   - Normalizes scores: min-max normalization to 0-1 range

3. **Score Fusion**:
   - Creates score maps: `{document_content: score}` for fast lookup
   - Combines: `combined_score = (hybrid_weight × vector_score) + ((1 - hybrid_weight) × bm25_score)`
   - **Weight Interpretation**:
     - `0.0` = Pure BM25 (keyword-only)
     - `1.0` = Pure vector (semantic-only)
     - `0.5` = Balanced

4. **Result Merging**:
   - Adds all vector results with combined scores
   - Adds top BM25 results not in vector results
   - Sorts by combined score, returns top K

**Score Normalization**:
- Both vector and BM25 scores use min-max normalization separately
- Formula: `normalized = (score - min) / (max - min)`
- Handles edge cases: empty lists, identical scores (returns zeros)
- Ensures fair combination of scores from different scales

**Chunking Defaults**:
| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `chunk_size` | 1000 characters | Balance between context and granularity |
| `chunk_overlap` | 150 characters | Prevents information loss at boundaries |

**Fallback Behavior**:
- If BM25 index not found, falls back to vector-only search
- Prints warning message to user

### Step 4: LLM Generation

**Goal**: Generate final answers grounded in retrieved context.

**Model**: Ollama (default: `llama3:8b`, configurable via `--model`)

**Prompt Template**:
```text
You are a helpful assistant that answers questions based ONLY on the provided context documents. Your answers must be grounded in the context provided below.

## Instructions:
1. Answer the question using ONLY information from the context provided
2. If the context contains the answer, provide a clear and accurate response
3. If multiple pieces of context are relevant, synthesize them into a coherent answer
4. If the context contradicts itself, acknowledge the contradiction and present both perspectives
5. If the context does NOT contain enough information to answer the question, explicitly state: "I don't have enough information in the provided documents to answer this question."
6. Do NOT use any knowledge outside of the provided context
7. Be specific and cite relevant details from the context when possible

## Context:
{context}

## Question:
{question}

## Answer:
```

**Process**:
1. Formats top K documents as context
2. Creates prompt with context and question
3. Invokes Ollama LLM with `temperature=0` (deterministic)
4. Returns answer with source citations

**Source Attribution**:
- Extracts sources from document metadata
- Removes duplicates (same document may appear in multiple chunks)

## Configuration

### Embedding Model

Change in both `create_database.py` and `query_data.py`:
```python
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
```

### Chunking

In `create_database.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=150,    # Overlap between chunks
)
```

### ChromaDB

Uses cosine distance (configured in `create_database.py`):
```python
collection_metadata={"hnsw:space": "cosine"}
```

### Command-Line Options

**`query_data.py`:**
- `query_text` (required): Your query
- `--model`: Ollama model (default: `llama3:8b`)
- `--k`: Number of documents (default: `5`)
- `--hybrid`: Enable hybrid search
- `--hybrid-weight`: Weight 0.0-1.0 (default: `0.7`)

**Examples:**
```bash
# Vector-only
python query_data.py "What is Python?"

# Hybrid (balanced)
python query_data.py "Python programming" --hybrid

# Hybrid (more semantic)
python query_data.py "How does ML work?" --hybrid --hybrid-weight 0.7

# More documents
python query_data.py "Explain databases" --k 10
```

## Testing

The system includes comprehensive unit tests with LLM-based evaluation to validate answer quality across different retrieval modes.

### Running Tests

```bash
# Run all tests with default evaluation model (mistral:7b)
python test_rag.py

# Use a different model for evaluation
python test_rag.py --eval-model llama3:8b
```

### Test Coverage

**Test Types**:
- **Positive Tests**: Verify the system correctly answers questions across various domains (anime, tech, science, history, etc.)
- **Negative Tests**: Verify the system correctly rejects wrong information (contradictions, incorrect facts)

**Retrieval Modes Tested**:
- Vector-only search
- BM25-only search (`hybrid_weight=0.0`)
- Hybrid search (`hybrid_weight=0.7`)

**Domains Covered**:
- Anime & Manga (Naruto, One Piece, etc.)
- Technical (Python, Machine Learning, SQL, REST API, etc.)
- Science (Quantum Computing, Photosynthesis, DNA, Evolution, etc.)
- History & Geography (World War II, Ancient Egypt, Mount Everest, etc.)
- Technology (Blockchain, Linux, Neural Networks, etc.)

### Evaluation Method

Tests use **LLM-based evaluation** where a separate model (default: `mistral:7b`) evaluates whether the actual response contains the expected information. The evaluator:
- Checks for contradictions (e.g., "centralized" vs "decentralized")
- Validates key facts, dates, and characteristics
- Handles different phrasings of the same information

**Test Structure**:
- Positive tests: `assert query_and_validate(...)` - should return `True`
- Negative tests: `assert not query_and_validate(...)` - should return `False` (system correctly rejects wrong info)

**Note**: LLM-based evaluation is not perfect and may occasionally misinterpret responses. Test results should be interpreted with this limitation in mind.

## Troubleshooting

**BM25 index not found**: Run `python create_database.py` to build it. System falls back to vector-only if missing.

**Low retrieval quality**: Try `--hybrid --hybrid-weight 0.3` (more keyword-focused) or increase `--k`.

**Ollama errors**: Ensure Ollama is running (`ollama serve`) and model is pulled (`ollama list`).

**Documents not found**: Verify files are in `data/` with `.pdf` or `.md` extensions.

## Tech Stack

- **Embeddings**: `BAAI/bge-small-en-v1.5`
- **Vector DB**: ChromaDB
- **Keyword Search**: BM25 algorithm (`rank-bm25`)
- **LLM**: Local Ollama model (any model)
- **Framework**: LangChain

## Possible Future Improvements

- Reranking with cross-encoders
- Query expansion techniques
- Better testing methods
- Metadata filtering