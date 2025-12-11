# Simple RAG System

A simple, open-source Retrieval-Augmented Generation (RAG) system that allows you to query your documents using natural language. Built with LangChain, ChromaDB, and Ollama for a fully local solution.

## Features

- 📄 **Multi-format Support**: Loads PDF and Markdown files from your data directory
- 🔍 **Hybrid Search**: Combines semantic (vector) and keyword/lexical (BM25) search for better retrieval
- 🤖 **Local LLM**: Powered by Ollama for completely offline operation
- 💾 **Persistent Storage**: ChromaDB vector database for efficient similarity search
- 🔄 **Incremental Updates**: Automatically detects and adds new documents without rebuilding the entire database

## Tech Stack

- **Embeddings**: `BAAI/bge-small-en-v1.5` (HuggingFace) - High-quality, open-source embedding model
- **Vector Database**: ChromaDB with cosine distance indexing
- **Keyword Search**: BM25 algorithm for exact term matching
- **LLM**: Ollama (supports any Ollama model)
- **Framework**: LangChain

## Prerequisites

- Python 3.8+
- An Ollama model pulled (e.g., `ollama pull llama3:8b`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gustavo30rocha/simple-rag.git
cd simple-rag
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama chromadb sentence-transformers pypdf "unstructured[md]" rank-bm25 wikipedia
```

4. Pull an Ollama model (if not already done):
```bash
ollama pull llama3:8b
```

## Project Structure

```
simple-rag/
├── data/                  # Place your PDF and Markdown files here
│   ├── *.pdf
│   ├── *.md
│   └── subdirectories/   # Supports recursive loading
├── chroma/               # ChromaDB database (auto-generated)
│   ├── bm25_index.pkl   # Pre-built BM25 index (auto-generated)
│   └── bm25_docs.pkl    # Document references for BM25 (auto-generated)
├── create_database.py    # Script to build/update the vector database
├── query_data.py         # Script to query your documents
├── scrape_wikipedia.py   # Script to download Wikipedia articles
└── README.md
```

## Usage

### 1. Prepare Your Documents

You can either:
- **Option A**: Place your PDF and Markdown files in the `data/` directory
- **Option B**: Use the Wikipedia scraper to download articles for testing

#### Option A: Manual Document Placement

Place your PDF and Markdown files in the `data/` directory. The system will recursively search all subdirectories.

```
data/
├── document1.pdf
├── document2.md
└── subfolder/
    └── document3.pdf
```

#### Option B: Download Wikipedia Articles

Use the included scraper to quickly build a test dataset:

```bash
python scrape_wikipedia.py
```

This will download Wikipedia articles specified in the `TOPICS` list and save them as Markdown files in the `data/` directory. You can customize the topics in `scrape_wikipedia.py`.

### 2. Create the Database

Run the database creation script to process and index your documents:

```bash
python create_database.py
```

This will:
- Load all PDF and Markdown files from the `data/` directory
- Split documents into chunks (1000 characters with 150 character overlap)
- Generate embeddings using BAAI/bge-small-en-v1.5
- Build a BM25 keyword search index
- Store everything in ChromaDB and save BM25 index to disk
- Only add new documents if the database already exists (incremental updates)

### 3. Query Your Documents

Ask questions about your documents:

```bash
python query_data.py "Your question here"
```

**Examples:**
```bash
# Basic query (vector search only)
python query_data.py "What type of evidence can we find in operating systems?"

# Hybrid search (combines vector + keyword search)
python query_data.py "Python programming" --hybrid

# Hybrid search with custom weight (70% vector, 30% keyword)
python query_data.py "Python machine learning" --hybrid --hybrid-weight 0.7

# Use a different Ollama model
python query_data.py "Explain file carving techniques" --model llama3.1

# Retrieve more documents
python query_data.py "How does memory forensics work?" --k 10
```

### Command-Line Options

**`query_data.py` options:**
- `query_text` (required): Your question
- `--model`: Ollama model to use (default: `llama3:8b`)
- `--k`: Number of documents to retrieve (default: `5`)
- `--hybrid`: Enable hybrid search (combines vector + keyword search)
- `--hybrid-weight`: Weight for hybrid search (0.0 = only keyword, 1.0 = only vector, default: `0.5`)

## How It Works

### Vector Search (Default)

1. **Document Processing**: Documents are split into manageable chunks (1000 chars with 150 char overlap)
2. **Embedding Generation**: Each chunk is converted to a vector using BAAI/bge-small-en-v1.5
3. **Vector Storage**: Embeddings are stored in ChromaDB
4. **Query Processing**: 
   - Your question is embedded using the same model
   - ChromaDB finds the most similar document chunks using cosine distance
   - Cosine distance (0-2 range) is converted to cosine similarity (-1 to 1), then normalized to (0-1)
   - Formula: `similarity = 1 - distance`, then `normalized = (similarity + 1) / 2`
5. **Response Generation**: The LLM generates an answer based on the retrieved context

### Hybrid Search (Optional)

When `--hybrid` is enabled, the system combines two search methods:

1. **Vector Search (Dense)**: Semantic similarity using embeddings (finds conceptually similar content)
2. **BM25 Search (Sparse)**: Keyword matching using the BM25 algorithm (finds exact term matches)
3. **Score Combination**: Both scores are normalized to 0-1 range using min-max normalization, then combined:
   - `combined_score = (hybrid_weight × vector_score) + ((1 - hybrid_weight) × bm25_score)`
4. **Result Merging**: Top results from both methods are merged and re-ranked by combined score
5. **Response Generation**: The LLM generates an answer based on the hybrid-retrieved context

**Benefits of Hybrid Search:**
- Better for queries with exact keywords (e.g., "Python", "REST API")
- Still maintains semantic understanding
- Combines the strengths of both approaches
- Configurable weighting between semantic and keyword matching

## Configuration

### Embedding Model

The embedding model is configured in both `create_database.py` and `query_data.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
```

You can change this to other models like:
- `sentence-transformers/all-mpnet-base-v2` (better quality, larger)
- `BAAI/bge-base-en-v1.5` (larger, better quality)

### Chunking Strategy

Chunking parameters in `create_database.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=150,     # Overlap between chunks
)
```

### ChromaDB Configuration

The database uses cosine distance for indexing (configured in `create_database.py`):

```python
collection_metadata={"hnsw:space": "cosine"}
```

**Note**: ChromaDB uses cosine **distance** (0-2 range, where lower = more similar), not cosine similarity. The query script automatically converts distance to similarity and normalizes it to a 0-1 range for easier interpretation.

## Score Normalization

The system converts cosine distance to normalized similarity scores:

1. **Cosine Distance** (from ChromaDB): Range 0-2, where:
   - 0 = identical vectors
   - 2 = opposite vectors
   - Lower distance = more similar

2. **Convert to Cosine Similarity**: `similarity = 1 - distance`
   - Range becomes -1 to 1
   - 1 = identical, 0 = orthogonal, -1 = opposite

3. **Normalize to 0-1**: `normalized = (similarity + 1) / 2`
   - Range becomes 0 to 1
   - 1 = most similar, 0 = least similar

## Features in Detail

### Hybrid Search

Hybrid search combines semantic (vector) and keyword (BM25) retrieval for improved results:

- **When to use**: Queries with specific terms, technical terminology, or when you need both semantic understanding and exact keyword matching
- **BM25 Index**: Built once during database creation and saved to disk for fast loading during queries
- **Normalization**: Both vector and BM25 scores use min-max normalization to ensure fair combination
- **Weighting**: Adjust `--hybrid-weight` to balance between semantic (1.0) and keyword (0.0) search

### Incremental Updates

The system automatically detects new documents and only adds them to the database, avoiding full rebuilds. Chunk IDs are generated based on source file, page number, and chunk index (format: `source:page:chunk_index`). The BM25 index is automatically rebuilt when new documents are added.

## Troubleshooting

### "Unable to find matching results"
- Try increasing `--k` to retrieve more documents
- Check that your database was created successfully
- Verify your query is related to the document content

### Ollama connection errors
- Ensure Ollama is running: `ollama serve`
- Verify the model is pulled: `ollama list`
- Pull the model if needed: `ollama pull llama3:8b`

### Low scores
- The embedding model might not match your domain
- Consider using a larger embedding model
- Adjust chunk size/overlap for better context

### BM25 index not found (hybrid search)
- Run `python create_database.py` to build the BM25 index
- Ensure the `chroma/` directory exists and contains `bm25_index.pkl` and `bm25_docs.pkl`
- The system will fall back to vector-only search if the index is missing

## Future Improvements

Potential enhancements:
- Multi-query retrieval for better coverage
- Reranking with cross-encoders
- Better prompt engineering
- Response streaming

