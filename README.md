# Simple RAG System

A simple, open-source Retrieval-Augmented Generation (RAG) system that allows you to query your documents using natural language. Built with LangChain, ChromaDB, and Ollama for a fully local solution.

## Features

- 📄 **Multi-format Support**: Loads PDF and Markdown files from your data directory
- 🔍 **Semantic Search**: Uses state-of-the-art embedding models for accurate document retrieval
- 🤖 **Local LLM**: Powered by Ollama for completely offline operation
- 💾 **Persistent Storage**: ChromaDB vector database for efficient similarity search
- 🔄 **Incremental Updates**: Automatically detects and adds new documents without rebuilding the entire database

## Tech Stack

- **Embeddings**: `BAAI/bge-small-en-v1.5` (HuggingFace) - High-quality, open-source embedding model
- **Vector Database**: ChromaDB with cosine distance indexing
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
pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama chromadb sentence-transformers pypdf "unstructured[md]"
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
├── create_database.py    # Script to build/update the vector database
├── query_data.py         # Script to query your documents
└── README.md
```

## Usage

### 1. Prepare Your Documents

Place your PDF and Markdown files in the `data/` directory. The system will recursively search all subdirectories.

```
data/
├── document1.pdf
├── document2.md
└── subfolder/
    └── document3.pdf
```

### 2. Create the Database

Run the database creation script to process and index your documents:

```bash
python create_database.py
```

This will:
- Load all PDF and Markdown files from the `data/` directory
- Split documents into chunks (1000 characters with 150 character overlap)
- Generate embeddings using BAAI/bge-small-en-v1.5
- Store everything in ChromaDB
- Only add new documents if the database already exists (incremental updates)

### 3. Query Your Documents

Ask questions about your documents:

```bash
python query_data.py "Your question here"
```

**Examples:**
```bash
# Basic query
python query_data.py "What type of evidence can we find in operating systems?"

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

## How It Works

1. **Document Processing**: Documents are split into manageable chunks (1000 chars with 150 char overlap)
2. **Embedding Generation**: Each chunk is converted to a vector using BAAI/bge-small-en-v1.5
3. **Vector Storage**: Embeddings are stored in ChromaDB
4. **Query Processing**: 
   - Your question is embedded using the same model
   - ChromaDB finds the most similar document chunks using cosine distance
   - Cosine distance (0-2 range) is converted to cosine similarity (-1 to 1), then normalized to (0-1)
   - Formula: `similarity = 1 - distance`, then `normalized = (similarity + 1) / 2`
5. **Response Generation**: The LLM generates an answer based on the retrieved context

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

### Incremental Updates

The system automatically detects new documents and only adds them to the database, avoiding full rebuilds. Chunk IDs are generated based on source file, page number, and chunk index (format: `source:page:chunk_index`).

### Recursive Document Loading

Both PDF and Markdown files are loaded recursively from subdirectories, so you can organize your documents in folders.

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

## Future Improvements

Potential enhancements:
- Multi-query retrieval for better coverage
- Reranking with cross-encoders
- Hybrid search (semantic + keyword)
- Better prompt engineering
- Response streaming

