import os
import re
import pickle
import argparse
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

DATA_PATH = "data"
CHROMA_PATH = "chroma"
BM25_INDEX_PATH = os.path.join(CHROMA_PATH, "bm25_index.pkl")
BM25_DOCS_PATH = os.path.join(CHROMA_PATH, "bm25_docs.pkl")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clears the database"
    )
    args = parser.parse_args()

    if args.clear:
        if os.path.exists(CHROMA_PATH):
            print("Clearing database...")
            shutil.rmtree(CHROMA_PATH)
            os.makedirs(CHROMA_PATH)
    else:
        generate_database()

def load_documents():
    """
    Load documents from the data folder.
    Supports both PDF and markdown files.
    """
    documents = []
    
    # Load PDF files
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH, recursive=True)
    pdf_documents = pdf_loader.load()
    documents.extend(pdf_documents)
    print(f"Loaded {len(pdf_documents)} PDF document(s)")
    
    # Load markdown files
    md_loader = DirectoryLoader(DATA_PATH, glob="*.md", recursive=True)
    md_documents = md_loader.load()
    documents.extend(md_documents)
    print(f"Loaded {len(md_documents)} markdown document(s)")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def tokenize(text):
    # Simple tokenizer for testing purposes
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def build_bm25_index(chunks: list[Document]):
    """
    Build BM25 index from chunks and save to disk.
    Also saves the document content and metadata for lookup.
    """
    print("Building BM25 index...")
    
    tokenized_docs = [tokenize(chunk.page_content) for chunk in chunks]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    # Save the index
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25, f)
    
    # Save document content and metadata for lookup so we can map scores back to documents
    docs_data = [
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "id": chunk.metadata.get("id", "")
        }
        for chunk in chunks
    ]
    
    with open(BM25_DOCS_PATH, 'wb') as f:
        pickle.dump(docs_data, f)
    
    #print(f"Saved BM25 index to {BM25_INDEX_PATH}")
    #print(f"Saved {len(docs_data)} chunks references to {BM25_DOCS_PATH}")

def save_to_chroma(chunks: list[Document]):
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Get existing documents IDs
    existing_items = db.get()
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    chunks = calculate_chunk_ids(chunks)

    # Check for existing documents and add new ones
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

        all_docs_data = db.get()
        all_chunks = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(
                all_docs_data["documents"],
                all_docs_data.get("metadatas", [{}] * len(all_docs_data["documents"]))
            )
        ]
        build_bm25_index(all_chunks)
    else:
        print("No new documents to add")
        # Still rebuild index to ensure it's up to date
        all_docs_data = db.get()
        all_chunks = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(
                all_docs_data["documents"],
                all_docs_data.get("metadatas", [{}] * len(all_docs_data["documents"]))
            )
        ]
        build_bm25_index(all_chunks)

    print((f"Saved {len(chunks)} chunks to Chroma database at {CHROMA_PATH}"))

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/xxx.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    # Only works for PDF files at the moment

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def generate_database():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()