from typing import Any


from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import os
import shutil

DATA_PATH = "data"
CHROMA_PATH = "chroma"

#has to be the same embedding model as the query_data.py
# Using BAAI/bge-small-en-v1.5 - better quality than MiniLM, same size
# Trained specifically for retrieval tasks, better semantic understanding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

def main():
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
        chunk_overlap=30,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def save_to_chroma(chunks: list[Document]):
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"}
    )

    existing_items = db.get()
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    chunks = calculate_chunk_ids(chunks)

    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

    print((f"Saved {len(chunks)} chunks to Chroma database at {CHROMA_PATH}"))

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/xxx.pdf:6:2"
    # Page Source : Page Number : Chunk Index

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