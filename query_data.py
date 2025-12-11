import argparse
import re
import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

CHROMA_PATH = "chroma"
BM25_INDEX_PATH = os.path.join(CHROMA_PATH, "bm25_index.pkl")
BM25_DOCS_PATH = os.path.join(CHROMA_PATH, "bm25_docs.pkl")

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context documents.

Use the following pieces of context to answer the question. If you don't know the answer based on the context alone, say that you don't have enough information in the provided documents to answer the question.

Context:
{context}

Question: {question}

Provide a clear, concise, and accurate answer based solely on the context provided above. If the context contains multiple relevant pieces of information, synthesize them into a coherent response. If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
"""

def distance_to_similarity(cosine_distance):
    """Convert cosine distance (0-2) to cosine similarity (-1 to 1), then normalize to (0-1)"""
    cosine_similarity = 1 - cosine_distance  # Convert distance to similarity
    normalized = (cosine_similarity + 1) / 2  # Normalize to 0-1
    return normalized

def tokenize(text):
    # Simple tokenizer for BM25: splits text into words
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3:8b", 
        help="Ollama model to use (default: llama3:8b)"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid search (combines vector (dense) + keyword (sparse) search)"
    )
    parser.add_argument(
        "--hybrid-weight",
        type=float,
        default=0.5,
        help="Weight for hybrid search: 0.0 = only keyword, 1.0 = only vector (default: 0.5)"
    )
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB - must use the same embedding model as create_database.py
    # Using BAAI/bge-small-en-v1.5 - better quality than MiniLM, same size
    # Trained specifically for retrieval tasks, better semantic understanding
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform search based on mode
    if args.hybrid:
        # Hybrid search: combine vector + keyword search
        results = hybrid_search(db, query_text, args.k, args.hybrid_weight)
    else:
        # Standard vector search
        raw_results = db.similarity_search_with_score(query_text, k=args.k)
        results = [
            (doc, distance_to_similarity(distance))
            for doc, distance in raw_results
        ]

    # Check if we got any results
    if len(results) == 0:
        return
    
    top_score = results[0][1]
    
    print(f"Top result score: {top_score:.4f}")

    # Format the retrieved documents as context to later use in the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(f"Context: {context_text}")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context= context_text, question= query_text)
    #print(f"Prompt: {prompt}")

    model = ChatOllama(model=args.model, temperature=0)
    response = model.invoke(prompt)
    sources = [doc.metadata["source"] for doc, _score in results]
    sources = set(sources) # Removes duplicates for cleaner output
    formatted_response = f"Response: {response.content}\nSources: {sources}"
    
    print(formatted_response)

def hybrid_search(db, query_text, k, hybrid_weight=0.5):
    """
    Perform hybrid search combining vector similarity and BM25 keyword search.
    Uses pre-built BM25 index from disk.
    
    Args:
        db: ChromaDB instance
        query_text: Query string
        k: Number of results to return
        hybrid_weight: Weight for vector search (0.0 = only BM25, 1.0 = only vector)
    
    Returns:
        List of (document, combined_score) tuples, sorted by score descending
    """
    # Load BM25 index and document references
    if not os.path.exists(BM25_INDEX_PATH) or not os.path.exists(BM25_DOCS_PATH):
        print("⚠️  BM25 index not found. Please run create_database.py first.")
        print("Falling back to vector search only.")
        raw_results = db.similarity_search_with_score(query_text, k=k)
        return [(doc, distance_to_similarity(score)) for doc, score in raw_results]
    
    print("Loading BM25 index...")
    with open(BM25_INDEX_PATH, 'rb') as f:
        bm25 = pickle.load(f)
    
    with open(BM25_DOCS_PATH, 'rb') as f:
        all_docs = pickle.load(f)
    
    # Perform BM25 search
    query_tokens = tokenize(query_text)
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Normalize BM25 scores to 0-1 range
    bm25_scores_list = bm25_scores.tolist() if hasattr(bm25_scores, 'tolist') else list(bm25_scores)
    if len(bm25_scores_list) == 0:
        bm25_scores_normalized = []
    else:
        min_bm25 = min(bm25_scores_list)
        max_bm25 = max(bm25_scores_list)
        
        if max_bm25 > min_bm25:
            # Min-max normalization: (score - min) / (max - min)
            bm25_scores_normalized = [
                (score - min_bm25) / (max_bm25 - min_bm25)
                for score in bm25_scores_list
            ]
        else:
            # All scores are the same, set all to 0.5 (or 1.0, your choice)
            bm25_scores_normalized = [1.0] * len(bm25_scores_list)
    
    # Create BM25 score map by document content
    bm25_scores_map = {
        doc["content"]: score
        for doc, score in zip(all_docs, bm25_scores_normalized)
    }
    
    # Perform vector search (get more results to merge with BM25)
    vector_results = db.similarity_search_with_score(query_text, k=k * 3)
    
    # Create vector scores map (normalized similarity scores)
    vector_scores_map = {
        doc.page_content: distance_to_similarity(score)
        for doc, score in vector_results
    }
    
    # Combine results: merge vector and BM25 scores
    combined_results = {}
    
    # Add all vector results
    for doc, raw_distance in vector_results:
        content = doc.page_content
        vector_score = distance_to_similarity(raw_distance)
        bm25_score = bm25_scores_map.get(content, 0.0)
        combined_score = (hybrid_weight * vector_score) + ((1 - hybrid_weight) * bm25_score)
        combined_results[content] = (doc, combined_score)
    
    # Add top BM25 results that might not be in vector results
    # Sort BM25 scores and get top k
    bm25_with_docs = list(zip(all_docs, bm25_scores_normalized))
    bm25_with_docs.sort(key=lambda x: x[1], reverse=True)
    
    for doc_data, bm25_score in bm25_with_docs[:k]:
        content = doc_data["content"]
        if content not in combined_results:
            doc = Document(
                page_content=content,
                metadata=doc_data["metadata"]
            )
            vector_score = vector_scores_map.get(content, 0.0)
            combined_score = (hybrid_weight * vector_score) + ((1 - hybrid_weight) * bm25_score)
            combined_results[content] = (doc, combined_score)
    
    # Convert to list and sort by combined score
    results = list(combined_results.values())
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return results[:k]
if __name__ == "__main__":
    main()